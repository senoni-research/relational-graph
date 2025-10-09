#!/usr/bin/env python3
"""Produce VN2 ordering features using learned scorer + optional calibration.

Outputs a CSV with columns:
  store_id,product_id,start_week,horizon_weeks,p_mean,p_sum,mu_hat_plus,sigma_hat_plus,mu_H,sigma_H

Where mu_H and sigma_H are the zero-inflated horizon mean/std using calibrated p(active) for
weeks [start_week, start_week + horizon-1].

Example:
  python -u scripts/orders_vn2.py \
    --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
    --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
    --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl \
    --train-end 2024-01-31 --start-week 20240318 --horizon 3 \
    --out orders_features.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
from datetime import date, timedelta
from typing import Dict, Tuple, Any

import networkx as nx
import torch

from graph_qa.io.loader import load_graph
from graph_qa.train.trainer import sample_subgraph_for_edge
from graph_qa.train.model import SimpleEdgeScorer
from graph_qa.train.model_v2 import EnhancedEdgeScorer


def yyyymmdd_to_date(yyyymmdd: int) -> date:
    y = yyyymmdd // 10000
    m = (yyyymmdd % 10000) // 100
    d = yyyymmdd % 100
    return date(y, m, d)


def next_monday(yyyymmdd: int) -> int:
    dt = yyyymmdd_to_date(yyyymmdd)
    days_ahead = (7 - dt.weekday()) % 7  # Monday=0
    if days_ahead == 0:
        days_ahead = 7
    nxt = dt + timedelta(days=days_ahead)
    return int(nxt.strftime("%Y%m%d"))


def build_model_from_ckpt(G: nx.Graph, ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    is_enhanced = "model_state" in ckpt and any(k.startswith("cat_embeds.") for k in ckpt["model_state"].keys())
    if is_enhanced:
        model = EnhancedEdgeScorer(
            node_types=ckpt["node_types"],
            categorical_attrs=ckpt.get("categorical_attrs", {}),
            hidden_dim=ckpt.get("hidden_dim", 64),
            num_layers=ckpt.get("num_layers", 3),
            recency_feature=ckpt.get("recency_feature", False),
            recency_norm=ckpt.get("recency_norm", 52.0),
        )
        model.fast_mode = ckpt.get("fast_mode", False)
        model.skip_hopdist = ckpt.get("skip_hopdist", False)
    else:
        model = SimpleEdgeScorer(
            node_types=ckpt["node_types"],
            hidden_dim=ckpt.get("hidden_dim", 64),
            num_layers=ckpt.get("num_layers", 2),
        )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


def score_edge(model: torch.nn.Module, G: nx.Graph, edge: Tuple[Any, Any, int], hops: int, K: int, calibrator: dict | None) -> float:
    sub = sample_subgraph_for_edge(G, edge, hops=hops, K=K)
    ev = (edge[0], edge[1], edge[2])
    with torch.no_grad():
        logit = model(sub, [ev]).squeeze(0)
        raw_logit = float(logit.item())
        prob = 1.0 / (1.0 + math.exp(-raw_logit))
        if calibrator is not None:
            method = calibrator.get("method", "isotonic")
            input_space = calibrator.get("input", "prob")
            model_obj = calibrator["model"]
            if input_space == "logit":
                X = [raw_logit]
            else:
                X = [prob]
            if hasattr(model_obj, "transform"):
                prob = float(model_obj.transform(X)[0])
            else:
                import numpy as np
                prob = float(model_obj.predict_proba(np.array(X).reshape(-1, 1))[0, 1])
        return float(prob)


def compute_mu_sigma_plus(G: nx.Graph, train_end: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Estimate per-product mu_plus, sigma_plus from training window (units>0 on 'sold')
    per_product_values: Dict[str, list] = {}
    for u, v, key, attrs in G.edges(data=True, keys=True):
        if attrs.get("rel") != "sold":
            continue
        t = int(attrs.get("time", 0))
        if t > train_end:
            continue
        try:
            units = float(attrs.get("units", 0.0))
        except Exception:
            units = 0.0
        if units <= 0.0:
            continue
        # identify product node
        prod = u if str(G.nodes[u].get("type")) == "product" else v
        per_product_values.setdefault(prod, []).append(units)

    mu_plus: Dict[str, float] = {}
    sigma_plus: Dict[str, float] = {}
    global_vals = [x for vals in per_product_values.values() for x in vals]
    global_mu = sum(global_vals) / max(1, len(global_vals)) if global_vals else 1.0
    global_var = (sum((x - global_mu) ** 2 for x in global_vals) / max(1, len(global_vals))) if global_vals else 1.0
    global_sigma = math.sqrt(max(1e-6, global_var))
    for prod, vals in per_product_values.items():
        if len(vals) >= 2:
            mu = sum(vals) / len(vals)
            var = sum((x - mu) ** 2 for x in vals) / len(vals)
            mu_plus[prod] = float(mu)
            sigma_plus[prod] = float(math.sqrt(max(1e-6, var)))
        elif len(vals) == 1:
            mu_plus[prod] = float(vals[0])
            sigma_plus[prod] = float(global_sigma)
    return mu_plus, sigma_plus


def main():
    ap = argparse.ArgumentParser(description="Generate VN2 order features from scorer + calibration")
    ap.add_argument("--graph", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--calibrator", default=None, type=str)
    ap.add_argument("--train-end", default="2024-01-31", type=str)
    ap.add_argument("--start-week", default=None, type=str, help="YYYYMMDD start week (default: next Monday after train_end)")
    ap.add_argument("--horizon", default=3, type=int)
    ap.add_argument("--hops", default=1, type=int)
    ap.add_argument("--K", default=30, type=int)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    print(f"Loading graph from {args.graph}...")
    G = load_graph(args.graph, multi=True)

    print(f"Loading model from {args.ckpt}...")
    model = build_model_from_ckpt(G, args.ckpt)

    calibrator = None
    if args.calibrator:
        with open(args.calibrator, "rb") as f:
            cal = pickle.load(f)
        if isinstance(cal, dict) and "model" in cal:
            calibrator = cal
        else:
            calibrator = {"method": "isotonic", "input": "prob", "model": cal}

    # Parse dates
    train_end_int = int(str(args.train_end).replace("-", ""))
    if args.start_week:
        start_week = int(str(args.start_week).replace("-", ""))
    else:
        start_week = next_monday(train_end_int)

    # Collect nodes
    store_nodes = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "store"]
    product_nodes = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "product"]
    print(f"Stores: {len(store_nodes)}, Products: {len(product_nodes)}")

    # Per-product mu+/sigma+ from training window
    mu_plus, sigma_plus = compute_mu_sigma_plus(G, train_end_int)

    rows = []
    for s in store_nodes:
        for p in product_nodes:
            # Skip if nodes not present (already filtered) or edge impossible by graph type
            # Compute p(active) for each week in horizon
            week = start_week
            p_list = []
            for _ in range(int(args.horizon)):
                prob = score_edge(model, G, (s, p, week), hops=args.hops, K=args.K, calibrator=calibrator)
                p_list.append(prob)
                # advance by 7 days
                d = yyyymmdd_to_date(week) + timedelta(days=7)
                week = int(d.strftime("%Y%m%d"))

            prod_mu = mu_plus.get(p, mu_plus.get(p, 1.0))
            prod_sigma = sigma_plus.get(p, sigma_plus.get(p, 1.0))

            # Zero-inflated mixture across horizon
            mu_H = 0.0
            var_H = 0.0
            for prob in p_list:
                mu_H += prob * prod_mu
                var_H += prob * (prod_sigma ** 2) + prob * (1 - prob) * (prod_mu ** 2)

            rows.append({
                "store_id": s.replace("store:", ""),
                "product_id": p.replace("product:", ""),
                "start_week": start_week,
                "horizon_weeks": int(args.horizon),
                "p_mean": sum(p_list) / len(p_list),
                "p_sum": sum(p_list),
                "mu_hat_plus": prod_mu,
                "sigma_hat_plus": prod_sigma,
                "mu_H": mu_H,
                "sigma_H": math.sqrt(max(1e-6, var_H)),
            })

    print(f"Writing {len(rows)} rows to {args.out}...")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Done.")


if __name__ == "__main__":
    main()


