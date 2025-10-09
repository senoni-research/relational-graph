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
    ap.add_argument("--submission-index", default=None, type=str, help="599-row Store×Product index CSV (store_id,product_id) in platform order")
    ap.add_argument("--state", default=None, type=str, help="CSV with columns store_id,product_id,onhand,onorder_le2")
    ap.add_argument("--submit", default=None, type=str, help="If provided, write 599-row orders.csv here with order_qty column")
    ap.add_argument("--beta", default=0.833, type=float, help="Critical fractile (default 1/(1+0.2))")
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

    # Optional: produce 599-row submission file in platform order
    if args.submission_index and args.submit:
        print(f"Producing submission from index {args.submission_index} -> {args.submit}")
        # Load features into a lookup
        feat_map: Dict[Tuple[str, str], Dict[str, float]] = {}
        for r in rows:
            key = (str(r["store_id"]), str(r["product_id"]))
            feat_map[key] = r

        # Load state if provided
        state_map: Dict[Tuple[str, str], Tuple[float, float]] = {}
        if args.state:
            with open(args.state, "r") as sf:
                sr = csv.DictReader(sf)
                for srow in sr:
                    key = (str(srow.get("store_id", "")), str(srow.get("product_id", "")))
                    try:
                        onhand = float(srow.get("onhand", 0) or 0)
                        onorder = float(srow.get("onorder_le2", 0) or 0)
                    except Exception:
                        onhand, onorder = 0.0, 0.0
                    state_map[key] = (onhand, onorder)

        # Load index and compute orders
        submissions: list[Dict[str, Any]] = []
        with open(args.submission_index, "r") as idxf:
            idxr = csv.DictReader(idxf)
            for idxrow in idxr:
                sid = str(idxrow.get("store_id", ""))
                pid = str(idxrow.get("product_id", ""))
                key = (sid, pid)
                feat = feat_map.get(key)
                if not feat:
                    # Missing feature row: default to zero order
                    mu_H = 0.0
                    sigma_H = 0.0
                else:
                    mu_H = float(feat["mu_H"])  # horizon mean
                    sigma_H = float(feat["sigma_H"])  # horizon std
                # Critical fractile to z (approx without scipy)
                # For beta=0.833, z ≈ 0.967. For general beta, use probit approximation.
                beta = max(1e-6, min(1 - 1e-6, float(args.beta)))
                # Abramowitz-Stegun approximation for inverse normal CDF
                def inv_norm_cdf(p: float) -> float:
                    # Source: Wichura algorithm approximation (simplified)
                    # Good enough for beta near 0.8–0.9
                    a1 = -39.69683028665376
                    a2 = 220.9460984245205
                    a3 = -275.9285104469687
                    a4 = 138.3577518672690
                    a5 = -30.66479806614716
                    a6 = 2.506628277459239
                    b1 = -54.47609879822406
                    b2 = 161.5858368580409
                    b3 = -155.6989798598866
                    b4 = 66.80131188771972
                    b5 = -13.28068155288572
                    c1 = -0.007784894002430293
                    c2 = -0.3223964580411365
                    c3 = -2.400758277161838
                    c4 = -2.549732539343734
                    c5 = 4.374664141464968
                    c6 = 2.938163982698783
                    d1 = 0.007784695709041462
                    d2 = 0.3224671290700398
                    d3 = 2.445134137142996
                    d4 = 3.754408661907416
                    plow = 0.02425
                    phigh = 1 - plow
                    if p < plow:
                        q = math.sqrt(-2 * math.log(p))
                        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
                            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
                        )
                    if p > phigh:
                        q = math.sqrt(-2 * math.log(1 - p))
                        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
                            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
                        )
                    q = p - 0.5
                    r = q * q
                    return (
                        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
                    ) / (
                        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
                    )

                z = inv_norm_cdf(beta)
                S = mu_H + z * sigma_H
                onhand, onorder = state_map.get(key, (0.0, 0.0))
                IP = onhand + onorder
                q = max(0.0, S - IP)
                submissions.append({"store_id": sid, "product_id": pid, "order_qty": int(round(q))})

        # Write submission
        with open(args.submit, "w", newline="") as sf:
            writer = csv.DictWriter(sf, fieldnames=["store_id", "product_id", "order_qty"])
            writer.writeheader()
            writer.writerows(submissions)
        print(f"Wrote submission {args.submit} with {len(submissions)} rows.")


if __name__ == "__main__":
    main()


