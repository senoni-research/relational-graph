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
import os
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

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
        # Match training feature config (including optional ID embeddings)
        store_ids = [str(n).replace("store:", "") for n, a in G.nodes(data=True) if str(a.get("type")) == "store"]
        product_ids = [str(n).replace("product:", "") for n, a in G.nodes(data=True) if str(a.get("type")) == "product"]
        model = EnhancedEdgeScorer(
            node_types=ckpt.get("node_types"),
            categorical_attrs=ckpt.get("categorical_attrs", {}),
            hidden_dim=ckpt.get("hidden_dim", 64),
            num_layers=ckpt.get("num_layers", 3),
            recency_feature=ckpt.get("recency_feature", False),
            recency_norm=ckpt.get("recency_norm", 52.0),
            rel_aware_attn=ckpt.get("rel_aware_attn", False),
            event_buckets=ckpt.get("event_buckets", None),
            store_ids=store_ids,
            product_ids=product_ids,
            id_emb_dim=ckpt.get("id_emb_dim", 16),
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


def score_edge(model: torch.nn.Module, G: nx.Graph, edge: Tuple[Any, Any, int], hops: int, K: int, calibrator: dict | None, calibrators_by_key: dict | None = None, seg_cols: list[str] | None = None) -> float:
    sub = sample_subgraph_for_edge(G, edge, hops=hops, K=K)
    ev = (edge[0], edge[1], edge[2])
    with torch.no_grad():
        logit = model(sub, [ev]).squeeze(0)
        raw_logit = float(logit.item())
        prob = 1.0 / (1.0 + math.exp(-raw_logit))
        if calibrator is not None or calibrators_by_key is not None:
            # Default calibrator
            method = calibrator.get("method", "isotonic") if calibrator else "isotonic"
            input_space = calibrator.get("input", "prob") if calibrator else "prob"
            model_obj = calibrator["model"] if calibrator else None
            # Segment-aware override
            if calibrators_by_key is not None and seg_cols:
                s, p = edge[0], edge[1]
                def _get(n, col):
                    try:
                        return str(G.nodes[n].get(col, ""))
                    except Exception:
                        return ""
                key = tuple(_get(s, c) for c in seg_cols)
                seg_cal = calibrators_by_key.get(key)
                if seg_cal is not None:
                    method = seg_cal.get("method", method)
                    input_space = seg_cal.get("input", input_space)
                    model_obj = seg_cal.get("model", model_obj)
            # Apply
            if model_obj is not None:
                X = [raw_logit] if input_space == "logit" else [prob]
                if hasattr(model_obj, "transform"):
                    prob = float(model_obj.transform(X)[0])
                else:
                    import numpy as np
                    prob = float(model_obj.predict_proba(np.array(X).reshape(-1, 1))[0, 1])
        return float(prob)


# === Sanity / Grid helpers ===
SQRT2PI = math.sqrt(2.0 * math.pi)

def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / SQRT2PI

def _Phi(z: np.ndarray) -> np.ndarray:
    # Normal CDF via math.erf; vectorized for numpy arrays (SciPy-free)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def _compute_mu_sigma_H_from_rows(rows: list[dict], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    # rows: list of feature dicts with p_t1..p_t3, mu_hat_plus, sigma_hat_plus
    df = {k: [] for k in ("mu_hat_plus","sigma_hat_plus","p_t1","p_t2","p_t3")}
    for r in rows:
        df["mu_hat_plus"].append(float(r.get("mu_hat_plus", 1.0)))
        df["sigma_hat_plus"].append(float(r.get("sigma_hat_plus", 1.0)))
        df["p_t1"].append(float(r.get("p_t1", r.get("p_mean", 0.5))))
        df["p_t2"].append(float(r.get("p_t2", r.get("p_mean", 0.5))))
        df["p_t3"].append(float(r.get("p_t3", r.get("p_mean", 0.5))))
    mu_plus = np.asarray(df["mu_hat_plus"], dtype=float)
    sigma_plus = np.asarray(df["sigma_hat_plus"], dtype=float)
    p1 = np.asarray(df["p_t1"], dtype=float)
    p2 = np.asarray(df["p_t2"], dtype=float)
    p3 = np.asarray(df["p_t3"], dtype=float)
    p_weeks = [p1, p2, p3]
    if horizon > 3:
        p_weeks = p_weeks + [p3] * (horizon - 3)
    p_weeks = p_weeks[:horizon]
    lam = [pw * mu_plus for pw in p_weeks]
    var = [pw * (sigma_plus ** 2) + pw * (1.0 - pw) * (mu_plus ** 2) for pw in p_weeks]
    mu_H = np.sum(lam, axis=0)
    var_H = np.sum(var, axis=0)
    sigma_H = np.sqrt(np.maximum(1e-12, var_H))
    return mu_H, sigma_H

def _expected_cost(mu_H: np.ndarray, sigma_H: np.ndarray, S: np.ndarray, cost_shortage: float, cost_holding_per_week: float, horizon: int) -> float:
    z = (S - mu_H) / sigma_H
    small = sigma_H < 1e-8
    under = sigma_H * _phi(z) + (mu_H - S) * (1.0 - _Phi(z))
    over  = sigma_H * _phi(z) + (S - mu_H) * _Phi(z)
    under[small] = np.maximum(0.0, mu_H[small] - S[small])
    over[small]  = np.maximum(0.0, S[small] - mu_H[small])
    Cu = float(cost_shortage)
    CoH = float(cost_holding_per_week) * float(horizon)
    return float(np.sum(Cu * under + CoH * over))

def _expected_cost_over_df(df: pd.DataFrame, q: np.ndarray, cost_shortage: float, cost_holding_per_week: float, horizon: int) -> float:
    # Build arrays; tolerate missing onhand/onorder by using zeros
    mu_H_arr = pd.to_numeric(df.get("mu_H", 0.0), errors="coerce").fillna(0.0).to_numpy()
    sigma_H_arr = pd.to_numeric(df.get("sigma_H", 1e-6), errors="coerce").fillna(1e-6).to_numpy()
    onhand_series = df["onhand"] if "onhand" in df.columns else pd.Series(0.0, index=df.index)
    onorder_series = df["onorder_le2"] if "onorder_le2" in df.columns else pd.Series(0.0, index=df.index)
    onhand = pd.to_numeric(onhand_series, errors="coerce").fillna(0.0).to_numpy()
    onorder = pd.to_numeric(onorder_series, errors="coerce").fillna(0.0).to_numpy()
    S = onhand + onorder + np.asarray(q, dtype=float)
    return _expected_cost(mu_H_arr, sigma_H_arr, S, cost_shortage, cost_holding_per_week, horizon)

def _expected_cost_mixture_over_df(df: pd.DataFrame, q: np.ndarray, cost_shortage: float, cost_holding_per_week: float, horizon: int) -> float:
    """Row-wise mixture expected cost using p_t3/p_mean and mu_hat_plus/sigma_hat_plus."""
    mu_pos = pd.to_numeric(df.get("mu_hat_plus", 0.0), errors="coerce").fillna(0.0).to_numpy()
    sigma_pos = pd.to_numeric(df.get("sigma_hat_plus", 1e-6), errors="coerce").fillna(1e-6).to_numpy()
    p = pd.to_numeric(df.get("p_t3", df.get("p_mean", 0.5)), errors="coerce").fillna(0.5).to_numpy()
    onhand_series = df["onhand"] if "onhand" in df.columns else pd.Series(0.0, index=df.index)
    onorder_series = df["onorder_le2"] if "onorder_le2" in df.columns else pd.Series(0.0, index=df.index)
    onhand = pd.to_numeric(onhand_series, errors="coerce").fillna(0.0).to_numpy()
    onorder = pd.to_numeric(onorder_series, errors="coerce").fillna(0.0).to_numpy()
    S = onhand + onorder + np.asarray(q, dtype=float)
    total = 0.0
    for i in range(len(S)):
        total += _expected_cost_mixture(float(S[i]), float(p[i]), float(mu_pos[i]), float(sigma_pos[i]), float(cost_shortage), float(cost_holding_per_week), int(horizon))
    return float(total)

# Fast inverse normal CDF approximation (Abramowitz-Stegun style)
def inv_norm_cdf(p: float) -> float:
    p = float(max(1e-12, min(1-1e-12, p)))
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

def _p_active_from_weeks(p_weeks: list[float]) -> float:
    prod_inactive = 1.0
    for pw in p_weeks:
        prod_inactive *= max(0.0, min(1.0, float(pw)))
    return 1.0 - prod_inactive

def _expected_cost_mixture(S: float, p_active: float, mu_pos: float, sigma_pos: float, cost_shortage: float, cost_holding_per_week: float, horizon: int) -> float:
    # Mixture: (1-p) * delta0 + p * N(mu_pos, sigma_pos^2)
    # Expected shortage/overage at stock level S
    p = max(0.0, min(1.0, float(p_active)))
    sigma = max(1e-8, float(sigma_pos))
    mu = float(mu_pos)
    z = (S - mu) / sigma
    # Normal parts
    phi = math.exp(-0.5 * z * z) / SQRT2PI
    # Use math.erf for Phi
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    e_short_pos = sigma * phi + (mu - S) * (1.0 - Phi)
    e_over_pos  = sigma * phi + (S - mu) * Phi
    # Mixture expectations
    e_short = p * e_short_pos
    e_over  = (1.0 - p) * max(0.0, S) + p * e_over_pos
    Cu = float(cost_shortage)
    CoH = float(cost_holding_per_week) * float(horizon)
    return Cu * e_short + CoH * e_over

def _mixture_base_stock(p_active: float, mu_pos: float, sigma_pos: float, cost_shortage: float, cost_holding_per_week: float, horizon: int) -> float:
    # Newsvendor on mixture: order 0 if p <= 1-beta; else quantile within positive component at u = (beta - (1-p))/p
    p = max(0.0, min(1.0, float(p_active)))
    # Convert costs to horizon-equivalent beta
    beta = float(cost_shortage) / (float(cost_shortage) + float(cost_holding_per_week) * float(horizon))
    if p <= max(0.0, 1.0 - beta):
        return 0.0
    u = (beta - (1.0 - p)) / max(p, 1e-8)
    u = min(max(u, 1e-6), 1.0 - 1e-6)
    z = inv_norm_cdf(u)
    return float(max(0.0, float(mu_pos) + float(sigma_pos) * z))

def _sanity_print(submissions: list[dict], hb_map: Dict[Tuple[str,str], int], k: int = 50) -> None:
    try:
        p_arr = np.array([float(s.get("p_t3", 0.0)) for s in submissions])
        hb_arr = np.array([hb_map.get((s["store_id"], s["product_id"]), 0) for s in submissions], dtype=float)
        bl_arr = np.array([s.get("order_qty_blend", s["order_qty"]) for s in submissions], dtype=float)
        top_idx = np.argsort(-p_arr)[:k]
        bot_idx = np.argsort(p_arr)[:k]
        top_keep = float((bl_arr[top_idx] >= hb_arr[top_idx]).mean()) if len(top_idx) else 0.0
        bot_shrink = float((bl_arr[bot_idx] < hb_arr[bot_idx]).mean()) if len(bot_idx) else 0.0
        print(f"Sanity: top-{k} keep-rate={top_keep:.2f}, bottom-{k} shrink-rate={bot_shrink:.2f}")
    except Exception as e:
        print(f"[warn] sanity calc failed: {e}")


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


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _merge_state(df: pd.DataFrame, state_path: str) -> pd.DataFrame:
    """Merge state (onhand, onorder) into df for cost calculations and order sizing."""
    st = pd.read_csv(state_path)
    cols = {c.lower(): c for c in st.columns}
    sid = cols.get("store_id") or cols.get("store")
    pid = cols.get("product_id") or cols.get("product")
    onh = cols.get("onhand") or cols.get("end inventory") or cols.get("inventory")
    it1 = cols.get("in transit w+1")
    it2 = cols.get("in transit w+2")
    st = st.rename(columns={sid: "store_id", pid: "product_id"})
    st["store_id"] = st["store_id"].astype(str)
    st["product_id"] = st["product_id"].astype(str)
    st["onhand"] = pd.to_numeric(st.get(onh, 0), errors="coerce").fillna(0.0)
    st["onorder_le2"] = (
        pd.to_numeric(st.get(it1, 0), errors="coerce").fillna(0.0) +
        pd.to_numeric(st.get(it2, 0), errors="coerce").fillna(0.0)
    )
    out = df.merge(st[["store_id", "product_id", "onhand", "onorder_le2"]],
                   on=["store_id", "product_id"], how="left")
    out["onhand"] = out["onhand"].fillna(0.0)
    out["onorder_le2"] = out["onorder_le2"].fillna(0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate VN2 order features from scorer + calibration")
    ap.add_argument("--graph", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--calibrator", default=None, type=str)
    ap.add_argument("--calibrator-dir", default=None, type=str, help="Directory with per-segment calibrators")
    ap.add_argument("--segment-cols", default=None, type=str, help="Comma-separated node attrs to segment (e.g., 'Region,Format')")
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
    ap.add_argument("--features-599", default=None, type=str, help="If provided, write 599-row features aligned to index (includes p_t1..p_t3)")
    ap.add_argument("--blend", default="none", choices=["none","hb","graph","gated","gated2","shrink","cap","mixture","mixture_min"], help="Blend submission strategy")
    ap.add_argument("--hb", default=None, type=str, help="Path to HB submission CSV (store_id,product_id,order_qty)")
    ap.add_argument("--tau", default=0.6, type=float, help="Gating threshold on p_t3 for blend=gated")
    ap.add_argument("--alpha", default=0.3, type=float, help="Shrink factor on HB when p<tau for blend=gated")
    ap.add_argument("--submit-blended", default=None, type=str, help="Path to write blended submission (599 rows)")
    # Mixture policy options
    ap.add_argument("--cost-shortage-mixture", type=float, default=None, help="Override shortage cost per unit for mixture policy (defaults to --cost-shortage)")
    ap.add_argument("--cost-holding-mixture", type=float, default=None, help="Override holding cost per unit per week for mixture policy (defaults to --cost-holding)")
    # Two-sided gating + ABC flags
    ap.add_argument("--tau-hi", type=float, default=None, help="High-probability threshold for two-sided gate (gated2)")
    ap.add_argument("--tau-lo", type=float, default=None, help="Low-probability threshold for two-sided gate (gated2)")
    ap.add_argument("--tau-margin", type=float, default=0.10, help="If --tau-lo is not set, use tau_hi - tau_margin")
    ap.add_argument("--abc-quantiles", type=str, default="0.6,0.9", help="Quantiles for A/B/C split on mu_H (e.g., '0.6,0.9')")
    ap.add_argument("--beta-a", type=float, default=0.88, help="Service level beta for A class")
    ap.add_argument("--beta-b", type=float, default=0.80, help="Service level beta for B class")
    ap.add_argument("--beta-c", type=float, default=0.70, help="Service level beta for C class")
    # Grid and cost-sim flags
    ap.add_argument("--grid", action="store_true", help="Run tau/alpha grid and pick best by expected cost or sanity score")
    ap.add_argument("--tau-grid", type=str, default=None, help="Comma-separated tau grid, e.g. '0.50,0.55,0.60'")
    ap.add_argument("--alpha-grid", type=str, default=None, help="Comma-separated alpha grid, e.g. '0.30,0.50,0.70'")
    ap.add_argument("--simulate-cost", action="store_true", help="Use expected-cost proxy (shortage=1, holding per week as given) when running --grid")
    ap.add_argument("--cost-shortage", type=float, default=1.0, help="Shortage cost per unit for grid cost proxy")
    ap.add_argument("--cost-holding", type=float, default=0.2, help="Holding cost per unit per week for grid cost proxy")
    ap.add_argument("--expected-cost-mode", type=str, default="normal", choices=["normal","mixture"], help="Which expected cost model to use for --simulate-cost and gated2 evaluation")
    ap.add_argument("--sanity-k", type=int, default=50, help="Top/Bottom K for sanity keep/shrink checks")
    args = ap.parse_args()

    print(f"Loading graph from {args.graph}...")
    G = load_graph(args.graph, multi=True)

    print(f"Loading model from {args.ckpt}...")
    model = build_model_from_ckpt(G, args.ckpt)

    calibrator = None
    calibrators_by_key = None
    seg_cols = [s.strip() for s in args.segment_cols.split(",")] if args.segment_cols else []
    if args.calibrator and not args.calibrator_dir:
        with open(args.calibrator, "rb") as f:
            cal = pickle.load(f)
        calibrator = cal if (isinstance(cal, dict) and "model" in cal) else {"method": "isotonic", "input": "prob", "model": cal}
    elif args.calibrator_dir:
        calibrators_by_key = {}
        for fn in os.listdir(args.calibrator_dir):
            if not fn.endswith((".pkl", ".pickle")):
                continue
            path = os.path.join(args.calibrator_dir, fn)
            try:
                with open(path, "rb") as f:
                    cal = pickle.load(f)
                key = cal.get("key")
                if key is not None:
                    calibrators_by_key[tuple(key)] = cal
            except Exception:
                continue

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
            p_weeks = []
            for _ in range(int(args.horizon)):
                prob = score_edge(model, G, (s, p, week), hops=args.hops, K=args.K, calibrator=calibrator, calibrators_by_key=calibrators_by_key, seg_cols=seg_cols)
                p_list.append(prob)
                p_weeks.append(prob)
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

            row = {
                "store_id": s.replace("store:", ""),
                "product_id": p.replace("product:", ""),
                "start_week": start_week,
                "horizon_weeks": int(args.horizon),
                "p_mean": sum(p_list) / len(p_list),
                "p_sum": sum(p_list),
                "p_t1": p_weeks[0] if len(p_weeks) > 0 else 0.0,
                "p_t2": p_weeks[1] if len(p_weeks) > 1 else (p_weeks[0] if p_weeks else 0.0),
                "p_t3": p_weeks[2] if len(p_weeks) > 2 else (p_weeks[-1] if p_weeks else 0.0),
                "mu_hat_plus": prod_mu,
                "sigma_hat_plus": prod_sigma,
                "mu_H": mu_H,
                "sigma_H": math.sqrt(max(1e-6, var_H)),
            }
            rows.append(row)

    print(f"Writing {len(rows)} rows to {args.out}...")
    _ensure_parent_dir(args.out)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Done.")

    # Optional: produce 599-row submission file in platform order
    # Optionally write 599-row features aligned to index
    if args.submission_index and args.features_599:
        print(f"Writing 599-row features to {args.features_599}")
        _ensure_parent_dir(args.features_599)
        feat_map: Dict[Tuple[str, str], Dict[str, float]] = {}
        for r in rows:
            feat_map[(str(r["store_id"]), str(r["product_id"]))] = r
        with open(args.submission_index, "r") as idxf, open(args.features_599, "w", newline="") as outf:
            idxr = csv.DictReader(idxf)
            fieldnames = ["store_id","product_id","p_t1","p_t2","p_t3","mu_hat_plus","sigma_hat_plus","mu_H","sigma_H"]
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            count = 0
            for idxrow in idxr:
                sid = str(idxrow.get("store_id", ""))
                pid = str(idxrow.get("product_id", ""))
                feat = feat_map.get((sid, pid), {})
                writer.writerow({
                    "store_id": sid,
                    "product_id": pid,
                    "p_t1": feat.get("p_t1", 0.0),
                    "p_t2": feat.get("p_t2", 0.0),
                    "p_t3": feat.get("p_t3", 0.0),
                    "mu_hat_plus": feat.get("mu_hat_plus", 0.0),
                    "sigma_hat_plus": feat.get("sigma_hat_plus", 0.0),
                    "mu_H": feat.get("mu_H", 0.0),
                    "sigma_H": feat.get("sigma_H", 0.0),
                })
                count += 1
        print(f"Wrote {count} feature rows to {args.features_599}")

    # === τ/α grid (if requested) — skip for gated2 (two-sided policy has its own grid) ===
    if args.grid and args.blend != "gated2":
        if not args.hb:
            raise SystemExit("--grid requires --hb")
        if not args.features_599:
            raise SystemExit("--grid requires --features-599")
        print(f"Running τ/α grid... (expected-cost-mode={args.expected_cost_mode})")
        tau_list = [float(x.strip()) for x in (args.tau_grid.split(",") if args.tau_grid else ["0.50","0.55","0.60"])]
        alpha_list = [float(x.strip()) for x in (args.alpha_grid.split(",") if args.alpha_grid else ["0.30","0.50","0.70"])]
        # Load features and HB for grid
        feat = pd.read_csv(args.features_599)
        hb = pd.read_csv(args.hb)
        for d in (feat, hb):
            cols = {c.lower(): c for c in d.columns}
            sid = cols.get("store_id", list(d.columns)[0])
            pid = cols.get("product_id", list(d.columns)[1])
            d.rename(columns={sid:"store_id", pid:"product_id"}, inplace=True)
            d["store_id"] = d["store_id"].astype(str)
            d["product_id"] = d["product_id"].astype(str)
        hb_qty_col = next((c for c in hb.columns if c.lower() in {"order_qty","orders","qty","0"}), hb.columns[-1])
        df = feat.merge(hb[["store_id","product_id", hb_qty_col]], on=["store_id","product_id"], how="left")
        hbq = pd.to_numeric(df[hb_qty_col], errors="coerce").fillna(0.0).to_numpy()
        # Merge state for cost calculations
        if args.state:
            df = _merge_state(df, args.state)
        pcol = "p_t3" if "p_t3" in df.columns else df.columns[2]
        p = pd.to_numeric(df[pcol], errors="coerce").fillna(0.5).to_numpy()
        
        best_tau, best_alpha, best_score = None, None, float("inf") if args.simulate_cost else -float("inf")
        print("τ/α grid:")
        for tau in tau_list:
            logs = []
            for alpha in alpha_list:
                w = np.where(p >= tau, 1.0, alpha)
                q = np.rint(np.clip(w * hbq, 0.0, None)).astype(int)
                if args.simulate_cost:
                    # Compute expected cost using chosen mode
                    if args.expected_cost_mode == "mixture":
                        cost = _expected_cost_mixture_over_df(df, q, float(args.cost_shortage), float(args.cost_holding), int(args.horizon))
                    else:
                        if "mu_H" in df.columns and "sigma_H" in df.columns:
                            mu_H_arr = pd.to_numeric(df["mu_H"], errors="coerce").fillna(0.0).to_numpy()
                            sigma_H_arr = pd.to_numeric(df["sigma_H"], errors="coerce").fillna(1e-6).to_numpy()
                        else:
                            mu_H_arr, sigma_H_arr = _compute_mu_sigma_H_from_rows(rows, args.horizon)
                        # Inventory position components (default to zeros if not provided)
                        onhand_series = df["onhand"] if "onhand" in df.columns else pd.Series(0.0, index=df.index)
                        onorder_series = df["onorder_le2"] if "onorder_le2" in df.columns else pd.Series(0.0, index=df.index)
                        onhand = pd.to_numeric(onhand_series, errors="coerce").fillna(0.0).to_numpy()
                        onorder = pd.to_numeric(onorder_series, errors="coerce").fillna(0.0).to_numpy()
                        cost = _expected_cost(mu_H_arr, sigma_H_arr, onhand + onorder + q, args.cost_shortage, args.cost_holding, args.horizon)
                    score = cost
                    logs.append(f"α={alpha:.2f} cost={cost:.2f}")
                    better = cost < best_score
                else:
                    top_idx = np.argsort(-p)[:args.sanity_k]
                    bot_idx = np.argsort(p)[:args.sanity_k]
                    keep_rate = float(np.mean(q[top_idx] >= hbq[top_idx]))
                    shrink_rate = float(np.mean(q[bot_idx] <= hbq[bot_idx]))
                    score = keep_rate + shrink_rate
                    logs.append(f"α={alpha:.2f} score={score:.3f} (keep={keep_rate:.2f}, shrink={shrink_rate:.2f})")
                    better = score > best_score
                if better:
                    best_score, best_tau, best_alpha = score, tau, alpha
            print(f"  τ={tau:.2f} -> " + " | ".join(logs))
        print(f"Best: τ={best_tau:.2f}, α={best_alpha:.2f}, {'cost' if args.simulate_cost else 'score'}={best_score:.3f}")
        
        # Write blended with best params
        if args.submit_blended:
            w = np.where(p >= best_tau, 1.0, best_alpha)
            if args.blend == "cap":
                # cap: low-p -> min(HB, Graph); high-p -> HB
                # Compute graph orders from features (base-stock)
                mu_H = pd.to_numeric(df.get("mu_H", 0.0), errors="coerce").fillna(0.0).to_numpy()
                sigma_H = pd.to_numeric(df.get("sigma_H", 1e-6), errors="coerce").fillna(1e-6).to_numpy()
                beta = 0.833
                z = inv_norm_cdf(beta)
                S = mu_H + z * sigma_H
                onhand_series = df["onhand"] if "onhand" in df.columns else pd.Series(0.0, index=df.index)
                onorder_series = df["onorder_le2"] if "onorder_le2" in df.columns else pd.Series(0.0, index=df.index)
                onhand = pd.to_numeric(onhand_series, errors="coerce").fillna(0.0).to_numpy()
                onorder = pd.to_numeric(onorder_series, errors="coerce").fillna(0.0).to_numpy()
                IP = onhand + onorder
                q_graph = np.rint(np.clip(S - IP, 0.0, None)).astype(int)
                # Apply cap: high-p -> HB; low-p -> min(HB, Graph)
                q = np.where(p >= best_tau, hbq, np.minimum(hbq, q_graph)).astype(int)
            else:
                q = np.rint(np.clip(w * hbq, 0.0, None)).astype(int)
            out = df[["store_id","product_id"]].copy()
            out["order_qty"] = q
            _ensure_parent_dir(args.submit_blended)
            out.to_csv(args.submit_blended, index=False)
            print(f"Wrote blended submission {args.submit_blended} with best τ={best_tau:.2f}, α={best_alpha:.2f}")
            # Local sanity: keep HB at high-p, shrink at low-p
            try:
                top_idx = np.argsort(-p)[:args.sanity_k]
                bot_idx = np.argsort(p)[:args.sanity_k]
                keep_rate = float(np.mean(q[top_idx] >= hbq[top_idx])) if len(top_idx) else 0.0
                shrink_rate = float(np.mean(q[bot_idx] <= hbq[bot_idx])) if len(bot_idx) else 0.0
                print(f"Sanity: top-{args.sanity_k} keep-rate={keep_rate:.2f}, bottom-{args.sanity_k} shrink-rate={shrink_rate:.2f}")
            except Exception as e:
                print(f"[warn] sanity calc failed: {e}")

    # === Two-sided gated policy (gated2) ===
    if args.submission_index and args.blend == "gated2":
        # Build df from features and, if available, state and HB
        feat = pd.read_csv(args.features_599) if args.features_599 else None
        if feat is None:
            # Construct features from in-memory rows
            feat = pd.DataFrame(rows)
            if args.submission_index:
                idx = pd.read_csv(args.submission_index)
                feat = idx.merge(feat, on=["store_id","product_id"], how="left")
        hb = pd.read_csv(args.hb) if args.hb else pd.DataFrame(columns=["store_id","product_id"]) 
        for d in (feat, hb):
            if len(d.columns) == 0:
                continue
            cols = {c.lower(): c for c in d.columns}
            sid = cols.get("store_id", list(d.columns)[0])
            pid = cols.get("product_id", list(d.columns)[1] if len(d.columns) > 1 else list(d.columns)[0])
            d.rename(columns={sid:"store_id", pid:"product_id"}, inplace=True)
            d["store_id"] = d["store_id"].astype(str)
            d["product_id"] = d["product_id"].astype(str)
        # pick HB qty col
        hb_qty_col = next((c for c in hb.columns if c.lower() in {"order_qty","orders","qty","0"}), (hb.columns[-1] if len(hb.columns) else None))
        df = feat.merge(hb[["store_id","product_id", hb_qty_col]] if hb_qty_col else hb, on=["store_id","product_id"], how="left")
        hbq = pd.to_numeric(df[hb_qty_col], errors="coerce").fillna(0).to_numpy().astype(int) if hb_qty_col else np.zeros(len(df), dtype=int)
        # Merge state for cost calculations
        if args.state:
            df = _merge_state(df, args.state)
        # probabilities
        p = pd.to_numeric(df.get("p_t3", df.get("p_mean", 0.5)), errors="coerce").fillna(0.5).to_numpy()
        # ABC segmentation by mu_H quantiles
        try:
            q1, q2 = (float(x) for x in str(args.abc_quantiles).split(","))
        except Exception:
            q1, q2 = 0.6, 0.9
        muH = pd.to_numeric(df.get("mu_H", 0.0), errors="coerce").fillna(0.0)
        t1, t2 = np.quantile(muH.to_numpy(), [q1, q2]) if len(muH) > 0 else (0.0, 0.0)
        classes = np.where(muH >= t2, "A", np.where(muH >= t1, "B", "C"))
        beta_vec = np.where(classes == "A", float(args.beta_a), np.where(classes == "B", float(args.beta_b), float(args.beta_c)))
        # Graph base-stock by class beta
        z = np.vectorize(inv_norm_cdf)(np.clip(beta_vec, 1e-6, 1-1e-6))
        mu_H_arr = pd.to_numeric(df.get("mu_H", 0.0), errors="coerce").fillna(0.0).to_numpy()
        sigma_H_arr = pd.to_numeric(df.get("sigma_H", 1e-6), errors="coerce").fillna(1e-6).to_numpy()
        onhand_series = df["onhand"] if "onhand" in df.columns else pd.Series(0.0, index=df.index)
        onorder_series = df["onorder_le2"] if "onorder_le2" in df.columns else pd.Series(0.0, index=df.index)
        onhand = pd.to_numeric(onhand_series, errors="coerce").fillna(0.0).to_numpy()
        onorder = pd.to_numeric(onorder_series, errors="coerce").fillna(0.0).to_numpy()
        S_graph = mu_H_arr + z * sigma_H_arr
        IP = onhand + onorder
        q_graph = np.rint(np.clip(S_graph - IP, 0.0, None)).astype(int)
        # two-sided grid if requested
        if args.grid:
            tau_list = [float(x.strip()) for x in (args.tau_grid.split(",") if args.tau_grid else ["0.55","0.60","0.65"])]
            best_tau_hi, best_tau_lo, best_cost = None, None, float("inf")
            print(f"Running two-sided τ_hi grid (gated2, expected-cost-mode={args.expected_cost_mode}):")
            for tau_hi in tau_list:
                tau_lo = float(args.tau_lo) if args.tau_lo is not None else max(0.0, tau_hi - float(args.tau_margin))
                q_tmp = hbq.copy()
                hi = p >= tau_hi
                lo = p <= tau_lo
                isA = (classes == "A")
                q_tmp[hi] = np.maximum(hbq[hi], q_graph[hi])
                lo_mask = lo & (~isA)
                q_tmp[lo_mask] = np.minimum(hbq[lo_mask], q_graph[lo_mask])
                if args.expected_cost_mode == "mixture":
                    cost = _expected_cost_mixture_over_df(df, q_tmp, float(args.cost_shortage), float(args.cost_holding), int(args.horizon))
                else:
                    cost = _expected_cost_over_df(df, q_tmp, float(args.cost_shortage), float(args.cost_holding), int(args.horizon))
                print(f"  τ_hi={tau_hi:.2f} (τ_lo={tau_lo:.2f}) -> cost={cost:.2f}")
                if cost < best_cost:
                    best_cost, best_tau_hi, best_tau_lo, q_final = cost, tau_hi, tau_lo, q_tmp
            print(f"Best(two-sided): τ_hi={best_tau_hi:.2f}, τ_lo={best_tau_lo:.2f}, cost={best_cost:.3f}")
        else:
            # thresholds single-shot
            tau_hi = float(args.tau_hi) if args.tau_hi is not None else 0.60
            tau_lo = float(args.tau_lo) if args.tau_lo is not None else max(0.0, tau_hi - float(args.tau_margin))
            q_final = hbq.copy()
            hi = p >= tau_hi
            lo = p <= tau_lo
            isA = (classes == "A")
            q_final[hi] = np.maximum(hbq[hi], q_graph[hi])
            lo_mask = lo & (~isA)
            q_final[lo_mask] = np.minimum(hbq[lo_mask], q_graph[lo_mask])
            best_tau_hi, best_tau_lo = tau_hi, tau_lo
            if args.expected_cost_mode == "mixture":
                best_cost = _expected_cost_mixture_over_df(df, q_final, float(args.cost_shortage), float(args.cost_holding), int(args.horizon))
            else:
                best_cost = _expected_cost_over_df(df, q_final, float(args.cost_shortage), float(args.cost_holding), int(args.horizon))
            print(f"Two-sided gated (τ_hi={tau_hi:.2f}, τ_lo={tau_lo:.2f}) -> expected cost={best_cost:.2f}")
        # write submission
        out_path = args.submit_blended or (args.submit if args.submit else "orders_blended_two_sided.csv")
        _ensure_parent_dir(out_path)
        out = df[["store_id","product_id"]].copy()
        out["order_qty"] = q_final
        out.to_csv(out_path, index=False)
        print(f"Wrote blended submission {out_path} with {len(out)} rows (strategy=gated2).")
        # sanity
        try:
            top_idx = np.argsort(-p)[:50]
            bot_idx = np.argsort(p)[:50]
            top_keep = float((q_final[top_idx] >= hbq[top_idx]).mean()) if len(top_idx) else 0.0
            bot_shrink = float((q_final[bot_idx] <= hbq[bot_idx]).mean()) if len(bot_idx) else 0.0
            print(f"Sanity: top-50 keep-rate={top_keep:.2f}, bottom-50 shrink-rate={bot_shrink:.2f}")
        except Exception as e:
            print(f"[warn] two-sided sanity failed: {e}")

    if args.submission_index and (args.submit or args.submit_blended or args.blend in ("hb","graph","gated","shrink")):
        if args.submit:
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
                # Retrieve positive component estimates and p_t3
                if feat:
                    mu_pos = float(feat.get("mu_hat_plus", 1.0))
                    sigma_pos = float(feat.get("sigma_hat_plus", 1.0))
                    p_t3 = float(feat.get("p_t3", feat.get("p_mean", 0.5)))
                else:
                    mu_pos, sigma_pos, p_t3 = 1.0, 1.0, 0.5
                # Critical fractile to z (approx without scipy)
                # For beta=0.833, z ≈ 0.967. For general beta, use probit approximation.
                beta = max(1e-6, min(1 - 1e-6, float(args.beta)))
                # Abramowitz-Stegun approximation for inverse normal CDF (local to avoid shadowing global)
                def _inv_norm_cdf_local(p: float) -> float:
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

                z = _inv_norm_cdf_local(beta)
                S = mu_H + z * sigma_H
                onhand, onorder = state_map.get(key, (0.0, 0.0))
                IP = onhand + onorder
                q_graph = max(0.0, S - IP)
                # Optional mixture policy computation
                if args.blend in ("mixture","mixture_min"):
                    c_s = float(args.cost_shortage_mixture if args.cost_shortage_mixture is not None else args.cost_shortage)
                    c_h = float(args.cost_holding_mixture if args.cost_holding_mixture is not None else args.cost_holding)
                    S_mix = _mixture_base_stock(p_t3, mu_pos, sigma_pos, c_s, c_h, int(args.horizon))
                    q_mix = max(0.0, S_mix - IP)
                    q_final = q_mix if args.blend == "mixture" else (q_mix if _expected_cost_mixture(IP + q_mix, p_t3, mu_pos, sigma_pos, c_s, c_h, int(args.horizon)) < _expected_cost_mixture(IP + q_graph, p_t3, mu_pos, sigma_pos, c_s, c_h, int(args.horizon)) else q_graph)
                    submissions.append({"store_id": sid, "product_id": pid, "order_qty": int(round(q_final)), "p_t3": p_t3})
                else:
                    submissions.append({"store_id": sid, "product_id": pid, "order_qty": int(round(q_graph)), "p_t3": p_t3})

        # Optional blending
        if args.blend in ("hb","graph","gated","shrink","cap","mixture","mixture_min"):
            # Load HB if needed
            hb_map: Dict[Tuple[str, str], int] = {}
            if args.blend in ("hb","gated"):
                if not args.hb:
                    raise SystemExit("--hb is required when --blend is hb or gated")
                with open(args.hb, "r") as hf:
                    hr = csv.DictReader(hf)
                    # Robust column detection
                    cols = [c for c in hr.fieldnames or []]
                    def pick(colnames, candidates):
                        for cand in candidates:
                            for c in colnames:
                                if c.lower().strip() == cand:
                                    return c
                        for c in colnames:
                            cl = c.lower()
                            if any(tok in cl for tok in candidates):
                                return c
                        return None
                    s_col = pick(cols, ["store_id","store","store id","store#"]) or cols[0]
                    p_col = pick(cols, ["product_id","product","product id","sku","item"]) or cols[1]
                    q_col = pick(cols, ["order_qty","order","orders","quantity","qty"]) or cols[-1]
                    for h in hr:
                        key = (str(h.get(s_col, "")), str(h.get(p_col, "")))
                        try:
                            hb_val = int(round(float(h.get(q_col, 0) or 0)))
                        except Exception:
                            hb_val = 0
                        hb_map[key] = hb_val

            # Build blended orders
            blended = []
            for s in submissions:
                key = (s["store_id"], s["product_id"])
                q_graph = s["order_qty"]
                if args.blend == "graph":
                    q_final = q_graph
                elif args.blend == "hb":
                    q_final = hb_map.get(key, 0)
                elif args.blend == "gated":
                    p = float(s.get("p_t3", 0.0))
                    tau = float(args.tau)
                    alpha = float(args.alpha)
                    # If HB missing, fall back to graph to avoid zeroing high-p items
                    q_hb = hb_map.get(key, q_graph)
                    w = 1.0 if p >= tau else alpha
                    q_final = int(round(max(0.0, w * q_hb + (1.0 - w) * q_graph)))
                elif args.blend == "shrink":  # if p>=tau keep HB; else shrink toward zero (alpha * HB)
                    p = float(s.get("p_t3", 0.0))
                    tau = float(args.tau)
                    alpha = float(args.alpha)
                    q_hb = hb_map.get(key, q_graph)
                    q_final = int(round(max(0.0, q_hb if p >= tau else alpha * q_hb)))
                elif args.blend == "cap":  # cap: low-p -> min(HB, Graph); high-p -> HB
                    p = float(s.get("p_t3", 0.0))
                    tau = float(args.tau)
                    q_hb = hb_map.get(key, q_graph)
                    q_final = int(round(max(0.0, q_hb if p >= tau else min(q_hb, q_graph))))
                elif args.blend == "mixture":
                    # Already computed in submissions if selected; here ensure HB fallback only if desired
                    q_final = s["order_qty"]
                elif args.blend == "mixture_min":
                    # Choose-min between HB and already computed mixture/graph order
                    q_hb = hb_map.get(key, s["order_qty"])  # if HB missing, use current
                    q_final = int(round(min(q_hb, s["order_qty"])) if float(s.get("p_t3", 0.0)) < float(args.tau) else q_hb)
                blended.append({"store_id": s["store_id"], "product_id": s["product_id"], "order_qty": int(q_final)})

            out_path = args.submit_blended or (args.submit if args.submit else "orders_blended.csv")
            _ensure_parent_dir(out_path)
            with open(out_path, "w", newline="") as bf:
                writer = csv.DictWriter(bf, fieldnames=["store_id","product_id","order_qty"])
                writer.writeheader()
                writer.writerows(blended)
            print(f"Wrote blended submission {out_path} with {len(blended)} rows (strategy={args.blend}).")
            # Sanity checks
            try:
                # Build quick lookup arrays for checks
                p_arr = np.array([float(s.get("p_t3", 0.0)) for s in submissions])
                hb_arr = np.array([hb_map.get((s["store_id"], s["product_id"]), 0) for s in submissions], dtype=float)
                gr_arr = np.array([s["order_qty"] for s in submissions], dtype=float)
                bl_arr = np.array([b["order_qty"] for b in blended], dtype=float)
                top_idx = np.argsort(-p_arr)[:50]
                bot_idx = np.argsort(p_arr)[:50]
                top_keep = float((bl_arr[top_idx] >= hb_arr[top_idx]).mean()) if len(top_idx) else 0.0
                bot_shrink = float((bl_arr[bot_idx] <= hb_arr[bot_idx]).mean()) if len(bot_idx) else 0.0
                print(f"Sanity: top-50 keep-rate={top_keep:.2f}, bottom-50 shrink-rate={bot_shrink:.2f}")
            except Exception:
                pass
        # Write pure-graph submission if requested
        if args.submit:
            _ensure_parent_dir(args.submit)
            with open(args.submit, "w", newline="") as sf:
                writer = csv.DictWriter(sf, fieldnames=["store_id", "product_id", "order_qty"])
                writer.writeheader()
                for s in submissions:
                    writer.writerow({"store_id": s["store_id"], "product_id": s["product_id"], "order_qty": s["order_qty"]})
            print(f"Wrote submission {args.submit} with {len(submissions)} rows.")


if __name__ == "__main__":
    main()


