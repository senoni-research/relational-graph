#!/usr/bin/env python3
"""Fit isotonic calibration on validation window and save the calibrator.

Usage:
  python -u scripts/calibrate_scorer.py \
    --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
    --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
    --train-end 2024-01-31 --val-end 2024-03-15 \
    --hops 1 --K 30 --time-aware \
    --out checkpoints/calibrators/iso_val.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from graph_qa.io.loader import load_graph
from graph_qa.train.dataset import build_time_aware_dataset, build_edge_dataset
from graph_qa.train.model import SimpleEdgeScorer
from graph_qa.train.model_v2 import EnhancedEdgeScorer
from graph_qa.train.trainer import sample_subgraph_for_edge


def main():
    p = argparse.ArgumentParser(description="Fit isotonic calibrator on validation window")
    p.add_argument("--graph", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--train-end", default="2024-01-31", type=str)
    p.add_argument("--val-end", default="2024-03-15", type=str)
    p.add_argument("--hops", default=1, type=int)
    p.add_argument("--K", default=30, type=int)
    p.add_argument("--time-aware", action="store_true")
    p.add_argument("--calibration-method", type=str, default="isotonic", choices=["isotonic", "platt"], help="Calibration method")
    p.add_argument("--input-space", type=str, default="prob", choices=["prob", "logit"], help="Calibrator input space")
    p.add_argument("--out", required=True, type=str)
    args = p.parse_args()

    print(f"Loading graph from {args.graph}...")
    G = load_graph(args.graph, multi=True)

    print("Building validation set...")
    if args.time_aware:
        train_data, val_data, _ = build_time_aware_dataset(G, args.train_end, args.val_end)
    else:
        train_data, val_data, _ = build_edge_dataset(G, args.train_end, args.val_end, num_negatives=1)
    print(f"Val samples: {len(val_data)}")

    print(f"Loading model from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    is_enhanced = "model_state" in ckpt and any(k.startswith("cat_embeds.") for k in ckpt["model_state"].keys())
    if is_enhanced:
        model = EnhancedEdgeScorer(
            node_types=ckpt["node_types"],
            categorical_attrs=ckpt.get("categorical_attrs", {}),
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            recency_feature=ckpt.get("recency_feature", False),
            recency_norm=ckpt.get("recency_norm", 52.0),
        )
        model.fast_mode = ckpt.get("fast_mode", False)
        model.skip_hopdist = ckpt.get("skip_hopdist", False)
    else:
        model = SimpleEdgeScorer(node_types=ckpt["node_types"], hidden_dim=ckpt["hidden_dim"], num_layers=ckpt["num_layers"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    print("Scoring validation set...")
    probs = []
    logits = []
    labels = []
    with torch.no_grad():
        for i, (edge, label) in enumerate(val_data):
            if i % 200 == 0:
                print(f"  {i}/{len(val_data)}", end="\r")
            sub = sample_subgraph_for_edge(G, edge, hops=args.hops, K=args.K)
            if isinstance(edge, (tuple, list)) and len(edge) == 3:
                ev = (edge[0], edge[1], edge[2])
            else:
                ev = edge
            logit = model(sub, [ev]).squeeze(0)
            prob = torch.sigmoid(logit).item()
            logits.append(float(logit.item()))
            probs.append(prob)
            labels.append(label)

    print("\nFitting calibrator on validation...")
    model_obj = None
    meta = {"method": args.calibration_method, "input": args.input_space}
    if args.calibration_method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        xs = logits if args.input_space == "logit" else probs
        _ = iso.fit_transform(xs, labels)
        model_obj = iso
    else:
        # Platt scaling via logistic regression on chosen input space
        xs = logits if args.input_space == "logit" else probs
        import numpy as np
        X = np.array(xs, dtype=float).reshape(-1, 1)
        y = np.array(labels, dtype=int)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(X, y)
        model_obj = lr

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"method": meta["method"], "input": meta["input"], "model": model_obj}, f)
    print(f"Saved calibrator to {args.out} ({meta['method']}, input={meta['input']})")


if __name__ == "__main__":
    main()


