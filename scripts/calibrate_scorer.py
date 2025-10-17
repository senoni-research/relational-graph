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
    p.add_argument("--segment-by", type=str, default=None, help="Comma-separated node attrs to segment calibrators (e.g., 'Region,Format')")
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
        # Match training feature config (including optional ID embeddings)
        store_ids = [str(n).replace("store:", "") for n, a in G.nodes(data=True) if str(a.get("type")) == "store"]
        product_ids = [str(n).replace("product:", "") for n, a in G.nodes(data=True) if str(a.get("type")) == "product"]
        model = EnhancedEdgeScorer(
            node_types=ckpt["node_types"],
            categorical_attrs=ckpt.get("categorical_attrs", {}),
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
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
        model = SimpleEdgeScorer(node_types=ckpt["node_types"], hidden_dim=ckpt["hidden_dim"], num_layers=ckpt["num_layers"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    print("Scoring validation set...")
    rows = []
    with torch.no_grad():
        for i, (edge, label) in enumerate(val_data):
            if i % 200 == 0:
                print(f"  {i}/{len(val_data)}", end="\r")
            sub = sample_subgraph_for_edge(G, edge, hops=args.hops, K=args.K)
            if isinstance(edge, (tuple, list)) and len(edge) == 3:
                ev = (edge[0], edge[1], edge[2])
                s, p = ev[0], ev[1]
            else:
                ev = edge
                s, p = ev[0], ev[1]
            logit = model(sub, [ev]).squeeze(0)
            prob = float(torch.sigmoid(logit).item())
            rows.append({
                "store": str(s),
                "product": str(p),
                "prob": prob,
                "logit": float(logit.item()),
                "label": int(label),
                # optional attributes for segmentation
                "Region": str(G.nodes.get(s, {}).get("Region", "")) if hasattr(G, 'nodes') and G.has_node(s) else "",
                "Format": str(G.nodes.get(s, {}).get("Format", "")) if hasattr(G, 'nodes') and G.has_node(s) else "",
            })

    print("\nFitting calibrator on validation...")
    meta = {"method": args.calibration_method, "input": args.input_space}
    segments = [s.strip() for s in args.segment_by.split(",")] if args.segment_by else None
    
    def _fit(xs, ys):
        if args.calibration_method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            _ = iso.fit_transform(xs, ys)
            return iso
        else:
            import numpy as np
            X = np.array(xs, dtype=float).reshape(-1, 1)
            y = np.array(ys, dtype=int)
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(X, y)
            return lr
    
    out_dir = os.path.dirname(args.out)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if not segments:
        # Single global calibrator
        xs = [r[args.input_space] for r in rows]
        ys = [r["label"] for r in rows]
        model_obj = _fit(xs, ys)
        with open(args.out, "wb") as f:
            pickle.dump({"method": meta["method"], "input": meta["input"], "model": model_obj}, f)
        print(f"Saved calibrator to {args.out} ({meta['method']}, input={meta['input']})")
    else:
        # Segmented calibrators
        for r in rows:
            r["_seg_key"] = tuple(str(r.get(col, "")) for col in segments)
        from collections import defaultdict
        bucket = defaultdict(list)
        for r in rows:
            bucket[r["_seg_key"]].append(r)
        for key, items in bucket.items():
            xs = [it[args.input_space] for it in items]
            ys = [it["label"] for it in items]
            if len(set(ys)) < 2:
                print(f"[warn] segment {key} has single-class labels; skipping")
                continue
            model_obj = _fit(xs, ys)
            seg_name = "__".join(f"{segments[i]}={key[i]}" for i in range(len(segments)))
            out_path = os.path.join(out_dir, f"{Path(args.out).stem}.{seg_name}{Path(args.out).suffix}")
            with open(out_path, "wb") as f:
                pickle.dump({"method": meta["method"], "input": meta["input"], "model": model_obj, "segment": segments, "key": key}, f)
            print(f"Saved segment calibrator: {out_path}")


if __name__ == "__main__":
    main()


