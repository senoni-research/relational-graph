#!/usr/bin/env python3
"""Evaluate trained scorer on test set."""

import argparse
import pickle
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from graph_qa.io.loader import load_graph
from graph_qa.train.dataset import build_edge_dataset, build_time_aware_dataset
from graph_qa.train.model import SimpleEdgeScorer
from graph_qa.train.model_v2 import EnhancedEdgeScorer
from graph_qa.train.trainer import sample_subgraph_for_edge


def evaluate_scorer(
    graph_path: str,
    ckpt_path: str,
    train_end: str = "2024-01-31",
    val_end: str = "2024-03-15",
    hops: int = 2,
    K: int = 150,
    time_aware: bool = False,
    calibrator_path: str | None = None,
):
    print(f"Loading graph from {graph_path}...")
    G = load_graph(graph_path, multi=True)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"Building test set (> {val_end})...")
    if time_aware:
        _, _, test_data = build_time_aware_dataset(G, train_end, val_end)
    else:
        _, _, test_data = build_edge_dataset(G, train_end, val_end, num_negatives=1)
    print(f"Test samples: {len(test_data)}")

    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Detect model type by checking for enhanced-specific keys
    is_enhanced = "cat_embeds.ProductGroup.weight" in ckpt["model_state"] if "model_state" in ckpt else False
    
    if is_enhanced:
        print("  → Detected enhanced model")
        # Use categorical attrs from checkpoint if available
        cat_attrs = ckpt.get("categorical_attrs")
        if not cat_attrs:
            # Fallback to placeholder (not ideal)
            cat_attrs = {}
            for key in ckpt["model_state"].keys():
                if key.startswith("cat_embeds.") and key.endswith(".weight"):
                    attr_name = key.split(".")[1]
                    vocab_size = ckpt["model_state"][key].shape[0] - 1
                    cat_attrs[attr_name] = list(range(vocab_size))
        
        model = EnhancedEdgeScorer(
            node_types=ckpt["node_types"],
            categorical_attrs=cat_attrs,
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            recency_feature=ckpt.get("recency_feature", False),
            recency_norm=ckpt.get("recency_norm", 52.0),
        )
        # Align runtime flags with training if present
        if "fast_mode" in ckpt:
            model.fast_mode = bool(ckpt["fast_mode"])  # type: ignore
        if "skip_hopdist" in ckpt:
            model.skip_hopdist = bool(ckpt["skip_hopdist"])  # type: ignore
    else:
        print("  → Detected simple baseline model")
        model = SimpleEdgeScorer(
            node_types=ckpt["node_types"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        )
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Use training hparams if provided
    hops = ckpt.get("hops", hops)
    K = ckpt.get("K", K)

    print("Evaluating on test set...")
    preds = []
    labels = []
    empty_subgraphs = 0

    with torch.no_grad():
        for i, (edge, label) in enumerate(test_data):
            if i % 100 == 0:
                print(f"  {i}/{len(test_data)}", end="\r")
            
            sub = sample_subgraph_for_edge(G, edge, hops=hops, K=K)
            # Pass (u,v,t) when available to enable t-anchored temporal features
            if isinstance(edge, (tuple, list)) and len(edge) == 3:
                ev = (edge[0], edge[1], edge[2])
            else:
                ev = edge
            # If empty, still score endpoints using node attrs only
            if sub.number_of_edges() == 0:
                empty_subgraphs += 1
            if sub.number_of_edges() == 0 and len(sub) == 0:
                import networkx as nx  # local import to avoid global dep
                sub = nx.Graph()
                if G.has_node(ev[0]):
                    sub.add_node(ev[0], **G.nodes[ev[0]])
                if G.has_node(ev[1]):
                    sub.add_node(ev[1], **G.nodes[ev[1]])
            logit = model(sub, [ev])
            
            prob = torch.sigmoid(logit).item()
            preds.append(prob)
            labels.append(label)

    # Apply optional calibrator
    if calibrator_path:
        try:
            with open(calibrator_path, "rb") as f:
                iso = pickle.load(f)
            preds = list(map(float, iso.transform(preds)))
            print(f"Applied calibrator: {calibrator_path}")
        except Exception as e:
            print(f"Warning: failed to load/apply calibrator {calibrator_path}: {e}")

    print("\nTest Results:")
    print(f"  Empty subgraphs: {empty_subgraphs} / {len(test_data)} ({100.0*empty_subgraphs/len(test_data):.1f}%)")
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    print(f"  AUC: {auc:.4f}")
    print(f"  AP:  {ap:.4f}")
    
    # Show a few predictions
    print("\nSample predictions:")
    for i in range(min(10, len(test_data))):
        edge, label = test_data[i]
        pred = preds[i]
        print(f"  {edge}: label={label}, pred={pred:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained edge scorer on test set")
    parser.add_argument("--graph", type=str, required=True, help="Path to JSONL graph")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--train-end", type=str, default="2024-01-31")
    parser.add_argument("--val-end", type=str, default="2024-03-15")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--K", type=int, default=150)
    parser.add_argument("--time-aware", action="store_true")
    parser.add_argument("--calibrator", type=str, default=None, help="Path to a saved isotonic calibrator (pickle)")
    args = parser.parse_args()

    evaluate_scorer(
        args.graph,
        args.ckpt,
        train_end=args.train_end,
        val_end=args.val_end,
        hops=args.hops,
        K=args.K,
        time_aware=args.time_aware,
        calibrator_path=args.calibrator,
    )


if __name__ == "__main__":
    main()

