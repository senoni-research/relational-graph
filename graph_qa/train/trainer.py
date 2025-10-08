from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import concurrent.futures
import os
from datetime import datetime

from graph_qa.io.loader import load_graph
from graph_qa.train.dataset import build_edge_dataset
from graph_qa.train.model import build_model
from graph_qa.train.model_v2 import build_enhanced_model
from graph_qa.sampling.temporal_egonet import sample_temporal_egonet


Edge = Tuple[Any, Any]


def sample_subgraph_for_edge(G: nx.Graph, edge: Edge, hops: int = 2, K: int = 150) -> nx.Graph:
    """Sample a temporal egonet around an edge for training."""
    u, v = edge
    anchors = [u, v]
    # Use the edge's time if available (get latest time for MultiGraph)
    if G.has_edge(u, v):
        if isinstance(G, nx.MultiGraph):
            # Get max time across all parallel edges
            times = [G[u][v][key].get("time", None) for key in G[u][v]]
            anchor_time = max(t for t in times if t is not None) if any(t is not None for t in times) else None
        else:
            anchor_time = G.edges[u, v].get("time", None)
    else:
        anchor_time = None
    sub = sample_temporal_egonet(G, anchors, hops=hops, K=K, anchor_time=anchor_time)
    return sub


# Multiprocessing helpers (parallel subgraph sampling)
_WORKER_G = None
_WORKER_HOPS = 2
_WORKER_K = 150

def _init_worker(G: nx.Graph, hops: int, K: int):
    global _WORKER_G, _WORKER_HOPS, _WORKER_K
    _WORKER_G = G
    _WORKER_HOPS = hops
    _WORKER_K = K

def _worker_sample(edge: Edge) -> nx.Graph:
    # Uses globals set in _init_worker; returns a small subgraph
    return sample_subgraph_for_edge(_WORKER_G, edge, hops=_WORKER_HOPS, K=_WORKER_K)


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    G: nx.Graph,
    train_data: List[Tuple[Edge, int]],
    batch_size: int = 32,
    hops: int = 2,
    K: int = 150,
    device: str = "cpu",
    log_interval: int = 100,
    num_workers: int = 0,
) -> float:
    """Train for one epoch; return average loss."""
    model.train()
    random.shuffle(train_data)
    total_loss = 0.0
    num_batches = 0

    total_batches = (len(train_data) + batch_size - 1) // batch_size
    
    # Create a persistent executor per epoch to avoid per-batch startup overhead
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    if num_workers and num_workers > 0:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, initializer=_init_worker, initargs=(G, hops, K)
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Parallel sampling enabled with {num_workers} workers")

    try:
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            batch_edges = [e for e, _ in batch]
            batch_labels = torch.FloatTensor([lbl for _, lbl in batch]).to(device)

            # Sample subgraphs and predict
            logits_list = []
            if executor is not None:
                subs = list(executor.map(_worker_sample, batch_edges))
            else:
                subs = [sample_subgraph_for_edge(G, edge, hops=hops, K=K) for edge in batch_edges]

            for edge, sub in zip(batch_edges, subs):
                logits = model(sub, [edge])  # (1,)
                logits_list.append(logits[0])
        
        logits = torch.stack(logits_list)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        # Progress reporting
        if num_batches % max(1, log_interval) == 0 or num_batches == 1:
            avg_loss = total_loss / num_batches
            progress_pct = 100 * num_batches / total_batches
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {num_batches}/{total_batches} ({progress_pct:.1f}%) - Avg Loss: {avg_loss:.4f}")

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return total_loss / max(num_batches, 1)


def eval_epoch(
    model: nn.Module,
    G: nx.Graph,
    val_data: List[Tuple[Edge, int]],
    batch_size: int = 64,
    hops: int = 2,
    K: int = 150,
    device: str = "cpu",
    num_workers: int = 0,
) -> Tuple[float, float]:
    """Evaluate on val; return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        executor: concurrent.futures.ProcessPoolExecutor | None = None
        if num_workers and num_workers > 0:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, initializer=_init_worker, initargs=(G, hops, K)
            )
        try:
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i : i + batch_size]
                batch_edges = [e for e, _ in batch]
                batch_labels = torch.FloatTensor([lbl for _, lbl in batch]).to(device)

                logits_list = []
                if executor is not None:
                    subs = list(executor.map(_worker_sample, batch_edges))
                else:
                    subs = [sample_subgraph_for_edge(G, edge, hops=hops, K=K) for edge in batch_edges]
                for edge, sub in zip(batch_edges, subs):
                    logits = model(sub, [edge])
                    logits_list.append(logits[0])
            
            logits = torch.stack(logits_list)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    avg_loss = total_loss / max(len(val_data) // batch_size, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train edge scorer on relational graph")
    parser.add_argument("--graph", type=str, required=True, help="Path to JSONL graph")
    parser.add_argument("--train-end", type=str, default="2024-01-31", help="Train cutoff date (YYYY-MM-DD)")
    parser.add_argument("--val-end", type=str, default="2024-03-15", help="Val cutoff date (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--hops", type=int, default=2, help="Subgraph hops")
    parser.add_argument("--K", type=int, default=150, help="Subgraph max nodes")
    parser.add_argument("--out", type=str, default="checkpoints/edge_scorer.pt", help="Output checkpoint path")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--use-enhanced", action="store_true", help="Use enhanced model with rich features")
    parser.add_argument("--fast-mode", action="store_true", help="Fast mode: skip attention and hop distance")
    parser.add_argument("--skip-hopdist", action="store_true", help="Skip hop-distance computation only (keep attention)")
    parser.add_argument("--log-interval", type=int, default=100, help="Batches between progress logs")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel workers for subgraph sampling (0=off)")
    args = parser.parse_args()

    print(f"Loading graph from {args.graph}...")
    G = load_graph(args.graph, multi=True)  # Use MultiGraph to preserve temporal edges
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (temporal multi-edges)")

    print("Building train/val/test splits...")
    train_data, val_data, test_data = build_edge_dataset(
        G, train_end=args.train_end, val_end=args.val_end, num_negatives=1
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Prefer MPS on Apple Silicon if available; fallback to CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  → Using MPS")
    else:
        device = torch.device("cpu")
        print("  → Using CPU")

    print("Building model...")
    if args.use_enhanced:
        print("  → Using enhanced model with attention and rich features")
        model = build_enhanced_model(G, hidden_dim=args.hidden_dim, num_layers=args.num_layers, fast_mode=args.fast_mode, skip_hopdist=args.skip_hopdist)
    else:
        print("  → Using simple baseline model")
        model = build_model(G, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{args.epochs} starting ...")
        train_loss = train_epoch(
            model, optimizer, G, train_data, batch_size=args.batch_size, hops=args.hops, K=args.K, device=str(device), log_interval=args.log_interval, num_workers=args.num_workers
        )
        val_loss, val_acc = eval_epoch(model, G, val_data, batch_size=args.batch_size * 2, hops=args.hops, K=args.K, device=str(device), num_workers=max(1, args.num_workers // 2))
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "node_types": model.node_types,
                    "categorical_attrs": getattr(model, "categorical_attrs", None),
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                out_path,
            )
            print(f"  → Saved checkpoint to {out_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()

