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
from graph_qa.train.dataset import build_edge_dataset, build_time_aware_dataset, mine_hard_negatives_timeaware
from graph_qa.train.model import build_model
from graph_qa.train.model_v2 import build_enhanced_model
from graph_qa.sampling.temporal_egonet import sample_temporal_egonet


Edge = Tuple[Any, Any]
EdgeT = Tuple[Any, Any, int]


def sample_subgraph_for_edge(G: nx.Graph, edge: Any, hops: int = 2, K: int = 150) -> nx.Graph:
    """Sample a temporal egonet around an edge for training."""
    # Support (u, v) or (u, v, t)
    if isinstance(edge, (tuple, list)) and len(edge) == 3:
        u, v, t = edge  # type: ignore
        anchor_time = t
    else:
        u, v = edge  # type: ignore
        t = None
        anchor_time = None
    anchors = [u, v]
    # If time not provided, fallback to edge-derived time
    if anchor_time is None and G.has_edge(u, v):
        if isinstance(G, nx.MultiGraph):
            times = [G[u][v][key].get("time", None) for key in G[u][v]]
            anchor_time = max(t for t in times if t is not None) if any(t is not None for t in times) else None
        else:
            anchor_time = G.edges[u, v].get("time", None)
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
    executor: concurrent.futures.ProcessPoolExecutor | None = None,
    rank_loss: bool = False,
    rank_margin: float = 0.1,
    rank_weight: float = 0.5,
) -> float:
    """Train for one epoch; return average loss."""
    model.train()
    random.shuffle(train_data)
    total_loss = 0.0
    num_batches = 0

    total_batches = (len(train_data) + batch_size - 1) // batch_size
    
    try:
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            # Build balanced batch with optional hard negatives when time-aware triples are used
            pos_edges = [e for e, y in batch if y == 1]
            neg_edges = [e for e, y in batch if y == 0]
            # If edges are (u,v,t) and we have positives, mine hard negatives near positives
            batch_edges: list[Any] = []
            batch_y: list[float] = []
            if pos_edges and isinstance(pos_edges[0], (tuple, list)) and len(pos_edges[0]) == 3:
                # hard-negative mining target count: 1:K per positive when rank_loss enabled else 1:1
                target_k = 1 if not rank_loss else 2
                hardnegs = mine_hard_negatives_timeaware(G, pos_edges, k=target_k)
                batch_edges.extend(pos_edges)
                batch_y.extend([1.0] * len(pos_edges))
                # Append mined hard negatives (label 0)
                if hardnegs:
                    batch_edges.extend(hardnegs)
                    batch_y.extend([0.0] * len(hardnegs))
                # Add any prebuilt negatives to keep diversity
                if neg_edges:
                    batch_edges.extend(neg_edges[: max(0, len(pos_edges) * target_k - len(hardnegs))])
                    batch_y.extend([0.0] * min(len(neg_edges), max(0, len(pos_edges) * target_k - len(hardnegs))))
            else:
                # Fallback: use provided edges
                batch_edges = [e for e, _ in batch]
                batch_y = [float(lbl) for _, lbl in batch]
            batch_labels = torch.FloatTensor(batch_y).to(device)

            # Sample subgraphs and predict
            logits_list = []
            if executor is not None:
                try:
                    chunksize = max(1, len(batch_edges) // max(1, getattr(executor, "_max_workers", num_workers) * 4))
                    subs = list(executor.map(_worker_sample, batch_edges, chunksize=chunksize))
                except concurrent.futures.process.BrokenProcessPool:
                    print("[warn] process pool broken; falling back to single-process sampling")
                    executor = None
                    subs = [sample_subgraph_for_edge(G, edge, hops=hops, K=K) for edge in batch_edges]
            else:
                subs = [sample_subgraph_for_edge(G, edge, hops=hops, K=K) for edge in batch_edges]

            for edge, sub in zip(batch_edges, subs):
                # Preserve t for t-anchored temporal features
                if isinstance(edge, (tuple, list)) and len(edge) == 3:
                    ev = (edge[0], edge[1], edge[2])
                else:
                    ev = edge
                # Ensure minimal graph on pathological empty sample
                if sub.number_of_edges() == 0 and len(sub) == 0:
                    H = nx.Graph()
                    if G.has_node(ev[0]):
                        H.add_node(ev[0], **G.nodes[ev[0]])
                    if G.has_node(ev[1]):
                        H.add_node(ev[1], **G.nodes[ev[1]])
                    sub = H
                logits = model(sub, [ev])  # (1,)
                logits_list.append(logits[0])

            logits = torch.stack(logits_list)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)

            # Optional pairwise ranking loss using in-batch positives and negatives
            if rank_loss:
                pos_idx = [i for i, y in enumerate(batch_y) if y == 1.0]
                neg_idx = [i for i, y in enumerate(batch_y) if y == 0.0]
                if pos_idx and neg_idx:
                    # Simple pairing: each pos against a random neg
                    import random as _r
                    s_pos = logits[pos_idx]
                    s_neg = logits[_r.choices(neg_idx, k=len(pos_idx))]
                    rank_term = torch.log1p(torch.exp(-((s_pos - s_neg) - rank_margin))).mean()
                    loss = loss + rank_weight * rank_term

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
        pass

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
    executor: concurrent.futures.ProcessPoolExecutor | None = None,
) -> Tuple[float, float]:
    """Evaluate on val; return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        num_batches = 0
        try:
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i : i + batch_size]
                batch_edges = [e for e, _ in batch]
                batch_labels = torch.FloatTensor([lbl for _, lbl in batch]).to(device)

                logits_list = []
                if executor is not None:
                    try:
                        chunksize = max(1, len(batch_edges) // max(1, getattr(executor, "_max_workers", num_workers) * 4))
                        subs = list(executor.map(_worker_sample, batch_edges, chunksize=chunksize))
                    except concurrent.futures.process.BrokenProcessPool:
                        print("[warn] process pool broken during eval; falling back to single-process sampling")
                        executor = None
                        subs = [sample_subgraph_for_edge(G, edge, hops=hops, K=K) for edge in batch_edges]
                else:
                    subs = [sample_subgraph_for_edge(G, edge, hops=hops, K=K) for edge in batch_edges]
                for edge, sub in zip(batch_edges, subs):
                    if isinstance(edge, (tuple, list)) and len(edge) == 3:
                        ev = (edge[0], edge[1], edge[2])
                    else:
                        ev = edge
                    if sub.number_of_edges() == 0 and len(sub) == 0:
                        H = nx.Graph()
                        if G.has_node(ev[0]):
                            H.add_node(ev[0], **G.nodes[ev[0]])
                        if G.has_node(ev[1]):
                            H.add_node(ev[1], **G.nodes[ev[1]])
                        sub = H
                    logits = model(sub, [ev])
                    logits_list.append(logits[0])

                logits = torch.stack(logits_list)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
                total_loss += loss.item()
                num_batches += 1

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)
                all_probs.append(probs.cpu())
                all_labels.append(batch_labels.cpu())

        finally:
            pass

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    # Optional: print AUC/AP if sklearn is available and we have data
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if all_probs and all_labels:
            import torch as _torch
            probs_cat = _torch.cat(all_probs).numpy()
            labels_cat = _torch.cat(all_labels).numpy()
            val_auc = roc_auc_score(labels_cat, probs_cat)
            val_ap = average_precision_score(labels_cat, probs_cat)
            print(f"  Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
    except Exception:
        pass
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
    parser.add_argument("--recency-feature", action="store_true", help="Append log-scaled recency scalar to edge features (requires retrain when enabled)")
    parser.add_argument("--recency-norm", type=float, default=52.0, help="Normalization scale for recency feature (e.g., weeks)")
    parser.add_argument("--skip-hopdist", action="store_true", help="Skip hop-distance computation only (keep attention)")
    parser.add_argument("--log-interval", type=int, default=100, help="Batches between progress logs")
    parser.add_argument("--time-aware", action="store_true", help="Use time-aware dataset with (u,v,t) and inventory-aware negatives")
    parser.add_argument("--negatives", type=str, default="inventory_only", choices=["inventory_only","all"], help="Negative policy for time-aware dataset")
    parser.add_argument("--rank-loss", action="store_true", help="Add pairwise ranking loss (BPR-style)")
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--event-buckets", type=str, default=None, help="Comma-separated weeks for short history features, e.g. '1,2,4,8'")
    parser.add_argument("--rel-aware-attn", action="store_true", help="Use relation-aware attention projections")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel workers for subgraph sampling (0=off)")
    args = parser.parse_args()

    print(f"Loading graph from {args.graph}...")
    G = load_graph(args.graph, multi=True)  # Use MultiGraph to preserve temporal edges
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (temporal multi-edges)")

    print("Building train/val/test splits...")
    if args.time_aware:
        train_data, val_data, test_data = build_time_aware_dataset(
            G, train_end=args.train_end, val_end=args.val_end, negatives=args.negatives
        )
    else:
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
        # Parse event buckets
        event_buckets = None
        if args.event_buckets:
            try:
                event_buckets = [int(x.strip()) for x in args.event_buckets.split(",") if x.strip()]
            except Exception:
                event_buckets = None
        model = build_enhanced_model(
            G,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fast_mode=args.fast_mode,
            skip_hopdist=args.skip_hopdist,
            recency_feature=args.recency_feature,
            recency_norm=args.recency_norm,
            rel_aware_attn=args.rel_aware_attn,
            event_buckets=event_buckets,
        )
    else:
        print("  → Using simple baseline model")
        model = build_model(G, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    # Create a long-lived process pool if requested
    mp_ctx = None
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    if args.num_workers and args.num_workers > 0:
        import multiprocessing as _mp
        mp_ctx = _mp.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(G, args.hops, args.K),
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Parallel sampling enabled with {args.num_workers} workers (spawn)")

    for epoch in range(1, args.epochs + 1):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{args.epochs} starting ...")
        train_loss = train_epoch(
            model, optimizer, G, train_data, batch_size=args.batch_size, hops=args.hops, K=args.K, device=str(device), log_interval=args.log_interval, num_workers=args.num_workers, executor=executor, rank_loss=args.rank_loss, rank_margin=args.rank_margin, rank_weight=args.rank_weight
        )
        val_loss, val_acc = eval_epoch(model, G, val_data, batch_size=args.batch_size * 2, hops=args.hops, K=args.K, device=str(device), num_workers=max(0, args.num_workers // 2), executor=executor)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "node_types": getattr(model, "node_types", None),
                    "categorical_attrs": getattr(model, "categorical_attrs", None),
                    "cat_val2idx": getattr(model, "cat_val2idx", None),
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "hops": args.hops,
                    "K": args.K,
                    "fast_mode": getattr(model, "fast_mode", False),
                    "skip_hopdist": getattr(model, "skip_hopdist", False),
                    "recency_feature": getattr(model, "recency_feature", False),
                    "recency_norm": getattr(model, "recency_norm", 52.0),
                    "event_buckets": [int(x.strip()) for x in (args.event_buckets.split(",") if args.event_buckets else []) if x.strip()] if hasattr(args, "event_buckets") else None,
                    "rel_aware_attn": bool(getattr(args, "rel_aware_attn", False)),
                    "id_emb_dim": int(getattr(model, "id_emb_dim", 0)),
                    "negatives_policy": getattr(args, "negatives", None),
                },
                out_path,
            )
            print(f"  → Saved checkpoint to {out_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Shutdown pool if created
    if executor is not None:
        executor.shutdown(wait=True)

    print("Training complete.")


if __name__ == "__main__":
    main()

