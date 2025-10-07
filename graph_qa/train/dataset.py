from __future__ import annotations

import random
from typing import List, Tuple, Dict, Any
from datetime import datetime

import networkx as nx


Edge = Tuple[Any, Any]


def parse_time_attr(t: Any) -> int:
    """Convert time attribute to int (YYYYMMDD)."""
    if isinstance(t, int):
        return t
    if isinstance(t, str):
        # Try YYYYMMDD format
        if t.isdigit() and len(t) == 8:
            return int(t)
        # Try YYYY-MM-DD
        try:
            dt = datetime.strptime(t, "%Y-%m-%d")
            return int(dt.strftime("%Y%m%d"))
        except Exception:
            pass
    return 0


def temporal_train_val_test_split(
    G: nx.Graph,
    train_end: str,  # YYYY-MM-DD or YYYYMMDD
    val_end: str,
) -> Tuple[List[Edge], List[Edge], List[Edge]]:
    """
    Split edges by time into train/val/test.
    - train: edges with time <= train_end
    - val: edges with train_end < time <= val_end
    - test: edges with time > val_end
    
    Handles both Graph and MultiGraph (MultiGraph edges include edge_key).
    """
    train_end_int = parse_time_attr(train_end.replace("-", ""))
    val_end_int = parse_time_attr(val_end.replace("-", ""))

    train_edges: List[Edge] = []
    val_edges: List[Edge] = []
    test_edges: List[Edge] = []

    is_multigraph = isinstance(G, nx.MultiGraph)
    
    if is_multigraph:
        # MultiGraph: iterate with keys to get (u, v, key, attrs)
        for u, v, key, attrs in G.edges(data=True, keys=True):
            t = parse_time_attr(attrs.get("time", 0))
            if t <= train_end_int:
                train_edges.append((u, v))
            elif t <= val_end_int:
                val_edges.append((u, v))
            else:
                test_edges.append((u, v))
    else:
        # Simple Graph: iterate without keys to get (u, v, attrs)
        for u, v, attrs in G.edges(data=True):
            t = parse_time_attr(attrs.get("time", 0))
            if t <= train_end_int:
                train_edges.append((u, v))
            elif t <= val_end_int:
                val_edges.append((u, v))
            else:
                test_edges.append((u, v))

    return train_edges, val_edges, test_edges


def sample_negatives(
    G: nx.Graph,
    positive_edges: List[Edge],
    num_negatives: int = 1,
    strategy: str = "type_match",
) -> List[Edge]:
    """
    Sample negative edges (non-existent) for each positive edge.
    strategy:
      - type_match: corrupt tail to another node of the same type
      - random: corrupt tail to any node
    """
    negatives: List[Edge] = []
    node_types: Dict[str, List[Any]] = {}
    for n, attrs in G.nodes(data=True):
        ntype = attrs.get("type", "unknown")
        node_types.setdefault(ntype, []).append(n)

    for u, v in positive_edges:
        v_type = G.nodes[v].get("type", "unknown")
        candidates = node_types.get(v_type, list(G.nodes))
        if not candidates:
            candidates = list(G.nodes)
        
        for _ in range(num_negatives):
            # Sample until we get a non-edge
            attempts = 0
            while attempts < 20:
                neg_v = random.choice(candidates)
                if not G.has_edge(u, neg_v) and neg_v != v:
                    negatives.append((u, neg_v))
                    break
                attempts += 1
            else:
                # Fallback: accept any distinct node
                neg_v = random.choice(list(G.nodes))
                if neg_v != v:
                    negatives.append((u, neg_v))

    return negatives


def build_edge_dataset(
    G: nx.Graph,
    train_end: str,
    val_end: str,
    num_negatives: int = 1,
) -> Tuple[List[Tuple[Edge, int]], List[Tuple[Edge, int]], List[Tuple[Edge, int]]]:
    """
    Build (edge, label) pairs for train/val/test.
    label=1 for positives, label=0 for negatives.
    """
    train_pos, val_pos, test_pos = temporal_train_val_test_split(G, train_end, val_end)
    
    train_neg = sample_negatives(G, train_pos, num_negatives=num_negatives)
    val_neg = sample_negatives(G, val_pos, num_negatives=num_negatives)
    test_neg = sample_negatives(G, test_pos, num_negatives=num_negatives)

    train_data = [(e, 1) for e in train_pos] + [(e, 0) for e in train_neg]
    val_data = [(e, 1) for e in val_pos] + [(e, 0) for e in val_neg]
    test_data = [(e, 1) for e in test_pos] + [(e, 0) for e in test_neg]

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data

