from __future__ import annotations

import random
from typing import List, Tuple, Dict, Any
from datetime import datetime

import networkx as nx


Edge = Tuple[Any, Any]
EdgeT = Tuple[Any, Any, int]


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
            # Use only 'sold' events as positives; ignore zero-sales since they are not generated now
            if attrs.get("rel") != "sold":
                continue
            units = attrs.get("units", None)
            if units is not None and float(units) <= 0.0:
                # Defensive: skip zero units if present
                continue
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
            if attrs.get("rel") != "sold":
                continue
            units = attrs.get("units", None)
            if units is not None and float(units) <= 0.0:
                continue
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


def build_time_aware_dataset(
    G: nx.Graph,
    train_end: str,
    val_end: str,
    negatives: str = "inventory_only",
) -> Tuple[List[Tuple[EdgeT, int]], List[Tuple[EdgeT, int]], List[Tuple[EdgeT, int]]]:
    """
    Time-aware dataset:
    - Positive: (u,v,t) where rel=='sold' and units>0
    - Negative: (u,v,t) where rel=='sold' and units==0 AND inventory present at t
    Splits by t: train (<= train_end), val (train_end < t <= val_end), test (> val_end)
    """
    train_end_int = parse_time_attr(train_end.replace("-", ""))
    val_end_int = parse_time_attr(val_end.replace("-", ""))

    sold_pos: List[EdgeT] = []
    sold_zero: List[EdgeT] = []
    inv_present: set[EdgeT] | set = set()

    is_multigraph = isinstance(G, nx.MultiGraph)

    # Collect inventory presence keys for quick lookup
    if is_multigraph:
        for u, v, key, attrs in G.edges(data=True, keys=True):
            if attrs.get("rel") == "has_inventory" and bool(attrs.get("present", False)):
                t = parse_time_attr(attrs.get("time", 0))
                inv_present.add((u, v, t))
    else:
        for u, v, attrs in G.edges(data=True):
            if attrs.get("rel") == "has_inventory" and bool(attrs.get("present", False)):
                t = parse_time_attr(attrs.get("time", 0))
                inv_present.add((u, v, t))

    # Collect sold events
    if is_multigraph:
        for u, v, key, attrs in G.edges(data=True, keys=True):
            if attrs.get("rel") != "sold":
                continue
            t = parse_time_attr(attrs.get("time", 0))
            try:
                units = float(attrs.get("units", 0.0))
            except Exception:
                units = 0.0
            if units > 0.0:
                sold_pos.append((u, v, t))
            else:
                # Keep zero only if inventory present; we filter after collection using inv_present
                sold_zero.append((u, v, t))
    else:
        for u, v, attrs in G.edges(data=True):
            if attrs.get("rel") != "sold":
                continue
            t = parse_time_attr(attrs.get("time", 0))
            try:
                units = float(attrs.get("units", 0.0))
            except Exception:
                units = 0.0
            if units > 0.0:
                sold_pos.append((u, v, t))
            else:
                sold_zero.append((u, v, t))

    # Negatives
    sold_zero_set = set(sold_zero)
    if negatives == "inventory_only":
        # True negatives: inventory present AND sold==0
        true_zero = list(sold_zero_set & inv_present)
    else:
        # All recorded sold==0 as negatives (less recommended)
        true_zero = list(sold_zero_set)

    # Split by time
    def split_by_time(samples: List[EdgeT]) -> Tuple[List[EdgeT], List[EdgeT], List[EdgeT]]:
        tr: List[EdgeT] = []
        va: List[EdgeT] = []
        te: List[EdgeT] = []
        for u, v, t in samples:
            if t <= train_end_int:
                tr.append((u, v, t))
            elif t <= val_end_int:
                va.append((u, v, t))
            else:
                te.append((u, v, t))
        return tr, va, te

    tr_pos, va_pos, te_pos = split_by_time(sold_pos)
    tr_neg, va_neg, te_neg = split_by_time(true_zero)

    # Create labeled datasets
    train_data = [((u, v, t), 1) for (u, v, t) in tr_pos] + [((u, v, t), 0) for (u, v, t) in tr_neg]
    val_data = [((u, v, t), 1) for (u, v, t) in va_pos] + [((u, v, t), 0) for (u, v, t) in va_neg]
    test_data = [((u, v, t), 1) for (u, v, t) in te_pos] + [((u, v, t), 0) for (u, v, t) in te_neg]

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def mine_hard_negatives_timeaware(
    G: nx.Graph,
    positives: List[EdgeT],
    k: int = 2,
) -> List[EdgeT]:
    """Mine simple hard negatives near positives using schema affinities.
    Heuristics:
      - same store, similar products by Department/Division
      - same product, similar stores by StoreFormat/Format
      Requires has_inventory at t and no sold>0 at t.
    """
    # Build similarity buckets
    product_buckets: Dict[str, List[Any]] = {}
    store_buckets: Dict[str, List[Any]] = {}
    for n, a in G.nodes(data=True):
        t = str(a.get("type", ""))
        if t == "product":
            key = str(a.get("Department", a.get("Division", "unknown")))
            product_buckets.setdefault(key, []).append(n)
        elif t == "store":
            key = str(a.get("StoreFormat", a.get("Format", "unknown")))
            store_buckets.setdefault(key, []).append(n)

    # Fast inventory-present lookup
    inv_present: set[EdgeT] | set = set()
    if isinstance(G, nx.MultiGraph):
        for u, v, key, attrs in G.edges(data=True, keys=True):
            if attrs.get("rel") == "has_inventory" and bool(attrs.get("present", False)):
                t = parse_time_attr(attrs.get("time", 0))
                inv_present.add((u, v, t))
    else:
        for u, v, attrs in G.edges(data=True):
            if attrs.get("rel") == "has_inventory" and bool(attrs.get("present", False)):
                t = parse_time_attr(attrs.get("time", 0))
                inv_present.add((u, v, t))

    negatives: List[EdgeT] = []
    for (s, p, t) in positives:
        # same store, similar products
        dept = str(G.nodes[p].get("Department", G.nodes[p].get("Division", "unknown")))
        for p2 in product_buckets.get(dept, [])[: 4 * k]:
            if p2 == p:
                continue
            if (s, p2, t) in inv_present:
                # check sold at t
                sold_pos = False
                if G.has_edge(s, p2):
                    if isinstance(G, nx.MultiGraph):
                        for key in G[s][p2]:
                            a = G[s][p2][key]
                            if a.get("rel") == "sold" and parse_time_attr(a.get("time", 0)) == t and float(a.get("units", 0.0)) > 0.0:
                                sold_pos = True
                                break
                    else:
                        a = G.edges[s, p2]
                        if a.get("rel") == "sold" and parse_time_attr(a.get("time", 0)) == t and float(a.get("units", 0.0)) > 0.0:
                            sold_pos = True
                if not sold_pos:
                    negatives.append((s, p2, t))
                    if len(negatives) >= k:
                        break
        if len(negatives) >= k:
            continue
        # same product, similar stores
        fmt = str(G.nodes[s].get("StoreFormat", G.nodes[s].get("Format", "unknown")))
        for s2 in store_buckets.get(fmt, [])[: 4 * k]:
            if s2 == s:
                continue
            if (s2, p, t) in inv_present:
                sold_pos = False
                if G.has_edge(s2, p):
                    if isinstance(G, nx.MultiGraph):
                        for key in G[s2][p]:
                            a = G[s2][p][key]
                            if a.get("rel") == "sold" and parse_time_attr(a.get("time", 0)) == t and float(a.get("units", 0.0)) > 0.0:
                                sold_pos = True
                                break
                    else:
                        a = G.edges[s2, p]
                        if a.get("rel") == "sold" and parse_time_attr(a.get("time", 0)) == t and float(a.get("units", 0.0)) > 0.0:
                            sold_pos = True
                if not sold_pos:
                    negatives.append((s2, p, t))
                    if len(negatives) >= k * 2:
                        break
    # Dedupe
    return list({(u, v, t) for (u, v, t) in negatives})

