from __future__ import annotations

from collections import deque, defaultdict
from typing import Iterable, Optional, Set, Dict, Any

import networkx as nx


def _node_time(G: nx.Graph, n) -> float:
    t = G.nodes[n].get("time")
    if t is None:
        return float("-inf")
    return float(t)


def _edge_time(G: nx.Graph, u, v) -> float:
    """Get earliest edge time for (u,v). Handles MultiGraph."""
    if not G.has_edge(u, v):
        return float("-inf")
    
    if isinstance(G, nx.MultiGraph):
        # MultiGraph: iterate over all parallel edges to find min time
        times = []
        for key in G[u][v]:
            attrs = G[u][v][key]
            times.append(attrs.get("time", float("-inf")))
        return min(times) if times else float("-inf")
    else:
        # Simple Graph
        t = G.edges[u, v].get("time")
        if t is None:
            return float("-inf")
        return float(t)


def sample_temporal_egonet(
    G: nx.Graph,
    seed_nodes: Iterable,
    hops: int = 2,
    K: int = 300,
    anchor_time: Optional[float] = None,
) -> nx.Graph:
    """
    Fixed-K temporal egonet sampler.
    - Only include nodes/edges with time <= anchor_time.
    - Explore up to 'hops' from seed_nodes via BFS on the temporally filtered graph.
    - If >K nodes, downsample with priority: recency (closer to anchor_time), degree, type diversity.
    """
    seeds = list(seed_nodes)
    if not seeds:
        return nx.Graph()

    if anchor_time is None:
        # Use max seed node time as anchor
        anchor_time = max(_node_time(G, s) for s in seeds)

    # Pre-filter nodes by time
    allowed_nodes: Set = {n for n in G.nodes if _node_time(G, n) <= anchor_time}

    # BFS limited by hops on the filtered node set
    visited: Set = set()
    dist: Dict[Any, int] = {}
    q = deque([(s, 0) for s in seeds if s in allowed_nodes])
    for s in seeds:
        if s in allowed_nodes:
            dist[s] = 0
            visited.add(s)

    while q:
        node, d = q.popleft()
        if d >= hops:
            continue
        for nbr in G.neighbors(node):
            if nbr not in allowed_nodes:
                continue
            # Only traverse edges with time < anchor_time (strictly before t)
            if _edge_time(G, node, nbr) >= anchor_time:
                continue
            if nbr not in visited:
                visited.add(nbr)
                dist[nbr] = d + 1
                q.append((nbr, d + 1))

    # Build induced subgraph on visited nodes with temporal edge filter
    # Keep MultiGraph to preserve multiple events; iterate edges once to avoid duplication
    H = nx.MultiGraph()
    for n in visited:
        H.add_node(n, **G.nodes[n])

    if isinstance(G, nx.MultiGraph):
        for u, v, key, attrs in G.edges(keys=True, data=True):
            if u in visited and v in visited:
                t = attrs.get("time", float("inf"))
                if t < anchor_time:
                    H.add_edge(u, v, **attrs)
    else:
        for u, v, attrs in G.edges(data=True):
            if u in visited and v in visited:
                t = attrs.get("time", float("inf"))
                if t < anchor_time:
                    H.add_edge(u, v, **attrs)

    if H.number_of_nodes() <= K:
        return H

    # Downsample nodes with priority
    # Priority: smaller recency gap (anchor_time - node_time), larger degree, type diversity
    def recency_gap(n):
        return abs(anchor_time - _node_time(G, n))

    degs = dict(H.degree())
    # Sort by (recency_gap asc, degree desc)
    sorted_nodes = sorted(H.nodes, key=lambda n: (recency_gap(n), -degs.get(n, 0)))

    # Greedy ensure type diversity
    by_type: Dict[str, list] = defaultdict(list)
    for n in sorted_nodes:
        t = str(H.nodes[n].get("type", "NA"))
        by_type[t].append(n)

    selected: Set = set()
    # include one per type first
    for t, arr in by_type.items():
        if len(selected) < K and arr:
            selected.add(arr[0])

    # fill remaining slots
    for n in sorted_nodes:
        if len(selected) >= K:
            break
        selected.add(n)

    H2 = H.subgraph(selected).copy()
    return H2


