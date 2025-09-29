from __future__ import annotations

from typing import List, Any, Optional
import networkx as nx


def top_k_paths(
    G: nx.Graph,
    src: Any,
    dst: Any,
    cost_attr: str = "cost",
    k: int = 3,
    cutoff_len: Optional[int] = 6,
) -> List[list]:
    """
    Return up to k shortest paths by edge attribute 'cost'.
    If cutoff_len is set, filter paths with hops <= cutoff_len.
    """
    if src not in G or dst not in G:
        return []
    if src == dst:
        return [[src]]

    try:
        gen = nx.shortest_simple_paths(G, source=src, target=dst, weight=cost_attr)
    except nx.NetworkXNoPath:
        return []

    paths = []
    for path in gen:
        if cutoff_len is not None and len(path) - 1 > cutoff_len:
            continue
        paths.append(path)
        if len(paths) >= k:
            break
    return paths


