from __future__ import annotations

from typing import Iterable, Tuple, Dict, Set
import networkx as nx

Edge = Tuple[object, object]


def complete_subgraph(
    G_sub: nx.Graph,
    anchors: Iterable,
    edge_probs: Dict[Edge, float],
    p_threshold: float = 0.5,
    max_nodes: int = 50,
) -> nx.Graph:
    """
    Simple masked-subgraph completion:
      - keep edges with p(e) >= threshold
      - ensure anchors are connected via kept edges if possible
      - prune to max_nodes by degree if needed
    """
    H = nx.Graph()
    for n in G_sub.nodes:
        H.add_node(n, **G_sub.nodes[n])

    kept_edges: Set[Edge] = set()
    for (u, v), p in edge_probs.items():
        if u in H and v in H and p >= p_threshold:
            kept_edges.add((u, v))

    H.add_edges_from(kept_edges)

    # If too big, prune by node degree
    if H.number_of_nodes() > max_nodes:
        deg_sorted = sorted(H.degree(), key=lambda kv: kv[1], reverse=True)
        keep = set(n for n, _ in deg_sorted[:max_nodes])
        H = H.subgraph(keep).copy()

    # Keep largest connected component to ensure feasibility
    if H.number_of_nodes() == 0:
        return H
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    H = H.subgraph(comps[0]).copy()
    return H


