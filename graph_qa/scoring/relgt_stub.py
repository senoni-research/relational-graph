from __future__ import annotations

from typing import Iterable, Tuple, Dict
import math

import networkx as nx

from .interfaces import Edge


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class StubRELGTScorer:
    """
    Deterministic logistic baseline over hand-crafted features:
      - time proximity: smaller |t_u - t_v| increases prob
      - degree centrality: higher deg(u)+deg(v) increases prob (contextual)
      - same type bonus
      - edge recency: closer edge time to max(node times) gets small boost
    """

    def __init__(self, w_time: float = -0.4, w_deg: float = 0.08, w_same: float = 0.6, w_edge_time: float = 0.1, bias: float = -0.5, temperature: float = 1.0):
        self.w_time = w_time
        self.w_deg = w_deg
        self.w_same = w_same
        self.w_edge_time = w_edge_time
        self.bias = bias
        self.temperature = temperature

    def score(self, G_sub: nx.Graph, candidate_edges: Iterable[Edge]) -> Dict[Edge, float]:
        if not isinstance(G_sub, nx.Graph):
            raise TypeError("G_sub must be a NetworkX Graph")

        deg = dict(G_sub.degree())
        # Anchor for edge recency
        max_node_time = max((G_sub.nodes[n].get("time", 0.0) for n in G_sub.nodes), default=0.0)

        out: Dict[Edge, float] = {}
        for u, v in candidate_edges:
            if not G_sub.has_node(u) or not G_sub.has_node(v):
                continue
            tu = float(G_sub.nodes[u].get("time", 0.0))
            tv = float(G_sub.nodes[v].get("time", 0.0))
            dt = abs(tu - tv)

            same_type = 1.0 if (G_sub.nodes[u].get("type") == G_sub.nodes[v].get("type")) else 0.0
            deg_sum = float(deg.get(u, 0) + deg.get(v, 0))

            et = float(G_sub.edges[u, v].get("time", min(tu, tv))) if G_sub.has_edge(u, v) else min(tu, tv)
            edge_recency = -abs(max_node_time - et)

            z = (
                self.bias
                + self.w_time * dt
                + self.w_deg * deg_sum
                + self.w_same * same_type
                + self.w_edge_time * edge_recency
            )
            z = z / max(self.temperature, 1e-6)
            p = _sigmoid(z)
            # clamp for numerical stability
            p = max(1e-6, min(1 - 1e-6, p))
            out[(u, v)] = p
        return out


