from __future__ import annotations

from typing import Protocol, Iterable, Tuple, Dict
import networkx as nx

Edge = Tuple[object, object]


class EdgeScorer(Protocol):
    def score(self, G_sub: nx.Graph, candidate_edges: Iterable[Edge]) -> Dict[Edge, float]:
        """
        Return probabilities p(e) in (0,1) for each candidate edge in G_sub.
        """
        ...


