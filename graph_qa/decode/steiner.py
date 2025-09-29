from __future__ import annotations

from typing import Iterable, Set, Tuple
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

Edge = Tuple[object, object]


def steiner_connect(G: nx.Graph, terminals: Iterable) -> Set[Edge]:
    T = set(terminals)
    if not T:
        return set()
    if len(T) == 1:
        return set()
    T_sub = steiner_tree(G, T, weight="cost")
    return set(T_sub.edges())


