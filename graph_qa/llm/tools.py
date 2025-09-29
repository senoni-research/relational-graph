from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, Optional, Set
from pydantic import BaseModel, Field

import networkx as nx

from ..io.loader import load_graph
from ..sampling.temporal_egonet import sample_temporal_egonet
from ..scoring.relgt_stub import StubRELGTScorer
from ..decode.costs import edge_costs
from ..decode.ksp import top_k_paths
from ..decode.steiner import steiner_connect
from ..decode.complete import complete_subgraph

Edge = Tuple[Any, Any]


class PathResult(BaseModel):
    path: List[Any] = Field(default_factory=list)
    hops: int
    edges: List[Tuple[Any, Any]] = Field(default_factory=list)


def find_paths(G: nx.Graph, a: Any, b: Any, max_len: int = 6, k: int = 3) -> List[PathResult]:
    # Assign uniform cost if not present
    H = G.copy()
    for u, v in H.edges:
        if "cost" not in H.edges[u, v]:
            H.edges[u, v]["cost"] = 1.0
    paths = top_k_paths(H, a, b, cost_attr="cost", k=k, cutoff_len=max_len)
    out: List[PathResult] = []
    for p in paths:
        e_list = [(p[i], p[i+1]) for i in range(len(p)-1)]
        out.append(PathResult(path=p, hops=len(p)-1, edges=e_list))
    return out


def predict_subgraph(
    G: nx.Graph,
    anchors: Iterable[Any],
    hops: int = 2,
    K: int = 300,
    anchor_time: Optional[float] = None,
    k_paths: int = 3,
    cutoff_len: int = 6,
) -> Dict[str, Any]:
    anchors = list(anchors)
    sub = sample_temporal_egonet(G, anchors, hops=hops, K=K, anchor_time=anchor_time)
    scorer = StubRELGTScorer()
    cand_edges: List[Edge] = list(sub.edges)
    probs = scorer.score(sub, cand_edges)
    costs = edge_costs(probs)

    # annotate costs on subgraph
    for (u, v), c in costs.items():
        if sub.has_edge(u, v):
            sub.edges[u, v]["cost"] = c

    if len(anchors) >= 2:
        # Try k-shortest paths between first two anchors
        a, b = anchors[0], anchors[1]
        paths = top_k_paths(sub, a, b, cost_attr="cost", k=k_paths, cutoff_len=cutoff_len)
        used_edges: Set[Edge] = set()
        for p in paths:
            for i in range(len(p) - 1):
                e = (p[i], p[i+1])
                if sub.has_edge(*e):
                    used_edges.add(e)
        if not used_edges:
            # fallback to steiner over all anchors
            used_edges = steiner_connect(sub, anchors)
    else:
        used_edges = steiner_connect(sub, anchors)

    edges_out = []
    for (u, v) in used_edges:
        p = probs.get((u, v), probs.get((v, u), 0.0))
        edges_out.append({"u": u, "v": v, "p": float(p)})

    nodes_out = []
    used_nodes = set()
    for u, v in used_edges:
        used_nodes.add(u)
        used_nodes.add(v)
    for n in used_nodes:
        nodes_out.append({"id": n, **sub.nodes[n]})

    return {
        "anchors": anchors,
        "nodes": nodes_out,
        "edges": edges_out,
        "method": "temporal_egonet + stub_relgt + ksp/steiner",
    }


def relgt_score_edges(G: nx.Graph, edges: Iterable[Edge]) -> Dict[str, Any]:
    scorer = StubRELGTScorer()
    probs = scorer.score(G, edges)
    return {"edges": [{"u": u, "v": v, "p": float(p)} for (u, v), p in probs.items()]}


