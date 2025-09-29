import math
import networkx as nx

from graph_qa.scoring.relgt_stub import StubRELGTScorer
from graph_qa.decode.costs import edge_costs
from graph_qa.decode.ksp import top_k_paths


def build_graph_for_path():
    # A-B-C is close in time and same type
    # A-X-C is more distant and mixed type -> lower prob
    G = nx.Graph()
    G.add_node("A", type="person", time=10)
    G.add_node("B", type="person", time=11)
    G.add_node("C", type="person", time=12)
    G.add_node("X", type="org", time=3)
    G.add_edge("A", "B", time=11)
    G.add_edge("B", "C", time=12)
    G.add_edge("A", "X", time=4)
    G.add_edge("X", "C", time=5)
    return G


def test_stub_scores_in_0_1():
    G = build_graph_for_path()
    scorer = StubRELGTScorer()
    probs = scorer.score(G, G.edges)
    for p in probs.values():
        assert 0.0 < p < 1.0


def test_ksp_prefers_high_prob_path():
    G = build_graph_for_path()
    scorer = StubRELGTScorer()
    probs = scorer.score(G, G.edges)
    costs = edge_costs(probs)
    for (u, v), c in costs.items():
        G.edges[u, v]["cost"] = c
    paths = top_k_paths(G, "A", "C", cost_attr="cost", k=1, cutoff_len=4)
    assert paths, "No path found"
    path = paths[0]
    # Should pick A-B-C (2 hops) over A-X-C
    assert path == ["A", "B", "C"]


