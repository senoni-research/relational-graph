import networkx as nx

from graph_qa.io.loader import load_graph
from graph_qa.sampling.temporal_egonet import sample_temporal_egonet


def build_tiny_graph():
    G = nx.Graph()
    # nodes with times
    G.add_node("A", type="person", time=10)
    G.add_node("B", type="person", time=11)
    G.add_node("C", type="person", time=12)
    G.add_node("X", type="org", time=5)
    # edges with times
    G.add_edge("A", "B", time=11, rel="knows")
    G.add_edge("B", "C", time=12, rel="knows")
    G.add_edge("A", "X", time=6, rel="member_of")
    return G


def test_temporal_cutoff_and_K():
    G = build_tiny_graph()
    # anchor at time=11 => node C (time 12) should be excluded
    H = sample_temporal_egonet(G, ["A"], hops=2, K=2, anchor_time=11)
    assert all(H.nodes[n]["time"] <= 11 for n in H.nodes)
    assert H.number_of_nodes() <= 2


def test_hops_respected():
    G = build_tiny_graph()
    H = sample_temporal_egonet(G, ["A"], hops=1, K=10, anchor_time=12)
    # with hops=1, from A we can reach B and X only
    assert set(H.nodes) <= {"A", "B", "X"}


