import json
import networkx as nx

from graph_qa.llm.tools import find_paths, predict_subgraph
from graph_qa.llm.router import Router


def build_demo():
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


def test_find_paths_schema():
    G = build_demo()
    res = find_paths(G, "A", "C", max_len=4, k=2)
    assert isinstance(res, list)
    assert res and res[0].path[0] == "A"


def test_predict_subgraph_shapes():
    G = build_demo()
    res = predict_subgraph(G, anchors=["A", "C"], hops=2, K=50, anchor_time=12, k_paths=2, cutoff_len=4)
    assert "nodes" in res and "edges" in res
    assert any(e["u"] == "A" and e["v"] in {"B", "X"} or e["v"] == "A" for e in res["edges"])


def test_router_relationship():
    G = build_demo()
    r = Router()
    ans = r.ask(G, "What is the relationship between A and C?")
    assert "result" in ans and "paths" in ans["result"]


