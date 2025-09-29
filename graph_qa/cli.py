from __future__ import annotations

import json
from typing import List, Optional

import typer
import networkx as nx

from .io.loader import load_graph
from .llm.tools import find_paths, predict_subgraph
from .llm.router import Router


app = typer.Typer(help="Graph QA CLI")


@app.command("find-paths")
def cli_find_paths(
    graph_path: str = typer.Option(..., "--graph", help="Path to JSONL graph"),
    a: str = typer.Option(..., "--a"),
    b: str = typer.Option(..., "--b"),
    max_len: int = typer.Option(6, "--max-len"),
    k: int = typer.Option(3, "--k"),
):
    G = load_graph(graph_path)
    paths = find_paths(G, a, b, max_len=max_len, k=k)
    print(json.dumps([p.model_dump() for p in paths], indent=2))


@app.command("predict-subgraph")
def cli_predict_subgraph(
    graph_path: str = typer.Option(..., "--graph", help="Path to JSONL graph"),
    a: Optional[str] = typer.Option(None, "--a"),
    b: Optional[str] = typer.Option(None, "--b"),
    anchor_time: Optional[float] = typer.Option(None, "--anchor-time"),
    hops: int = typer.Option(2, "--hops"),
    K: int = typer.Option(300, "--K"),
    k_paths: int = typer.Option(3, "--k"),
    cutoff_len: int = typer.Option(6, "--max-len"),
):
    G = load_graph(graph_path)
    anchors: List[str] = [x for x in [a, b] if x is not None]
    res = predict_subgraph(G, anchors, hops=hops, K=K, anchor_time=anchor_time, k_paths=k_paths, cutoff_len=cutoff_len)
    print(json.dumps(res, indent=2))


@app.command("ask")
def cli_ask(
    graph_path: str = typer.Option(..., "--graph", help="Path to JSONL graph"),
    question: str = typer.Option(..., "--question"),
):
    G = load_graph(graph_path)
    router = Router()
    ans = router.ask(G, question)
    print(json.dumps(ans, indent=2))


