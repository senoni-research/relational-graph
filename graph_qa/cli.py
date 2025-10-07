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
    scorer_ckpt: Optional[str] = typer.Option(None, "--scorer-ckpt", help="Path to trained scorer checkpoint"),
):
    G = load_graph(graph_path)
    anchors: List[str] = [x for x in [a, b] if x is not None]
    res = predict_subgraph(G, anchors, hops=hops, K=K, anchor_time=anchor_time, k_paths=k_paths, cutoff_len=cutoff_len, scorer_ckpt=scorer_ckpt)
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


@app.command("train-scorer")
def cli_train_scorer(
    graph_path: str = typer.Option(..., "--graph", help="Path to JSONL graph"),
    train_end: str = typer.Option("2024-01-31", "--train-end", help="Train cutoff (YYYY-MM-DD)"),
    val_end: str = typer.Option("2024-03-15", "--val-end", help="Val cutoff (YYYY-MM-DD)"),
    epochs: int = typer.Option(10, "--epochs"),
    batch_size: int = typer.Option(32, "--batch-size"),
    lr: float = typer.Option(0.001, "--lr"),
    hidden_dim: int = typer.Option(64, "--hidden-dim"),
    num_layers: int = typer.Option(2, "--num-layers"),
    hops: int = typer.Option(2, "--hops"),
    K: int = typer.Option(150, "--K"),
    out: str = typer.Option("checkpoints/edge_scorer.pt", "--out"),
    patience: int = typer.Option(3, "--patience"),
    use_enhanced: bool = typer.Option(False, "--use-enhanced", help="Use enhanced model with attention"),
):
    """Train an edge scorer on the graph."""
    import sys
    from .train.trainer import main as trainer_main
    
    # Monkey-patch sys.argv for argparse in trainer_main
    argv = [
        "train-scorer",
        "--graph", graph_path,
        "--train-end", train_end,
        "--val-end", val_end,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--hops", str(hops),
        "--K", str(K),
        "--out", out,
        "--patience", str(patience),
    ]
    if use_enhanced:
        argv.append("--use-enhanced")
    sys.argv = argv
    trainer_main()


