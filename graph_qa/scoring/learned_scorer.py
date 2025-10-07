from __future__ import annotations

from typing import Dict, Any, Tuple
from pathlib import Path

import torch
import networkx as nx

from graph_qa.train.model import SimpleEdgeScorer


Edge = Tuple[Any, Any]


class LearnedEdgeScorer:
    """
    Loads a trained SimpleEdgeScorer checkpoint and scores edges.
    Conforms to EdgeScorer protocol: score(G_sub, edges) -> {edge: p}.
    """

    def __init__(self, ckpt_path: str | Path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model = SimpleEdgeScorer(
            node_types=ckpt["node_types"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def score(self, G_sub: nx.Graph, edges: list) -> Dict[Edge, float]:
        """Return {edge: p(edge)} for candidate edges in G_sub."""
        if not edges:
            return {}
        with torch.no_grad():
            logits = self.model(G_sub, edges)
            probs = torch.sigmoid(logits).numpy()
        return {e: float(p) for e, p in zip(edges, probs)}

