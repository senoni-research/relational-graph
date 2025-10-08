from __future__ import annotations

from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import networkx as nx


Edge = Tuple[Any, Any]


class SimpleEdgeScorer(nn.Module):
    """
    Minimal edge scorer: encode node features + neighbor context, predict edge existence.
    For VN2: uses node type, degree, and optional product hierarchy features.
    """

    def __init__(
        self,
        node_types: List[str],
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_types = node_types
        self.type_to_idx = {t: i for i, t in enumerate(node_types)}
        self.hidden_dim = hidden_dim

        # Type embedding
        self.type_embed = nn.Embedding(len(node_types), hidden_dim)
        
        # Simple aggregator: mean of neighbor embeddings
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, G_sub: nx.Graph, nodes: List[Any]) -> torch.Tensor:
        """Encode a list of nodes from G_sub."""
        type_indices = []
        for n in nodes:
            ntype = G_sub.nodes[n].get("type", "unknown")
            idx = self.type_to_idx.get(ntype, 0)
            type_indices.append(idx)
        
        device = next(self.parameters()).device
        type_tensor = torch.LongTensor(type_indices).to(device)
        x = self.type_embed(type_tensor)  # (num_nodes, hidden_dim)
        
        # Simple GNN: aggregate neighbor embeddings
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        for layer in self.gnn_layers:
            x_new = x.clone()
            for i, n in enumerate(nodes):
                neighbors = list(G_sub.neighbors(n))
                if neighbors:
                    neighbor_indices = [node_to_idx[nb] for nb in neighbors if nb in node_to_idx]
                    if neighbor_indices:
                        neighbor_embeds = x[neighbor_indices]
                        x_new[i] = x_new[i] + neighbor_embeds.mean(dim=0)
            x = torch.relu(layer(x_new))
        
        return x

    def forward(self, G_sub: nx.Graph, edges: List[Edge]) -> torch.Tensor:
        """
        Predict edge probabilities for a list of edges in G_sub.
        Returns: (num_edges,) tensor of logits
        """
        # Collect all nodes involved
        nodes = list(G_sub.nodes)
        node_embeds = self.encode_nodes(G_sub, nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        edge_logits = []
        for u, v in edges:
            u_idx = node_to_idx.get(u, 0)
            v_idx = node_to_idx.get(v, 0)
            u_emb = node_embeds[u_idx]
            v_emb = node_embeds[v_idx]
            concat = torch.cat([u_emb, v_emb], dim=0)
            logit = self.edge_mlp(concat).squeeze(-1)
            edge_logits.append(logit)
        
        return torch.stack(edge_logits)


def build_model(G: nx.Graph, hidden_dim: int = 64, num_layers: int = 2) -> SimpleEdgeScorer:
    """Build a scorer from a graph (infer node types)."""
    node_types_set = set()
    for _, attrs in G.nodes(data=True):
        node_types_set.add(attrs.get("type", "unknown"))
    node_types = sorted(node_types_set)
    return SimpleEdgeScorer(node_types, hidden_dim=hidden_dim, num_layers=num_layers)

