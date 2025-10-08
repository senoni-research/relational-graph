from __future__ import annotations

from typing import Dict, Any, List, Tuple
import math

import torch
import torch.nn as nn
import networkx as nx


Edge = Tuple[Any, Any]


class EnhancedEdgeScorer(nn.Module):
    """
    Enhanced edge scorer with:
    - Rich node features (type, hierarchy, format, degree)
    - Edge features (units, temporal decay)
    - Attention-based neighbor aggregation
    """

    def __init__(
        self,
        node_types: List[str],
        categorical_attrs: Dict[str, List[Any]],  # e.g., {"ProductGroup": [...], "Department": [...]}
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_types = node_types
        self.type_to_idx = {t: i for i, t in enumerate(node_types)}
        self.categorical_attrs = categorical_attrs
        self.hidden_dim = hidden_dim
        self.fast_mode = False  # can be toggled by builder
        self.skip_hopdist = False  # optionally skip hop-distance computation

        # Type embedding
        self.type_embed = nn.Embedding(len(node_types), hidden_dim // 2)
        
        # Categorical embeddings for product hierarchy
        self.cat_embeds = nn.ModuleDict()
        cat_dim = hidden_dim // (2 * max(len(categorical_attrs), 1))
        for attr_name, values in categorical_attrs.items():
            vocab_size = len(values) + 1  # +1 for unknown
            self.cat_embeds[attr_name] = nn.Embedding(vocab_size, cat_dim)
        
        # Degree encoding (log scale)
        self.degree_mlp = nn.Linear(1, hidden_dim // 4)
        
        # Combine all features
        total_dim = hidden_dim // 2 + cat_dim * len(categorical_attrs) + hidden_dim // 4
        self.feature_proj = nn.Linear(total_dim, hidden_dim)
        
        # Attention-based GNN layers
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True))
        
        # Edge predictor with temporal/distance features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),  # +2 for temporal and hop distance
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_node_features(self, G_sub: nx.Graph, nodes: List[Any]) -> torch.Tensor:
        """Encode node features from graph attributes."""
        type_embeds = []
        cat_embeds_list = []
        degrees = []
        
        for n in nodes:
            # Type embedding
            ntype = G_sub.nodes[n].get("type", "unknown")
            type_idx = self.type_to_idx.get(ntype, 0)
            type_embeds.append(type_idx)
            
            # Categorical embeddings
            cat_feats = []
            for attr_name, values in self.categorical_attrs.items():
                val = G_sub.nodes[n].get(attr_name, None)
                if val in values:
                    idx = values.index(val) + 1
                else:
                    idx = 0  # unknown
                cat_feats.append(idx)
            cat_embeds_list.append(cat_feats)
            
            # Degree (log scale)
            deg = G_sub.degree(n)
            degrees.append(math.log(deg + 1))
        
        # Convert to tensors
        device = next(self.parameters()).device
        type_tensor = torch.LongTensor(type_embeds).to(device)
        type_emb = self.type_embed(type_tensor)  # (N, hidden_dim//2)
        
        cat_emb_parts = []
        for i, (attr_name, _) in enumerate(self.categorical_attrs.items()):
            cat_indices = torch.LongTensor([c[i] for c in cat_embeds_list]).to(device)
            cat_emb_parts.append(self.cat_embeds[attr_name](cat_indices))
        cat_emb = torch.cat(cat_emb_parts, dim=-1) if cat_emb_parts else torch.zeros(len(nodes), 0, device=device)
        
        degree_tensor = torch.FloatTensor(degrees).unsqueeze(-1).to(device)
        degree_emb = torch.relu(self.degree_mlp(degree_tensor))
        
        # Concatenate all features
        all_feats = torch.cat([type_emb, cat_emb, degree_emb], dim=-1)
        x = self.feature_proj(all_feats)
        return x

    def encode_nodes(self, G_sub: nx.Graph, nodes: List[Any]) -> torch.Tensor:
        """Encode nodes with attention-based aggregation."""
        x = self.encode_node_features(G_sub, nodes)  # (N, hidden_dim)
        device = x.device
        
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Fast mode: skip attention aggregation entirely
        if not self.fast_mode:
            # Vectorized neighbor attention on device (avoids Python loops per node)
            # Build padded neighbor index matrix once and reuse across layers
            neighbor_indices_list = []
            max_len = 1
            for i, n in enumerate(nodes):
                neighbors = [node_to_idx[nb] for nb in G_sub.neighbors(n) if nb in node_to_idx]
                # Include self at position 0
                indices = [i] + neighbors
                neighbor_indices_list.append(indices)
                if len(indices) > max_len:
                    max_len = len(indices)

            # Create (N, max_len) index tensor and mask
            context_indices = torch.full((len(nodes), max_len), fill_value=0, dtype=torch.long, device=device)
            key_padding_mask = torch.ones((len(nodes), max_len), dtype=torch.bool, device=device)
            for i, indices in enumerate(neighbor_indices_list):
                L = len(indices)
                context_indices[i, :L] = torch.tensor(indices, dtype=torch.long, device=device)
                key_padding_mask[i, :L] = False  # not padded

            for attn_layer in self.attn_layers:
                # Gather neighbor contexts: (N, max_len, H)
                context = x[context_indices]
                # Queries: (N, 1, H)
                query = x.unsqueeze(1)
                attn_out, _ = attn_layer(query, context, context, key_padding_mask=key_padding_mask)
                x = torch.relu(attn_out.squeeze(1))  # (N, H)
        
        return x

    def forward(self, G_sub: nx.Graph, edges: List[Edge]) -> torch.Tensor:
        """
        Predict edge probabilities for a list of edges in G_sub.
        Returns: (num_edges,) tensor of logits
        """
        nodes = list(G_sub.nodes)
        node_embeds = self.encode_nodes(G_sub, nodes)
        device = node_embeds.device
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        edge_logits = []
        for u, v in edges:
            u_idx = node_to_idx.get(u, 0)
            v_idx = node_to_idx.get(v, 0)
            u_emb = node_embeds[u_idx]
            v_emb = node_embeds[v_idx]
            
            # Add edge features: temporal proximity, hop distance
            if G_sub.has_edge(u, v):
                # For positive edges, use actual attributes
                # Handle both Graph and MultiGraph
                if isinstance(G_sub, nx.MultiGraph):
                    edge_dict = G_sub.edges[u, v]
                    edge_time = min(attrs.get("time", 0) for attrs in edge_dict.values()) if edge_dict else 0
                else:
                    edge_time = G_sub.edges[u, v].get("time", 0)
                temporal_feat = 1.0  # Present edge
                hop_dist = 1.0
            else:
                # For negative/candidate edges, compute distance
                temporal_feat = 0.0
                if self.fast_mode or self.skip_hopdist:
                    hop_dist = 2.0  # constant to avoid nx.shortest_path_length overhead
                else:
                    try:
                        hop_dist = float(nx.shortest_path_length(G_sub, u, v))
                    except:
                        hop_dist = 10.0  # Disconnected
            
            edge_feats = torch.FloatTensor([temporal_feat, hop_dist]).to(device)
            concat = torch.cat([u_emb, v_emb, edge_feats], dim=0)
            logit = self.edge_mlp(concat).squeeze(-1)
            edge_logits.append(logit)
        
        return torch.stack(edge_logits)


def build_enhanced_model(G: nx.Graph, hidden_dim: int = 128, num_layers: int = 3, fast_mode: bool = False, skip_hopdist: bool = False) -> EnhancedEdgeScorer:
    """Build an enhanced scorer from a graph (infer node types and categorical attributes)."""
    node_types_set = set()
    cat_attrs: Dict[str, set] = {}
    
    for _, attrs in G.nodes(data=True):
        node_types_set.add(attrs.get("type", "unknown"))
        # Collect categorical attributes
        for k, v in attrs.items():
            if k in ("ProductGroup", "Division", "Department", "DepartmentGroup", "StoreFormat", "Format"):
                if isinstance(v, (int, str)):
                    cat_attrs.setdefault(k, set()).add(v)
    
    node_types = sorted(node_types_set)
    categorical_attrs = {k: sorted(v) for k, v in cat_attrs.items()}
    
    model = EnhancedEdgeScorer(node_types, categorical_attrs, hidden_dim=hidden_dim, num_layers=num_layers)
    model.fast_mode = fast_mode
    model.skip_hopdist = skip_hopdist
    return model

