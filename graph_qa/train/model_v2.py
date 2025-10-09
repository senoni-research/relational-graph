from __future__ import annotations

from typing import Dict, Any, List, Tuple
import math
import datetime as _dt

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
        recency_feature: bool = False,
        recency_norm: float = 52.0,
    ):
        super().__init__()
        # Ensure explicit UNK handling and stable categorical normalization
        self.node_types = ["unknown"] + [t for t in node_types if t != "unknown"]
        self.type_to_idx = {t: i for i, t in enumerate(self.node_types)}
        self.categorical_attrs = {k: [str(v) for v in vals] for k, vals in categorical_attrs.items()}
        self.hidden_dim = hidden_dim
        self.fast_mode = False  # can be toggled by builder
        self.skip_hopdist = False  # optionally skip hop-distance computation
        # Optional temporal recency scalar (log-scaled) appended to edge features
        self.recency_feature = bool(recency_feature)
        self.recency_norm = float(recency_norm)

        # Type embedding
        self.type_embed = nn.Embedding(len(self.node_types), hidden_dim // 2)
        
        # Categorical embeddings for product hierarchy
        self.cat_embeds = nn.ModuleDict()
        # Precompute value->index maps (0=UNK) for O(1) lookup
        self.cat_val2idx: Dict[str, Dict[str, int]] = {}
        cat_dim = hidden_dim // (2 * max(len(self.categorical_attrs), 1))
        for attr_name, values in self.categorical_attrs.items():
            vocab_size = len(values) + 1  # +1 for UNK index 0
            self.cat_embeds[attr_name] = nn.Embedding(vocab_size, cat_dim)
            self.cat_val2idx[attr_name] = {str(v): (i + 1) for i, v in enumerate(values)}
        
        # Degree encoding (log scale)
        self.degree_mlp = nn.Linear(1, hidden_dim // 4)
        
        # Combine all features
        total_dim = hidden_dim // 2 + cat_dim * len(categorical_attrs) + hidden_dim // 4
        self.feature_proj = nn.Linear(total_dim, hidden_dim)
        
        # Attention-based GNN layers
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True))
        
        # Edge predictor with temporal/distance features (+ optional recency)
        edge_feat_dim = 2 + (1 if self.recency_feature else 0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
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
            
            # Categorical embeddings (O(1) lookup with UNK handling)
            cat_feats = []
            for attr_name, values in self.categorical_attrs.items():
                val = G_sub.nodes[n].get(attr_name, None)
                idx = self.cat_val2idx.get(attr_name, {}).get(str(val), 0)
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

    def forward(self, G_sub: nx.Graph, edges: List[Tuple[Any, ...]], anchor_times: List[int] | None = None) -> torch.Tensor:
        """
        Predict edge probabilities for a list of edges in G_sub.
        Returns: (num_edges,) tensor of logits
        """
        nodes = list(G_sub.nodes)
        node_embeds = self.encode_nodes(G_sub, nodes)
        device = node_embeds.device
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        edge_logits = []
        
        def _delta_in_weeks(t_new: int, t_old: int) -> int:
            """Return non-negative difference in weeks between t_new and t_old.
            Supports YYYYMMDD ints or week indices.
            """
            try:
                if 19000101 <= t_new <= 21001231 and 19000101 <= t_old <= 21001231:
                    y1, m1, d1 = t_new // 10000, (t_new % 10000) // 100, t_new % 100
                    y0, m0, d0 = t_old // 10000, (t_old % 10000) // 100, t_old % 100
                    days = max(0, (_dt.date(y1, m1, d1).toordinal() - _dt.date(y0, m0, d0).toordinal()))
                    return days // 7
                return max(0, int(t_new) - int(t_old))
            except Exception:
                return max(0, int(t_new) - int(t_old))
        for idx_edge, e in enumerate(edges):
            # Support (u, v) or (u, v, t_anchor)
            if isinstance(e, (tuple, list)) and len(e) == 3:
                u, v, t_anchor = e  # type: ignore
            else:
                u, v = e  # type: ignore
                t_anchor = None if anchor_times is None else anchor_times[idx_edge]

            u_idx = node_to_idx.get(u, 0)
            v_idx = node_to_idx.get(v, 0)
            u_emb = node_embeds[u_idx]
            v_emb = node_embeds[v_idx]
            
            # Add edge features: seen-before (anchored to t), optional recency, and hop distance
            seen_before = False
            recency_scalar = 0.0
            if G_sub.has_edge(u, v):
                if isinstance(G_sub, nx.MultiGraph):
                    if t_anchor is None:
                        seen_before = True
                    else:
                        data = G_sub.get_edge_data(u, v)
                        prev_ts = [attrs.get("time", -1) for attrs in data.values() if attrs.get("time", -1) is not None]
                        prev_ts = [int(t) for t in prev_ts if int(t) < int(t_anchor)]
                        if prev_ts:
                            seen_before = True
                            if self.recency_feature:
                                delta = max(0, int(t_anchor) - max(prev_ts))
                                # log-scaled recency normalized by recency_norm
                                recency_scalar = math.log1p(delta) / math.log1p(self.recency_norm)
                else:
                    # Graph subgraph is already strictly pre-t; presence implies seen before
                    seen_before = True

            temporal_feat = 1.0 if seen_before else 0.0

            if self.fast_mode or self.skip_hopdist:
                hop_dist = 2.0  # constant to avoid nx.shortest_path_length overhead
            else:
                try:
                    hop_dist = float(nx.shortest_path_length(G_sub, u, v))
                except:
                    hop_dist = 10.0  # Disconnected

            edge_feat_list = [temporal_feat, hop_dist]
            if self.recency_feature:
                # Convert recency to weeks if timestamps are calendar dates
                if t_anchor is not None and seen_before:
                    # prev_ts were already filtered to < t_anchor above
                    # Recompute weeks from the max previous timestamp to anchor
                    try:
                        weeks = _delta_in_weeks(int(t_anchor), int(max(prev_ts)) if 'prev_ts' in locals() and prev_ts else int(t_anchor))
                    except Exception:
                        weeks = 0
                    denom = math.log1p(max(1.0, float(self.recency_norm)))
                    recency_scalar = math.log1p(float(weeks)) / denom
                edge_feat_list.append(recency_scalar)
            edge_feats = torch.tensor(edge_feat_list, dtype=torch.float32, device=device)
            concat = torch.cat([u_emb, v_emb, edge_feats], dim=0)
            logit = self.edge_mlp(concat).squeeze(-1)
            edge_logits.append(logit)
        
        return torch.stack(edge_logits)


def build_enhanced_model(
    G: nx.Graph,
    hidden_dim: int = 128,
    num_layers: int = 3,
    fast_mode: bool = False,
    skip_hopdist: bool = False,
    recency_feature: bool = False,
    recency_norm: float = 52.0,
) -> EnhancedEdgeScorer:
    """Build an enhanced scorer from a graph (infer node types and categorical attributes)."""
    node_types_set = set()
    cat_attrs: Dict[str, set] = {}
    
    for _, attrs in G.nodes(data=True):
        node_types_set.add(attrs.get("type", "unknown"))
        # Collect categorical attributes (normalize to strings)
        for k, v in attrs.items():
            if k in ("ProductGroup", "Division", "Department", "DepartmentGroup", "StoreFormat", "Format"):
                cat_attrs.setdefault(k, set()).add(str(v))
    
    node_types = ["unknown"] + sorted(t for t in node_types_set if t != "unknown")
    categorical_attrs = {k: sorted(v) for k, v in cat_attrs.items()}
    
    model = EnhancedEdgeScorer(
        node_types,
        categorical_attrs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        recency_feature=recency_feature,
        recency_norm=recency_norm,
    )
    model.fast_mode = fast_mode
    model.skip_hopdist = skip_hopdist
    return model

