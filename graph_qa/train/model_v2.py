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
        rel_aware_attn: bool = False,
        event_buckets: List[int] | None = None,
        store_ids: List[Any] | None = None,
        product_ids: List[Any] | None = None,
        id_emb_dim: int = 16,
        use_seq_encoder: bool = False,
        seq_len: int = 0,
        peer_features: bool = False,
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
        
        # Relation-aware attention toggle
        self.rel_aware_attn = bool(rel_aware_attn)
        # Event-bucket windows (weeks)
        self.event_buckets = list(event_buckets) if event_buckets else []
        # Optional peer feature toggle (expects peer_mean_w{w} and peer_max_w{w} if enabled)
        self.peer_features = bool(peer_features)

        # Optional last-N event sequence encoder (u,v history)
        self.use_seq_encoder = bool(use_seq_encoder and seq_len and seq_len > 0)
        self.seq_len = int(seq_len if self.use_seq_encoder else 0)
        self.seq_hidden_dim = hidden_dim // 4 if self.use_seq_encoder else 0
        if self.use_seq_encoder:
            # Input: [delta_weeks, units, inv_flag, woy_sin]
            self.seq_gru = nn.GRU(input_size=4, hidden_size=self.seq_hidden_dim, batch_first=True)

        # Optional ID embeddings (store/product)
        self.has_id_embeddings = store_ids is not None and product_ids is not None and id_emb_dim > 0
        if self.has_id_embeddings:
            # Reserve index 0 for UNK
            self.store_id_to_idx: Dict[Any, int] = {sid: i + 1 for i, sid in enumerate(store_ids or [])}
            self.product_id_to_idx: Dict[Any, int] = {pid: i + 1 for i, pid in enumerate(product_ids or [])}
            self.store_emb = nn.Embedding(len(self.store_id_to_idx) + 1, id_emb_dim)
            self.prod_emb = nn.Embedding(len(self.product_id_to_idx) + 1, id_emb_dim)
            self.id_emb_dim = id_emb_dim
        else:
            self.store_id_to_idx = {}
            self.product_id_to_idx = {}
            self.id_emb_dim = 0

        # Attention-based GNN layers
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True))

        # Optional relation embeddings / gating for attention modulation
        # Map known relation names to indices; default to 'other'
        self.rel_to_idx: Dict[str, int] = {"sold": 0, "has_inventory": 1, "state": 2, "other": 3}
        if self.rel_aware_attn:
            # Lightweight gating path: project relation presence to a scalar gate per neighbor
            rel_gate_dim = max(16, hidden_dim // 8)
            self.rel_gate_embed = nn.Embedding(len(self.rel_to_idx), rel_gate_dim)
            self.rel_gate_proj = nn.Linear(rel_gate_dim, 1)
        
        # Edge predictor with temporal/distance features (+ optional recency + buckets)
        bucket_dim = len(self.event_buckets) * 4 if self.event_buckets else 0  # sales, stockout, inbound, onhand summaries
        # Optional peer mean/max per window
        peer_dim = (len(self.event_buckets) * 2) if (self.event_buckets and self.peer_features) else 0
        edge_feat_dim = 2 + (1 if self.recency_feature else 0) + bucket_dim + peer_dim + (self.seq_hidden_dim)
        # Add learned ID biases (store/product) via a small bias head
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim + (2 * self.id_emb_dim), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Keep hashed bias as fallback if ID embeddings are disabled
        if not self.has_id_embeddings:
            self.store_bias = nn.Embedding(65536, 1)  # large enough; will index via hashed ids
            self.product_bias = nn.Embedding(131072, 1)

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

            # Pre-compute relation-aware neighbor gates once (constant across layers)
            rel_gates = None
            if self.rel_aware_attn:
                rel_gates = torch.ones((len(nodes), max_len), dtype=torch.float32, device=device)
                # Helper to get a scalar gate for (src, dst)
                def _gate_for_pair(src_node: Any, dst_node: Any) -> float:
                    if src_node == dst_node:
                        return 1.0
                    try:
                        if G_sub.has_edge(src_node, dst_node):
                            # Aggregate unique relation types across multi-edges
                            rel_set = set()
                            if isinstance(G_sub, nx.MultiGraph):
                                data = G_sub.get_edge_data(src_node, dst_node)
                                for d in data.values():
                                    r = str(d.get("rel", "other"))
                                    rel_set.add(r)
                            else:
                                r = str(G_sub.edges[src_node, dst_node].get("rel", "other"))
                                rel_set.add(r)
                            if not rel_set:
                                rel_set.add("other")
                            emb_sum = torch.zeros(self.rel_gate_embed.embedding_dim, device=device)
                            for r in rel_set:
                                idx = self.rel_to_idx.get(r, self.rel_to_idx["other"])
                                emb_sum = emb_sum + self.rel_gate_embed.weight[idx]
                            g = torch.sigmoid(self.rel_gate_proj(emb_sum)).squeeze()
                            return float(g.item())
                        else:
                            return 1.0
                    except Exception:
                        return 1.0
                # Fill gates
                for i, indices in enumerate(neighbor_indices_list):
                    src = nodes[i]
                    for j, nb_idx in enumerate(indices):
                        dst = nodes[nb_idx]
                        rel_gates[i, j] = _gate_for_pair(src, dst)

            for attn_layer in self.attn_layers:
                # Gather neighbor contexts: (N, max_len, H)
                context = x[context_indices]
                # Relation-aware gating (scale neighbor messages)
                if self.rel_aware_attn and rel_gates is not None:
                    context = context * rel_gates.unsqueeze(-1)
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
            # Event-bucket summaries (cheap aggregations over pre-t subgraph)
            peer_means = []
            peer_maxes = []
            if self.event_buckets:
                # Consume precomputed v2 history attrs from the original graph nodes/edges if present
                # We read from G_sub edge attrs for (u,v) at the last pre-t time if available
                sales_buckets = []
                stockout_buckets = []
                inbound_buckets = []
                onhand_buckets = []
                # Initialize with zeros
                for _ in self.event_buckets:
                    sales_buckets.append(0.0)
                    stockout_buckets.append(0.0)
                    inbound_buckets.append(0.0)
                    onhand_buckets.append(0.0)
                # Try to fetch lag_sum_w{w} and inv_present_now from edge attrs (closest pre-t)
                try:
                    if G_sub.has_edge(u, v):
                        if isinstance(G_sub, nx.MultiGraph):
                            # choose edge with max time < t_anchor
                            best_edge = None
                            best_time = -1
                            data = G_sub.get_edge_data(u, v)
                            for d in data.values():
                                tt = d.get("time", -1)
                                if t_anchor is None or (tt is not None and int(tt) < int(t_anchor)):
                                    if int(tt) > best_time:
                                        best_time = int(tt)
                                        best_edge = d
                            src = best_edge if best_edge is not None else {}
                        else:
                            src = G_sub.edges[u, v]
                        for idx_w, w in enumerate(self.event_buckets):
                            val = src.get(f"lag_sum_w{w}")
                            if val is not None:
                                sales_buckets[idx_w] = float(val)
                            if self.peer_features:
                                pm = src.get(f"peer_mean_w{w}")
                                px = src.get(f"peer_max_w{w}")
                                peer_means.append(float(pm) if pm is not None else 0.0)
                                peer_maxes.append(float(px) if px is not None else 0.0)
                        inv_now = src.get("inv_present_now")
                        if inv_now is not None:
                            onhand_buckets = [float(inv_now)] * len(self.event_buckets)
                        stockout_last_w1 = src.get("stockout_last_w1")
                        if stockout_last_w1 is not None:
                            stockout_buckets = [float(bool(stockout_last_w1))] * len(self.event_buckets)
                except Exception:
                    pass
                edge_feat_list.extend(sales_buckets + stockout_buckets + inbound_buckets + onhand_buckets)
                if self.peer_features:
                    # Ensure consistent length across windows even if some are missing
                    # If not populated above (e.g., no src), fill zeros
                    if len(peer_means) != len(self.event_buckets):
                        # normalize to length |event_buckets|
                        peer_means = (peer_means + [0.0] * len(self.event_buckets))[: len(self.event_buckets)]
                    if len(peer_maxes) != len(self.event_buckets):
                        peer_maxes = (peer_maxes + [0.0] * len(self.event_buckets))[: len(self.event_buckets)]
                    edge_feat_list.extend(peer_means + peer_maxes)

            # Optional last-N sequence features via GRU
            seq_vec = None
            if self.use_seq_encoder and t_anchor is not None and isinstance(G_sub, nx.MultiGraph) and G_sub.has_edge(u, v):
                try:
                    data = G_sub.get_edge_data(u, v)
                    # collect pre-anchor events sorted by time
                    events = []
                    for d in data.values():
                        tt = d.get("time", None)
                        if tt is None or int(tt) >= int(t_anchor):
                            continue
                        units = float(d.get("units", 0.0) or 0.0)
                        inv_flag = 1.0 if bool(d.get("present", False)) else (1.0 if units > 0.0 else 0.0)
                        woy_sin = float(d.get("woy_sin", 0.0) or 0.0)
                        events.append((int(tt), units, inv_flag, woy_sin))
                    events.sort(key=lambda x: x[0])
                    # take last-N
                    last = events[-self.seq_len :] if self.seq_len > 0 else events
                    seq = []
                    prev_t = int(last[0][0]) if last else int(t_anchor)
                    for (tt, units, inv_flag, woy_sin) in last:
                        delta_w = 0
                        try:
                            delta_w = max(0, int(tt) - int(prev_t))
                        except Exception:
                            delta_w = 0
                        prev_t = tt
                        seq.append([float(delta_w), float(units), float(inv_flag), float(woy_sin)])
                    if not seq:
                        seq = [[0.0, 0.0, 0.0, 0.0]]
                    seq_t = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
                    _, h_n = self.seq_gru(seq_t)  # h_n: (1, 1, H)
                    seq_vec = h_n.squeeze(0).squeeze(0)  # (H,)
                except Exception:
                    seq_vec = None

            edge_scalars = torch.tensor(edge_feat_list, dtype=torch.float32, device=device)
            parts = [u_emb, v_emb, edge_scalars]
            if self.use_seq_encoder and seq_vec is not None:
                parts.append(seq_vec)
            concat = torch.cat(parts, dim=0)
            # Optional ID embeddings concatenated
            if self.has_id_embeddings:
                sid_idx = self.store_id_to_idx.get(str(u).replace("store:", ""), 0)
                pid_idx = self.product_id_to_idx.get(str(v).replace("product:", ""), 0)
                id_vec = torch.cat([
                    self.store_emb.weight[sid_idx],
                    self.prod_emb.weight[pid_idx],
                ], dim=0).to(device)
                concat = torch.cat([concat, id_vec], dim=0)
                logit = self.edge_mlp(concat).squeeze(-1)
            else:
                logit = self.edge_mlp(concat).squeeze(-1)
                # Fallback hashed bias terms
                try:
                    sb = abs(hash(str(u))) % 65536
                    pb = abs(hash(str(v))) % 131072
                    logit = logit + self.store_bias.weight[sb].squeeze() + self.product_bias.weight[pb].squeeze()
                except Exception:
                    pass
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
    rel_aware_attn: bool = False,
    event_buckets: List[int] | None = None,
    id_emb_dim: int = 16,
    use_seq_encoder: bool = False,
    seq_len: int = 0,
    peer_features: bool = False,
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
    
    # Collect ID vocab for optional embeddings
    store_ids = [str(n).replace("store:", "") for n, a in G.nodes(data=True) if str(a.get("type")) == "store"]
    product_ids = [str(n).replace("product:", "") for n, a in G.nodes(data=True) if str(a.get("type")) == "product"]

    model = EnhancedEdgeScorer(
        node_types,
        categorical_attrs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        recency_feature=recency_feature,
        recency_norm=recency_norm,
        rel_aware_attn=rel_aware_attn,
        event_buckets=event_buckets,
        store_ids=store_ids,
        product_ids=product_ids,
        id_emb_dim=id_emb_dim,
        use_seq_encoder=use_seq_encoder,
        seq_len=seq_len,
        peer_features=peer_features,
    )
    model.fast_mode = fast_mode
    model.skip_hopdist = skip_hopdist
    return model

