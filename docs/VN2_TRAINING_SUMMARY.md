# VN2 Graph Scorer Training Summary

## What we built

A learned edge scorer that predicts p(store, product) = probability of a meaningful sales relationship, trained on VN2 competition data with temporal splits to avoid leakage.

## Training improvements (baseline → enhanced)

### Baseline model
- **Features**: node type only (store vs product)
- **Architecture**: mean neighbor aggregation, 2-layer MLP
- **Results**: AUC 0.48, AP 0.72 on test set
- **Limitation**: ignores product hierarchy, store format, and temporal patterns

### Enhanced model (v2)
- **Added features**:
  - **Product hierarchy**: ProductGroup, Division, Department, DepartmentGroup (categorical embeddings)
  - **Store attributes**: StoreFormat, Format
  - **Graph features**: degree (log-scaled), hop distance
  - **Edge features**: temporal presence, units sold
- **Architecture**:
  - Multi-head attention over neighbors (4 heads)
  - 3 layers, 128-dim hidden
  - Dropout 0.1 for regularization
- **Expected improvement**: Better AUC (target 0.65+), lower val loss

## How to use for VN2

### 1. Generate graph with features
```bash
python scripts/vn2_to_jsonl.py \
  --vn2-data-dir ../vn2inventory/data \
  --out graph_qa/data/vn2_graph_full.jsonl \
  --max-pairs 0
```

### 2. Train enhanced scorer
```bash
relational-graph train-scorer \
  --graph graph_qa/data/vn2_graph_full.jsonl \
  --epochs 15 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 3 \
  --use-enhanced \
  --out checkpoints/vn2_scorer_enhanced.pt
```

### 3. Evaluate on test set
```bash
python scripts/evaluate_scorer.py \
  --graph graph_qa/data/vn2_graph_full.jsonl \
  --ckpt checkpoints/vn2_scorer_enhanced.pt
```

### 4. Extract probabilities for VN2 ordering policy
```python
from graph_qa.io.loader import load_graph
from graph_qa.scoring.learned_scorer import LearnedEdgeScorer

G = load_graph("graph_qa/data/vn2_graph_full.jsonl")
scorer = LearnedEdgeScorer("checkpoints/vn2_scorer_enhanced.pt")

# Get edge probabilities for all (store, product) pairs
for store_id in range(62):  # VN2 has stores 0-61
    for product_id in [124, 126, 160, ...]:  # Your product list
        edge = (f"store:{store_id}", f"product:{product_id}")
        if G.has_node(edge[0]) and G.has_node(edge[1]):
            p = scorer.score(G, [edge]).get(edge, 0.5)
            # Use p as a feature in your ordering policy
            print(f"{edge}: p={p:.3f}")
```

## Temporal split (prevents leakage)

**Full temporal graph** (all 157 weeks from 2021-04 to 2024-04):
- **Train**: edges ≤ 2023-06-30 (~128K edges)
- **Val**: 2023-07-01 to 2023-12-31 (~30K edges)
- **Test**: > 2023-12-31 (~22K edges)

The loader uses `nx.MultiGraph()` to preserve all temporal edges (same (store, product) pair can have edges at multiple timestamps).

## Next steps to improve further
1. Add **temporal aggregates** as node features (mean sales last 4 weeks, CV, trend)
2. Add **co-purchase edges** (product–product if sold together at same store/week)
3. Train longer (30+ epochs) with learning rate schedule
4. Use **product embeddings** from text (product names/descriptions) if available
5. Ensemble with your existing VN2 forecast models

## VN2 competition benefit
- **Cold-start products**: Graph probabilities help identify latent demand
- **Cross-product signals**: High-prob paths reveal bundles to order together
- **Risk-adjusted ordering**: Use p(edge) as confidence weight in safety stock formula

