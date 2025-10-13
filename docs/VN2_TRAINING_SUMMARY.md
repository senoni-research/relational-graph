# VN2 Graph Ordering Guide

This guide explains the complete pipeline for the VN2 inventory competition: building a temporal graph, training a graph classifier to predict sales activation probabilities, and generating optimal orders.

## 1) Problem

- **Objective**: Decide weekly order quantities for 599 Store×Product pairs to minimize total cost.
- **Timing**: Weekly review; lead time L=2 weeks; protection horizon H=L+R=3 weeks.
- **Costs**: Shortage = 1.0/unit; Holding = 0.2/unit/week.
- **Submission**: 599-row CSV with `order_qty` column (non-negative integers, exact platform order).

## 2) Our approach

We use a temporal graph neural network to predict p(active | store, product, week), calibrate probabilities, and convert them to base-stock orders via a zero-inflated newsvendor model. Optionally, we blend with Hierarchical Bayes (HB) forecasts using a conservative shrink gate.

**Pipeline**:
1. Build enhanced temporal graph (v2) with history features and state edges.
2. Train time-aware classifier with inventory-aware negatives, ranking loss, ID embeddings, and event buckets.
3. Calibrate probabilities (isotonic) on validation.
4. Generate orders (base-stock) and optionally blend with HB (τ/α grid).

**Key benefits**:
- Graph focuses on intermittency (activation); HB provides magnitude.
- Calibrated probabilities improve safety-stock sizing.
- Conservative blend never over-orders low-p SKUs.

## 3) Quickstart (production pipeline)

### 1) Build enhanced v2 graph
```bash
python scripts/vn2_to_jsonl.py \
  --vn2-data-dir ../vn2inventory/data \
  --out artifacts/vn2_graph_full_temporal_v2.jsonl \
  --max-pairs 0 --v2 --add-history-features --history-windows 1,2,4,8 \
  --initial-state "../vn2inventory/data/Week 0 - 2024-04-08 - Initial State.csv"
```

### 2) Train (ID embeddings, ranking loss, event buckets)
```bash
python -u -m graph_qa.train.trainer \
  --graph artifacts/vn2_graph_full_temporal_v2.jsonl \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --epochs 12 --patience 2 --batch-size 256 \
  --hidden-dim 64 --num-layers 3 --K 30 --hops 1 \
  --use-enhanced --recency-feature --recency-norm 52 \
  --time-aware --negatives inventory_only \
  --rank-loss --rank-margin 0.2 --rank-weight 0.2 \
  --event-buckets 1,2,4,8 --num-workers 8 \
  --out artifacts/checkpoints/v2_scorer_idemb.pt
```

### 3) Evaluate (with slices)
```bash
python -u scripts/evaluate_scorer.py \
  --graph artifacts/vn2_graph_full_temporal_v2.jsonl \
  --ckpt artifacts/checkpoints/v2_scorer_idemb.pt \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --hops 1 --K 30 --time-aware
```

### 4) Calibrate (isotonic)
```bash
python -u scripts/calibrate_scorer.py \
  --graph artifacts/vn2_graph_full_temporal_v2.jsonl \
  --ckpt artifacts/checkpoints/v2_scorer_idemb.pt \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --hops 1 --K 30 --time-aware \
  --calibration-method isotonic --input-space prob \
  --out artifacts/checkpoints/calibrators/iso_val_2024-03-15_idemb.pkl
```

### 5) Orders (two-sided gated recommended; mixture available)
```bash
python -u scripts/orders_vn2.py \
  --graph artifacts/vn2_graph_full_temporal_v2.jsonl \
  --ckpt artifacts/checkpoints/v2_scorer_idemb.pt \
  --calibrator artifacts/checkpoints/calibrators/iso_val_2024-03-15_idemb.pkl \
  --train-end 2024-01-31 --start-week 20240408 --horizon 3 \
  --hops 1 --K 30 \
  --submission-index artifacts/sales_index.csv \
  --state "../vn2inventory/data/Week 0 - 2024-04-08 - Initial State.csv" \
  --out artifacts/orders_features_idemb_gated2.csv \
  --features-599 artifacts/orders_features_599_idemb_gated2.csv \
  --hb ../vn2inventory/submissions/orders_hierarchical_final_store_cv.csv \
  --blend gated2 --grid --tau-grid 0.55,0.60,0.65 --tau-margin 0.10 \
  --abc-quantiles 0.6,0.9 --beta-a 0.88 --beta-b 0.80 --beta-c 0.70 \
  --simulate-cost --cost-shortage 1.0 --cost-holding 0.2 \
  --submit-blended artifacts/orders_blended_two_sided.csv
```

Notes:
- Two-sided gated (gated2) lifts HB when p is high (avoid under-ordering) and caps only low-p B/C items; A-class is never capped.
- Alternative: mixture policies
  - `--blend mixture` (pure mixture base-stock using calibrated p)
  - `--blend mixture_min` (conservative cap: min(HB, Mixture) for p below τ)

## 4) Production checkpoint (ID-embed, Oct 2025)

- **Graph**: `artifacts/vn2_graph_full_temporal_v2.jsonl` (v2 generator: history features, week-of-year, state edge)
- **Checkpoint**: `artifacts/checkpoints/v2_scorer_idemb.pt`
- **Calibrator**: `artifacts/checkpoints/calibrators/iso_val_2024-03-15_idemb.pkl`
- **Snapshot**: `artifacts/runs/20251012_183620_v2_idemb_h1k30_recency_buckets_rank/`

**Train flags**:
- `--epochs 12 --patience 2 --batch-size 256 --hidden-dim 64 --num-layers 3 --K 30 --hops 1`
- `--use-enhanced --recency-feature --recency-norm 52`
- `--time-aware --negatives inventory_only`
- `--rank-loss --rank-margin 0.2 --rank-weight 0.2`
- `--event-buckets 1,2,4,8 --num-workers 8`

**Metrics**:
- Val AUC: 0.6627, AP: 0.7792 (epoch 6)
- Test AUC: 0.6593, AP: 0.7574 (raw)
- Test AUC: 0.6628, AP: 0.7341 (calibrated isotonic)
- Slices (calibrated):
  - Seen-before: AUC 0.7082, AP 0.8198 (1306 samples)
  - Cold: AUC 0.5950, AP 0.6452 (1082 samples)
  - Inv-present: AUC 0.6603, AP 0.7573 (all 2388)

**Orders recommendation (current run)**:
- Submit: `artifacts/orders_blended_two_sided.csv`
- Two-sided: τ_hi=0.55, τ_lo=0.45 (ABC betas: A=0.88, B=0.80, C=0.70)
- Expected cost: ≈ 3509.40 vs HB ≈ 4603.93 (∆ ≈ −1094)
- Sanity: top-50 keep-rate ≈ 1.00; bottom-50 shrink-rate ≈ 0.00 (lift-only in this run)

## 5) Key features

**Graph v2 enhancements**:
- Tri-state inventory parsing (True/False/NA)
- History on sold edges: seen_before, recency_weeks, lag_sum_w{1,2,4,8}, inv_present_now, stockout_last_w1
- Week-of-year seasonality (sin/cos)
- State edge from Initial State (onhand, in-transit) at anchor week
- Fixed Master split: products get ProductGroup/Division/Department/DepartmentGroup only; stores get StoreFormat/Format/Region

**Model improvements**:
- Store/product ID embeddings (16-d) for learned fixed effects
- Consumes v2 history features from edges (lag sums, recency, WoY)
- Relation-aware attention (optional)
- Event-bucket hooks for short-history summaries

**Training**:
- Inventory-aware negatives (sold==0 ∧ has_inventory)
- Hard-negative mining (same store/similar products; same product/similar stores)
- Pairwise ranking loss (BPR) alongside BCE
- MultiGraph egonets without duplication (all pre-t parallel edges preserved)

**Evaluation**:
- Slice metrics (seen-before vs cold; inv-present vs no-inv)
- Isotonic calibration on validation; applied at inference

## 6) Temporal split (prevents leakage)

- **Train**: edges ≤ 2024-01-31
- **Val**: 2024-02-01 to 2024-03-15
- **Test**: > 2024-03-15
- **Sampler**: strict pre‑t (time < anchor_time); MultiGraph preserved

## 7) Submission checklist

- 599 rows in platform's Store×Product order
- Single `order_qty` column (non-negative integers)
- Start week 2024-04-08 aligns with Initial State
- Blended sanity: top-50 keep-rate ≈ 1.00; bottom-50 shrink-rate small (≈ 0.0–0.3 depending on policy)

### Convert to platform header
If needed, convert `store_id,product_id,order_qty` to `Store,Product,0`:
```bash
python - << 'PY'
import pandas as pd
sub = pd.read_csv('artifacts/orders_blended_two_sided.csv')
sub = sub.rename(columns={'store_id':'Store','product_id':'Product','order_qty':'0'})
sub[['Store','Product','0']].to_csv('artifacts/orders_ensembled.csv', index=False)
print('Wrote artifacts/orders_ensembled.csv')
PY
```

## 8) Next improvements (optional)

- Add peer velocity features (by product category across peer stores) to lift cold-slice AUC.
- Try lower rank-loss weight (0.1) if AUC plateaus.
- Optionally: 2-hop (K=24, time-strict) or explicit no_inventory edges with relation-aware attention on.

---

For reference: [VN2_TRAINING_SUMMARY.md](https://github.com/senoni-research/relational-graph/blob/main/docs/VN2_TRAINING_SUMMARY.md)