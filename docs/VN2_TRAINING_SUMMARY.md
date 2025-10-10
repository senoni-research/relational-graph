# VN2 Graph Ordering Guide

This guide explains what we are building for the VN2 inventory competition, why it matters, and exactly how to run the pipeline end‑to‑end. It is written for newcomers—no prior context required.

## 1) Problem we are solving

- Objective: Decide weekly order quantities for each Store×Product to minimize total cost.
- Timing: Weekly review; lead time L=2 weeks; we plan for a protection horizon H=L+R=3 weeks.
- Costs: Shortage = 1.0 per unit; Holding = 0.2 per unit per week.
- Submission: One CSV with 599 rows (exact Store×Product order specified by the platform) and a single integer column `order_qty`.

## 2) Our approach (big picture)

We use a graph model to estimate the probability that a Store×Product is “active” (has sales) in a given week, then convert probabilistic demand into orders with a cost‑aware base‑stock rule. Optionally, we blend these orders with an existing Hierarchical Bayes (HB) forecast using a conservative gate.

Pipeline overview:
1. Build a temporal graph from VN2 data (multiple edges across weeks preserved, no dedup).
2. Train a time‑aware graph classifier to predict p(active) for (store, product, week) without peeking into the future.
3. Calibrate probabilities on the validation window for reliability (isotonic calibration).
4. Convert probabilities to orders using a zero‑inflated mixture and the critical fractile β≈0.833.
5. (Optional) Blend with HB using a simple gate: keep HB when probability is high; shrink HB when probability is low.

Why this helps:
- The classifier focuses on intermittency (activation), complementing magnitude‑focused forecasts.
- Calibrated probabilities improve safety‑stock sizing and reduce decision errors.

## 3) Quickstart (copy‑paste commands)

1) Build the full temporal graph
```bash
python scripts/vn2_to_jsonl.py \
  --vn2-data-dir ../vn2inventory/data \
  --out graph_qa/data/vn2_graph_full_temporal.jsonl \
  --max-pairs 0 \
  # Zero-sales policy:
  # (default) keep zeros only when inventory is present
  # To drop all zeros: --no-sold-zeros-if-instock
  # To keep all zeros regardless of inventory: --include-sold-zeros
```

2) Train the graph classifier (time‑aware, 1‑hop; auto‑uses MPS if available)
```bash
python -u -m graph_qa.train.trainer \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --train-end 2024-01-31 \
  --val-end 2024-03-15 \
  --epochs 10 \
  --batch-size 256 \
  --hidden-dim 64 \
  --num-layers 3 \
  --K 30 \
  --hops 1 \
  --use-enhanced \
  --recency-feature --recency-norm 52 \
  --lr 0.001 \
  --patience 3 \
  --log-interval 10 \
  --num-workers 6 \
  --time-aware \
  --out checkpoints/vn2_temporal_scorer_timeaware_recency.pt
```

3) Evaluate (sanity metrics)
```bash
python -u scripts/evaluate_scorer.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --train-end 2024-01-31 \
  --val-end 2024-03-15 \
  --hops 1 --K 30 \
  --time-aware
```

4) Calibrate probabilities (isotonic on validation)
```bash
python -u scripts/calibrate_scorer.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --hops 1 --K 30 --time-aware \
  --calibration-method isotonic --input-space prob \
  --out checkpoints/calibrators/iso_val_2024-03-15.pkl
```

5) Generate features and orders (599 rows, platform order)
```bash
python -u scripts/orders_vn2.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl \
  --train-end 2024-01-31 --start-week 20240408 --horizon 3 \
  --hops 1 --K 30 \
  --submission-index sales_index.csv --state current_state.csv \
  --out orders_features.csv \
  --features-599 orders_features_599.csv \
  --submit orders_graph.csv
```

6) Optional: Blend with HB using a safe “shrink” gate
```bash
python -u scripts/orders_vn2.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl \
  --train-end 2024-01-31 --start-week 20240408 --horizon 3 \
  --hops 1 --K 30 \
  --submission-index sales_index.csv --state current_state.csv \
  --out orders_features.csv --features-599 orders_features_599.csv \
  --blend shrink --hb ../vn2inventory/submissions/orders_hierarchical_final_store_cv.csv \
  --tau 0.55 --alpha 0.5 \
  --submit-blended orders_blended.csv
```

7) Optional: small gate grid (pick best τ/α by expected cost)
```bash
python -u scripts/orders_vn2.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl \
  --train-end 2024-01-31 --start-week 20240408 --horizon 3 \
  --hops 1 --K 30 \
  --submission-index sales_index.csv --state current_state.csv \
  --out orders_features.csv --features-599 orders_features_599.csv \
  --blend shrink --hb ../vn2inventory/submissions/orders_hierarchical_final_store_cv.csv \
  --grid --tau-grid 0.50,0.55,0.60 --alpha-grid 0.30,0.50,0.70 \
  --simulate-cost --cost-shortage 1.0 --cost-holding 0.2 \
  --submit-blended orders_blended_best.csv
```

## 4) What each component does

- Temporal graph (JSONL): nodes are stores and products; edges carry weekly events (e.g., sold>0, inventory present) and time. We preserve multiple edges over time to keep history.
- Graph classifier: estimates p(active) for (store, product, week). We split by time so training never sees future weeks.
- Calibration: isotonic regression fitted on validation makes probabilities reliable (improves ECE/Brier/NLL).
- Orders (base‑stock): compute horizon mean μ(H) and std σ(H) from a zero‑inflated mixture—probability p times conditional size—then S=μ+zβσ with β=1/(1+0.2)=0.833 and zβ≈0.967, and order q=max(0,S−InventoryPosition).
- HB blend (optional): if p is high, keep HB; if p is low, shrink HB (do not increase low‑p SKUs).

## 5) Current results (sanity)

- Validation (best epoch ~10): AUC 0.6446, AP 0.7609
- Test (held‑out > 2024‑03‑15): AUC 0.6444, AP 0.7312 (empty subgraphs 0%)
- With isotonic calibration on validation (applied consistently): Test AUC ≈ 0.647–0.650. AP may dip slightly (expected for piecewise‑constant calibrators).
- Ablation: 2‑hop with K=24 and skipping hop‑distance underperformed here (Test AUC ≈ 0.615, AP ≈ 0.702). Stick with 1‑hop, K=30.

## 6) Reproducible artifacts

- Full graph: `graph_qa/data/vn2_graph_full_temporal.jsonl`
- Checkpoint: `checkpoints/vn2_temporal_scorer_timeaware_recency.pt`
- Calibrator: `checkpoints/calibrators/iso_val_2024-03-15.pkl`
- Frozen copy: `checkpoints/prod/vn2_temporal_scorer_timeaware_recency_2025-10-09.pt`

## 7) Submission checklist

- 599 rows in the platform’s Store×Product order; single `order_qty` column; integers; non‑negative.
- Start week aligns with the state file (e.g., 2024‑04‑08 corresponds to “Week 0 – 2024‑04‑08 – Initial State.csv”).
- If blending, sanity looks healthy: Top‑50 keep‑rate ≈ 1.00; Bottom‑50 shrink‑rate ≈ 1.00.

## 8) Concepts (in plain English)

- Time‑aware split: split by date so training never peeks beyond `train_end`; validation never peeks beyond `val_end`.
- Strict pre‑t subgraphs: every sampled subgraph includes only edges with time < t to avoid leakage when scoring (u,v,t).
- Zero‑inflated mixture: demand is a mixture of zeros and continuous positive sales; we keep its moments per week and sum across weeks.
- Critical fractile: with shortage=1.0 and holding=0.2/week, β=1/(1+0.2)=0.833 ⇒ zβ≈0.967 for the base‑stock target.

—

For reference in GitHub: [VN2_TRAINING_SUMMARY.md](https://github.com/senoni-research/relational-graph/blob/main/docs/VN2_TRAINING_SUMMARY.md)
# VN2 Graph Scorer Training Summary

## What we built

A learned edge scorer that predicts p(store, product) = probability of a meaningful sales relationship, trained on VN2 competition data with temporal splits to avoid leakage.

## Training improvements (enhanced)

Note: early baseline results are omitted; the production pipeline is the enhanced MultiGraph model below.

### Enhanced model v2 (MultiGraph, ~172K temporal edges)
- **Added features**:
  - **Product hierarchy**: ProductGroup, Division, Department, DepartmentGroup (categorical embeddings)
  - **Store attributes**: StoreFormat, Format
  - **Graph features**: degree (log-scaled), hop distance
  - **Edge features**: temporal presence, units sold
- **Architecture**:
  - Multi-head attention over neighbors (4 heads)
  - 3 layers, 64–128-dim hidden (latest run: 64)
  - Dropout 0.1 for regularization
- **Data**: Full temporal MultiGraph (~172K edges) across 157 weeks (2021–2024); keeps zero-sales only when inventory is present (normalized IDs)
- **Training**: Time-aware samples with positives = sold>0 and negatives = (inventory present ∧ sold==0), temporal split. Current run: Train 77,561 · Val 3,577 · Test 2,388 (samples)
- **Results**: Training infrastructure complete and tested; full training requires GPU or overnight CPU run (~2-4 hours for 15 epochs on CPU; <30 min with GPU)

## How to use for VN2

### Models: baseline vs enhanced

- Baseline model (`graph_qa/train/model.py`)
  - Fast, minimal features (node type + mean neighbor aggregation)
  - Useful for smoke tests, CI, and very limited hardware
  - Selected by default when you do NOT pass `--use-enhanced`
- Enhanced model (`graph_qa/train/model_v2.py`)
  - Rich categorical features (product hierarchy, store format), degree, attention
  - Preferred for VN2; higher accuracy, heavier compute
  - Selected with the `--use-enhanced` flag

Why keep both? The baseline provides a quick, deterministic check and a fallback for constrained environments; the enhanced model is the recommended scorer for real results.

### 1. Generate full temporal graph
```bash
python scripts/vn2_to_jsonl.py \
  --vn2-data-dir ../vn2inventory/data \
  --out graph_qa/data/vn2_graph_full_temporal.jsonl \
  --max-pairs 0 \
  # Zero-sales policy:
  # (default) keep zeros only when inventory is present
  # To drop all zeros: --no-sold-zeros-if-instock
  # To keep all zeros regardless of inventory: --include-sold-zeros
```

### 2. Train scorer (temporal)

Fast baseline (CPU-friendly):
```bash
relational-graph train-scorer \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --train-end 2023-06-30 \
  --val-end 2023-12-31 \
  --epochs 10 \
  --batch-size 512 \
  --hidden-dim 32 \
  --num-layers 2 \
  --K 30 \
  --hops 1 \
  --out checkpoints/vn2_scorer_baseline.pt
```

Enhanced (recommended):
```bash
python -u -m graph_qa.train.trainer \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --train-end 2024-01-31 \
  --val-end 2024-03-15 \
  --epochs 10 \
  --batch-size 256 \
  --hidden-dim 64 \
  --num-layers 3 \
  --K 30 \
  --hops 1 \
  --use-enhanced \
  --recency-feature --recency-norm 52 \
  --lr 0.001 \
  --patience 3 \
  --log-interval 10 \
  --num-workers 6 \
  --time-aware \
  --out checkpoints/vn2_temporal_scorer_timeaware_recency.pt
```
This trains on time-aware samples (u,v,t): positives sold>0, negatives inventory∧sold==0, using temporal splits.

### 3. Evaluate on test set
```bash
python -u scripts/evaluate_scorer.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --train-end 2024-01-31 \
  --val-end 2024-03-15 \
  --hops 1 --K 30 \
  --time-aware
```
Reports AUC/AP on temporally held-out (u,v,t) samples in 2024.

Latest results (time-aware, enhanced + recency):
- Val AUC: 0.6446, Val AP: 0.7609 (best epoch ~10)
- Test AUC: 0.6444, Test AP: 0.7312 (empty subgraphs: 0.0%)

### 4. (Optional) Calibrate probabilities with isotonic regression

Fit on the validation window and save the calibrator:
```bash
python -u scripts/calibrate_scorer.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --hops 1 --K 30 --time-aware \
  --calibration-method isotonic --input-space prob \
  --out checkpoints/calibrators/iso_val_2024-03-15.pkl
```

Apply the calibrator at evaluation/inference time:
```bash
python -u scripts/evaluate_scorer.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --hops 1 --K 30 --time-aware \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl
```

### Production checkpoint

- Checkpoint: `checkpoints/vn2_temporal_scorer_timeaware_recency.pt`
- Frozen copy: `checkpoints/prod/vn2_temporal_scorer_timeaware_recency_2025-10-09.pt`
- Calibrator: `checkpoints/calibrators/iso_val_2024-03-15.pkl`
- Train flags: `--epochs 10 --batch-size 256 --hidden-dim 64 --num-layers 3 --K 30 --hops 1 --use-enhanced --recency-feature --recency-norm 52 --lr 0.001 --patience 3 --log-interval 10 --num-workers 6 --time-aware`
- Eval flags: `--train-end 2024-01-31 --val-end 2024-03-15 --hops 1 --K 30 --time-aware`
- Metrics:
  - Raw Test AUC 0.6444, AP 0.7312
  - Calibrated Test AUC 0.6499, AP 0.7182

Example (calibrated inference):
```bash
python -u scripts/evaluate_scorer.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/prod/vn2_temporal_scorer_timeaware_recency_2025-10-09.pt \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --hops 1 --K 30 --time-aware \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl
```

### 4. Generate orders (features and submissions)

Produce features and a pure-graph submission (599 rows, exact platform order):
```bash
python -u scripts/orders_vn2.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl \
  --train-end 2024-01-31 --start-week 20240408 --horizon 3 \
  --hops 1 --K 30 \
  --submission-index sales_index.csv --state current_state.csv \
  --out orders_features.csv \
  --features-599 orders_features_599.csv \
  --submit orders_graph.csv
```

Blend with HB using a strict shrink gate (keep HB if p≥τ; else α·HB):
```bash
python -u scripts/orders_vn2.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl \
  --train-end 2024-01-31 --start-week 20240408 --horizon 3 \
  --hops 1 --K 30 \
  --submission-index sales_index.csv --state current_state.csv \
  --out orders_features.csv --features-599 orders_features_599.csv \
  --blend shrink --hb /Users/senoni/noni/vn2inventory/submissions/orders_hierarchical_final_store_cv.csv \
  --tau 0.55 --alpha 0.5 \
  --submit-blended orders_blended.csv
```

Optional: small τ/α grid with expected-cost proxy to pick a gate:
```bash
python -u scripts/orders_vn2.py \
  --graph graph_qa/data/vn2_graph_full_temporal.jsonl \
  --ckpt checkpoints/vn2_temporal_scorer_timeaware_recency.pt \
  --calibrator checkpoints/calibrators/iso_val_2024-03-15.pkl \
  --train-end 2024-01-31 --start-week 20240408 --horizon 3 \
  --hops 1 --K 30 \
  --submission-index sales_index.csv --state current_state.csv \
  --out orders_features.csv --features-599 orders_features_599.csv \
  --blend shrink --hb /Users/senoni/noni/vn2inventory/submissions/orders_hierarchical_final_store_cv.csv \
  --grid --tau-grid 0.50,0.55,0.60 --alpha-grid 0.30,0.50,0.70 \
  --simulate-cost --cost-shortage 1.0 --cost-holding 0.2 \
  --submit-blended orders_blended_best.csv
```

## Temporal split (prevents leakage)

**Full temporal graph** (all 157 weeks from 2021-04 to 2024-04):
- **Train**: edges ≤ 2024-01-31
- **Val**: 2024-02-01 to 2024-03-15
- **Test**: > 2024-03-15

The loader uses `nx.MultiGraph()` to preserve all temporal edges (same (store, product) pair can have edges at multiple timestamps).

## Key achievement: MultiGraph temporal support

**Problem solved**: NetworkX `Graph()` de-duplicates edges, so 182K temporal edges collapsed to 1.2K (only latest per pair).

**Solution**: 
- Switched to `nx.MultiGraph()` to preserve all temporal edges
- Fixed all code paths: dataset iterator, sampler edge access, model forward pass
- Result: proper training on full 3-year history (2021-2024)

## Performance notes
- **CPU training**: ~2-4 hours for 15 epochs (257K samples, batch=512, K=30, hidden=32)
- **Apple Silicon (MPS)**: Auto-detected; training runs on MPS if available and typically 1.5–3× faster than CPU
- **Recommended**: Run overnight or use GPU/MPS/Colab for faster iteration
- **Quick baseline**: Use simple model (hidden=32, layers=2, K=30) for POC; scale up later

## Next steps to improve further
1. **Run full training**: Overnight CPU run or cloud GPU (Colab, AWS) for production checkpoint
2. **Change task to regression**: predict `units sold` instead of binary edge existence → direct demand forecast
3. Add **temporal aggregates** as node features (rolling mean last 4 weeks, CV, trend)
4. Add **co-purchase edges** (product–product if sold together at same store/week)
5. Train longer (20-30 epochs) with learning rate decay
6. Ensemble with your existing VN2 forecast models

## Ablations

- 2-hop with K=24 and skip-hopdist underperformed on this graph:
  - Test AUC ≈ 0.6149, AP ≈ 0.7017
- Recommended for production: 1-hop, K=30, enhanced + recency, calibrated (as above).

## VN2 competition benefit
- **Richer demand signals**: Graph context captures co-movement patterns across stores/products
- **Cold-start handling**: Sparse products get better estimates from neighbor signals
- **Explainable decisions**: Query paths to understand why a prediction was made
- **Risk-adjusted ordering**: Use p(edge) as confidence weight in safety stock formulas

