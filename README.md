## Relational Graph for VN2: Temporal Graph Training and Cost-Aware Ordering

### Overview
This repo builds a temporal MultiGraph from VN2 data, trains a calibrated edge scorer, and produces cost-aware order recommendations. It also includes subgraph decoding and a minimal QA interface, but the current focus is the VN2 competition pipeline.

What's implemented now:
- VN2 → JSONL v2 generator with ID normalization and inventory-aware zeros
- Time-aware training with enhanced features (recency, event buckets, relation-aware attention, optional ID embeddings)
- Proper temporal egonet sampling (strict pre-t), MPS-compatible training, and parallel sampling
- Checkpointed categorical vocab and calibrators (isotonic/Platt)
- Order generation with multiple policies: HB baseline, mixture base-stock, and two-sided gated (lift high‑p, cap low‑p)

### Install
- Python ≥ 3.10
- Create a venv and install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
```

### 1) Generate the temporal graph (v2)
```bash
python scripts/vn2_to_jsonl.py \
  --vn2-data-dir ../vn2inventory/data \
  --out artifacts/vn2_graph_full_temporal_v2.jsonl \
  --max-pairs 0 \
  --v2 \
  --initial-state "../vn2inventory/data/Week 0 - 2024-04-08 - Initial State.csv"
# Defaults: keep zero-sales only when inventory present; emits MultiGraph temporal edges
```

### 2) Train the scorer (time-aware, enhanced)
```bash
python -u -m graph_qa.train.trainer \
  --graph artifacts/vn2_graph_full_temporal_v2.jsonl \
  --epochs 10 --batch-size 512 --K 30 --hops 1 \
  --hidden-dim 64 --num-layers 3 --use-enhanced \
  --lr 0.001 --patience 3 --log-interval 25 --num-workers 4 \
  --time-aware \
  --out artifacts/checkpoints/v2_scorer_idemb.pt
```

### 3) Calibrate (optional but recommended)
```bash
python -u scripts/calibrate_scorer.py \
  --graph artifacts/vn2_graph_full_temporal_v2.jsonl \
  --ckpt artifacts/checkpoints/v2_scorer_idemb.pt \
  --train-end 2024-01-31 --val-end 2024-03-15 \
  --calibration-method isotonic --input-space prob \
  --out artifacts/checkpoints/calibrators/iso_val_2024-03-15_idemb.pkl
```

### 4) Generate orders (recommended: two-sided gated)
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
This prints expected-cost comparisons and sanity metrics. The resulting 599-row CSV uses `store_id,product_id,order_qty`.

To produce the platform header:
```bash
python - << 'PY'
import pandas as pd
sub = pd.read_csv('artifacts/orders_blended_two_sided.csv')
sub = sub.rename(columns={'store_id':'Store','product_id':'Product','order_qty':'0'})
sub[['Store','Product','0']].to_csv('artifacts/orders_ensembled.csv', index=False)
print('Wrote artifacts/orders_ensembled.csv')
PY
```

Alternative policies available:
- `--blend mixture` (pure mixture base-stock using calibrated p)
- `--blend mixture_min` (conservative cap: min(HB, Mixture) when p below τ)
- `--blend cap` or `--blend shrink` (legacy)

### 5) Evaluate trained models
```bash
python scripts/evaluate_scorer.py \
  --graph artifacts/vn2_graph_full_temporal_v2.jsonl \
  --ckpt artifacts/checkpoints/v2_scorer_idemb.pt \
  --time-aware
```
Reports AUC/AP (and calibrated metrics if a calibrator is supplied).

### Repo structure (selected)
```
graph_qa/
  io/loader.py                 # JSONL → NetworkX (MultiGraph support)
  sampling/temporal_egonet.py  # strict pre-t temporal egonet sampler
  train/                       # dataset, trainer, model_v2 (enhanced scorer)
  scoring/                     # calibration, interfaces
scripts/
  vn2_to_jsonl.py              # VN2 → JSONL v2 generator
  orders_vn2.py                # order generation (policies: HB, mixture, gated2)
  evaluate_scorer.py           # evaluation and slicing
  calibrate_scorer.py          # probability calibration (isotonic/Platt)
```

### Notes
- Mac (MPS) supported for training; parallel sampling uses a long‑lived process pool with fallbacks
- Time-aware positives: sold>0; hard negatives: inventory-present zeros at time t
- Checkpoints save categorical vocab and feature flags; evaluators rebuild model config accordingly

### License
This project is licensed under the MIT License. See `LICENSE` for details.
