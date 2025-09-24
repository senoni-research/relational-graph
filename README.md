# Titanic: KumoRFM vs LightGBM

A short, reproducible comparison between KumoRFM and a LightGBM baseline on the Titanic dataset. The notebook runs end‑to‑end and reports AUROC for both methods using the same train/test split.

## Quick start

Prereqs: Python 3.9+ and a working C compiler (for LightGBM wheels, pip will fetch prebuilt binaries on macOS/arm64).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

## How to run

1. Open `notebooks/titanic.ipynb` in Jupyter Lab.
2. Select the `.venv` or `Python (kumo)` kernel.
3. Run all cells. The notebook will:
   - Load Titanic from `seaborn` and create a primary key column `id`.
   - Create a single, shared stratified train/test split keyed by `id`.
   - KumoRFM: mask test labels, predict with a `PREDICT ... FOR table.id IN (...)` query, and evaluate AUROC by aligning on the returned `ENTITY` IDs.
   - LightGBM: train with native categoricals and early stopping callbacks, then evaluate AUROC on the same test IDs.

## Authentication (Kumo)

Set your API key once in the environment, or use interactive auth when prompted by the notebook:

```bash
export KUMO_API_KEY="<your_api_key>"
```

## Contents

- `notebooks/titanic.ipynb` — the end‑to‑end experiment.

## Notes

- Both methods use the exact same stratified train/test split keyed by `id`.
- To silence LightGBM training logs, the notebook sets `verbosity=-1` and uses callbacks.
- For background on single‑table usage in KumoRFM, see the reference tutorial notebook: https://github.com/kumo-ai/kumo-rfm/blob/master/notebooks/single_table.ipynb
