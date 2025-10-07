## Relational Graph: Subgraph Prediction & LLM QA (MVP)

### Overview
Graph-structured QA that predicts structure (subgraphs), not just links. We score edges in a fixed-K temporal egonet and decode the most likely connecting subgraph between anchors using k-shortest paths and a Steiner-tree fallback. An LLM-style router calls tools to answer natural-language questions with method, evidence, and caveats.

Key ideas:
- Fixed-K temporal egonets (≤ anchor_time) to avoid future leakage and keep compute predictable.
- Pluggable edge scorer interface (RELGT-ready), with a deterministic stub for the MVP.
- Decoders over costs c(e) = −log p(e): k-shortest paths and Steiner approximations.
- ReAct-style tool routing for QA; causal questions return clearly labeled non-causal proxies unless a causal engine is configured.

### Install
- Python ≥ 3.10
- Editable install:
```bash
pip install -e .
```

### Quickstart
Demo data lives in `graph_qa/data/demo_graph.jsonl`. A simple demo notebook is in `notebooks/demo.ipynb`.

- Relationship QA (top-k paths):
```bash
relational-graph find-paths --graph graph_qa/data/demo_graph.jsonl --a A --b C --max-len 4 --k 2
```

- Predict subgraph (structure):
```bash
relational-graph predict-subgraph --graph graph_qa/data/demo_graph.jsonl --a A --b C --anchor-time 12 --hops 2 --K 50
```

- LLM router (stubbed deterministic):
```bash
relational-graph ask --graph graph_qa/data/demo_graph.jsonl --question "What is the relationship between A and C?"
relational-graph ask --graph graph_qa/data/demo_graph.jsonl --question "What is the likelihood that A causes C?"
```

### Team action step (POC with VN2 data)

**Quick demo (5 min)**
```bash
source .venv/bin/activate
relational-graph ask --graph graph_qa/data/vn2_graph_sample.jsonl --question "What is the relationship between store:0 and product:126?"
```

**Full workflow (train your own scorer)**

1) Convert VN2 CSVs to graph
```bash
python scripts/vn2_to_jsonl.py \
  --vn2-data-dir ../vn2inventory/data \
  --out graph_qa/data/vn2_graph_full.jsonl \
  --max-pairs 0  # 0 = all pairs
```

2) Train edge scorer (masked-edge objective)
```bash
relational-graph train-scorer \
  --graph graph_qa/data/vn2_graph_full.jsonl \
  --train-end 2024-01-31 \
  --val-end 2024-03-15 \
  --epochs 10 \
  --batch-size 32 \
  --out checkpoints/vn2_scorer.pt
```
This learns p(edge | graph context) on temporal train split; validates on held-out weeks.

3) Evaluate on test set
```bash
python scripts/evaluate_scorer.py \
  --graph graph_qa/data/vn2_graph_full.jsonl \
  --ckpt checkpoints/vn2_scorer.pt
```
Reports AUC/AP on edges after 2024-03-15 (temporal test split).

4) Query with learned probabilities
```bash
relational-graph predict-subgraph \
  --graph graph_qa/data/vn2_graph_full.jsonl \
  --a store:5 --b product:124 \
  --scorer-ckpt checkpoints/vn2_scorer.pt
```
Returns edges with learned p(e); paths ranked by cost = −log p(e).

**Next milestone**: Add non-causal "influence" tool for "likelihood/causes" queries using time-respecting paths.

### What’s inside
```
graph_qa/
  io/loader.py                 # JSONL → NetworkX (node.type, node.time, edge.time)
  sampling/temporal_egonet.py  # fixed-K egonet sampler (≤ anchor_time)
  scoring/
    interfaces.py              # EdgeScorer protocol
    relgt_stub.py              # logistic baseline p(e) ∈ (0,1)
    calibrate.py               # temperature/prob calibration helpers
  decode/
    costs.py                   # c(e) = -log p(e)
    ksp.py                     # top-k shortest paths over cost
    steiner.py                 # Steiner connector (networkx approx)
    path_score.py              # path scoring (NLL / geometric mean)
    complete.py                # thresholded completion (MVP)
  llm/
    tools.py                   # find_paths, predict_subgraph, relgt_score_edges
    router.py                  # minimal ReAct-like router
    prompts.py                 # system & user templates
  cli.py                       # Typer CLI entry point
  data/demo_graph.jsonl        # tiny demo graph
```

### Architecture (MVP)
```
  Graph (NetworkX)
        │
        ▼
  Temporal Egonet (≤ T, K nodes)
        │
        ▼
  Edge Scorer (RELGT-ready) ──► p(e)
        │                         │
        ▼                         ▼
      Costs c(e) = −log p(e)   Evidence (edge probs)
        │
        ├── k-shortest paths (anchors a↔b)
        └── Steiner tree (multi-anchors)
        ▼
  Predicted Subgraph + Paths
        │
        ▼
  LLM Router → tools (find_paths / predict_subgraph / score_edges)
        │
        ▼
  Answer = result + method + evidence + caveats
```

### Extending (RELGT / Griffin)
- Implement `graph_qa/scoring/relgt_wrapper.py` or `graph_qa/scoring/griffin_wrapper.py` to conform to `EdgeScorer.score(G_sub, candidate_edges) -> dict[edge, float]`. The decoders and tools work unchanged.
- Add tokenization (type, hop, Δtime, local GNN-PE) in the scorer to mirror RELGT inputs. Use temperature scaling/isotonic if calibration matters.

### Tests
```bash
pytest -q
```

### Notes & Caveats
- Causal questions: the router returns association/influence proxies and lists assumptions unless a causal engine is enabled.
- `torch_geometric` is optional (not required for MVP); we rely on NetworkX for decoding.
- The CLI script name is `relational-graph` (see `[project.scripts]` in `pyproject.toml`).

### References
- RELGT: paper [`arxiv.org/abs/2505.10960`](https://arxiv.org/abs/2505.10960) · code [`github.com/snap-stanford/relgt`](https://github.com/snap-stanford/relgt)
- ContextGNN: [`github.com/kumo-ai/ContextGNN`](https://github.com/kumo-ai/ContextGNN/blob/master/contextgnn/nn/models/contextgnn.py)
- Griffin: [`github.com/yanxwb/Griffin`](https://github.com/yanxwb/Griffin)
- Decoders: NetworkX `shortest_simple_paths`, `approximation.steinertree.steiner_tree`


