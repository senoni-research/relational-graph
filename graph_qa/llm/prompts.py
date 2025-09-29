SYSTEM_PROMPT = """You are a graph QA assistant for relational/temporal graphs. You can call tools:
- find_paths(a,b,max_len,k)
- predict_subgraph(anchors)
- relgt_score_edges(edges, context)

Always return: (1) result, (2) method (tools used), (3) evidence (edges/nodes/attributes), (4) caveats.
For causal questions, clarify assumptions and return influence/association proxies unless a causal model is configured.
"""

RELATIONSHIP_TEMPLATE = (
    "What is the relationship between {a} and {b}? "
    "Please include the top-{k} paths (≤ {L} hops) and summarize in plain English."
)

CAUSAL_TEMPLATE = (
    "What is the likelihood that {a} causes {b} in window {t0}..{t1}? "
    "If strong causal assumptions are required, list them; "
    "otherwise provide association/influence estimate and clearly label it as non-causal."
)

PREDICT_STRUCTURE_TEMPLATE = (
    "Predict a minimal subgraph connecting {anchors} at anchor_time={T} with at most {K} nodes. "
    "Return edges with probabilities and the decoding method used."
)


