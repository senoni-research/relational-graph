from __future__ import annotations

from typing import Any, Dict, Optional, Iterable
import re
import networkx as nx

from .prompts import SYSTEM_PROMPT
from .tools import find_paths, predict_subgraph, relgt_score_edges


class Router:
    """
    Minimal, deterministic ReAct-like router stub for tests.
    """

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    def ask(
        self,
        G: nx.Graph,
        question: str,
        *,
        max_len: int = 6,
        k_paths: int = 3,
        anchors: Optional[Iterable] = None,
    ) -> Dict[str, Any]:
        # Use lowercase only for intent matching; preserve original for entity extraction
        q_lower = question.lower().strip()

        # Relationship: route to find_paths if two tokens A-B present
        m = re.findall(r"between\s+([^\s]+)\s+and\s+([^\s\?\.]+)", question, flags=re.IGNORECASE)
        if "relationship" in q_lower and m:
            a_raw, b_raw = m[0]
            # Map tokens case-insensitively to existing node IDs
            node_map = {str(n).lower(): n for n in G.nodes}
            a = node_map.get(str(a_raw).lower(), a_raw)
            b = node_map.get(str(b_raw).lower(), b_raw)
            paths = find_paths(G, a, b, max_len=max_len, k=k_paths)
            return {
                "result": {"paths": [p.model_dump() for p in paths]},
                "method": ["find_paths"],
                "evidence": {"k": k_paths, "max_len": max_len},
                "caveats": "Path-based association only; not causal.",
            }

        # Predict structure
        if "predict" in q_lower and "subgraph" in q_lower:
            anc = list(anchors or [])
            res = predict_subgraph(G, anc)
            return {
                "result": res,
                "method": ["predict_subgraph"],
                "evidence": {"anchors": anc},
                "caveats": "Uses heuristic scorer; probabilities are not calibrated.",
            }

        # Causal queries: return influence proxy disclaimer
        if "cause" in q_lower or "causes" in q_lower or "causal" in q_lower:
            return {
                "result": {"message": "Causal inference not configured; returning non-causal influence proxy."},
                "method": [],
                "evidence": {},
                "caveats": "Requires strong assumptions; use a causal engine for valid estimates.",
            }

        # Default: noop
        return {
            "result": {"message": "No route matched."},
            "method": [],
            "evidence": {},
            "caveats": "Provide a relationship or predict-subgraph query.",
        }


