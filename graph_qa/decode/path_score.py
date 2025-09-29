from __future__ import annotations

import math
from typing import Dict, List, Tuple

Edge = Tuple[object, object]


def path_nll(path: List, edge_probs: Dict[Edge, float]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        e_rev = (path[i + 1], path[i])
        p = edge_probs.get(e, edge_probs.get(e_rev, 1e-6))
        total += -math.log(max(1e-8, min(1 - 1e-8, p)))
    return total


def score_path(path: List, edge_probs: Dict[Edge, float], method: str = "sum", prior: float = 1.0) -> float:
    """
    Smaller is better if method == 'sum' (NLL).
    If method == 'geom', return geometric mean probability (larger is better).
    """
    if method == "sum":
        return path_nll(path, edge_probs) - math.log(max(1e-8, prior))
    elif method == "geom":
        logp = 0.0
        L = max(1, len(path) - 1)
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            e_rev = (path[i + 1], path[i])
            p = edge_probs.get(e, edge_probs.get(e_rev, 1e-6))
            logp += math.log(max(1e-8, min(1 - 1e-8, p)))
        return math.exp(logp / L) * prior
    else:
        raise ValueError(f"Unknown method: {method}")


