from __future__ import annotations

import math
from typing import Dict, Tuple

Edge = Tuple[object, object]


def edge_costs(edge_probs: Dict[Edge, float]) -> Dict[Edge, float]:
    return {e: -math.log(max(1e-8, min(1 - 1e-8, p))) for e, p in edge_probs.items()}


