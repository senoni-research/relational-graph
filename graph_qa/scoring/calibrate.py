from __future__ import annotations

from typing import Iterable, Sequence
import math


class TemperatureScaler:
    def __init__(self, T: float = 1.0):
        self.T = max(1e-6, float(T))

    def set_temperature(self, T: float) -> None:
        self.T = max(1e-6, float(T))

    def apply(self, logits: Iterable[float]) -> list[float]:
        return [1.0 / (1.0 + math.exp(-z / self.T)) for z in logits]


def temperature_scale_probs(probs: Sequence[float], T: float) -> list[float]:
    # Convert to logits, scale, convert back
    T = max(1e-6, float(T))
    out = []
    for p in probs:
        p = min(max(p, 1e-6), 1 - 1e-6)
        logit = math.log(p / (1 - p))
        logit /= T
        p2 = 1.0 / (1.0 + math.exp(-logit))
        out.append(p2)
    return out


