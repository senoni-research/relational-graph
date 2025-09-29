from typing import Optional
from pydantic import BaseModel


class SamplerConfig(BaseModel):
    hops: int = 2
    K: int = 300
    anchor_time: Optional[float] = None


class DecodeConfig(BaseModel):
    k_paths: int = 3
    cutoff_len: int = 6
    p_threshold: float = 0.5


