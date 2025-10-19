"""Metadata and reproducibility utilities for VN2 artifacts."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def write_meta_json(
    output_path: str | Path,
    as_of: str | int,
    train_end: str | int | None = None,
    val_end: str | int | None = None,
    horizon: int | None = None,
    hops: int | None = None,
    K: int | None = None,
    seed: int | None = None,
    counts: Dict[str, int] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Write metadata JSON next to output artifact for reproducibility and traceability.
    
    Args:
        output_path: Path to artifact (will write .meta.json next to it)
        as_of: Temporal boundary (YYYY-MM-DD or YYYYMMDD) - no data after this date
        train_end: Training cutoff
        val_end: Validation cutoff
        horizon: Forecast horizon weeks
        hops: Graph sampling hops
        K: Max nodes in subgraph
        seed: Random seed
        counts: Dict of counts (e.g., {"total_edges": 123, "positives": 45})
        extra: Additional metadata
    """
    meta = {
        "as_of": str(as_of),
        "generated_at": datetime.now().isoformat(),
        "code_version": get_git_sha(),
    }
    
    if train_end is not None:
        meta["train_end"] = str(train_end)
    if val_end is not None:
        meta["val_end"] = str(val_end)
    if horizon is not None:
        meta["horizon_weeks"] = int(horizon)
    if hops is not None:
        meta["hops"] = int(hops)
    if K is not None:
        meta["K"] = int(K)
    if seed is not None:
        meta["seed"] = int(seed)
    if counts is not None:
        meta["counts"] = counts
    if extra is not None:
        meta.update(extra)
    
    # Write to .meta.json
    out_path = Path(output_path)
    meta_path = out_path.parent / f"{out_path.stem}.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"📋 Metadata written to {meta_path}")


def load_meta_json(artifact_path: str | Path) -> Dict[str, Any]:
    """Load metadata from artifact's .meta.json sidecar."""
    path = Path(artifact_path)
    meta_path = path.parent / f"{path.stem}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)

