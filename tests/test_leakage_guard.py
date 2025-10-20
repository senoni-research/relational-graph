"""Test temporal leakage detection in graph loader."""

import json
import tempfile
from pathlib import Path

import pytest

from graph_qa.io.loader import load_graph


def test_loader_detects_leakage():
    """Loader should raise ValueError when edges violate as_of boundary."""
    
    # Create a test graph with a leakage edge
    records = [
        {"type": "node", "id": "store:0", "attrs": {"type": "store"}},
        {"type": "node", "id": "product:1", "attrs": {"type": "product"}},
        {"type": "edge", "u": "store:0", "v": "product:1", "attrs": {"rel": "sold", "time": 20240101, "units": 5.0}},
        {"type": "edge", "u": "store:0", "v": "product:1", "attrs": {"rel": "sold", "time": 20240415, "units": 3.0}},  # After as_of
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    
    try:
        # Should succeed without as_of
        G = load_graph(tmp_path, multi=True, as_of=None)
        assert G.number_of_edges() == 2
        
        # Should fail with as_of=20240410 (one edge is 20240415)
        with pytest.raises(ValueError, match="Temporal leakage detected"):
            load_graph(tmp_path, multi=True, as_of=20240410)
        
        # Should succeed with as_of=20240420 (both edges before)
        G2 = load_graph(tmp_path, multi=True, as_of=20240420)
        assert G2.number_of_edges() == 2
        
    finally:
        Path(tmp_path).unlink()


def test_loader_passes_without_time_attr():
    """Loader should not fail on edges without 'time' attribute."""
    
    records = [
        {"type": "node", "id": "a", "attrs": {}},
        {"type": "node", "id": "b", "attrs": {}},
        {"type": "edge", "u": "a", "v": "b", "attrs": {"weight": 1.0}},  # No time
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    
    try:
        # Should succeed even with as_of (no time to check)
        G = load_graph(tmp_path, as_of=20240101)
        assert G.number_of_edges() == 1
    finally:
        Path(tmp_path).unlink()


if __name__ == "__main__":
    test_loader_detects_leakage()
    test_loader_passes_without_time_attr()
    print("✅ All leakage guard tests passed")

