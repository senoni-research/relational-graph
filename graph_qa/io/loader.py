from __future__ import annotations

import json
import os
from typing import Any, Dict

import networkx as nx


def load_graph(path: str, multi: bool = False, as_of: int | None = None) -> nx.Graph:
    """
    Load a graph from a JSONL file with records:
      - {"type":"node","id":<node_id>,"attrs":{...}}
      - {"type":"edge","u":<u>,"v":<v>,"attrs":{...}}
    Returns an undirected NetworkX Graph (or MultiGraph if multi=True) with node/edge attributes.
    
    Args:
        path: Path to JSONL graph
        multi: If True, return MultiGraph
        as_of: Optional temporal boundary (YYYYMMDD int); if provided, assert all edge.time < as_of
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    G = nx.MultiGraph() if multi else nx.Graph()
    leakage_violations = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec: Dict[str, Any] = json.loads(line)
                rtype = rec.get("type")
                if rtype == "node":
                    nid = rec["id"]
                    attrs = rec.get("attrs", {})
                    G.add_node(nid, **attrs)
                elif rtype == "edge":
                    u = rec["u"]
                    v = rec["v"]
                    attrs = rec.get("attrs", {})
                    # P0: as-of leakage guard
                    if as_of is not None and "time" in attrs:
                        edge_time = int(attrs["time"])
                        if edge_time >= as_of:
                            leakage_violations.append((u, v, edge_time))
                    G.add_edge(u, v, **attrs)
                else:
                    # Ignore meta or unknown records gracefully
                    continue
        
        # P0: Hard fail on leakage
        if leakage_violations:
            raise ValueError(
                f"Temporal leakage detected: {len(leakage_violations)} edges with time >= as_of ({as_of}). "
                f"First few: {leakage_violations[:5]}"
            )
        
        return G

    # Minimal edge list fallback: 'u v' per line, no attrs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[:2]
            G.add_edge(u, v)
    return G


