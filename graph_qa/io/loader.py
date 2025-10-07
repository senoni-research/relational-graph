from __future__ import annotations

import json
import os
from typing import Any, Dict

import networkx as nx


def load_graph(path: str, multi: bool = False) -> nx.Graph:
    """
    Load a graph from a JSONL file with records:
      - {"type":"node","id":<node_id>,"attrs":{...}}
      - {"type":"edge","u":<u>,"v":<v>,"attrs":{...}}
    Returns an undirected NetworkX Graph (or MultiGraph if multi=True) with node/edge attributes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    G = nx.MultiGraph() if multi else nx.Graph()
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
                    G.add_edge(u, v, **attrs)
                else:
                    raise ValueError(f"Unknown record type: {rtype}")
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


