#!/usr/bin/env python3
"""
Compute demand transfer weights for VN2:
- Product substitution weights K_prod(p->p') within the same store/category
- Store proximity/transfer weights K_store(s->s') for the same product

Outputs a compact Parquet with two tables: product_subst, store_neighbor.

Notes:
- Uses weekly 'sold' edges and 'has_inventory' (or derived) to proxy OOS windows
- Smooths weights with taxonomy and distance priors; caps to top-K per source
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

from graph_qa.io.loader import load_graph


def _oos_weeks(G) -> set[Tuple[str, str, int]]:
    # Collect (store, product, week) where has_inventory==False if present; fallback: zero sales following positive history
    oos = set()
    # pass 1: explicit no_inventory edges if available
    for u, v, key, a in G.edges(data=True, keys=True):
        if a.get("rel") == "has_inventory" and (not bool(a.get("val", True))):
            t = int(a.get("time", 0))
            s = u if str(G.nodes[u].get("type")) == "store" else v
            p = v if s == u else u
            oos.add((str(s), str(p), t))
    # heuristic fallback skipped for brevity; rely on explicit signals where available
    return oos


def _taxonomy_key(attrs: dict) -> tuple:
    return (
        str(attrs.get("DepartmentGroup", "")),
        str(attrs.get("Department", "")),
    )


def compute_product_substitution(G, topk: int = 5) -> pd.DataFrame:
    # Aggregate weekly sales by (store, product, week)
    rows = []
    for u, v, key, a in G.edges(data=True, keys=True):
        if a.get("rel") != "sold":
            continue
        t = int(a.get("time", 0))
        units = float(a.get("units", 0.0))
        s = u if str(G.nodes[u].get("type")) == "store" else v
        p = v if s == u else u
        rows.append((str(s), str(p), t, units))
    df = pd.DataFrame(rows, columns=["store", "product", "week", "units"]) if rows else pd.DataFrame(columns=["store","product","week","units"]) 
    if df.empty:
        return pd.DataFrame(columns=["store","product","product_sub","weight"]) 

    # Taxonomy per product
    prod_tax = {}
    for n, a in G.nodes(data=True):
        if str(a.get("type")) == "product":
            prod_tax[str(n)] = _taxonomy_key(a)

    # Build candidate pairs within same store and taxonomy bucket
    subs_records = []
    for store, g in df.groupby("store"):
        prods = list(g["product"].unique())
        # baseline mean per product
        base = g.groupby("product")["units"].mean()
        # weekly pivot
        pivot = g.pivot_table(index="week", columns="product", values="units", aggfunc="sum").fillna(0.0)
        weeks = list(pivot.index)
        # OOS proxy set (if available)
        # For now, skip explicit OOS filter; use co-lift normalized by baseline
        for p in prods:
            # candidate substitutes by taxonomy
            cand = [q for q in prods if q != p and prod_tax.get(q) == prod_tax.get(p)]
            lifts = []
            for q in cand:
                # co-lift: correlation-weighted uplift relative to baseline
                x = pivot.get(p, pd.Series(0.0, index=weeks)).values
                y = pivot.get(q, pd.Series(0.0, index=weeks)).values
                if x.std() < 1e-6 or y.std() < 1e-6:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(x, y)[0, 1])
                uplift = max(0.0, y.mean() - base.get(q, 0.0))
                score = max(0.0, corr) * uplift
                if score > 0:
                    lifts.append((q, score))
            if not lifts:
                continue
            lifts.sort(key=lambda t: t[1], reverse=True)
            total = sum(w for _, w in lifts[:topk]) or 1.0
            for q, w in lifts[:topk]:
                subs_records.append((store, p, q, float(w / total)))

    return pd.DataFrame(subs_records, columns=["store","product","product_sub","weight"]) if subs_records else pd.DataFrame(columns=["store","product","product_sub","weight"]) 


def compute_store_neighbors(G, topk: int = 3, kappa: float = 0.2) -> pd.DataFrame:
    # Build store list with optional (x,y) coords if present; else use region/format proximity
    stores = []
    for n, a in G.nodes(data=True):
        if str(a.get("type")) == "store":
            stores.append((str(n), a))
    if not stores:
        return pd.DataFrame(columns=["store","store_nbr","weight"]) 

    # Distance kernel using simple region match; if coords available, prefer Euclidean
    def dist(a1, a2) -> float:
        x1, y1 = a1.get("x"), a1.get("y")
        x2, y2 = a2.get("x"), a2.get("y")
        if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
            return math.hypot(x1 - x2, y1 - y2)
        # fallback: 0 if same region/format, else 1
        return 0.0 if (a1.get("Region") == a2.get("Region") and a1.get("Format") == a2.get("Format")) else 1.0

    records = []
    for i, (s, a) in enumerate(stores):
        dists = []
        for j, (t, b) in enumerate(stores):
            if s == t:
                continue
            d = dist(a, b)
            w = math.exp(-kappa * d)
            dists.append((t, w))
        dists.sort(key=lambda t: t[1], reverse=True)
        total = sum(w for _, w in dists[:topk]) or 1.0
        for t, w in dists[:topk]:
            records.append((s, t, float(w / total)))
    return pd.DataFrame(records, columns=["store","store_nbr","weight"]) 


def main():
    ap = argparse.ArgumentParser(description="Compute VN2 demand transfer weights")
    ap.add_argument("--graph", required=True, type=str)
    ap.add_argument("--subst-topk", type=int, default=5)
    ap.add_argument("--store-topk", type=int, default=3)
    ap.add_argument("--kappa", type=float, default=0.2)
    ap.add_argument("--out", required=True, type=str, help="Parquet path to write weights")
    args = ap.parse_args()

    print(f"Loading graph from {args.graph}...")
    G = load_graph(args.graph, multi=True)

    print("Computing product substitution weights...")
    df_sub = compute_product_substitution(G, topk=int(args.subst_topk))
    print(f"product_substitution: {len(df_sub)} rows")

    print("Computing store neighbor weights...")
    df_store = compute_store_neighbors(G, topk=int(args.store_topk), kappa=float(args.kappa))
    print(f"store_neighbor: {len(df_store)} rows")

    print(f"Writing weights to {args.out}...")
    with pd.ExcelWriter(args.out.replace('.parquet', '.xlsx')) as xl:
        df_sub.to_excel(xl, sheet_name='product_subst', index=False)
        df_store.to_excel(xl, sheet_name='store_neighbor', index=False)
    # Also write parquet with two tables by concatenating with a tag
    df_sub['_table'] = 'product_subst'
    df_store['_table'] = 'store_neighbor'
    df_all = pd.concat([df_sub, df_store], ignore_index=True)
    df_all.to_parquet(args.out, index=False)
    print("Done.")


if __name__ == "__main__":
    main()


