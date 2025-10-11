#!/usr/bin/env python3
"""Visualize a tiny temporal MultiGraph: 2 stores × 3 products × 3 weeks.

Creates a NetworkX MultiGraph with:
- Nodes: store:1, store:2, product:101, product:102, product:103
- Edges: per-week 'sold' (units) and 'has_inventory' (present=True), with time=YYYYMMDD

Output: artifacts/graph_example_2s3p_3w.png

This focuses on showing how we encode:
- Node types (store vs product)
- Edge types (sold vs has_inventory)
- Temporal edges (multiple weeks) with curvature and color
- Units on 'sold' edges via line width
"""
from __future__ import annotations

import os
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import networkx as nx


def build_example_graph(num_stores: int = 2, num_products: int = 3, weeks: list[int] | None = None) -> nx.MultiGraph:
    G = nx.MultiGraph()

    # Weeks (YYYYMMDD)
    if not weeks:
        weeks = [20240408, 20240415, 20240422]

    # Nodes
    stores = [f"store:{i+1}" for i in range(num_stores)]
    products = [f"product:{101+j}" for j in range(num_products)]
    for s in stores:
        G.add_node(s, type="store", StoreFormat="S", Format="A")
    for p in products:
        G.add_node(p, type="product", Department="Snacks", Division="Grocery")

    def add_week(u: str, v: str, week: int, units: float | int, inv_present: bool) -> None:
        G.add_edge(u, v, rel="sold", time=week, units=float(units))
        if inv_present:
            G.add_edge(u, v, rel="has_inventory", time=week, present=True)

    # Special, clear 1×1 pattern
    if num_stores == 1 and num_products == 1:
        s, p = stores[0], products[0]
        pattern = [3, 0, 2]
        for w, u in zip(weeks, pattern):
            add_week(s, p, w, units=u, inv_present=True)
        return G

    # Generic pattern for NxM: vary units by indices and week; always show some zeros
    for i, s in enumerate(stores):
        for j, p in enumerate(products):
            base = (i + 1) + (j + 1)
            for k, w in enumerate(weeks):
                u = max(0, (base + k) % 5 - 1)  # values in {0,1,2,3}
                add_week(s, p, w, units=u, inv_present=True)

    return G


def bipartite_positions(G: nx.Graph) -> dict:
    # Left: stores; Right: products
    stores = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "store"]
    products = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "product"]
    stores = sorted(stores)
    products = sorted(products)

    pos = {}
    # Increase horizontal and vertical spacing to avoid overlap
    x_left = 0.0
    x_right = 5.0
    store_gap = 2.0
    product_gap = 1.6
    for i, s in enumerate(stores):
        pos[s] = (x_left, -i * store_gap)
    for j, p in enumerate(products):
        pos[p] = (x_right, -j * product_gap + 1.0)
    return pos


def compute_product_mu_sigma_plus(G: nx.MultiGraph) -> dict:
    """Mimic orders_vn2.compute_mu_sigma_plus for the tiny example.
    Returns mapping: product_node -> {mu_hat_plus, sigma_hat_plus}.
    """
    values = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("rel") != "sold":
            continue
        units = float(d.get("units", 0.0))
        if units <= 0:
            continue
        prod = u if str(G.nodes[u].get("type")) == "product" else v
        values.setdefault(prod, []).append(units)
    # Global fallback
    all_vals = [x for vv in values.values() for x in vv]
    g_mu = sum(all_vals) / max(1, len(all_vals)) if all_vals else 1.0
    g_sigma = (sum((x - g_mu) ** 2 for x in all_vals) / max(1, len(all_vals))) ** 0.5 if all_vals else 1.0
    out = {}
    for p, lst in values.items():
        if len(lst) >= 2:
            mu = sum(lst) / len(lst)
            var = sum((x - mu) ** 2 for x in lst) / len(lst)
            sigma = max(1e-6, var) ** 0.5
        elif len(lst) == 1:
            mu = float(lst[0])
            sigma = g_sigma
        else:
            mu, sigma = g_mu, g_sigma
        out[p] = {"mu_hat_plus": float(mu), "sigma_hat_plus": float(sigma)}
    return out


def draw_graph(G: nx.MultiGraph, out_path: Path) -> None:
    pos = bipartite_positions(G)

    fig = plt.figure(figsize=(9, 5), dpi=160)
    ax = plt.gca()
    # Dynamic title
    stores = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "store"]
    products = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "product"]
    weeks = sorted({int(d.get("time", 0)) for _, _, _, d in G.edges(keys=True, data=True)})
    ax.set_title(f"Temporal MultiGraph: {len(stores)} Stores × {len(products)} Products × {len(weeks)} Weeks")
    ax.axis("off")

    # Nodes: stores (squares) vs products (circles)
    stores = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "store"]
    products = [n for n, a in G.nodes(data=True) if str(a.get("type")) == "product"]
    # Scale node size by (log1p degree)
    def _node_size(n: str) -> float:
        deg = G.degree(n)
        return 1000.0 + 400.0 * (max(0.0, (deg))) ** 0.5
    store_sizes = [_node_size(s) for s in stores]
    product_sizes = [_node_size(p) for p in products]
    nx.draw_networkx_nodes(G, pos, nodelist=stores, node_color="#1f77b4", node_shape="s", node_size=store_sizes, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=products, node_color="#ff7f0e", node_shape="o", node_size=product_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white")

    # Edge style mapping
    weeks = sorted({int(d.get("time", 0)) for _, _, _, d in G.edges(keys=True, data=True)})
    # Use modern colormaps API (avoid deprecation warning)
    cmap = mpl.colormaps.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=min(weeks), vmax=max(weeks))
    week_to_color = {w: cmap(norm(w)) for w in weeks}
    # Different curvature per week so parallel edges are visible
    rad_values = [-0.35, 0.0, 0.35]
    week_to_rad = {w: rad_values[i % len(rad_values)] for i, w in enumerate(weeks)}

    # Draw each edge individually to control curvature and style
    for u, v, k, d in G.edges(keys=True, data=True):
        w = int(d.get("time", 0))
        rel = d.get("rel", "")
        if rel == "sold":
            units = float(d.get("units", 0.0))
            lw = 1.0 + 1.5 * max(0.0, units)  # thicker for more units; zero-sales still visible
            color = week_to_color.get(w, (0, 0, 0, 0.4))
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=lw,
                edge_color=[color],
                connectionstyle=f"arc3,rad={week_to_rad.get(w, 0.0)}",
                alpha=0.9,
            )
        elif rel == "has_inventory":
            # light, dashed to indicate availability
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=1.2,
                style="dashed",
                edge_color="#999999",
                connectionstyle=f"arc3,rad={week_to_rad.get(w, 0.0) * 0.7}",
                alpha=0.6,
            )

    # --- Annotate features from our pipeline (example) ---
    # Per-product mu+/sigma+ (computed from positive sold units)
    stats = compute_product_mu_sigma_plus(G)
    for p in products:
        x, y = pos[p]
        s = stats.get(p, {"mu_hat_plus": 0.0, "sigma_hat_plus": 0.0})
        text = f"μ+={s['mu_hat_plus']:.1f}\nσ+={s['sigma_hat_plus']:.1f}\nDept={G.nodes[p].get('Department','')}"
        ax.text(
            x + 0.3,
            y + 0.6,
            text,
            fontsize=8,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fffbe6", ec="#e0d9a6", alpha=0.9),
        )

    # Per-store display: Format and degree
    for s in stores:
        x, y = pos[s]
        deg = G.degree(s)
        text = f"Format={G.nodes[s].get('Format','')}\nDegree={deg}"
        ax.text(
            x - 0.3,
            y + 0.6,
            text,
            fontsize=8,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#eef6ff", ec="#bcd4f6", alpha=0.9),
        )

    # Legend
    sold_line = mlines.Line2D([], [], color=cmap(0.8), lw=3, label="sold (thickness ∝ units)")
    inv_line = mlines.Line2D([], [], color="#999999", lw=2, linestyle="--", label="has_inventory (present=True)")
    store_pt = mlines.Line2D([], [], color="#1f77b4", marker="s", linestyle="None", markersize=10, label="store")
    prod_pt = mlines.Line2D([], [], color="#ff7f0e", marker="o", linestyle="None", markersize=10, label="product")
    ax.legend(handles=[store_pt, prod_pt, sold_line, inv_line], loc="upper left", frameon=False)

    # Colorbar for weeks
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("week (YYYYMMDD)")
    cbar.set_ticks(weeks)
    cbar.set_ticklabels([str(w) for w in weeks])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize a tiny temporal MultiGraph example")
    ap.add_argument("--stores", type=int, default=2)
    ap.add_argument("--products", type=int, default=3)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "artifacts" / "graph_example.png"))
    args = ap.parse_args()

    G = build_example_graph(num_stores=args.stores, num_products=args.products)
    out = Path(args.out)
    draw_graph(G, out)


if __name__ == "__main__":
    main()


