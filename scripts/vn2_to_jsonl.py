from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import math
from datetime import date as _date


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def ymd_to_int(ymd: str) -> int:
    return int(ymd.replace("-", ""))


def iter_date_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if DATE_RE.match(c)]


def write_jsonl(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def build_graph_records(
    sales_path: Path,
    stock_path: Path,
    master_path: Path,
    *,
    max_pairs: int | None = 1000,
    start_date: str | None = None,
    end_date: str | None = None,
    include_sold_zeros: bool = False,
    sold_zeros_if_instock: bool = True,
) -> List[dict]:
    sales = pd.read_csv(sales_path)
    instock = pd.read_csv(stock_path)
    master = pd.read_csv(master_path)

    sales_dates = iter_date_columns(sales)
    stock_dates = iter_date_columns(instock)

    if start_date is not None:
        sales_dates = [d for d in sales_dates if d >= start_date]
        stock_dates = [d for d in stock_dates if d >= start_date]
    if end_date is not None:
        sales_dates = [d for d in sales_dates if d <= end_date]
        stock_dates = [d for d in stock_dates if d <= end_date]

    # Limit pairs if requested
    if max_pairs is not None and max_pairs > 0 and len(sales) > max_pairs:
        sales = sales.head(max_pairs)
        # Align instock/master to the same first pairs by Store+Product
        head_pairs = set(map(tuple, sales[["Store", "Product"]].values.tolist()))
        instock = instock[instock.apply(lambda r: (r["Store"], r["Product"]) in head_pairs, axis=1)]
        master = master[master.apply(lambda r: (r["Store"], r["Product"]) in head_pairs, axis=1)]

    # Helper to canonicalize IDs like "0.0" -> 0
    def canon_num(x: object) -> object:
        try:
            s = str(x)
            if s.replace('.', '', 1).isdigit():
                return int(float(s))
        except Exception:
            pass
        return x

    # Collect node ids (canonicalized)
    stores: Set[str] = set()
    products: Set[str] = set()
    for _, r in sales.iterrows():
        stores.add(f"store:{canon_num(r['Store'])}")
        products.add(f"product:{canon_num(r['Product'])}")

    # Build node records (store/product). Attach product taxonomy and store format from master
    product_attr_cols = [c for c in master.columns if c not in ("Store", "Product")]
    # Use first occurrence per product for attributes
    prod_attrs = (
        master.drop(columns=["Store"]).drop_duplicates(subset=["Product"]).set_index("Product")[product_attr_cols]
        if product_attr_cols
        else pd.DataFrame()
    )
    
    # Get store formats (use first occurrence per store)
    store_formats = (
        master.drop(columns=["Product"]).drop_duplicates(subset=["Store"]).set_index("Store")[["StoreFormat", "Format"]]
        if "StoreFormat" in master.columns
        else pd.DataFrame()
    )

    records: List[dict] = []
    for sid in sorted(stores):
        store_num_str = sid.split(":")[1]
        try:
            store_num = int(float(store_num_str)) if store_num_str.replace(".", "").isdigit() else store_num_str
        except ValueError:
            store_num = store_num_str
        attrs = {"type": "store"}
        if not store_formats.empty and store_num in store_formats.index:
            for col, val in store_formats.loc[store_num].items():
                if pd.notna(val):
                    attrs[col] = int(val) if isinstance(val, (int, float)) else val
        records.append({"type": "node", "id": sid, "attrs": attrs})
    
    def safe_sort_key(x):
        try:
            return (x.split(":")[0], int(float(x.split(":")[1])))
        except:
            return (x.split(":")[0], x.split(":")[1])
    
    for pid in sorted(products, key=safe_sort_key):
        pnum_str = pid.split(":")[1]
        try:
            pnum = int(float(pnum_str))
        except:
            pnum = pnum_str
        attrs = {"type": "product"}
        if not prod_attrs.empty and pnum in prod_attrs.index:
            for col, val in prod_attrs.loc[pnum].items():
                # Keep all hierarchy attributes
                if pd.notna(val):
                    attrs[col] = int(val) if isinstance(val, (int, float)) else val
        records.append({"type": "node", "id": pid, "attrs": attrs})

    # Prepare inventory-present keys for conditional zero-sales retention
    instock_bool = instock.copy()
    # Coerce to booleans if needed
    for d in stock_dates:
        if instock_bool[d].dtype != bool:
            instock_bool[d] = instock_bool[d].astype(str).str.lower().isin(["true", "1", "yes"])
    inv_present_keys: Set[tuple] = set()
    for _, r in instock_bool.iterrows():
        s = canon_num(r['Store'])
        p = canon_num(r['Product'])
        for d in stock_dates:
            if bool(r[d]):
                inv_present_keys.add((s, p, ymd_to_int(d)))

    # Edges: sold events (by default drop zero-sales; optionally keep when inventory is present)
    for _, r in sales.iterrows():
        u = f"store:{canon_num(r['Store'])}"
        v = f"product:{canon_num(r['Product'])}"
        for d in sales_dates:
            val = r[d]
            try:
                units = float(val)
            except Exception:
                continue
            if pd.notna(units):
                if units == 0.0 and not include_sold_zeros:
                    # Keep zero-sales only if inventory is present at that week and policy allows
                    if not sold_zeros_if_instock:
                        continue
                    s_key = canon_num(r['Store'])
                    p_key = canon_num(r['Product'])
                    t_key = ymd_to_int(d)
                    if (s_key, p_key, t_key) not in inv_present_keys:
                        continue
                records.append(
                    {
                        "type": "edge",
                        "u": u,
                        "v": v,
                        "attrs": {"rel": "sold", "time": ymd_to_int(d), "units": float(units)},
                    }
                )

    # Edges: has_inventory (boolean True)
    for _, r in instock_bool.iterrows():
        u = f"store:{canon_num(r['Store'])}"
        v = f"product:{canon_num(r['Product'])}"
        for d in stock_dates:
            present = bool(r[d])
            if present:
                records.append(
                    {
                        "type": "edge",
                        "u": u,
                        "v": v,
                        "attrs": {"rel": "has_inventory", "time": ymd_to_int(d), "present": True},
                    }
                )

    return records


def build_graph_records_v2(
    sales_path: Path,
    stock_path: Path,
    master_path: Path,
    *,
    max_pairs: int | None = 1000,
    start_date: str | None = None,
    end_date: str | None = None,
    include_sold_zeros: bool = False,
    sold_zeros_if_instock: bool = True,
    add_history_features: bool = True,
    history_windows: Iterable[int] = (1, 2, 4, 8),
) -> List[dict]:
    """
    Enhanced generator:
      - Enforces inventory-aware negatives by default (include_sold_zeros=False and sold_zeros_if_instock=True).
      - Adds compact past-history features to 'sold' edges WITHOUT leaking future info.
      - Keeps existing node/edge schema; only adds attrs; emits a 'meta' record first.
    """
    sales = pd.read_csv(sales_path, dtype={"Store": str, "Product": str})
    instock = pd.read_csv(stock_path, dtype={"Store": str, "Product": str})
    master = pd.read_csv(master_path, dtype={"Store": str, "Product": str})

    # Dates
    sales_dates = sorted([c for c in sales.columns if DATE_RE.match(c)])
    stock_dates = sorted([c for c in instock.columns if DATE_RE.match(c)])
    if start_date is not None:
        sales_dates = [d for d in sales_dates if d >= start_date]
        stock_dates = [d for d in stock_dates if d >= start_date]
    if end_date is not None:
        sales_dates = [d for d in sales_dates if d <= end_date]
        stock_dates = [d for d in stock_dates if d <= end_date]

    # Limit pairs and align
    if max_pairs is not None and max_pairs > 0 and len(sales) > max_pairs:
        sales = sales.head(max_pairs)
        head_pairs = set(map(tuple, sales[["Store", "Product"]].values.tolist()))
        instock = instock[instock.apply(lambda r: (r["Store"], r["Product"]) in head_pairs, axis=1)]
        master = master[master.apply(lambda r: (r["Store"], r["Product"]) in head_pairs, axis=1)]

    def canon_id(prefix: str, x: object) -> str:
        s = str(x).strip()
        try:
            if s.replace(".", "", 1).isdigit():
                s = str(int(float(s)))
        except Exception:
            pass
        return f"{prefix}:{s}"

    # Node ids
    stores: Set[str] = {canon_id("store", r["Store"]) for _, r in sales.iterrows()}
    products: Set[str] = {canon_id("product", r["Product"]) for _, r in sales.iterrows()}

    # Attributes (fixed split: product-level vs store-level)
    prod_cols = [c for c in ["ProductGroup","Division","Department","DepartmentGroup"] if c in master.columns]
    store_cols = [c for c in ["StoreFormat","Format","Region"] if c in master.columns]
    prod_attrs = (
        master[["Product"] + prod_cols].drop_duplicates(subset=["Product"]).set_index("Product")[prod_cols]
        if prod_cols else pd.DataFrame()
    )
    store_formats = (
        master[["Store"] + store_cols].drop_duplicates(subset=["Store"]).set_index("Store")[store_cols]
        if store_cols else pd.DataFrame()
    )

    # Meta + nodes
    records: List[dict] = []
    records.append({
        "type": "meta",
        "schema_version": "1.1",
        "attrs": {
            "start_date": start_date,
            "end_date": end_date,
            "history_windows": list(history_windows),
            "inventory_aware_zeros": bool(sold_zeros_if_instock and not include_sold_zeros),
        },
    })

    def _safe_sort(ids):
        def key(x):
            p, s = x.split(":")
            try:
                return (p, int(s))
            except Exception:
                return (p, s)
        return sorted(ids, key=key)

    for sid in _safe_sort(stores):
        store_key = sid.split(":")[1]
        attrs = {"type": "store"}
        if not store_formats.empty and store_key in store_formats.index:
            for col, val in store_formats.loc[store_key].items():
                if pd.notna(val):
                    attrs[col] = val
        records.append({"type": "node", "id": sid, "attrs": attrs})

    for pid in _safe_sort(products):
        prod_key = pid.split(":")[1]
        attrs = {"type": "product"}
        if not prod_attrs.empty and prod_key in prod_attrs.index:
            for col, val in prod_attrs.loc[prod_key].items():
                if pd.notna(val):
                    attrs[col] = val
        records.append({"type": "node", "id": pid, "attrs": attrs})

    # Inventory booleans index (preserve NA vs explicit false)
    instock_idx = instock.set_index(["Store", "Product"]) if not instock.empty else None

    sales_dates_sorted = sales_dates
    date_ints = [ymd_to_int(d) for d in sales_dates_sorted]

    # Optional: precompute peer aggregates per (Product, Date) across stores
    peer_keyed: dict[tuple[str, str], dict[str, float]] = {}
    try:
        d2i = {d: i for i, d in enumerate(sales_dates_sorted)}
        # Build matrix of units by (Product, Date) across stores
        units_by_prod_date: dict[tuple[str, str], list[float]] = {}
        for _, row in sales.iterrows():
            prod = str(row["Product"]).strip()
            for d in sales_dates_sorted:
                val = row.get(d, 0)
                try:
                    units = float(val) if pd.notna(val) else 0.0
                except Exception:
                    units = 0.0
                units_by_prod_date.setdefault((prod, d), []).append(units)
        # Compute rolling peer mean/max for requested windows
        windows = list(history_windows)
        for (prod, d), arr in units_by_prod_date.items():
            t = d2i.get(d, None)
            if t is None:
                continue
            peer_keyed[(prod, d)] = {}
            for w in windows:
                s = max(0, t - int(w))
                window_vals = arr[s:t]
                if window_vals:
                    peer_keyed[(prod, d)][f"peer_mean_w{w}"] = float(pd.Series(window_vals).mean())
                    peer_keyed[(prod, d)][f"peer_max_w{w}"] = float(pd.Series(window_vals).max())
                else:
                    peer_keyed[(prod, d)][f"peer_mean_w{w}"] = 0.0
                    peer_keyed[(prod, d)][f"peer_max_w{w}"] = 0.0
    except Exception:
        peer_keyed = {}

    # Iterate and build edges with history
    for _, row in sales.iterrows():
        u = canon_id("store", row["Store"])
        v = canon_id("product", row["Product"])
        pair_key = (row["Store"], row["Product"]) if instock_idx is not None else None

        # Units series
        s_units: List[float] = []
        for d in sales_dates_sorted:
            val = row.get(d, 0)
            try:
                s_units.append(float(val) if pd.notna(val) else 0.0)
            except Exception:
                s_units.append(0.0)

        # Inventory series aligned
        if instock_idx is not None and pair_key in instock_idx.index:
            inv_series = []
            for d in sales_dates_sorted:
                raw = instock_idx.loc[pair_key].get(d, None)
                s = str(raw).strip().lower()
                if s in ("true","1","yes","y","t"):
                    inv_series.append(True)
                elif s in ("false","0","no","n","f"):
                    inv_series.append(False)
                else:
                    inv_series.append(False)  # treat NA as not explicitly present
        else:
            inv_series = [False] * len(sales_dates_sorted)

        # Prefix sums for rolling
        last_sale_idx = -1
        cumsums = [0.0] * (len(sales_dates_sorted) + 1)
        for t in range(len(sales_dates_sorted)):
            cumsums[t + 1] = cumsums[t] + (s_units[t] if not pd.isna(s_units[t]) else 0.0)

        for t, d in enumerate(sales_dates_sorted):
            units = s_units[t] if s_units[t] is not None else 0.0
            inv_present = inv_series[t]

            emit_sold = (units > 0.0) or (include_sold_zeros or (sold_zeros_if_instock and inv_present))
            if emit_sold:
                attrs: dict = {"rel": "sold", "time": date_ints[t], "units": float(units)}
                if add_history_features:
                    seen_before = last_sale_idx >= 0
                    attrs["seen_before"] = bool(seen_before)
                    if seen_before:
                        attrs["recency_weeks"] = int(t - last_sale_idx)
                    for w in history_windows:
                        start = max(0, t - int(w))
                        attrs[f"lag_sum_w{w}"] = float(cumsums[t] - cumsums[start])
                    attrs["inv_present_now"] = bool(inv_present)
                    if t > 0:
                        attrs["stockout_last_w1"] = bool(not inv_series[t - 1])
                    # Add peer stats if available for (Product, Date)
                    try:
                        key_pd = (str(row["Product"]).strip(), sales_dates_sorted[t])
                        pvals = peer_keyed.get(key_pd, {})
                        for w in history_windows:
                            pm = pvals.get(f"peer_mean_w{w}")
                            px = pvals.get(f"peer_max_w{w}")
                            if pm is not None:
                                attrs[f"peer_mean_w{w}"] = float(pm)
                            if px is not None:
                                attrs[f"peer_max_w{w}"] = float(px)
                    except Exception:
                        pass
                # week-of-year seasonality (sin/cos)
                try:
                    iso_week = _date.fromisoformat(d).isocalendar().week
                    attrs["woy_sin"] = math.sin(2 * math.pi * iso_week / 52.0)
                    attrs["woy_cos"] = math.cos(2 * math.pi * iso_week / 52.0)
                except Exception:
                    pass
                records.append({"type": "edge", "u": u, "v": v, "attrs": attrs})

            if inv_present:
                records.append({"type": "edge", "u": u, "v": v, "attrs": {"rel": "has_inventory", "time": date_ints[t], "present": True}})
            else:
                # Optional: explicit no-inventory edge emitted later via flag; we record via a temporary marker
                pass

            if units > 0.0:
                last_sale_idx = t

    return records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert VN2 CSVs to JSONL graph for relational-graph repo.")
    ap.add_argument("--vn2-data-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "vn2inventory" / "data"))
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "graph_qa" / "data" / "vn2_graph_sample.jsonl"))
    ap.add_argument("--max-pairs", type=int, default=200, help="Limit number of (Store,Product) pairs")
    ap.add_argument("--start-date", type=str, default=None, help="Inclusive YYYY-MM-DD filter")
    ap.add_argument("--end-date", type=str, default=None, help="Inclusive YYYY-MM-DD filter")
    ap.add_argument("--include-sold-zeros", action="store_true", help="Keep zero-sales 'sold' edges regardless of inventory (default off)")
    # Default: keep zero-sales only when inventory is present; allow disabling
    ap.add_argument("--sold-zeros-if-instock", dest="sold_zeros_if_instock", action="store_true")
    ap.add_argument("--no-sold-zeros-if-instock", dest="sold_zeros_if_instock", action="store_false")
    ap.set_defaults(sold_zeros_if_instock=True)
    # v2 enhancements
    ap.add_argument("--v2", action="store_true", help="Use enhanced generator with history features and meta record")
    ap.add_argument("--add-history-features", dest="add_history_features", action="store_true")
    ap.add_argument("--no-history-features", dest="add_history_features", action="store_false")
    ap.set_defaults(add_history_features=True)
    ap.add_argument("--history-windows", type=str, default="1,2,4,8", help="Comma-separated weeks for rolling sums")
    ap.add_argument("--initial-state", type=str, default=None, help="Path to 'Week 0 - 2024-04-08 - Initial State.csv' to emit a state edge at anchor week")
    ap.add_argument("--emit-no-inventory", action="store_true", help="Emit explicit no_inventory edges when source is an explicit false/0")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.vn2_data_dir)
    sales_path = data_dir / "Week 0 - 2024-04-08 - Sales.csv"
    stock_path = data_dir / "Week 0 - In Stock.csv"
    master_path = data_dir / "Week 0 - Master.csv"

    if not sales_path.exists() or not stock_path.exists() or not master_path.exists():
        raise FileNotFoundError(f"Missing VN2 CSVs under {data_dir}")

    if args.v2:
        try:
            windows = [int(x.strip()) for x in (args.history_windows.split(",") if args.history_windows else []) if x.strip()]
        except Exception:
            windows = [1, 2, 4, 8]
        records = build_graph_records_v2(
            sales_path=sales_path,
            stock_path=stock_path,
            master_path=master_path,
            max_pairs=args.max_pairs,
            start_date=args.start_date,
            end_date=args.end_date,
            include_sold_zeros=args.include_sold_zeros,
            sold_zeros_if_instock=args.sold_zeros_if_instock,
            add_history_features=args.add_history_features,
            history_windows=windows,
        )
        # Optional: add a single state edge per pair from Initial State at anchor week
        if args.initial_state:
            try:
                state = pd.read_csv(args.initial_state, dtype={"Store": str, "Product": str})
                anchor_date = "2024-04-08"
                anchor_int = ymd_to_int(anchor_date)
                for _, r in state.iterrows():
                    u = f"store:{str(r.get('Store')).strip()}"
                    v = f"product:{str(r.get('Product')).strip()}"
                    attrs = {
                        "rel": "state",
                        "time": anchor_int,
                        "start_inv": float(r.get("Start Inventory", 0) or 0),
                        "in_transit_w1": float(r.get("In Transit W+1", 0) or 0),
                        "in_transit_w2": float(r.get("In Transit W+2", 0) or 0),
                        "missed_sales": float(r.get("Missed Sales", 0) or 0),
                        "holding_cost": float(r.get("Holding Cost", 0) or 0),
                        "shortage_cost": float(r.get("Shortage Cost", 0) or 0),
                    }
                    records.append({"type": "edge", "u": u, "v": v, "attrs": attrs})
            except Exception as e:
                print(f"[warn] failed to add initial state edges: {e}")
    else:
        records = build_graph_records(
            sales_path=sales_path,
            stock_path=stock_path,
            master_path=master_path,
            max_pairs=args.max_pairs,
            start_date=args.start_date,
            end_date=args.end_date,
            include_sold_zeros=args.include_sold_zeros,
            sold_zeros_if_instock=args.sold_zeros_if_instock,
        )
    out_path = Path(args.out)
    write_jsonl(records, out_path)
    print(f"Wrote {out_path} with {len(records)} records")


if __name__ == "__main__":
    main()


