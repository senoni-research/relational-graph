from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd


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

    # Collect node ids
    stores: Set[str] = set()
    products: Set[str] = set()
    for _, r in sales.iterrows():
        stores.add(f"store:{r['Store']}")
        products.add(f"product:{r['Product']}")

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

    # Edges: sold (create edge for ALL weeks, including zero sales for temporal history)
    for _, r in sales.iterrows():
        u = f"store:{r['Store']}"
        v = f"product:{r['Product']}"
        for d in sales_dates:
            val = r[d]
            try:
                units = float(val)
            except Exception:
                continue
            if pd.notna(units):
                # Create edge even if units=0 to capture full temporal history
                records.append(
                    {
                        "type": "edge",
                        "u": u,
                        "v": v,
                        "attrs": {"rel": "sold", "time": ymd_to_int(d), "units": float(units)},
                    }
                )

    # Edges: has_inventory (boolean True)
    # Coerce to booleans if needed
    instock_bool = instock.copy()
    for d in stock_dates:
        if instock_bool[d].dtype != bool:
            instock_bool[d] = instock_bool[d].astype(str).str.lower().isin(["true", "1", "yes"])

    for _, r in instock_bool.iterrows():
        u = f"store:{r['Store']}"
        v = f"product:{r['Product']}"
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert VN2 CSVs to JSONL graph for relational-graph repo.")
    ap.add_argument("--vn2-data-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "vn2inventory" / "data"))
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "graph_qa" / "data" / "vn2_graph_sample.jsonl"))
    ap.add_argument("--max-pairs", type=int, default=200, help="Limit number of (Store,Product) pairs")
    ap.add_argument("--start-date", type=str, default=None, help="Inclusive YYYY-MM-DD filter")
    ap.add_argument("--end-date", type=str, default=None, help="Inclusive YYYY-MM-DD filter")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.vn2_data_dir)
    sales_path = data_dir / "Week 0 - 2024-04-08 - Sales.csv"
    stock_path = data_dir / "Week 0 - In Stock.csv"
    master_path = data_dir / "Week 0 - Master.csv"

    if not sales_path.exists() or not stock_path.exists() or not master_path.exists():
        raise FileNotFoundError(f"Missing VN2 CSVs under {data_dir}")

    records = build_graph_records(
        sales_path=sales_path,
        stock_path=stock_path,
        master_path=master_path,
        max_pairs=args.max_pairs,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    out_path = Path(args.out)
    write_jsonl(records, out_path)
    print(f"Wrote {out_path} with {len(records)} records")


if __name__ == "__main__":
    main()


