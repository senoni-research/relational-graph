#!/usr/bin/env python3
"""
Convert VN2 CSVs to temporal snapshot graphs.
Creates one graph per time window (e.g., monthly) with aggregated sales/inventory.
This gives us many training samples instead of 1 static graph.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def ymd_to_int(ymd: str) -> int:
    return int(ymd.replace("-", ""))


def create_snapshot_graph(
    sales: pd.DataFrame,
    instock: pd.DataFrame,
    master: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict:
    """Create a graph snapshot for a date range with aggregated edges."""
    
    # Filter columns in range
    date_cols = [c for c in sales.columns if c >= start_date and c <= end_date and c not in ("Store", "Product")]
    
    if not date_cols:
        return None
    
    # Aggregate sales over the window
    sales_agg = sales[["Store", "Product"] + date_cols].copy()
    sales_agg["total_units"] = sales_agg[date_cols].sum(axis=1)
    sales_agg["mean_units"] = sales_agg[date_cols].mean(axis=1)
    sales_agg = sales_agg[sales_agg["total_units"] > 0]  # Only pairs with sales
    
    # Nodes
    stores = set(sales_agg["Store"].unique())
    products = set(sales_agg["Product"].unique())
    
    # Get product attributes
    prod_attrs = master.drop(columns=["Store"]).drop_duplicates(subset=["Product"]).set_index("Product")
    store_formats = master.drop(columns=["Product"]).drop_duplicates(subset=["Store"]).set_index("Store")
    
    records = []
    
    # Store nodes
    for sid in sorted(stores):
        try:
            store_num = int(float(sid))
        except:
            store_num = sid
        attrs = {"type": "store"}
        if store_num in store_formats.index:
            for col in ["StoreFormat", "Format"]:
                if col in store_formats.columns:
                    val = store_formats.loc[store_num][col]
                    if pd.notna(val):
                        attrs[col] = int(val)
        records.append({"type": "node", "id": f"store:{sid}", "attrs": attrs})
    
    # Product nodes
    for pid in sorted(products):
        try:
            pnum = int(float(pid))
        except:
            pnum = pid
        attrs = {"type": "product"}
        if pnum in prod_attrs.index:
            for col in ["ProductGroup", "Division", "Department", "DepartmentGroup"]:
                if col in prod_attrs.columns:
                    val = prod_attrs.loc[pnum][col]
                    if pd.notna(val):
                        attrs[col] = int(val)
        records.append({"type": "node", "id": f"product:{pid}", "attrs": attrs})
    
    # Edges with aggregated sales
    for _, r in sales_agg.iterrows():
        u = f"store:{r['Store']}"
        v = f"product:{r['Product']}"
        records.append({
            "type": "edge",
            "u": u,
            "v": v,
            "attrs": {
                "rel": "sold",
                "time": ymd_to_int(end_date),  # Use end of window as timestamp
                "units": float(r["total_units"]),
                "mean_units": float(r["mean_units"]),
            },
        })
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vn2-data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="graph_qa/data/vn2_snapshots")
    parser.add_argument("--window-weeks", type=int, default=4, help="Weeks per snapshot")
    args = parser.parse_args()
    
    data_dir = Path(args.vn2_data_dir)
    sales = pd.read_csv(data_dir / "Week 0 - 2024-04-08 - Sales.csv")
    instock = pd.read_csv(data_dir / "Week 0 - In Stock.csv")
    master = pd.read_csv(data_dir / "Week 0 - Master.csv")
    
    # Get all date columns
    date_cols = sorted([c for c in sales.columns if c not in ("Store", "Product") and "-" in c])
    print(f"Found {len(date_cols)} weeks: {date_cols[0]} to {date_cols[-1]}")
    
    # Create snapshots
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    snapshots_created = 0
    for i in range(0, len(date_cols), args.window_weeks):
        window_dates = date_cols[i : i + args.window_weeks]
        if not window_dates:
            continue
        
        start = window_dates[0]
        end = window_dates[-1]
        
        snapshot = create_snapshot_graph(sales, instock, master, start, end)
        if snapshot and len(snapshot["records"]) > 100:  # Skip empty snapshots
            out_file = out_dir / f"snapshot_{start}_{end}.jsonl"
            with out_file.open("w") as f:
                for rec in snapshot["records"]:
                    f.write(json.dumps(rec) + "\n")
            snapshots_created += 1
            print(f"  Created {out_file.name} ({len(snapshot['records'])} records)")
    
    print(f"\nTotal snapshots created: {snapshots_created}")
    print(f"Use these for time-series training (each snapshot = one training sample)")


if __name__ == "__main__":
    main()

