#!/usr/bin/env python
"""One-time download & cleanup of NYC TLC 2023 yellow-taxi data.

Produces:
    data/train.parquet       -- 11.5 months of 2023, ~37M trips after cleaning
    data/dev.parquet         -- last 2 weeks of 2023, ~1M trips (for local grading)
    data/sample_1M.parquet   -- 1M-row subset of train for fast iteration

The held-out Eval set (a 2024 slice) is kept by Gobblecube and never distributed.

Takes ~5 minutes on a fast connection, ~20 minutes on a slow one.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = [f"2023-{m:02d}" for m in range(1, 13)]

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

CUTOFF = pd.Timestamp("2023-12-18")   # dev = last ~2 weeks of Dec
SAMPLE_SIZE = 1_000_000


def download_month(yyyymm: str) -> Path:
    RAW_DIR.mkdir(exist_ok=True)
    url = f"{BASE_URL}/yellow_tripdata_{yyyymm}.parquet"
    out = RAW_DIR / f"yellow_{yyyymm}.parquet"
    if out.exists():
        print(f"  cached   {out.name}")
        return out
    print(f"  fetching {url}")
    urlretrieve(url, out)
    return out


def clean(paths: list[Path]) -> pa.Table:
    """Process and clean all monthly parquets, returning Arrow table."""
    print("  concatenating & filtering...", flush=True)
    result_table = None
    
    for i, p in enumerate(paths, 1):
        print(f"    processing {i}/{len(paths)}: {p.name}", flush=True)
        df = pd.read_parquet(
            p,
            columns=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "passenger_count",
            ],
        )
        
        print(f"      transforming...", flush=True)
        duration = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds()

        clean_df = pd.DataFrame({
            "pickup_zone":      df["PULocationID"].astype("int32"),
            "dropoff_zone":     df["DOLocationID"].astype("int32"),
            "requested_at":     df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "passenger_count":  df["passenger_count"].fillna(1).astype("int8"),
            "duration_seconds": duration.astype("float64"),
            "_ts":              df["tpep_pickup_datetime"],
        })

        mask = (
            (clean_df["duration_seconds"] >= 30)
            & (clean_df["duration_seconds"] <= 3 * 3600)
            & (clean_df["pickup_zone"].between(1, 265))
            & (clean_df["dropoff_zone"].between(1, 265))
            & (clean_df["_ts"].dt.year == 2023)
        )
        clean_df = clean_df.loc[mask]
        
        print(f"      converting to arrow...", flush=True)
        table = pa.Table.from_pandas(clean_df)
        
        if result_table is None:
            result_table = table
        else:
            print(f"      merging...", flush=True)
            result_table = pa.concat_tables([result_table, table])
        
        del df, clean_df, table  # Free memory immediately
    
    print(f"  cleaned: {result_table.num_rows:,} rows", flush=True)
    return result_table


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["_ts"] < CUTOFF].drop(columns=["_ts"]).reset_index(drop=True)
    dev = df[df["_ts"] >= CUTOFF].drop(columns=["_ts"]).reset_index(drop=True)
    return train, dev


def split_arrow(table: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Split Arrow table on timestamp without converting to pandas."""
    ts_col = table.column("_ts")
    mask = pc.less(ts_col, pa.scalar(CUTOFF.isoformat()))
    train = table.filter(mask).drop(["_ts"])
    dev = table.filter(pc.invert(mask)).drop(["_ts"])
    return train, dev


def main() -> None:
    print("Step 1: download monthly parquets")
    paths = [download_month(m) for m in MONTHS]

    print("\nStep 2: clean & combine")
    result_table = clean(paths)

    print("\nStep 3: train/dev split")
    train_table, dev_table = split_arrow(result_table)
    print(f"  train: {train_table.num_rows:,} rows", flush=True)
    print(f"  dev:   {dev_table.num_rows:,} rows", flush=True)
    
    print("  writing train.parquet...", flush=True)
    pq.write_table(train_table, DATA_DIR / "train.parquet")
    print(f"  train.parquet written", flush=True)
    
    print("  writing dev.parquet...", flush=True)
    pq.write_table(dev_table, DATA_DIR / "dev.parquet")
    print(f"  dev.parquet written", flush=True)

    print("\nStep 4: 1M-row training sample")
    print("  converting train to pandas for sampling...", flush=True)
    train_df = train_table.to_pandas()
    sample = train_df.sample(n=min(SAMPLE_SIZE, len(train_df)), random_state=42)
    sample.reset_index(drop=True).to_parquet(
        DATA_DIR / "sample_1M.parquet", index=False
    )
    print(f"  sample_1M.parquet: {len(sample):,} rows", flush=True)

    print("\nDone. Next: `python baseline.py`")


if __name__ == "__main__":
    main()
