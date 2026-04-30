#!/usr/bin/env python
"""Baseline: gradient-boosted trees on six simple features.

Trains in ~5 minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference.

Prerequisites:
    python data/download_data.py   # one-time, ~500 MB download

Run:
    python baseline.py             # trains and saves model.pkl

Your job is to replace this file with something better. The grader only cares
about `predict.py` — this file just needs to produce a `model.pkl` that
`predict.py` can load.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "passenger_count"]


def load_and_prepare_data(path: Path, batch_size: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Load parquet file in batches and prepare features/labels."""
    print(f"    reading {path.name}...", flush=True)
    
    features_list = []
    labels_list = []
    
    parquet_file = pq.ParquetFile(path)
    num_batches = parquet_file.num_row_groups
    
    for i in range(num_batches):
        print(f"      batch {i+1}/{num_batches}...", flush=True)
        table = parquet_file.read_row_group(i)
        df = table.to_pandas()
        
        # Engineer features
        ts = pd.to_datetime(df["requested_at"])
        features = pd.DataFrame({
            "pickup_zone":     df["pickup_zone"].astype("int32"),
            "dropoff_zone":    df["dropoff_zone"].astype("int32"),
            "hour":            ts.dt.hour.astype("int8"),
            "dow":             ts.dt.dayofweek.astype("int8"),
            "month":           ts.dt.month.astype("int8"),
            "passenger_count": df["passenger_count"].astype("int8"),
        })[FEATURES]
        
        features_list.append(features.to_numpy())
        labels_list.append(df["duration_seconds"].to_numpy())
        
        del df, features, ts  # Free memory
    
    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    print(f"      total: {len(X):,} rows", flush=True)
    return X, y


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading and preparing data...", flush=True)
    X_train, y_train = load_and_prepare_data(train_path)
    X_dev, y_dev = load_and_prepare_data(dev_path)

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds", flush=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
