#!/usr/bin/env python
"""Improved ETA model with advanced feature engineering and ensemble methods.

This replaces baseline.py with:
- Enhanced temporal features (holidays, peak hours, day patterns)
- Zone pair interactions and distance estimation
- Outlier removal and data filtering
- LightGBM + XGBoost ensemble for better predictions
- Target: ~250-280s MAE on dev (vs baseline ~350s)

Prerequisites:
    python data/download_data.py  # one-time, ~500 MB download

Run:
    python train_improved.py       # trains and saves model.pkl
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
CACHE_PATH = Path(__file__).parent / "data_cache.pkl"

# US holidays in 2023
HOLIDAYS_2023 = {
    (1, 1), (1, 16), (2, 20), (3, 29), (5, 29), (7, 4), 
    (9, 4), (10, 9), (11, 23), (11, 24), (12, 25),
}

# Numeric feature names - order matters!
NUMERIC_FEATURES = [
    "pickup_zone", "dropoff_zone", "passenger_count", "hour", "minute", 
    "dow", "month", "day_of_year", "is_weekend", "is_holiday", 
    "is_peak_hour", "is_night", "rush_hour_multiplier", "same_zone",
    "is_outer_borough_pickup", "is_outer_borough_dropoff", "cross_borough"
]


def is_holiday(ts: pd.Series) -> np.ndarray:
    """Check if date is a US holiday."""
    dates = np.column_stack([ts.dt.month, ts.dt.day])
    holidays = np.array(list(HOLIDAYS_2023))
    return np.any((dates[:, np.newaxis] == holidays[np.newaxis, :]).all(axis=2), axis=1)


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and return as DataFrame."""
    ts = pd.to_datetime(df["requested_at"])
    
    is_weekend = (ts.dt.weekday >= 5).astype("int8")
    is_holiday_flag = is_holiday(ts).astype("int8")
    is_peak = ((ts.dt.hour >= 7) & (ts.dt.hour <= 10) | 
               (ts.dt.hour >= 17) & (ts.dt.hour <= 19)).astype("int8")
    is_night = ((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype("int8")
    
    rush_multiplier = np.where(is_peak > 0, 1.5, 1.0)
    
    features = pd.DataFrame({
        "pickup_zone": df["pickup_zone"].astype("int32"),
        "dropoff_zone": df["dropoff_zone"].astype("int32"),
        "passenger_count": df["passenger_count"].astype("int8"),
        "hour": ts.dt.hour.astype("int8"),
        "minute": ts.dt.minute.astype("int8"),
        "dow": ts.dt.weekday.astype("int8"),
        "month": ts.dt.month.astype("int8"),
        "day_of_year": ts.dt.dayofyear.astype("int16"),
        "is_weekend": is_weekend,
        "is_holiday": is_holiday_flag,
        "is_peak_hour": is_peak,
        "is_night": is_night,
        "rush_hour_multiplier": rush_multiplier.astype("float32"),
        "same_zone": (df["pickup_zone"] == df["dropoff_zone"]).astype("int8"),
        "is_outer_borough_pickup": (df["pickup_zone"] > 100).astype("int8"),
        "is_outer_borough_dropoff": (df["dropoff_zone"] > 100).astype("int8"),
        "cross_borough": ((df["pickup_zone"] > 100) != (df["dropoff_zone"] > 100)).astype("int8"),
    })
    
    return features[NUMERIC_FEATURES]  # Ensure column order


def load_and_prepare_data(path: Path, batch_size: int = 1_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Load parquet file in batches and prepare features/labels directly as numpy."""
    print(f"    reading {path.name}...", flush=True)
    
    X_list = []
    y_list = []
    
    parquet_file = pq.ParquetFile(path)
    num_batches = parquet_file.num_row_groups
    
    for i in range(num_batches):
        print(f"      batch {i+1}/{num_batches}...", flush=True)
        table = parquet_file.read_row_group(i)
        df = table.to_pandas()
        
        # Feature engineering
        features_df = engineer_features_df(df)
        X_batch = features_df.values.astype('float32')
        y_batch = df["duration_seconds"].to_numpy()
        
        X_list.append(X_batch)
        y_list.append(y_batch)
        
        del df, features_df, X_batch, y_batch
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"      total: {len(X):,} rows", flush=True)
    
    return X, y


def remove_outliers(X: np.ndarray, y: np.ndarray, percentile: float = 98) -> Tuple[np.ndarray, np.ndarray]:
    """Remove trips with extreme durations."""
    threshold = np.percentile(y, percentile)
    mask = y <= threshold
    print(f"    removed {(~mask).sum():,} outlier trips (>{threshold:.0f}s)")
    return X[mask], y[mask]


def load_or_prepare_data(train_path: Path, dev_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load from cache if exists, otherwise prepare and cache."""
    if CACHE_PATH.exists():
        print("Loading cleaned data from cache...", flush=True)
        with open(CACHE_PATH, "rb") as f:
            X_train, y_train, X_dev, y_dev = pickle.load(f)
        print(f"  X_train: {X_train.shape}, X_dev: {X_dev.shape}", flush=True)
        return X_train, y_train, X_dev, y_dev
    
    # Load and prepare data
    print("Loading and preparing data...", flush=True)
    X_train, y_train = load_and_prepare_data(train_path)
    X_dev, y_dev = load_and_prepare_data(dev_path)
    
    print("\nCleaning data...", flush=True)
    X_train, y_train = remove_outliers(X_train, y_train, percentile=98)
    
    # Save to cache
    print(f"Caching cleaned data to {CACHE_PATH}...", flush=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump((X_train, y_train, X_dev, y_dev), f)
    
    return X_train, y_train, X_dev, y_dev


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run `python data/download_data.py` first.")

    X_train, y_train, X_dev, y_dev = load_or_prepare_data(train_path, dev_path)
    
    print(f"  train: {len(y_train):,} samples, mean duration: {y_train.mean():.0f}s")
    print(f"  dev:   {len(y_dev):,} samples, mean duration: {y_dev.mean():.0f}s")

    print("\nData shape:", flush=True)
    print(f"  X_train: {X_train.shape}")
    print(f"  X_dev:   {X_dev.shape}")
    print(f"  {X_train.shape[1]} features: {', '.join(NUMERIC_FEATURES)}")

    # ============ Model 1: LightGBM (often superior to XGBoost for tabular data) ============
    print("\nTraining LightGBM model...", flush=True)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.85,
        colsample_bytree=0.85,
        lambda_l1=0.5,
        lambda_l2=0.5,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
        metric='mae',
    )
    t0 = time.time()
    lgb_model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], 
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)])
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)
    
    lgb_preds_dev = lgb_model.predict(X_dev)

    # ============ Model 2: XGBoost (tuned for better performance) ============
    print("\nTraining XGBoost model...", flush=True)
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    xgb_model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)],
                  early_stopping=50, verbose=False)
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)
    
    xgb_preds_dev = xgb_model.predict(X_dev)

    # ============ Ensemble: weighted average of both models ============
    print("\nCreating ensemble...", flush=True)
    # LightGBM typically performs slightly better
    ensemble_preds = 0.55 * lgb_preds_dev + 0.45 * xgb_preds_dev
    
    # Evaluation
    from sklearn.metrics import mean_absolute_error
    mae_lgb = mean_absolute_error(y_dev, lgb_preds_dev)
    mae_xgb = mean_absolute_error(y_dev, xgb_preds_dev)
    mae_ensemble = mean_absolute_error(y_dev, ensemble_preds)
    
    print(f"\nDev set MAE:")
    print(f"  LightGBM:    {mae_lgb:.1f}s")
    print(f"  XGBoost:     {mae_xgb:.1f}s")
    print(f"  Ensemble:    {mae_ensemble:.1f}s")
    
    # Save ensemble with feature engineering function as a wrapper class
    class EnsembleModel:
        """Wrapper that handles feature engineering + ensemble prediction."""
        def __init__(self, lgb_model, xgb_model, feature_names):
            self.lgb_model = lgb_model
            self.xgb_model = xgb_model
            self.feature_names = feature_names
            # For XGBoost compatibility check
            self.get_booster = None
        
        def predict(self, X):
            """Predict on raw feature array."""
            lgb_preds = self.lgb_model.predict(X)
            xgb_preds = self.xgb_model.predict(X)
            return 0.55 * lgb_preds + 0.45 * xgb_preds
    
    ensemble = EnsembleModel(lgb_model, xgb_model, NUMERIC_FEATURES)
    
    print(f"\nSaving ensemble to {MODEL_PATH}...", flush=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ensemble, f)
    
    print("✓ Done! Model saved.")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
