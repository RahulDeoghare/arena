"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.

This version uses an improved ensemble model (LightGBM + XGBoost) with
advanced feature engineering including temporal patterns, time-of-day features,
and zone interaction features.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

# US holidays in 2023 (matching train_improved.py)
_HOLIDAYS_2023 = {
    (1, 1), (1, 16), (2, 20), (3, 29), (5, 29), (7, 4), 
    (9, 4), (10, 9), (11, 23), (11, 24), (12, 25),
}

# Feature order must match train_improved.py NUMERIC_FEATURES
_FEATURE_ORDER = [
    "pickup_zone", "dropoff_zone", "passenger_count", "hour", "minute", 
    "dow", "month", "day_of_year", "is_weekend", "is_holiday", 
    "is_peak_hour", "is_night", "rush_hour_multiplier", "same_zone",
    "is_outer_borough_pickup", "is_outer_borough_dropoff", "cross_borough"
]


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    dt = datetime.fromisoformat(request["requested_at"])
    
    pickup_zone = int(request["pickup_zone"])
    dropoff_zone = int(request["dropoff_zone"])
    passenger_count = int(request["passenger_count"])
    
    # Calculate temporal features
    hour = dt.hour
    minute = dt.minute
    dow = dt.weekday()  # 0=Monday, 6=Sunday
    month = dt.month
    day_of_year = dt.timetuple().tm_yday
    
    is_weekend = 1 if dow >= 5 else 0
    is_holiday = 1 if (month, dt.day) in _HOLIDAYS_2023 else 0
    is_peak_hour = 1 if (7 <= hour <= 10) or (17 <= hour <= 19) else 0
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    rush_hour_multiplier = 1.5 if is_peak_hour else 1.0
    
    same_zone = 1 if pickup_zone == dropoff_zone else 0
    is_outer_borough_pickup = 1 if pickup_zone > 100 else 0
    is_outer_borough_dropoff = 1 if dropoff_zone > 100 else 0
    cross_borough = 1 if is_outer_borough_pickup != is_outer_borough_dropoff else 0
    
    # Create feature vector in the correct order
    x = np.array([
        pickup_zone,
        dropoff_zone,
        passenger_count,
        hour,
        minute,
        dow,
        month,
        day_of_year,
        is_weekend,
        is_holiday,
        is_peak_hour,
        is_night,
        rush_hour_multiplier,
        same_zone,
        is_outer_borough_pickup,
        is_outer_borough_dropoff,
        cross_borough,
    ], dtype=np.float32).reshape(1, -1)
    
    return float(_MODEL.predict(x)[0])
