"""Create one-hour-ahead features: each row predicts demand at its timestamp."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data/features/station_hourly_demand.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data/features/forecasting_dataset.parquet"
LAGS = (1, 2, 3, 24, 25, 48, 168, 169)


def create_station_features(group: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    group = group.sort_values("timestamp")
    hours = pd.date_range(group.timestamp.iloc[0], end, freq="h")
    station = group.set_index("timestamp").reindex(hours)
    station.index.name = "timestamp"
    last_trip = pd.Series(station.index.where(station["demand"].notna()), index=station.index).ffill()
    recently_active = (station.index.to_series() - last_trip).dt.total_seconds().le(30 * 86_400)
    station["target"] = station["demand"].fillna(0).astype("float32")
    station["station_id"] = group.station_id.iloc[0]
    station["latitude"] = group.latitude.iloc[0]
    station["longitude"] = group.longitude.iloc[0]
    station = station[["station_id", "latitude", "longitude", "target"]]
    station["hour"] = station.index.hour.astype("int8")
    station["day_of_week"] = station.index.day_of_week.astype("int8")
    station["month"] = station.index.month.astype("int8")
    station["is_weekend"] = (station.index.day_of_week >= 5).astype("int8")
    for name, value, period in (("hour", station.index.hour, 24),
                                ("dow", station.index.day_of_week, 7),
                                ("month", station.index.month - 1, 12)):
        angle = 2 * np.pi * value / period
        station[f"{name}_sin"] = np.sin(angle).astype("float32")
        station[f"{name}_cos"] = np.cos(angle).astype("float32")
    for lag in LAGS:
        station[f"demand_lag_{lag}"] = station["target"].shift(lag)
    history = station["target"].shift(1)
    for window in (3, 24, 168):
        station[f"rolling_mean_{window}"] = history.rolling(window).mean().astype("float32")
    station["rolling_std_24"] = history.rolling(24).std().astype("float32")
    return station.iloc[169:].loc[recently_active.iloc[169:]].reset_index()


def create_features(demand: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "timestamp", "demand", "latitude", "longitude"}
    if missing := required - set(demand.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if demand.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("Duplicate station-hour rows")
    parts = []
    end = demand.timestamp.max()
    for i, (_, group) in enumerate(demand.groupby("station_id", sort=False), 1):
        if len(group) >= 2:
            parts.append(create_station_features(group, end))
        if i % 250 == 0:
            print(f"Processed {i:,} stations", flush=True)
    result = pd.concat(parts, ignore_index=True)
    result = result.dropna(subset=[f"demand_lag_{lag}" for lag in LAGS])
    result["station_id"] = result["station_id"].astype("category")
    return result.sort_values("timestamp", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    df = create_features(pd.read_parquet(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df):,} rows for {df.station_id.nunique():,} stations: {args.output}")
    print(f"Period: {df.timestamp.min()} to {df.timestamp.max()}")


if __name__ == "__main__":
    main()
