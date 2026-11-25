"""Build station-hour departures from all Divvy monthly CSVs with bounded RAM."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw/divvy"
OUTPUT_FILE = PROJECT_ROOT / "data/features/station_hourly_demand.parquet"
COLUMNS = ["ride_id", "started_at", "ended_at", "start_station_id", "start_station_name", "start_lat", "start_lng"]


def build_hourly_demand(raw_dir: Path, chunksize: int = 250_000) -> pd.DataFrame:
    files = sorted(raw_dir.glob("2024??-divvy-tripdata.csv"))
    if len(files) != 12:
        raise ValueError(f"Expected 12 monthly files in {raw_dir}; found {len(files)}")

    monthly_parts = []
    for file in files:
        start = perf_counter()
        parts = []
        seen_ride_ids: set[str] = set()
        for chunk in pd.read_csv(file, usecols=COLUMNS, chunksize=chunksize, low_memory=False,
                                 dtype={"ride_id": "string", "start_station_id": "string",
                                        "start_station_name": "string"}):
            chunk = chunk.dropna(subset=["ride_id", "start_station_id", "started_at", "ended_at"])
            fresh = ~chunk["ride_id"].isin(seen_ride_ids)
            chunk = chunk.loc[fresh].drop_duplicates("ride_id")
            seen_ride_ids.update(chunk["ride_id"].tolist())
            chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
            chunk["ended_at"] = pd.to_datetime(chunk["ended_at"], errors="coerce")
            duration = (chunk["ended_at"] - chunk["started_at"]).dt.total_seconds()
            chunk = chunk.loc[duration.between(0.001, 86_400)]
            chunk = chunk.loc[chunk["started_at"].dt.year.eq(2024)]
            chunk["timestamp"] = chunk["started_at"].dt.floor("h")
            part = chunk.groupby(["start_station_id", "timestamp"], sort=False, as_index=False).agg(
                demand=("ride_id", "size"), station_name=("start_station_name", "first"),
                latitude=("start_lat", "first"), longitude=("start_lng", "first"),
            )
            parts.append(part)
        if parts:
            monthly_parts.append(pd.concat(parts, ignore_index=True))
        print(f"{file.name}: {sum(map(len, parts)):,} grouped rows, {perf_counter() - start:.1f}s", flush=True)

    demand = pd.concat(monthly_parts, ignore_index=True).groupby(
        ["start_station_id", "timestamp"], sort=False, as_index=False
    ).agg(demand=("demand", "sum"), station_name=("station_name", "first"),
          latitude=("latitude", "first"), longitude=("longitude", "first"))
    demand = demand.rename(columns={"start_station_id": "station_id"})
    demand["demand"] = demand["demand"].astype("int16")
    demand[["latitude", "longitude"]] = demand[["latitude", "longitude"]].astype("float32")
    return demand.sort_values(["station_id", "timestamp"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    demand = build_hourly_demand(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(args.output, index=False)
    print(f"Saved {len(demand):,} station-hours for {demand.station_id.nunique():,} stations: {args.output}")
    print(f"Period: {demand.timestamp.min()} to {demand.timestamp.max()}")


if __name__ == "__main__":
    main()
