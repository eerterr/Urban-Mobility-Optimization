from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/divvy")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "divvy_2024.parquet"


def load_data():
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    dfs = []

    for file in files:
        print(f"Loading {file.name}...")
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    print(f"\nLoaded {len(data):,} rows")
    print(f"Columns: {list(data.columns)}")

    return data


def clean_data(df):
    print("\n--- Data cleaning ---")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Parse timestamps
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")

    # Remove rows with missing timestamps
    df = df.dropna(subset=["started_at", "ended_at"])

    # Calculate trip duration in minutes
    df["duration_min"] = (
        df["ended_at"] - df["started_at"]
    ).dt.total_seconds() / 60

    # Remove invalid durations
    df = df[
        (df["duration_min"] > 0)
        & (df["duration_min"] <= 24 * 60)
    ]

    # Remove duplicate trips
    if "ride_id" in df.columns:
        df = df.drop_duplicates(subset="ride_id")

    # Add temporal features
    df["date"] = df["started_at"].dt.date
    df["hour"] = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.dayofweek
    df["month"] = df["started_at"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    print(f"Rows after cleaning: {len(df):,}")

    return df


def main():
    data = load_data()

    print("\n--- Initial data overview ---")
    print(data.info())
    print("\nMissing values:")
    print(data.isna().sum())

    cleaned = clean_data(data)

    print("\n--- Final dataset ---")
    print(cleaned.head())
    print(cleaned.describe())

    cleaned.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Final shape: {cleaned.shape}")


if __name__ == "__main__":
    main()