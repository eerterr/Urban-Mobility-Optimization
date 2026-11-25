from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = Path("data/processed/divvy_2024.parquet")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    df = pd.read_parquet(DATA_FILE)
    print(f"Dataset shape: {df.shape}")
    return df


def basic_analysis(df):
    print("\n========== BASIC STATISTICS ==========")
    print(f"Total rides: {len(df):,}")
    print(f"Date range: {df['started_at'].min()} — {df['started_at'].max()}")
    print(f"Average duration: {df['duration_min'].mean():.2f} min")
    print(f"Median duration: {df['duration_min'].median():.2f} min")

    print("\nRideable types:")
    print(df["rideable_type"].value_counts())

    print("\nUser types:")
    print(df["member_casual"].value_counts())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(10))


def plot_hourly_demand(df):
    hourly = df.groupby("hour").size()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=hourly.index, y=hourly.values, marker="o")
    plt.title("Bike-sharing Demand by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hourly_demand.png", dpi=300)
    plt.close()


def plot_weekday_demand(df):
    weekday_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    daily = df.groupby("day_of_week").size()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=daily.index, y=daily.values)
    plt.title("Bike-sharing Demand by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Trips")
    plt.xticks(range(7), weekday_names, rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weekday_demand.png", dpi=300)
    plt.close()


def plot_monthly_demand(df):
    monthly = df.groupby("month").size()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=monthly.index, y=monthly.values)
    plt.title("Monthly Bike-sharing Demand")
    plt.xlabel("Month")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_demand.png", dpi=300)
    plt.close()


def plot_top_stations(df):
    stations = (
        df.groupby("start_station_name")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(x=stations.values, y=stations.index)
    plt.title("Top 15 Stations by Number of Departures")
    plt.xlabel("Number of Trips")
    plt.ylabel("Station")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_stations.png", dpi=300)
    plt.close()


def plot_weekend_comparison(df):
    comparison = df.groupby("is_weekend")["ride_id"].count()

    labels = ["Weekday", "Weekend"]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=comparison.values)
    plt.title("Weekday vs Weekend Demand")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weekend_comparison.png", dpi=300)
    plt.close()


def plot_hourly_heatmap(df):
    heatmap_data = df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="ride_id",
        aggfunc="count"
    )

    plt.figure(figsize=(16, 7))
    sns.heatmap(heatmap_data, cmap="YlOrRd")
    plt.title("Demand Heatmap: Day of Week vs Hour")
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demand_heatmap.png", dpi=300)
    plt.close()


def main():
    df = load_data()

    basic_analysis(df)

    print("\nGenerating plots...")
    plot_hourly_demand(df)
    plot_weekday_demand(df)
    plot_monthly_demand(df)
    plot_top_stations(df)
    plot_weekend_comparison(df)
    plot_hourly_heatmap(df)

    print(f"\nPlots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()