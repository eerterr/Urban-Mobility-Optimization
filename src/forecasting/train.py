"""Tune and evaluate one-hour-ahead station demand with chronological splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/features/forecasting_dataset.parquet"
REPORTS = ROOT / "results/reports"
FIGURES = ROOT / "results/figures"
MODELS = ROOT / "results/models"
SEED = 42
NUMERIC = [
    "latitude", "longitude", "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_24",
    "demand_lag_25", "demand_lag_48", "demand_lag_168", "demand_lag_169",
    "rolling_mean_3", "rolling_mean_24", "rolling_mean_168", "rolling_std_24",
]
FOLDS = [
    ("august", "2024-08-01", "2024-08-15"),
    ("october", "2024-10-01", "2024-10-15"),
    ("november", "2024-11-01", "2024-11-15"),
]
CONFIGS = [
    {"name": "hgb_100", "family": "hgb", "params": {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31, "max_depth": 8, "l2_regularization": 0.0}},
    {"name": "hgb_180", "family": "hgb", "params": {"max_iter": 180, "learning_rate": 0.06, "max_leaf_nodes": 63, "max_depth": 10, "l2_regularization": 1.0}},
    {"name": "cat_rmse", "family": "catboost", "params": {"iterations": 300, "depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 3.0, "loss_function": "RMSE"}},
    {"name": "cat_poisson", "family": "catboost", "params": {"iterations": 300, "depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 3.0, "loss_function": "Poisson"}},
]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = y - pred
    absolute = np.abs(error)
    denominator = np.sum(y)
    variance = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(absolute.sum() / denominator) if denominator else float("nan"),
        "r2": float(1 - np.sum(error**2) / variance) if variance else float("nan"),
    }


def sample_train(train: pd.DataFrame, limit: int) -> pd.DataFrame:
    return train.sample(n=min(limit, len(train)), random_state=SEED) if len(train) > limit else train


def fit_model(config: dict, train: pd.DataFrame, threads: int):
    if config["family"] == "hgb":
        model = HistGradientBoostingRegressor(
            **config["params"], early_stopping=False, random_state=SEED,
        )
        model.fit(train[NUMERIC], train["target"].to_numpy())
    else:
        model = CatBoostRegressor(
            **config["params"], thread_count=threads, random_seed=SEED,
            verbose=False, allow_writing_files=False,
        )
        # CatBoost handles high-cardinality station identity without one-hot expansion.
        data = train[["station_id", *NUMERIC]].copy()
        data["station_id"] = data["station_id"].astype(str)
        model.fit(data, train["target"].to_numpy(), cat_features=["station_id"])
    return model


def predict(model, family: str, frame: pd.DataFrame) -> np.ndarray:
    if family == "catboost":
        data = frame[["station_id", *NUMERIC]].copy()
        data["station_id"] = data["station_id"].astype(str)
    else:
        data = frame[NUMERIC]
    return np.maximum(0, np.asarray(model.predict(data), dtype=np.float32))


def evaluation_rows(name: str, fold: str, train: pd.DataFrame,
                    valid: pd.DataFrame, pred: np.ndarray, seconds: float) -> list[dict]:
    y = valid["target"].to_numpy()
    activity = train.groupby("station_id", observed=True)["target"].mean()
    busy = set(activity.nlargest(max(1, int(np.ceil(len(activity) * 0.1)))).index)
    segments = {
        "all": np.ones(len(valid), dtype=bool),
        "zero_demand": y == 0,
        "busy_stations": valid["station_id"].isin(busy).to_numpy(),
        "peak_hours": valid["hour"].isin((7, 8, 9, 16, 17, 18, 19)).to_numpy(),
        "weekday": (valid["is_weekend"] == 0).to_numpy(),
        "weekend": (valid["is_weekend"] == 1).to_numpy(),
    }
    return [{"model": name, "fold": fold, "segment": segment, "rows": int(mask.sum()),
             "fit_seconds": round(seconds, 2), **metrics(y[mask], pred[mask])}
            for segment, mask in segments.items() if mask.any()]


def tune(df: pd.DataFrame, max_rows: int, threads: int) -> tuple[list[dict], list[dict]]:
    train = sample_train(df.loc[df.timestamp < "2024-07-01"], max_rows)
    valid = df.loc[(df.timestamp >= "2024-07-01") & (df.timestamp < "2024-07-15")]
    if train.empty or valid.empty:
        raise ValueError("July tuning split is empty; use the complete 2024 dataset")
    candidates = [c for c in CONFIGS if c["family"] != "catboost" or CatBoostRegressor is not None]
    rows = []
    for config in candidates:
        start = perf_counter()
        model = fit_model(config, train, threads)
        pred = predict(model, config["family"], valid)
        row = {"name": config["name"], "family": config["family"],
               "train_rows": len(train), "valid_rows": len(valid),
               "fit_seconds": round(perf_counter() - start, 2), **metrics(valid.target.to_numpy(), pred)}
        rows.append(row)
        print(f"Tune {config['name']}: MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} {row['fit_seconds']:.0f}s", flush=True)
    winners = []
    for family in {c["family"] for c in candidates}:
        family_rows = [r for r in rows if r["family"] == family]
        best = min(family_rows, key=lambda r: (r["mae"], r["rmse"]))
        winners.append(next(c for c in candidates if c["name"] == best["name"]))
    return rows, winners


def plot_comparison(summary: pd.DataFrame, output: Path) -> None:
    table = summary.sort_values("mae_mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, label in zip(axes, ("mae", "rmse"), ("MAE", "RMSE")):
        ax.barh(table.model, table[f"{metric}_mean"], xerr=table[f"{metric}_std"],
                color=["#2066a3" if not name.startswith("lag_") else "#adb9c5" for name in table.model],
                capsize=3)
        ax.invert_yaxis()
        ax.set_xlabel(f"{label}, bikes per station-hour")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Chronological validation · August, October, November 2024")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--max-train-rows", type=int, default=800_000)
    parser.add_argument("--tune-rows", type=int, default=400_000)
    parser.add_argument("--final-train-rows", type=int, default=None,
                        help="Historical row cap for the final model; choose using training_size_experiment.py")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--reuse-results", action="store_true",
                        help="Reuse existing July tuning and fold metrics; rerun final selection and holdout only")
    args = parser.parse_args()
    if min(args.max_train_rows, args.tune_rows, args.threads) < 1:
        parser.error("row limits and threads must be positive")
    if args.final_train_rows is not None and args.final_train_rows < 1:
        parser.error("--final-train-rows must be positive")
    for folder in (REPORTS, FIGURES, MODELS):
        folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    missing = set(["timestamp", "station_id", "target", *NUMERIC]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    print(f"Data: {len(df):,} rows; {df.timestamp.min()} to {df.timestamp.max()}; {df.station_id.nunique():,} stations", flush=True)
    if df.timestamp.max() < pd.Timestamp("2024-12-31"):
        raise ValueError("Full-year features required; rebuild demand and features from all 12 CSVs")
    if args.reuse_results:
        tuning = pd.read_csv(REPORTS / "tuning.csv").to_dict("records")
        fold_metrics = pd.read_csv(REPORTS / "fold_metrics.csv")
        print("Reusing completed tuning and chronological validation", flush=True)
    else:
        tuning, _ = tune(df, args.tune_rows, args.threads)
        pd.DataFrame(tuning).to_csv(REPORTS / "tuning.csv", index=False)
        rows = []
        winners_for_folds = []
        for family in set(row["family"] for row in tuning):
            best = min((row for row in tuning if row["family"] == family),
                       key=lambda row: (row["mae"], row["rmse"]))
            winners_for_folds.append(next(c for c in CONFIGS if c["name"] == best["name"]))
        for fold, start, end in FOLDS:
            historical = df.loc[df.timestamp < start]
            train = sample_train(historical, args.max_train_rows)
            valid = df.loc[(df.timestamp >= start) & (df.timestamp < end)]
            if train.empty or valid.empty:
                raise ValueError(f"Empty fold: {fold}")
            print(f"Fold {fold}: train {len(train):,}/{len(historical):,}, valid {len(valid):,}", flush=True)
            for lag in (1, 24, 168):
                pred = valid[f"demand_lag_{lag}"].to_numpy()
                rows += evaluation_rows(f"lag_{lag}", fold, train, valid, pred, 0)
            for config in winners_for_folds:
                tick = perf_counter()
                model = fit_model(config, train, args.threads)
                seconds = perf_counter() - tick
                pred = predict(model, config["family"], valid)
                rows += evaluation_rows(config["name"], fold, train, valid, pred, seconds)
                print(f"  {config['name']}: MAE={metrics(valid.target.to_numpy(), pred)['mae']:.4f}, fit={seconds:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(REPORTS / "fold_metrics.csv", index=False)
        fold_metrics = pd.DataFrame(rows)

    winners = []
    for family in set(row["family"] for row in tuning):
        best = min((row for row in tuning if row["family"] == family),
                   key=lambda row: (row["mae"], row["rmse"]))
        winners.append(next(c for c in CONFIGS if c["name"] == best["name"]))
    all_rows = fold_metrics.loc[fold_metrics.segment.eq("all")]
    summary = all_rows.groupby("model", as_index=False).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        wape_mean=("wape", "mean"), r2_mean=("r2", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )
    summary["selection_score"] = summary.mae_mean + 0.25 * summary.mae_std + 0.05 * summary.rmse_mean
    summary = summary.sort_values("selection_score")
    summary.to_csv(REPORTS / "model_comparison.csv", index=False)
    plot_comparison(summary, FIGURES / "model_comparison.png")
    candidate_names = {c["name"] for c in winners}
    eligible = summary.loc[summary.model.isin(candidate_names)]
    near_best = eligible.loc[eligible.selection_score <= eligible.selection_score.min() * 1.01]
    selected_name = near_best.sort_values(["fit_seconds_mean", "selection_score"]).iloc[0].model
    selected = next(c for c in winners if c["name"] == selected_name)

    # December is evaluated once, after model selection on earlier periods.
    historical = df.loc[df.timestamp < "2024-12-01"]
    train = sample_train(historical, args.final_train_rows or args.max_train_rows)
    holdout = df.loc[df.timestamp >= "2024-12-01"]
    tick = perf_counter()
    model = fit_model(selected, train, args.threads)
    seconds = perf_counter() - tick
    holdout_rows = []
    for lag in (1, 24, 168):
        holdout_rows += evaluation_rows(f"lag_{lag}", "december_holdout", train, holdout,
                                        holdout[f"demand_lag_{lag}"].to_numpy(), 0)
    holdout_rows += evaluation_rows(selected_name, "december_holdout", train, holdout,
                                    predict(model, selected["family"], holdout), seconds)
    holdout_table = pd.DataFrame(holdout_rows)
    holdout_table.to_csv(REPORTS / "holdout_metrics.csv", index=False)
    holdout_table.loc[holdout_table.segment.eq("all"), ["model", "mae", "rmse"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE"}
    ).to_csv(REPORTS / "forecasting_metrics.csv", index=False)
    model_path = MODELS / ("best_model.cbm" if selected["family"] == "catboost" else "best_model.joblib")
    if selected["family"] == "catboost":
        model.save_model(str(model_path))
    else:
        joblib.dump(model, model_path)
    manifest = {
        "forecast_horizon": "next hour", "target": "hourly station departures",
        "selected_model": selected_name, "family": selected["family"], "parameters": selected["params"],
        "features": (["station_id"] if selected["family"] == "catboost" else []) + NUMERIC,
        "model_path": str(model_path.relative_to(ROOT)), "training_cutoff": "2024-12-01",
        "training_rows_sampled": len(train), "historical_rows_available": len(historical),
        "cv_train_row_cap": args.max_train_rows,
        "folds": FOLDS, "holdout": "2024-12-01 through 2024-12-31",
        "selection_rule": "Among models within 1% of the best CV score (mean MAE + 0.25 * std MAE + 0.05 * mean RMSE), choose the fastest fit",
        "dependencies": {"scikit_learn": sklearn.__version__,
                         "catboost": "1.2.10 (used in comparison)" if any(
                             row["family"] == "catboost" for row in tuning) else "not used"},
    }
    (REPORTS / "selected_model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Selected {selected_name}; saved {model_path}", flush=True)
    print(holdout_table.query("segment == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
