"""Check whether more historical rows improve the selected fast model."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from train import CONFIGS, DATA, FOLDS, REPORTS, fit_model, metrics, predict, sample_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--rows", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    df = pd.read_parquet(args.data)
    config = next(c for c in CONFIGS if c["name"] == "hgb_180")
    results = []
    for fold, start, end in FOLDS:
        train = sample_train(df.loc[df.timestamp < start], args.rows)
        valid = df.loc[(df.timestamp >= start) & (df.timestamp < end)]
        tick = perf_counter()
        model = fit_model(config, train, threads=1)
        fit_seconds = perf_counter() - tick
        result = {"fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                  "fit_seconds": round(fit_seconds, 2),
                  **metrics(valid.target.to_numpy(), predict(model, "hgb", valid))}
        results.append(result)
        print(result, flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "training_size_experiment.csv"
    table = pd.DataFrame(results)
    if output.exists():
        table = pd.concat([pd.read_csv(output), table], ignore_index=True)
        table = table.drop_duplicates(["fold", "train_rows"], keep="last")
    table.sort_values(["train_rows", "fold"]).to_csv(output, index=False)


if __name__ == "__main__":
    main()
