"""Small regression checks for time alignment and forecast availability."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.create_features import create_features  # noqa: E402
from forecasting.train import metrics  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_lags_refer_to_prediction_hour_and_use_only_past(self):
        demand = pd.DataFrame({
            "station_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-10 00:00"]),
            "demand": [5, 7, 11],
            "latitude": [41.0] * 3,
            "longitude": [-87.0] * 3,
        })
        features = create_features(demand).set_index("timestamp")
        row = features.loc[pd.Timestamp("2024-01-10 00:00")]
        self.assertEqual(row["target"], 11)
        self.assertEqual(row["demand_lag_1"], 0)
        self.assertEqual(row["demand_lag_168"], 0)
        self.assertEqual(row["rolling_mean_24"], 0)
        before = features.loc[pd.Timestamp("2024-01-09 00:00")]
        self.assertEqual(before["target"], 0)
        self.assertEqual(before["demand_lag_168"], 7)

    def test_inactive_station_excluded_after_30_days_without_trip(self):
        demand = pd.DataFrame({
            "station_id": ["A", "A", "B"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-10"]),
            "demand": [1, 1, 1], "latitude": [41.0] * 3, "longitude": [-87.0] * 3,
        })
        features = create_features(demand)
        a = features.loc[features.station_id.eq("A"), "timestamp"]
        self.assertEqual(a.max(), pd.Timestamp("2024-02-01 00:00"))

    def test_error_metrics(self):
        result = metrics(np.array([0, 2]), np.array([0, 1]))
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5))
        self.assertAlmostEqual(result["wape"], 0.5)


if __name__ == "__main__":
    unittest.main()
