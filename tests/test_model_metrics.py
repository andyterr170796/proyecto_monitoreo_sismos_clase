import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train_model import train_model_pipeline


class ModelMetricsTests(unittest.TestCase):
    @patch("src.models.train_model.setup_logging")
    @patch(
        "src.models.train_model._get_candidate_models",
        return_value={"baseline": DummyRegressor(strategy="mean")},
    )
    def test_training_records_all_validation_metrics(self, _candidate_models, _setup_logging):
        features = ["feature_a", "feature_b"]
        X_train = pd.DataFrame(
            [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1], [5, 0]],
            columns=features,
        )
        y_train = pd.Series([1, 2, 2, 4, 3, 6])
        X_validation = pd.DataFrame([[6, 1], [7, 0], [8, 1]], columns=features)
        y_validation = pd.Series([5, 8, 7])

        bundle = train_model_pipeline(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
        row = bundle["results"][0]
        predictions = bundle["pipeline"].predict(X_validation)
        errors = y_validation.to_numpy() - predictions
        expected = {
            "mae": mean_absolute_error(y_validation, predictions),
            "mse": mean_squared_error(y_validation, predictions),
            "rmse": np.sqrt(mean_squared_error(y_validation, predictions)),
            "r2": r2_score(y_validation, predictions),
        }

        self.assertTrue(bundle["run_timestamp"])
        for metric, value in expected.items():
            self.assertAlmostEqual(row[metric], value)
        self.assertAlmostEqual(
            row["pinball_loss"],
            np.mean(np.maximum(0.8 * errors, -0.2 * errors)),
        )


if __name__ == "__main__":
    unittest.main()
