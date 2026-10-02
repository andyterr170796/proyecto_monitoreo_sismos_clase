import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import main_pipeline


class WinnerOutputTests(unittest.TestCase):
    @patch("src.main_pipeline.predicciones_en_produccion")
    @patch("src.main_pipeline.save_model")
    @patch("src.main_pipeline.train_model_pipeline", return_value={"model": "candidate"})
    @patch("src.main_pipeline.split_features_target")
    @patch("src.main_pipeline.pd.read_excel")
    @patch("src.main_pipeline.transformar_datos")
    @patch("src.main_pipeline.carga_batch_sismos")
    @patch("src.main_pipeline.setup_logging")
    def test_winner_outputs_are_generated_only_when_candidate_is_promoted(
        self,
        _setup_logging,
        _load_events,
        transform_data,
        read_validation,
        split_features,
        _train_model,
        save_model,
        predict_production,
    ):
        data = pd.DataFrame({
            **{feature: [1.0, 2.0, 3.0, 4.0] for feature in main_pipeline.NUMERIC_FEATURES},
            main_pipeline.TARGET: [1, 2, 3, 4],
            "fecha": pd.date_range("2025-01-01", periods=4),
        })
        transform_data.return_value = data
        read_validation.return_value = data.tail(1).copy()
        split_features.side_effect = lambda frame, features, target: (
            frame[features], frame[target]
        )
        save_model.side_effect = [
            {"promoted": False, "metrics_path": Path("metrics.xlsx")},
            {
                "promoted": True,
                "metrics_path": Path("metrics.xlsx"),
                "version_path": Path("winner_model_version.pkl"),
            },
        ]

        main_pipeline.run_pipeline()
        main_pipeline.run_pipeline()

        predict_production.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()