import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from src.models.train_model import save_model


class ModelPromotionTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.winner_path = self.root / "winner_model.pkl"
        self.versions_dir = self.root / "versions"

    @staticmethod
    def make_bundle(score, model_name):
        return {
            "pipeline": model_name,
            "model": model_name,
            "scaler": "scaler",
            "best_model_name": model_name,
            "pinball_loss": score,
            "alpha": 0.8,
            "interpretability": None,
            "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results": [{
                "name": model_name,
                "pipeline": model_name,
                "pinball_loss": score,
                "mae": 0.2,
                "mse": 0.09,
                "rmse": 0.3,
                "r2": 0.5,
            }],
        }

    @patch("src.models.train_model.setup_logging")
    def test_archives_each_candidate_and_only_promotes_lower_loss(self, _setup_logging):
        first = save_model(
            self.make_bundle(0.5, "initial"),
            self.winner_path,
            self.versions_dir,
            self.root / "model_metrics.xlsx",
        )
        worse = save_model(
            self.make_bundle(0.7, "worse"),
            self.winner_path,
            self.versions_dir,
            self.root / "model_metrics.xlsx",
        )
        better = save_model(
            self.make_bundle(0.3, "better"),
            self.winner_path,
            self.versions_dir,
            self.root / "model_metrics.xlsx",
        )

        self.assertTrue(first["promoted"])
        self.assertFalse(worse["promoted"])
        self.assertTrue(better["promoted"])
        self.assertEqual(joblib.load(self.winner_path)["model"], "better")
        self.assertEqual(len(list(self.versions_dir.glob("winner_model_*.pkl"))), 3)
        metrics = pd.read_excel(self.root / "model_metrics.xlsx")
        self.assertEqual(len(metrics), 3)
        self.assertTrue(metrics["fecha_ejecucion"].notna().all())
        self.assertTrue({"pinball_loss", "mae", "mse", "rmse", "r2"}.issubset(metrics.columns))
        self.assertEqual(metrics["promovido_a_winner"].tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
