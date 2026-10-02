import unittest

import pandas as pd

from src.config.config import PATHS


class ProcessedEarthquakeDataTests(unittest.TestCase):
    def test_required_event_fields_have_no_missing_values(self):
        processed_path = PATHS["processed_excel"]
        if not processed_path.exists():
            self.skipTest("Ejecuta el pipeline para generar el Excel procesado.")

        data = pd.read_excel(processed_path)
        required_columns = ["mag", "depth_km", "latitude", "longitude"]
        missing_columns = [column for column in required_columns if column not in data]
        self.assertFalse(missing_columns, f"Faltan columnas: {missing_columns}")

        missing_counts = data[required_columns].isna().sum()
        columns_with_missing = missing_counts[missing_counts.gt(0)].to_dict()
        self.assertFalse(
            columns_with_missing,
            f"Hay valores faltantes en el Excel procesado: {columns_with_missing}",
        )


if __name__ == "__main__":
    unittest.main()
