import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_STORE = MODEL_DIR / "model_lr.pkl"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PARQUET = RAW_DATA_DIR / "db_terremotos.parquet"
PROCESSED_EXCEL = PROCESSED_DATA_DIR / "terremotos_procesados.xlsx"
PROCESSED_EXCEL_RESHAPE = PROCESSED_DATA_DIR / "terremotos_procesados_reshape.xlsx"
PROCESSED_EXCEL_RESHAPE_TEST = PROCESSED_DATA_DIR / "terremotos_procesados_reshape_test.xlsx"
PROCESSED_PREDICTIONS = PROCESSED_DATA_DIR / "predicciones.xlsx"
LOGS_DIR = PROJECT_ROOT / "logs"
GENERAL_LOG = LOGS_DIR / "general.log"
GET_DATA_LOG = GENERAL_LOG
MAX_DATE_LOG = GENERAL_LOG
TRANSFORM_DATA_LOG = GENERAL_LOG
LIMITE_POR_PAGINA = 20_000
MAGNITUD_MINIMA = 3.0
NUMERIC_FEATURES = ["L1_sismos", "L2_sismos", "MA_sismos", "MA2_sismos", "EMA_sismos", "EMA2_sismos"]
TARGET = "sismos"


def setup_logging():
	LOGS_DIR.mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
		handlers=[
			logging.FileHandler(GENERAL_LOG, encoding="utf-8"),
			logging.StreamHandler(),
		],
		force=True,
	)
