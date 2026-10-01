import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PATHS = {
    "project_root": PROJECT_ROOT,
    "model_dir": PROJECT_ROOT / "models",
    "model_store": PROJECT_ROOT / "models" / "model_lr.pkl",
    "model_scaler": PROJECT_ROOT / "models" / "standard_scaler.pkl",
    "outputs_dir": PROJECT_ROOT / "outputs",
    "comparison_plot": PROJECT_ROOT / "outputs" / "real_vs_prediccion.png",
    "model_interpretability": PROJECT_ROOT / "outputs" / "interpretabilidad_modelo.xlsx",
    "data_dir": PROJECT_ROOT / "data",
    "raw_data_dir": PROJECT_ROOT / "data" / "raw",
    "processed_data_dir": PROJECT_ROOT / "data" / "processed",
    "db_parquet": PROJECT_ROOT / "data" / "raw" / "db_terremotos.parquet",
    "processed_excel": PROJECT_ROOT / "data" / "processed" / "terremotos_procesados.xlsx",
    "processed_excel_reshape": PROJECT_ROOT / "data" / "processed" / "terremotos_procesados_reshape.xlsx",
    "processed_excel_reshape_test": PROJECT_ROOT / "data" / "processed" / "terremotos_procesados_reshape_test.xlsx",
    "processed_predictions": PROJECT_ROOT / "data" / "processed" / "predicciones.xlsx",
    "logs_dir": PROJECT_ROOT / "logs",
    "general_log": PROJECT_ROOT / "logs" / "general.log",
}

LIMITE_POR_PAGINA = 20_000
MAGNITUD_MINIMA = 3.0
TEST_SIZE = 0.02
NUMERIC_FEATURES = [
    "L1_sismos",
    "L2_sismos",
    "L3_sismos",
    "L4_sismos",
    "L5_sismos",
    "L6_sismos",
    "L7_sismos",
    "L8_sismos",
    "L9_sismos",
    "L10_sismos",
    "L11_sismos",
    "L12_sismos",
    "MA_sismos",
    "MA2_sismos",
    "EMA_sismos",
    "EMA2_sismos",
]
TARGET = "sismos"


def setup_logging():
    PATHS["logs_dir"].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(PATHS["general_log"], encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
