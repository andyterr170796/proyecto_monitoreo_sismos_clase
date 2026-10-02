import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import (
    NUMERIC_FEATURES,
    PATHS,
    TARGET,
    setup_logging,
)
from src.data.get_data import carga_batch_sismos
from src.data.transform_data import transformar_datos
from src.features.build_features import split_features_target
from src.models.forecast_model import forecast_next_days
from src.models.predict_model import predicciones_en_produccion
from src.models.train_model import save_model, train_model_pipeline


def run_pipeline():
    """Ejecuta extracción, transformación, entrenamiento y evaluación en orden."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando pipeline integral de monitoreo sísmico")

    carga_batch_sismos()
    training_data = transformar_datos()
    validation_data = pd.read_excel(PATHS["processed_excel_reshape_test"])

    if validation_data.empty or len(validation_data) >= len(training_data):
        raise ValueError("Los conjuntos de entrenamiento y validación no son válidos.")
    training_data = training_data.iloc[:-len(validation_data)]
    logger.info(
        "Train: %s filas | Validation: %s filas",
        len(training_data),
        len(validation_data),
    )

    X_train, y_train = split_features_target(
        training_data, NUMERIC_FEATURES, TARGET
    )
    X_validation, y_validation = split_features_target(
        validation_data, NUMERIC_FEATURES, TARGET
    )
    model_bundle = train_model_pipeline(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    promotion = save_model(model_bundle, PATHS["model_store"])
    logger.info("Historial de métricas actualizado en %s", promotion["metrics_path"])
    if promotion["promoted"]:
        logger.info("Candidato promovido desde %s", promotion["version_path"])
        predicciones_en_produccion()
    else:
        logger.info(
            "Se mantiene el modelo ganador y sus artefactos de interpretabilidad y comparación"
        )

    logger.info("Pipeline finalizado correctamente")
    return model_bundle


if __name__ == "__main__":
    run_pipeline()
