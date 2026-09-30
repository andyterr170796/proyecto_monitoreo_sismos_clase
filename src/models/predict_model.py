import logging
import os
import sys
import pickle
import pandas as pd

# Ajuste path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config import (
    MODEL_STORE,
    NUMERIC_FEATURES,
    PROCESSED_EXCEL_RESHAPE_TEST,
    PROCESSED_PREDICTIONS,
    TARGET,
    setup_logging,
)

def setup_logger():
    setup_logging()

def predicciones_en_produccion(model=None):
    """Evalúa el modelo con el conjunto de prueba y guarda datos reales y predichos."""
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando predicciones sobre el conjunto de prueba")

    if model is None:
        with open(MODEL_STORE, "rb") as model_file:
            model = pickle.load(model_file)
        logger.info("Modelo cargado desde: %s", MODEL_STORE)
    df_test = pd.read_excel(PROCESSED_EXCEL_RESHAPE_TEST)
    columnas_requeridas = [*NUMERIC_FEATURES, TARGET]
    columnas_faltantes = [columna for columna in columnas_requeridas if columna not in df_test]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el conjunto de prueba: {columnas_faltantes}")

    X_test = df_test.loc[:, NUMERIC_FEATURES]
    y_test = df_test[TARGET]
    logger.info("Conjunto de prueba cargado: X_test=%s, y_test=%s", X_test.shape, y_test.shape)
    resultados = X_test.copy()
    resultados["y_test"] = y_test.to_numpy()
    resultados["predicciones"] = model.predict(X_test)
    logger.info("R^2 de predicción en prueba: %s", model.score(X_test, y_test))

    PROCESSED_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_excel(PROCESSED_PREDICTIONS, index=False)
    logger.info("Predicciones y valores reales guardados en: %s", PROCESSED_PREDICTIONS)
    logger.info("Filas evaluadas: %s", len(resultados))
    print("Predicciones, X_test e y_test guardados en:", PROCESSED_PREDICTIONS)
    return resultados

if __name__ == '__main__':
    predicciones_en_produccion()
