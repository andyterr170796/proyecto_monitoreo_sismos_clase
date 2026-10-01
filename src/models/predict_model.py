import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression

# Ajuste path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config import (
    PATHS,
    NUMERIC_FEATURES,
    TARGET,
    setup_logging,
)


def setup_logger():
    setup_logging()


def _resolve_model_and_scaler(model):
    if isinstance(model, dict):
        bundle = model
        model_obj = bundle.get("model")
        scaler = bundle.get("scaler")
        return model_obj, scaler
    return model, None


def predicciones_en_produccion(model=None):
    """Evalúa el modelo con el conjunto de prueba y guarda datos reales y predichos."""
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando predicciones sobre el conjunto de prueba")

    if model is None:
        bundle = joblib.load(PATHS["model_store"])
        model, scaler = _resolve_model_and_scaler(bundle)
        logger.info("Modelo cargado desde: %s", PATHS["model_store"])
    else:
        bundle = model if isinstance(model, dict) else {}
        model, scaler = _resolve_model_and_scaler(model)

    if scaler is None:
        scaler = joblib.load(PATHS["model_scaler"])
        logger.info("Scaler cargado desde: %s", PATHS["model_scaler"])

    df_test = pd.read_excel(PATHS["processed_excel_reshape_test"])
    columnas_requeridas = [*NUMERIC_FEATURES, TARGET]
    columnas_faltantes = [columna for columna in columnas_requeridas if columna not in df_test]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el conjunto de prueba: {columnas_faltantes}")

    X_test = df_test.loc[:, NUMERIC_FEATURES]
    y_test = df_test[TARGET]
    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and list(scaler_features) != NUMERIC_FEATURES:
        raise ValueError(
            "El scaler guardado fue entrenado con features distintas. "
            "Vuelve a entrenar el modelo ejecutando: python -m src.main_pipeline"
        )
    X_test_scaled = scaler.transform(X_test)

    logger.info("Conjunto de prueba cargado: X_test=%s, y_test=%s", X_test.shape, y_test.shape)
    resultados = X_test.copy()
    resultados["y_test"] = y_test.to_numpy()
    resultados["predicciones"] = model.predict(X_test_scaled)

    best_model_name = bundle.get("best_model_name") or type(model).__name__
    logger.info("Modelo ganador cargado para predicción: %s", best_model_name)
    if best_model_name == "linear_regression" or isinstance(model, LinearRegression):
        interpretability = bundle.get("interpretability")
        if interpretability is None:
            raise ValueError(
                "El bundle no contiene la inferencia del modelo lineal. "
                "Vuelve a entrenar ejecutando: python -m src.main_pipeline"
            )
    else:
        shap_values = shap.TreeExplainer(model).shap_values(X_test_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim != 2 or shap_values.shape != X_test_scaled.shape:
            raise ValueError(f"Forma inesperada de SHAP: {shap_values.shape}")
        feature_importances = getattr(model, "feature_importances_", None)
        if feature_importances is None:
            raise ValueError(
                f"El modelo {best_model_name} no expone feature_importances_."
            )
        columna_fecha = "fecha" if "fecha" in df_test.columns else "time"
        fechas = pd.to_datetime(df_test[columna_fecha]).to_numpy()
        interpretability = pd.DataFrame({
            "fecha": np.repeat(fechas, len(NUMERIC_FEATURES)),
            "feature": np.tile(NUMERIC_FEATURES, len(X_test)),
            "valor_feature": X_test.to_numpy().ravel(),
            "shap_value": shap_values.ravel(),
            "feature_importance": np.tile(feature_importances, len(X_test)),
        })

    errors = y_test.to_numpy() - resultados["predicciones"].to_numpy()
    alpha = 0.8
    pinball_loss = np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))
    logger.info("Pinball Loss (alpha=0.8) en prueba: %s", pinball_loss)

    PATHS["outputs_dir"].mkdir(parents=True, exist_ok=True)
    comparativa = resultados[["y_test", "predicciones"]].copy()
    comparativa.columns = ["real", "prediccion"]
    columna_fecha = "fecha" if "fecha" in df_test.columns else "time"
    comparativa["fecha"] = pd.to_datetime(df_test[columna_fecha])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        comparativa["fecha"],
        comparativa["real"].to_numpy(),
        label="Real",
        marker="o",
        linewidth=1.5,
    )
    ax.plot(
        comparativa["fecha"],
        comparativa["prediccion"].to_numpy(),
        label="Predicción",
        marker="x",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_title("Comparativa real vs predicción")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("N° Sismos")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PATHS["comparison_plot"], dpi=200)
    plt.close(fig)

    PATHS["processed_predictions"].parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(PATHS["processed_predictions"], engine="openpyxl") as writer:
        resultados.to_excel(writer, sheet_name="predicciones", index=False)
        interpretability.to_excel(writer, sheet_name="interpretabilidad", index=False)
    interpretability.to_excel(PATHS["model_interpretability"], index=False)
    logger.info("Gráfico real vs predicción guardado en: %s", PATHS["comparison_plot"])
    logger.info("Predicciones y valores reales guardados en: %s", PATHS["processed_predictions"])
    logger.info("Tabla de interpretabilidad guardada en: %s", PATHS["model_interpretability"])
    logger.info("Filas evaluadas: %s", len(resultados))
    print("Gráfico real vs predicción guardado en:", PATHS["comparison_plot"])
    print("Predicciones e interpretabilidad guardadas en:", PATHS["processed_predictions"])
    print("Tabla de interpretabilidad guardada en:", PATHS["model_interpretability"])
    if best_model_name == "linear_regression" or isinstance(model, LinearRegression):
        print("\nInterpretabilidad de Linear Regression:")
        print(interpretability.to_string(index=False))
    return comparativa


if __name__ == '__main__':
    predicciones_en_produccion()
