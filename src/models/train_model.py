import logging
import os

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

from src.config.config import PATHS, setup_logging


def _get_candidate_models():
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )
    return models


def pinball_loss(y_true, y_pred, alpha=0.8):
    """Calcula la Pinball Loss para cuantiles con alpha especificado."""
    errors = y_true - y_pred
    return np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))


def _linear_model_interpretability(X_train_scaled, y_train, feature_names):
    design = pd.DataFrame(X_train_scaled, columns=feature_names, index=y_train.index)
    design = sm.add_constant(design, has_constant="add")
    inference = sm.OLS(y_train, design).fit()
    confidence_intervals = inference.conf_int(alpha=0.05)
    return pd.DataFrame({
        "feature": inference.params.index,
        "coeficiente": inference.params.to_numpy(),
        "p_valor": inference.pvalues.to_numpy(),
        "IC_95_inferior": confidence_intervals.iloc[:, 0].to_numpy(),
        "IC_95_superior": confidence_intervals.iloc[:, 1].to_numpy(),
    })


def train_models(X_train, y_train, X_validation, y_validation, alpha=0.8):
    """Entrena modelos y selecciona el mejor usando el conjunto de validación recibido."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando entrenamiento de modelos con validación por Pinball Loss (alpha=0.8)...")

    logger.info(
        "Train: X=%s, y=%s | Validation: X=%s, y=%s",
        X_train.shape,
        y_train.shape,
        X_validation.shape,
        y_validation.shape,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    results = []
    for name, model in _get_candidate_models().items():
        model.fit(X_train_scaled, y_train)
        predicciones = model.predict(X_validation_scaled)
        loss = pinball_loss(y_validation.to_numpy(), predicciones, alpha=alpha)
        results.append({
            "name": name,
            "model": model,
            "pinball_loss": loss,
        })
        logger.info("Modelo %s entrenado con Pinball Loss (alpha=0.8): %.6f", name, loss)

    ganador = min(results, key=lambda item: item["pinball_loss"])
    logger.info(
        "Modelo ganador por Pinball Loss: %s con loss %.6f",
        ganador["name"],
        ganador["pinball_loss"],
    )
    interpretability = None
    if ganador["name"] == "linear_regression":
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [
            f"feature_{index + 1}" for index in range(X_train_scaled.shape[1])
        ]
        interpretability = _linear_model_interpretability(
            X_train_scaled,
            y_train,
            feature_names,
        )

    return {
        "model": ganador["model"],
        "scaler": scaler,
        "best_model_name": ganador["name"],
        "pinball_loss": ganador["pinball_loss"],
        "alpha": alpha,
        "results": results,
        "interpretability": interpretability,
    }


def save_model(model_bundle, filepath):
    """Guarda el mejor modelo y su scaler, seleccionando por Pinball Loss."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Guardando modelo ganador...")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({
        "model": model_bundle["model"],
        "scaler": model_bundle["scaler"],
        "best_model_name": model_bundle["best_model_name"],
        "pinball_loss": model_bundle["pinball_loss"],
        "alpha": model_bundle["alpha"],
        "interpretability": model_bundle["interpretability"],
    }, filepath)

    scaler_path = PATHS["model_scaler"]
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(model_bundle["scaler"], scaler_path)

    logger.info("Modelo ganador guardado en: %s | Pinball Loss: %.6f", filepath, model_bundle["pinball_loss"])
    logger.info("Scaler guardado en: %s", scaler_path)
    print(f"Modelo ganador guardado en: {filepath} | Pinball Loss: {model_bundle['pinball_loss']:.6f}")
    print(f"Scaler guardado en: {scaler_path}")
    return filepath

