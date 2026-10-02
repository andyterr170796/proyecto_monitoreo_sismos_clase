import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
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


def train_model_pipeline(X_train, y_train, X_validation, y_validation, alpha=0.8):
    """Entrena pipelines completos y selecciona el mejor con el conjunto de validación."""
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

    results = []
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for name, estimator in _get_candidate_models().items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predicciones = pipeline.predict(X_validation)
        y_true = y_validation.to_numpy(dtype=float)
        loss = float(pinball_loss(y_true, predicciones, alpha=alpha))
        mse = float(mean_squared_error(y_true, predicciones))
        results.append({
            "name": name,
            "pipeline": pipeline,
            "pinball_loss": loss,
            "mae": float(mean_absolute_error(y_true, predicciones)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, predicciones)) if len(y_true) > 1 else float("nan"),
        })
        logger.info(
            "Modelo %s | Pinball Loss: %.6f | MAE: %.6f | MSE: %.6f | RMSE: %.6f | R2: %.6f",
            name,
            loss,
            results[-1]["mae"],
            mse,
            results[-1]["rmse"],
            results[-1]["r2"],
        )

    ganador = min(results, key=lambda item: item["pinball_loss"])
    logger.info(
        "Modelo ganador por Pinball Loss: %s con loss %.6f",
        ganador["name"],
        ganador["pinball_loss"],
    )
    pipeline = ganador["pipeline"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    interpretability = None
    if ganador["name"] == "linear_regression":
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [
            f"feature_{index + 1}" for index in range(X_train.shape[1])
        ]
        interpretability = _linear_model_interpretability(
            scaler.transform(X_train),
            y_train,
            feature_names,
        )

    return {
        "pipeline": pipeline,
        "model": model,
        "scaler": scaler,
        "best_model_name": ganador["name"],
        "pinball_loss": ganador["pinball_loss"],
        "alpha": alpha,
        "results": results,
        "run_timestamp": run_timestamp,
        "interpretability": interpretability,
    }


def _append_metrics_history(model_bundle, promoted, metrics_path):
    rows = []
    for result in model_bundle["results"]:
        rows.append({
            "fecha_ejecucion": model_bundle["run_timestamp"],
            "modelo": result["name"],
            "pinball_loss": result["pinball_loss"],
            "mae": result["mae"],
            "mse": result["mse"],
            "rmse": result["rmse"],
            "r2": result["r2"],
            "seleccionado_en_ejecucion": result["name"] == model_bundle["best_model_name"],
            "promovido_a_winner": (
                promoted and result["name"] == model_bundle["best_model_name"]
            ),
        })

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    current_history = (
        pd.read_excel(metrics_path) if metrics_path.exists() else pd.DataFrame()
    )
    history = pd.concat([current_history, pd.DataFrame(rows)], ignore_index=True)
    history.to_excel(metrics_path, index=False, sheet_name="metricas")


def save_model(model_bundle, filepath, versions_dir=None, metrics_path=None):
    """Versiona cada candidato y promueve al ganador solo si mejora Pinball Loss."""
    setup_logging()
    logger = logging.getLogger(__name__)
    filepath = Path(filepath)
    versions_dir = Path(versions_dir or PATHS["model_versions_dir"])
    metrics_path = Path(metrics_path or PATHS["model_metrics"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    versions_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": model_bundle["pipeline"],
        "model": model_bundle["model"],
        "scaler": model_bundle["scaler"],
        "best_model_name": model_bundle["best_model_name"],
        "pinball_loss": model_bundle["pinball_loss"],
        "alpha": model_bundle["alpha"],
        "interpretability": model_bundle["interpretability"],
        "run_timestamp": model_bundle["run_timestamp"],
        "validation_metrics": [
            {key: value for key, value in result.items() if key != "pipeline"}
            for result in model_bundle["results"]
        ],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    version_path = versions_dir / f"winner_model_{timestamp}.pkl"
    joblib.dump(artifact, version_path)

    previous_loss = float("inf")
    if filepath.exists():
        current_artifact = joblib.load(filepath)
        previous_loss = float(current_artifact.get("pinball_loss", float("inf")))

    candidate_loss = float(model_bundle["pinball_loss"])
    promoted = candidate_loss < previous_loss
    _append_metrics_history(model_bundle, promoted, metrics_path)
    if promoted:
        temporary_path = filepath.with_name(f".{filepath.stem}_{timestamp}.tmp.pkl")
        try:
            joblib.dump(artifact, temporary_path)
            os.replace(temporary_path, filepath)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        logger.info(
            "Nuevo ganador promovido: Pinball Loss %.6f (anterior: %.6f)",
            candidate_loss,
            previous_loss,
        )
    else:
        logger.info(
            "Se conserva el ganador actual: Pinball Loss %.6f; candidato: %.6f",
            previous_loss,
            candidate_loss,
        )

    logger.info("Candidato versionado en: %s", version_path)
    return {
        "promoted": promoted,
        "winner_path": filepath,
        "version_path": version_path,
        "previous_pinball_loss": previous_loss,
        "candidate_pinball_loss": candidate_loss,
        "metrics_path": metrics_path,
    }

