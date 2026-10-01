import numpy as np
import pandas as pd

from src.config.config import NUMERIC_FEATURES


def _next_day_features(history, forecast_date, ema_states):
    counts = history["sismos"].to_numpy(dtype=float)
    values = {}
    for feature in NUMERIC_FEATURES:
        if feature.startswith("L") and feature.endswith("_sismos"):
            lag = int(feature[1:feature.index("_")])
            values[feature] = counts[-lag]
        elif feature == "mes":
            values[feature] = forecast_date.month
        elif feature == "anio":
            values[feature] = forecast_date.year
        elif feature == "MA_sismos":
            values[feature] = counts[-3:].mean()
        elif feature == "MA2_sismos":
            values[feature] = counts[-5:].mean()
        elif feature == "EMA_sismos":
            values[feature] = ema_states[3]
        elif feature == "EMA2_sismos":
            values[feature] = ema_states[5]
        else:
            raise ValueError(f"No se conoce cómo construir la feature: {feature}")
    return pd.DataFrame([values], columns=NUMERIC_FEATURES)


def forecast_next_days(history, model_bundle, days=3):
    """Pronostica recursivamente el conteo diario usando el bundle entrenado."""
    required_columns = {"fecha", "sismos", "EMA_sismos", "EMA2_sismos"}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas para pronosticar: {sorted(missing_columns)}")
    if len(history) < 12:
        raise ValueError("Se necesitan al menos 12 días de historial para generar los rezagos.")

    history = history.sort_values("fecha").reset_index(drop=True).copy()
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    ema_states = {
        3: float(history["EMA_sismos"].iloc[-1]),
        5: float(history["EMA2_sismos"].iloc[-1]),
    }
    alpha_by_span = {span: 2 / (span + 1) for span in ema_states}
    forecasts = []
    last_date = pd.Timestamp(history["fecha"].iloc[-1]).normalize()

    for step in range(1, days + 1):
        forecast_date = last_date + pd.Timedelta(days=step)
        previous_count = float(history["sismos"].iloc[-1])
        for span, alpha in alpha_by_span.items():
            ema_states[span] = alpha * previous_count + (1 - alpha) * ema_states[span]

        features = _next_day_features(history, forecast_date, ema_states)
        scaled_features = scaler.transform(features)
        prediction = max(0.0, float(model.predict(scaled_features)[0]))
        forecasts.append({
            "fecha": forecast_date,
            "prediccion": prediction,
            "prediccion_redondeada": int(np.rint(prediction)),
        })
        history.loc[len(history)] = {
            "fecha": forecast_date,
            "sismos": prediction,
            "EMA_sismos": ema_states[3],
            "EMA2_sismos": ema_states[5],
        }

    return pd.DataFrame(forecasts)
