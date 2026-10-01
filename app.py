from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config.config import NUMERIC_FEATURES, PATHS, TARGET
from src.models.forecast_model import forecast_next_days


st.set_page_config(
    page_title="Sismos | Centro de monitoreo",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #1e302c;
        --muted: #73817b;
        --green: #157a62;
        --green-soft: #e7f2ed;
        --coral: #dc7057;
        --gold: #bc8c37;
        --line: #dce5df;
        --paper: #f5f8f5;
        --white: #ffffff;
    }
    html, body, [class*="css"] { font-family: 'Manrope', 'Segoe UI', sans-serif; color: var(--ink); }
    .stApp {
        background-color: var(--paper);
        background-image: linear-gradient(rgba(21, 122, 98, .035) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(21, 122, 98, .035) 1px, transparent 1px);
        background-size: 34px 34px;
    }
    [data-testid="stHeader"] { background: rgba(245, 248, 245, .94); }
    [data-testid="stSidebar"] { background: #edf3ee; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: var(--ink); }
    .block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 3rem; }
    .masthead { border-bottom: 1px solid var(--line); padding: 0 0 1.25rem; margin-bottom: 1.25rem; }
    .eyebrow { color: var(--green); font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
    .masthead h1 { font-size: 2.05rem; font-weight: 800; letter-spacing: 0; margin: .35rem 0 .15rem; color: var(--ink); }
    .masthead p { color: var(--muted); margin: 0; font-size: .92rem; }
    [data-testid="stMetric"] { background: var(--white); border: 1px solid var(--line); border-top: 3px solid var(--green); padding: 1rem 1.1rem; border-radius: 5px; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .8rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'DM Mono', monospace; font-size: 1.55rem; }
    [data-testid="stMetricDelta"] { font-size: .75rem; }
    div[data-testid="stTabs"] button { font-weight: 700; color: var(--muted); }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--green); border-bottom-color: var(--green); }
    .section-title { font-size: 1.13rem; font-weight: 800; color: var(--ink); margin: .4rem 0 .2rem; }
    .section-note { color: var(--muted); font-size: .84rem; margin: 0 0 .9rem; }
    .forecast-date { font-family: 'DM Mono', monospace; color: var(--green); font-weight: 500; }
    .stDownloadButton button, .stButton button { border-radius: 4px; border: 1px solid var(--green); color: var(--green); font-weight: 700; }
    .stDownloadButton button:hover, .stButton button:hover { background: var(--green); color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_excel(path, modified):
    return pd.read_excel(path)


@st.cache_resource(show_spinner=False)
def _load_bundle(path, modified):
    return joblib.load(path)


def _read_artifacts():
    history_path = PATHS["processed_excel_reshape"]
    validation_path = PATHS["processed_excel_reshape_test"]
    model_path = PATHS["model_store"]
    required_paths = (history_path, validation_path, model_path)
    missing = [str(path) for path in required_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos del pipeline. Ejecuta `python -m src.main_pipeline`. "
            + "; ".join(missing)
        )

    history = _load_excel(str(history_path), history_path.stat().st_mtime_ns)
    validation = _load_excel(str(validation_path), validation_path.stat().st_mtime_ns)
    bundle = _load_bundle(str(model_path), model_path.stat().st_mtime_ns)
    return history, validation, bundle


def _pinball_loss(actual, predicted, alpha=0.8):
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.maximum(alpha * errors, (alpha - 1) * errors)))


def _validate_artifacts(history, validation, bundle):
    for label, frame in (("histórico", history), ("validación", validation)):
        missing = [column for column in ["fecha", TARGET, *NUMERIC_FEATURES] if column not in frame]
        if missing:
            raise ValueError(f"Faltan columnas en {label}: {missing}")
    if not isinstance(bundle, dict) or not {"model", "scaler"}.issubset(bundle):
        raise ValueError("El archivo del modelo no contiene el bundle esperado. Vuelve a entrenar.")
    scaler_features = getattr(bundle["scaler"], "feature_names_in_", None)
    if scaler_features is not None and list(scaler_features) != NUMERIC_FEATURES:
        raise ValueError("El modelo guardado usa otro esquema de features. Vuelve a entrenar el pipeline.")
    dates = pd.to_datetime(history["fecha"]).sort_values()
    if dates.diff().dropna().dt.days.gt(1).any():
        raise ValueError("El histórico tiene días ausentes; actualiza la transformación antes de pronosticar.")


def _score_validation(validation, bundle):
    X = validation.loc[:, NUMERIC_FEATURES]
    scaled = bundle["scaler"].transform(X)
    predicted = bundle["model"].predict(scaled)
    actual = validation[TARGET].to_numpy(dtype=float)
    errors = actual - predicted
    score_frame = pd.DataFrame({
        "fecha": pd.to_datetime(validation["fecha"]),
        "real": actual,
        "prediccion": predicted,
        "error": errors,
    })
    metrics = {
        "pinball": _pinball_loss(actual, predicted, bundle.get("alpha", 0.8)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
    return score_frame, metrics


def _style_figure(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=30, b=8),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Manrope, Segoe UI, sans-serif", color="#1e302c", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor="#dce5df", tickfont=dict(color="#73817b")),
        yaxis=dict(showgrid=True, gridcolor="#e5ece7", zeroline=False, tickfont=dict(color="#73817b")),
    )
    return fig


def _history_forecast_chart(history, forecast, days_to_show):
    visible = history.tail(days_to_show)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=visible["fecha"], y=visible[TARGET], name="Observado",
        mode="lines", line=dict(color="#157a62", width=2.5),
        hovertemplate="%{x|%d-%m-%Y}<br>Observado: %{y:.0f}<extra></extra>",
    ))
    forecast_dates = [visible["fecha"].iloc[-1], *forecast["fecha"].tolist()]
    forecast_values = [float(visible[TARGET].iloc[-1]), *forecast["prediccion"].tolist()]
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values, name="Pronóstico",
        mode="lines+markers", line=dict(color="#dc7057", width=2.5, dash="dash"),
        marker=dict(size=8, color="#dc7057"),
        hovertemplate="%{x|%d-%m-%Y}<br>Pronóstico: %{y:.1f}<extra></extra>",
    ))
    fig.add_vrect(x0=forecast["fecha"].iloc[0], x1=forecast["fecha"].iloc[-1],
                  fillcolor="#dc7057", opacity=.07, line_width=0)
    fig.update_layout(title="Sismos observados y próximos 3 días", yaxis_title="Eventos diarios")
    return _style_figure(fig)


def _render_interpretability(bundle, validation):
    model_name = bundle.get("best_model_name", type(bundle["model"]).__name__)
    st.markdown('<div class="section-title">Interpretabilidad del ganador</div>', unsafe_allow_html=True)
    if model_name == "linear_regression":
        table = bundle.get("interpretability")
        if table is None:
            st.warning("Este bundle no tiene inferencia estadística. Reentrena con `python -m src.main_pipeline`.")
            return
        st.caption("Regresión OLS sobre features estandarizadas. El IC corresponde al 95%.")
        st.dataframe(table, use_container_width=True, hide_index=True, height=520)
        coefficients = table.loc[table["feature"] != "const"].copy()
        coefficients = coefficients.reindex(coefficients["coeficiente"].abs().sort_values().index)
        fig = go.Figure(go.Bar(
            x=coefficients["coeficiente"], y=coefficients["feature"], orientation="h",
            marker_color=np.where(coefficients["coeficiente"] >= 0, "#157a62", "#dc7057"),
            hovertemplate="%{y}<br>Coeficiente: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(title="Coeficientes estandarizados", xaxis_title="Cambio estimado en sismos")
        st.plotly_chart(_style_figure(fig, height=460), use_container_width=True)
        return

    import shap

    X = validation.loc[:, NUMERIC_FEATURES]
    scaled = bundle["scaler"].transform(X)
    values = shap.TreeExplainer(bundle["model"]).shap_values(scaled)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    importance = getattr(bundle["model"], "feature_importances_", None)
    if importance is None:
        st.warning("Este modelo no expone importancias de variables.")
        return
    summary = pd.DataFrame({
        "feature": NUMERIC_FEATURES,
        "SHAP_abs_medio": np.abs(values).mean(axis=0),
        "SHAP_medio": values.mean(axis=0),
        "feature_importance": importance,
    }).sort_values("SHAP_abs_medio", ascending=False)
    st.caption("SHAP medio absoluto resume el impacto; SHAP medio conserva su dirección en la validación.")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    top = summary.head(12).sort_values("SHAP_abs_medio")
    fig = go.Figure(go.Bar(
        x=top["SHAP_abs_medio"], y=top["feature"], orientation="h",
        marker_color="#bc8c37", hovertemplate="%{y}<br>Impacto medio: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(title="Features con mayor impacto SHAP", xaxis_title="Mean |SHAP|")
    st.plotly_chart(_style_figure(fig, height=430), use_container_width=True)


def main():
    st.sidebar.markdown("### SISMO / OBSERVATORIO")
    st.sidebar.caption("Conteo diario · modelo predictivo")
    history_window = st.sidebar.selectbox("Ventana histórica", [30, 60, 90, 180, 365], index=2, format_func=lambda n: f"Últimos {n} días")
    if st.sidebar.button("Actualizar artefactos", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption(f"Fuente procesada\n{PATHS['processed_excel_reshape'].name}")

    try:
        history, validation, bundle = _read_artifacts()
        history["fecha"] = pd.to_datetime(history["fecha"])
        validation["fecha"] = pd.to_datetime(validation["fecha"])
        _validate_artifacts(history, validation, bundle)
        forecast = forecast_next_days(history, bundle, days=3)
        validation_scores, metrics = _score_validation(validation, bundle)
    except Exception as exc:
        st.error(str(exc))
        st.info("Genera los artefactos del modelo con `python -m src.main_pipeline` y vuelve a cargar la aplicación.")
        st.stop()

    model_name = bundle.get("best_model_name", type(bundle["model"]).__name__)
    model_label = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
    }.get(model_name, model_name)
    latest_date = history["fecha"].iloc[-1]
    latest_count = float(history[TARGET].iloc[-1])
    mean_7d = float(history[TARGET].tail(7).mean())
    forecast_total = float(forecast["prediccion"].sum())

    st.markdown(
        f"""
        <div class="masthead">
            <div class="eyebrow">USGS · MONITOREO SÍSMICO · {latest_date:%d %b %Y}</div>
            <h1>Actividad sísmica</h1>
            <p>Histórico diario, rendimiento del modelo y proyección recursiva a 3 días.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    kpis = st.columns(4)
    kpis[0].metric("Último día observado", f"{latest_count:,.0f}", latest_date.strftime("%d-%m-%Y"))
    kpis[1].metric("Promedio · 7 días", f"{mean_7d:,.1f}", "eventos / día")
    kpis[2].metric("Pronóstico · 3 días", f"{forecast_total:,.1f}", "eventos esperados")
    kpis[3].metric("Pinball Loss · validación", f"{metrics['pinball']:.3f}", f"{model_label}")

    overview_tab, forecast_tab, performance_tab, interpretation_tab = st.tabs(
        ["Panorama", "Pronóstico 3 días", "Rendimiento", "Interpretabilidad"]
    )

    with overview_tab:
        left, right = st.columns([1.7, 1], gap="large")
        with left:
            st.plotly_chart(_history_forecast_chart(history, forecast, history_window), use_container_width=True)
        with right:
            st.markdown('<div class="section-title">Próximos días</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-note">Estimación recursiva; cada predicción alimenta el siguiente día.</div>', unsafe_allow_html=True)
            view = forecast[["fecha", "prediccion", "prediccion_redondeada"]].copy()
            view.columns = ["Fecha", "Esperado", "Aproximado"]
            view["Fecha"] = view["Fecha"].dt.strftime("%d-%m-%Y")
            view["Esperado"] = view["Esperado"].map(lambda value: f"{value:.1f}")
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.markdown(f"**Modelo activo**  \n{model_label}")
            st.markdown(f"**Último dato**  \n{latest_date:%d-%m-%Y}")

    with forecast_tab:
        st.markdown('<div class="section-title">Proyección día por día</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Los rezagos, medias móviles y medias exponenciales se recalculan para cada paso futuro.</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=forecast["fecha"], y=forecast["prediccion"],
            marker_color=["#157a62", "#bc8c37", "#dc7057"],
            text=forecast["prediccion"].map(lambda value: f"{value:.1f}"), textposition="outside",
            hovertemplate="%{x|%d-%m-%Y}<br>Esperado: %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(title="Conteo esperado de sismos", yaxis_title="Sismos", showlegend=False)
        st.plotly_chart(_style_figure(fig, height=420), use_container_width=True)
        forecast_table = forecast.copy()
        forecast_table["fecha"] = forecast_table["fecha"].dt.strftime("%d-%m-%Y")
        forecast_table = forecast_table.rename(columns={
            "fecha": "fecha", "prediccion": "conteo_esperado", "prediccion_redondeada": "conteo_aproximado"
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar pronóstico CSV",
            data=forecast_table.to_csv(index=False).encode("utf-8"),
            file_name="pronostico_sismos_3_dias.csv",
            mime="text/csv",
        )

    with performance_tab:
        st.markdown('<div class="section-title">Evaluación sobre la validación reservada</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">La misma partición usada para seleccionar el ganador; no representa una prueba independiente.</div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Pinball Loss", f"{metrics['pinball']:.4f}", f"α = {bundle.get('alpha', 0.8):.1f}")
        metric_cols[1].metric("MAE", f"{metrics['mae']:.2f}", "sismos")
        metric_cols[2].metric("RMSE", f"{metrics['rmse']:.2f}", "sismos")
        metric_cols[3].metric("Filas evaluadas", f"{len(validation_scores)}", model_label)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=validation_scores["fecha"], y=validation_scores["real"], name="Real", mode="lines+markers", line=dict(color="#157a62", width=2)))
        fig.add_trace(go.Scatter(x=validation_scores["fecha"], y=validation_scores["prediccion"], name="Predicción", mode="lines+markers", line=dict(color="#dc7057", width=2, dash="dash")))
        fig.update_layout(title="Real vs. predicción · validación", yaxis_title="Sismos diarios")
        st.plotly_chart(_style_figure(fig, height=430), use_container_width=True)
        errors = validation_scores[["fecha", "error"]].copy()
        errors["fecha"] = errors["fecha"].dt.strftime("%d-%m-%Y")
        st.dataframe(errors.rename(columns={"fecha": "Fecha", "error": "Error (real - predicción)"}), use_container_width=True, hide_index=True)

    with interpretation_tab:
        _render_interpretability(bundle, validation)

    st.caption(f"Histórico: {len(history):,} días · última actualización de artefactos {pd.Timestamp(PATHS['processed_excel_reshape'].stat().st_mtime, unit='s'):%d-%m-%Y %H:%M}")


if __name__ == "__main__":
    main()
