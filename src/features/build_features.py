import logging
from src.config.config import setup_logging

def split_features_target(df, feature_cols, target_col):
    """Separa el dataset en matriz de predictoras (X) y vector objetivo (y)."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando separación de features y target...")
    try:
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        logger.info("Features: %s, Target: %s", X.shape, y.shape if y is not None else "N/A")
        return X, y
    except KeyError as e:
        logger.exception("Error al separar features y target: %s", e)
        return None, None