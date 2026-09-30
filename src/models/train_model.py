import logging
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.config.config import setup_logging

def train_linear_regression(X, y):
    """Entrenamiento de modelo con control de logs."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando entrenamiento del modelo - Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
    logger.info("Train: X=%s, y=%s | Test: X=%s, y=%s", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = LinearRegression()
    logger.info("Iniciando entrenamiento del modelo - Fit...")
    model.fit(X_train, y_train)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    logger.info("Modelo entrenado. Coeficientes: %s, R^2 Train: %s, R^2 Test: %s", model.coef_, r2_train, r2_test)
    print(f"Modelo entrenado. Coeficientes: {model.coef_}, R^2 Train: {r2_train}, R^2 Test: {r2_test}")
    return model

def save_model(model, filepath):
    """Serializa y exporta el archivo de modelo (reproducibilidad)."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Guardando modelo...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Modelo guardado en: %s", filepath)
    print(f"Modelo guardado en: {filepath}")
