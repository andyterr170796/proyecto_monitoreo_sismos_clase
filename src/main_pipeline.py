import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import MODEL_STORE, NUMERIC_FEATURES, TARGET, setup_logging
from src.data.get_data import carga_batch_sismos
from src.data.transform_data import transformar_datos

from src.features.build_features import split_features_target
from src.models.train_model import train_linear_regression, save_model
from src.models.predict_model import predicciones_en_produccion

def setup_logger():
    setup_logging()

def run_pipeline():
    """Orquestador maestro: une y ejecuta todos los bloques evitando Spaghetti code (Slide 41)."""
    setup_logger()
    logging.getLogger(__name__)
    logging.info("Iniciando Pipeline BIKES...")
    df_load = carga_batch_sismos()
    df_clean = transformar_datos()
    X, y = split_features_target(df_clean, NUMERIC_FEATURES, TARGET)
    model_lr = train_linear_regression(X, y)
    save_model(model_lr, MODEL_STORE)
    predicciones_en_produccion(model_lr)

if __name__ == '__main__':
    run_pipeline()
