import logging
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import PATHS, NUMERIC_FEATURES, TARGET, setup_logging
from src.data.get_data import carga_batch_sismos
from src.data.transform_data import transformar_datos

from src.features.build_features import split_features_target
from src.models.train_model import train_models, save_model
from src.models.predict_model import predicciones_en_produccion

def setup_logger():
    setup_logging()

def run_pipeline():
    """Orquestador maestro: une y ejecuta todos los bloques evitando Spaghetti code (Slide 41)."""
    setup_logger()
    logging.getLogger(__name__)
    logging.info("Iniciando Pipeline SISMOS...")
    df_load = carga_batch_sismos()
    df_clean = transformar_datos()
    df_validation = pd.read_excel(PATHS["processed_excel_reshape_test"])
    df_train = df_clean.iloc[:-len(df_validation)]
    logging.info(
        "Train: %s filas | Validation: %s filas",
        len(df_train),
        len(df_validation),
    )
    X_train, y_train = split_features_target(df_train, NUMERIC_FEATURES, TARGET)
    X_validation, y_validation = split_features_target(
        df_validation,
        NUMERIC_FEATURES,
        TARGET,
    )
    model_lr = train_models(X_train, y_train, X_validation, y_validation)
    save_model(model_lr, PATHS["model_store"])
    predicciones_en_produccion(model_lr)

if __name__ == '__main__':
    run_pipeline()
