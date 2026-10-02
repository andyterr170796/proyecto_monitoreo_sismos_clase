# Monitoreo de sismos

Pipeline en Python que obtiene eventos sismicos de USGS, los transforma en una serie diaria y compara estimadores para predecir la cantidad de sismos. Cada candidato integra el escalado y el estimador en un `sklearn.Pipeline`; se selecciona el mejor por Pinball Loss y se reutiliza para validacion, produccion y pronostico.

## Instalacion

Desde la raiz del proyecto:

```powershell
python -m pip install -r requirements.txt
```

## Ejecucion

```powershell
python -m src.main_pipeline
```

La ejecucion requiere conexion a Internet para consultar USGS. Los resultados se guardan en `data/processed/` y los registros de ejecucion en `logs/general.log`.

## Plataforma Streamlit

Instala las dependencias del proyecto y ejecuta desde la raiz:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La plataforma presenta el histórico diario, el pronóstico recursivo de los próximos 3 días, métricas sobre la validación reservada e interpretabilidad del modelo ganador. Requiere los artefactos generados por `python -m src.main_pipeline`.

## Pruebas

Desde la raíz del proyecto:

```powershell
python -m unittest discover -s tests -v
```

La prueba de calidad comprueba que el Excel procesado no tenga valores faltantes en `mag`, `depth_km`, `latitude` ni `longitude`. Omite esa comprobación si todavía no existe el archivo procesado.

## Promoción y versiones del modelo

Cada ejecución guarda un bundle versionado en `models/versions/winner_model_<fecha_hora>.pkl`. `models/winner_model.pkl` solo se reemplaza cuando el candidato reduce la Pinball Loss frente al ganador actual. Un valor menor es mejor.

El historial acumulativo de evaluación se guarda en `outputs/model_metrics.xlsx`, con una fila por cada modelo ejecutado y fecha/hora, Pinball Loss, MAE, MSE, RMSE, R², selección de la ejecución y estado de promoción.

## Docker

Requiere Docker con Compose. Construye la imagen y ejecuta el pipeline una vez:

```powershell
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml --profile pipeline run --rm pipeline
```

Después de la primera ejecución del pipeline, inicia la aplicación y el scheduler diario en segundo plano. El scheduler actualiza datos, entrenamiento y artefactos todos los días a las 08:00, hora de Perú (`America/Lima`):

```powershell
docker compose -f docker/compose.yaml up -d app scheduler
```

La aplicación queda disponible en `http://localhost:8501`. El scheduler permanece activo y reinicia automáticamente con Docker.

Los directorios `data/`, `models/`, `outputs/` y `logs/` se montan desde el proyecto para conservar los datos y artefactos fuera del contenedor. Para correr las pruebas en Docker:

```powershell
docker compose -f docker/compose.yaml --profile pipeline run --rm pipeline python -m unittest discover -s tests -v
```
