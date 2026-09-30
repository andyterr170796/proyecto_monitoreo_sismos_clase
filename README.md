# Monitoreo de sismos

Pipeline en Python que obtiene eventos sismicos de USGS, los transforma en una serie diaria y entrena un modelo de regresion lineal para estimar la cantidad de sismos.

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
