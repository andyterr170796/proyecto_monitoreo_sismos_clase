import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.config.config import (
    DB_PARQUET,
    LIMITE_POR_PAGINA,
    MAGNITUD_MINIMA,
    RAW_DATA_DIR,
    setup_logging,
)

URL_USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def configurar_logs():
    setup_logging()
    return logging.getLogger("get_data"), logging.getLogger("max_date")

def obtener_sismos(fecha_inicio, fecha_fin):
    """Obtiene todos los eventos de USGS entre las fechas indicadas."""
    sismos = []
    offset = 1

    while True:
        parametros = {
            "format": "geojson",
            "starttime": fecha_inicio.isoformat(),
            "endtime": fecha_fin.isoformat(),
            "minmagnitude": MAGNITUD_MINIMA,
            "limit": LIMITE_POR_PAGINA,
            "offset": offset,
        }

        response = requests.get(URL_USGS, params=parametros, timeout=60)
        response.raise_for_status()
        pagina = response.json().get("features", [])
        sismos.extend(pagina)

        if len(pagina) < LIMITE_POR_PAGINA:
            break

        offset += LIMITE_POR_PAGINA

    return sismos


def convertir_a_filas(sismos):
    """Convierte features GeoJSON en filas con el formato de la tabla final."""
    filas = []
    for sismo in sismos:
        propiedades = sismo.get("properties", {})
        coordenadas = sismo.get("geometry", {}).get("coordinates", [])
        longitud = coordenadas[0] if len(coordenadas) > 0 else None
        latitud = coordenadas[1] if len(coordenadas) > 1 else None
        profundidad = coordenadas[2] if len(coordenadas) > 2 else None

        identificador = sismo.get("id") or "|".join(
            str(valor)
            for valor in (
                propiedades.get("time"),
                latitud,
                longitud,
                profundidad,
                propiedades.get("mag"),
            )
        )

        filas.append({
            "_event_key": identificador,
            "id": sismo.get("id"),
            "mag": propiedades.get("mag"),
            "place": propiedades.get("place"),
            "time": propiedades.get("time"),
            "updated": propiedades.get("updated"),
            "url": propiedades.get("url"),
            "latitude": latitud,
            "longitude": longitud,
            "depth_km": profundidad,
            "status": propiedades.get("status"),
            "tsunami": propiedades.get("tsunami"),
            "type": propiedades.get("type"),
        })

    return pd.DataFrame(filas)


def deduplicar(df):
    """Conserva una sola fila por evento y no expone la clave técnica."""
    if df.empty:
        return df.drop(columns=["_event_key"], errors="ignore")

    df["_event_key"] = df["id"]
    sin_clave = df["_event_key"].isna()
    if sin_clave.any():
        columnas_clave = [
            "time",
            "latitude",
            "longitude",
            "depth_km",
            "mag",
        ]
        df.loc[sin_clave, "_event_key"] = (
            df.loc[sin_clave, columnas_clave]
            .astype("string")
            .fillna("")
            .agg("|".join, axis=1)
        )

    cantidad_antes = len(df)
    df = df.drop_duplicates(subset="_event_key", keep="first")
    print(f"Se eliminaron {cantidad_antes - len(df)} registros duplicados.")
    return df.drop(columns="_event_key")


def carga_batch_sismos():
    logger_detalle, logger_fecha_maxima = configurar_logs()
    inicio_ejecucion = datetime.now()
    logger_detalle.info("Inicio de ejecución: %s", inicio_ejecucion.isoformat())
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ahora = datetime.now()

    if DB_PARQUET.exists():
        fecha_inicio = ahora - timedelta(days=7)
        tipo_carga = "incremental"
    else:
        fecha_inicio = ahora - timedelta(days=365 * 2)
        tipo_carga = "inicial de 2 años"

    fecha_fin = ahora
    print(
        f"Iniciando carga {tipo_carga} desde "
        f"{fecha_inicio:%Y-%m-%d %H:%M:%S} hasta {fecha_fin:%Y-%m-%d %H:%M:%S}..."
    )
    logger_detalle.info(
        "Tipo de carga: %s | Ventana: %s a %s",
        tipo_carga,
        fecha_inicio.isoformat(),
        fecha_fin.isoformat(),
    )

    try:
        sismos = obtener_sismos(fecha_inicio, fecha_fin)
        datos_nuevos = convertir_a_filas(sismos)
        logger_detalle.info("Registros extraídos desde USGS: %s", len(datos_nuevos))
        print(f"Se extrajeron {len(datos_nuevos)} registros desde USGS.")

        if DB_PARQUET.exists():
            datos_existentes = pd.read_parquet(DB_PARQUET)
            cantidad_existente = len(datos_existentes)
            datos = pd.concat([datos_existentes, datos_nuevos], ignore_index=True)
        else:
            cantidad_existente = 0
            datos_existentes = pd.DataFrame()
            datos = datos_nuevos

        datos = deduplicar(datos)
        datos.to_parquet(DB_PARQUET, index=False)

        cantidad_insertada = max(len(datos) - cantidad_existente, 0)
        cantidad_descartada = cantidad_existente + len(datos_nuevos) - len(datos)
        fecha_maxima = pd.to_datetime(datos["time"], unit="ms", utc=True).max()

        logger_detalle.info(
            "Registros insertados: %s | Registros descartados: %s | Total almacenado: %s",
            cantidad_insertada,
            cantidad_descartada,
            len(datos),
        )
        logger_detalle.info("Fin de ejecución: %s", datetime.now().isoformat())
        logger_fecha_maxima.info(
            "ejecucion=%s | fecha_maxima_datos=%s | total_datos=%s",
            inicio_ejecucion.isoformat(),
            fecha_maxima.isoformat(),
            len(datos),
        )

        print(f"Base Parquet actualizada: {DB_PARQUET}")
        print(f"Total de registros únicos almacenados: {len(datos)}")
        return datos
    except Exception:
        logger_detalle.exception("La ejecución terminó con error")
        raise


if __name__ == "__main__":
    carga_batch_sismos()
