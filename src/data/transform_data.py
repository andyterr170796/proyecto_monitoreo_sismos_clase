import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.config import (
	PATHS,
	TEST_SIZE,
	setup_logging,
)

def transformar_datos():
	setup_logging()
	logger = logging.getLogger("transform_data")

	df = pd.read_parquet(PATHS["db_parquet"])
	registros_iniciales = len(df)

	for columna in ("time", "updated"):
		df[columna] = pd.to_datetime(
			df[columna],
			unit="ms",
			errors="coerce",
			utc=True,
		).dt.tz_localize(None)

	PATHS["processed_data_dir"].mkdir(parents=True, exist_ok=True)

	logger.info("Registros leídos: %s", registros_iniciales)
	print(f"Registros leídos: {registros_iniciales}")
	logger.info("Registros sin magnitud descartados: %s", registros_iniciales - len(df))
	print(f"Registros sin magnitud descartados: {registros_iniciales - len(df)}")
	logger.info("Registros procesados: %s", len(df))
	print(f"Registros procesados: {len(df)}")
	logger.info("Excel generado: %s", PATHS["processed_excel"])
	print(f"Excel generado: {PATHS['processed_excel']}")
	logger.info("min date - %s | max date - %s", df['time'].min(), df['time'].max())
	print(f"min date - {df['time'].min()} | max date - {df['time'].max()}")
	for columna in ('mag', 'depth_km', 'latitude', 'longitude', 'tsunami'):
		logger.info("missing values in %s: %s", columna, df[columna].isna().sum())
		print(f"missing values in {columna}: {df[columna].isna().sum()}")
		df = df.dropna(subset=[columna]).copy()
	duplicados = df.duplicated(subset=['id']).sum()
	logger.info("Verificar duplicados en 'id': %s", duplicados)
	print("Verificar duplicados en 'id':", duplicados)
	df = df.drop_duplicates(subset=['id'], keep='first')
	df.to_excel(PATHS["processed_excel"], index=False, sheet_name="sismos")
	logger.info("Transformación finalizada correctamente")
	df = (
		df.assign(fecha=df['time'].dt.normalize())
		.groupby('fecha', as_index=False)
		.size()
		.rename(columns={'size': 'sismos'})
	)
	df["mes"] = df["fecha"].dt.month
	df["anio"] = df["fecha"].dt.year

	for lag in range(1, 13):
		df[f"L{lag}_sismos"] = df["sismos"].shift(lag)

	df["MA_sismos"] = df["sismos"].shift(1).rolling(window=3).mean()
	df["MA2_sismos"] = df["sismos"].shift(1).rolling(window=5).mean()
	df["EMA_sismos"] = df["sismos"].shift(1).ewm(span=3, adjust=False).mean()
	df["EMA2_sismos"] = df["sismos"].shift(1).ewm(span=5, adjust=False).mean()
	df.dropna(inplace=True)
	logger.info("Reshape a día finalizado correctamente")
	df.to_excel(PATHS["processed_excel_reshape"], index=False, sheet_name="sismos")
	_, df_test = train_test_split(df, test_size=TEST_SIZE, shuffle=False)
	df_test.to_excel(PATHS["processed_excel_reshape_test"], index=False, sheet_name="sismos")
	logger.info("Conjunto de prueba guardado en: %s", PATHS["processed_excel_reshape_test"])
	return df

if __name__ == "__main__":
	transformar_datos()
