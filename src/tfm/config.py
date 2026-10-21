"""Configuración central del proyecto: rutas, columnas y parámetros del TFM."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_PATH = DATA_DIR / "geo" / "madrid_districts.geojson"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

PROCESSED_CSV = PROCESSED_DIR / "datos_procesados_final.csv"
MODEL_PATH = MODELS_DIR / "gbr_tfm.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "predicciones_test.csv"

# Ficheros descargados de la API de Idealista (feb-may 2024) y mes asociado
RAW_FILES = {
    "idealista_02.csv": "Febrero",
    "idealista_03.csv": "Marzo",
    "idealista_04.csv": "Abril",
    "idealista_05.csv": "Mayo",
}

# Columnas que se conservan al fusionar (mismo orden que en el TFM)
COLUMN_ORDER = [
    "propertyCode", "price", "floor", "propertyType", "size", "exterior", "rooms",
    "bathrooms", "province", "district", "neighborhood", "latitude", "longitude",
    "distance", "status", "hasLift", "priceByArea", "parkingSpace", "Month",
]

# Filtro de alcance del estudio
PROVINCE = "Madrid"
MAX_PRICE = 800_000

TARGET = "price"
ID_COL = "propertyCode"
DROP_COLS = [ID_COL, TARGET, "Month", "province"]

# Variables del modelo final (sección 6.8 del TFM)
FINAL_NUMERIC = ["priceByArea", "size", "bathrooms", "bathroomSize"]
FINAL_CATEGORICAL = ["neighborhood"]
FINAL_FEATURES = FINAL_NUMERIC + FINAL_CATEGORICAL

# Mejores hiperparámetros del GradientBoostingRegressor obtenidos en el TFM
# (RandomizedSearchCV, n_iter=50, cv=5, random_state=42)
FINAL_GBR_PARAMS = {
    "learning_rate": 0.11681788387508839,
    "max_depth": 7,
    "min_samples_leaf": 1,
    "min_samples_split": 11,
    "n_estimators": 221,
    "subsample": 0.6220627611238871,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
