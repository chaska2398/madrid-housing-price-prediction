"""Predicción con el modelo entrenado (Business Cases del TFM)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .data import procesamiento_datos
from .features import feature_engineering


@lru_cache(maxsize=2)
def load_model(path: str | Path = config.MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python scripts/run_pipeline.py")
    return joblib.load(path)


def predict_price(neighborhood: str, priceByArea: float, size: float, bathrooms: int,
                  bathroomSize: float | None = None, model=None) -> float:
    """Business Case 2: estimar el valor de un inmueble (misma firma que el TFM)."""
    model = model or load_model()
    if bathroomSize is None:
        bathroomSize = size / bathrooms
    row = pd.DataFrame({"priceByArea": [priceByArea], "size": [size], "bathrooms": [bathrooms],
                        "bathroomSize": [bathroomSize], "neighborhood": [neighborhood]})
    return float(model.predict(row[config.FINAL_FEATURES])[0])


def predict_dataframe(df: pd.DataFrame, model=None, raw: bool = False) -> pd.DataFrame:
    """Predice un lote de inmuebles.

    raw=True  -> df tiene el formato de la API de Idealista (se limpia y se
                 generan variables con el mismo pipeline del TFM).
    raw=False -> df ya contiene las columnas del modelo (bathroomSize se
                 calcula si falta).
    """
    model = model or load_model()
    df = df.copy()
    if raw:
        cols = [c for c in config.COLUMN_ORDER if c != "Month"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas de Idealista: {missing}")
        df = df[cols]
        df["Month"] = "Nuevo"
        df = feature_engineering(procesamiento_datos(df.drop_duplicates(config.ID_COL, keep="last")))
    elif "bathroomSize" not in df.columns:
        df["bathroomSize"] = df["size"] / df["bathrooms"]

    missing = [c for c in config.FINAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para el modelo: {missing}")
    df["Predicted_Price"] = model.predict(df[config.FINAL_FEATURES])
    if config.TARGET in df.columns:
        df["Difference"] = df[config.TARGET] - df["Predicted_Price"]
    return df
