"""Carga, fusión y limpieza de los datos de Idealista.

Réplica de las funciones `merge_and_clean_data` y `procesamiento_datos`
del notebook final del TFM (Notebook_11), con rutas relativas y sin
dependencia de Google Colab.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# 1. Fusión de los CSV mensuales
# ---------------------------------------------------------------------------
def merge_and_clean_data(
    files: Iterable[str | Path],
    months: Iterable[str],
    column_order: list[str] = config.COLUMN_ORDER,
) -> pd.DataFrame:
    """Une varios CSV de Idealista, añade la columna 'Month' y elimina
    duplicados por 'propertyCode' (se queda con la última aparición)."""
    dfs = []
    for file, month in zip(files, months):
        df = pd.read_csv(file, low_memory=False)
        df["Month"] = month
        missing = [c for c in column_order if c not in df.columns]
        if missing:
            raise ValueError(f"{Path(file).name}: faltan columnas {missing}")
        dfs.append(df[column_order])

    merged = pd.concat(dfs, ignore_index=True)
    return merged.drop_duplicates(subset=config.ID_COL, keep="last")


def load_raw(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Carga los cuatro CSV originales del TFM (feb-may 2024)."""
    files = [raw_dir / f for f in config.RAW_FILES]
    return merge_and_clean_data(files, list(config.RAW_FILES.values()))


# ---------------------------------------------------------------------------
# 2. Procesamiento (limpieza de columnas)
# ---------------------------------------------------------------------------
def _to_flag(series: pd.Series) -> pd.Series:
    """True/'True'/1 -> 1 ; False/'False'/NaN -> 0 (equivale al
    fillna('False') + replace + astype(int) del TFM)."""
    return series.map(lambda v: 1 if v is True or str(v).strip().lower() in {"true", "1", "1.0"} else 0).astype(int)


def _process_floor(df: pd.DataFrame) -> pd.DataFrame:
    floor = df["floor"].astype("object")
    floor = floor.replace({"bj": "0", "en": "0", "ss": "-1", "60": "6"})
    # 'st' (sobretecho/entreplanta) se elimina: no se conoce la planta real
    df = df.loc[floor != "st"].copy()
    df["floor"] = pd.to_numeric(floor.loc[df.index], errors="coerce")

    median_floors = df.groupby("propertyType")["floor"].median()

    def fill_floor(row):
        if pd.isna(row["floor"]):
            return 0 if row["propertyType"] == "chalet" else median_floors[row["propertyType"]]
        return row["floor"]

    df["floor"] = df.apply(fill_floor, axis=1).astype(int)
    return df


def _parse_parking(value) -> pd.Series:
    has_space, included, price = False, False, 0
    if not (isinstance(value, float) and np.isnan(value)) and value is not None:
        try:
            d = value if isinstance(value, dict) else ast.literal_eval(str(value))
            has_space = d.get("hasParkingSpace", False)
            included = d.get("isParkingSpaceIncludedInPrice", False)
            price = d.get("parkingSpacePrice", 0)
        except (SyntaxError, ValueError):
            pass
    return pd.Series(
        {"hasParkingSpace": has_space, "isParkingSpaceIncludedInPrice": included, "parkingSpacePrice": price}
    )


def procesamiento_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza del TFM (sección 5.2):
    floor -> numérico (bj/en=0, ss=-1, se quitan 'st', NaN = mediana por tipo),
    exterior / hasLift -> 0/1, se quitan barrios nulos y viviendas con 0 baños,
    parkingSpace -> hasParkingSpace, isParkingSpaceIncludedInPrice, parkingSpacePrice.
    """
    df = df.copy()
    df = _process_floor(df)
    df["exterior"] = _to_flag(df["exterior"])
    df = df[df["neighborhood"].notna()]
    df = df[df["bathrooms"] != 0]
    df["hasLift"] = _to_flag(df["hasLift"])

    parking = df["parkingSpace"].apply(_parse_parking)
    df = pd.concat([df.drop(columns=["parkingSpace"]), parking], axis=1)
    df["hasParkingSpace"] = _to_flag(df["hasParkingSpace"])
    df["isParkingSpaceIncludedInPrice"] = _to_flag(df["isParkingSpaceIncludedInPrice"])
    df["parkingSpacePrice"] = pd.to_numeric(df["parkingSpacePrice"], errors="coerce").fillna(0)
    return df


def filter_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Alcance del estudio: provincia de Madrid y precio <= 800.000 €."""
    return df[(df["province"] == config.PROVINCE) & (df[config.TARGET] <= config.MAX_PRICE)]
