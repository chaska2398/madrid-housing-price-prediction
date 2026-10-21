"""Feature engineering del TFM (sección 6.2)."""
from __future__ import annotations

import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Añade:
    - priceByAreaByDistrict / priceByAreaByneighborhood: media de €/m² por zona.
    - roomSize = size / rooms, bathroomSize = size / bathrooms,
      bathroomRoom = bathrooms / rooms, liftFloorInteraction = hasLift * floor.
    """
    df = df.copy()
    df["priceByAreaByDistrict"] = df["district"].map(df.groupby("district")["priceByArea"].mean())
    df["priceByAreaByneighborhood"] = df["neighborhood"].map(df.groupby("neighborhood")["priceByArea"].mean())

    df["roomSize"] = 0.0
    df["bathroomSize"] = 0.0
    df["bathroomRoom"] = 0.0
    rooms_nz = df["rooms"] != 0
    baths_nz = df["bathrooms"] != 0
    df.loc[rooms_nz, "roomSize"] = df.loc[rooms_nz, "size"] / df.loc[rooms_nz, "rooms"]
    df.loc[baths_nz, "bathroomSize"] = df.loc[baths_nz, "size"] / df.loc[baths_nz, "bathrooms"]
    df.loc[rooms_nz, "bathroomRoom"] = df.loc[rooms_nz, "bathrooms"] / df.loc[rooms_nz, "rooms"]
    df["liftFloorInteraction"] = df["hasLift"] * df["floor"]

    studio = (df["propertyType"] == "studio") & (df["rooms"] == 0)
    df.loc[studio, ["roomSize", "bathroomRoom"]] = 0
    return df


def neighborhood_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de referencia por barrio (para autocompletar entradas en la app)."""
    return (
        df.groupby(["district", "neighborhood"])
        .agg(priceByArea_media=("priceByArea", "mean"), ofertas=("price", "size"),
             precio_mediano=("price", "median"), size_mediano=("size", "median"))
        .reset_index()
        .sort_values(["district", "neighborhood"])
    )
