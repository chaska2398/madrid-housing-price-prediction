"""Modelos del TFM: benchmark (sección 6.5-6.6) y modelo final (6.7-6.8)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config

warnings.filterwarnings("ignore", category=FutureWarning)


def metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
    }


def split(data: pd.DataFrame, features: list[str] | None = None):
    """train/test 80/20 con random_state=42, igual que en el TFM."""
    X = data.drop(columns=config.DROP_COLS) if features is None else data[features]
    y = data[config.TARGET]
    return train_test_split(X, y, train_size=1 - config.TEST_SIZE, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE)


def make_pipeline(regressor, numeric, categorical, smoothing: float | None = 0.5) -> Pipeline:
    enc = TargetEncoder(cols=categorical, smoothing=smoothing) if smoothing is not None else TargetEncoder()
    pre = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", enc, categorical)])
    return Pipeline([("preprocessor", pre), ("regressor", regressor)])


BENCHMARK_MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoostingRegressor": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBRegressor": lambda: XGBRegressor(n_estimators=100, random_state=42),
}


def benchmark(data: pd.DataFrame, dropna: bool = False) -> dict:
    """Entrena los 4 modelos con todas las variables disponibles."""
    if dropna:
        data = data.dropna()
    X_train, X_test, y_train, y_test = split(data)
    numeric = list(X_train.select_dtypes(exclude=["object", "string"]).columns)
    categorical = list(X_train.select_dtypes(include=["object", "string"]).columns)
    X_train, X_test = X_train.copy(), X_test.copy()
    for c in categorical:
        X_train[c] = X_train[c].astype("object")
        X_test[c] = X_test[c].astype("object")

    results = {}
    for name, factory in BENCHMARK_MODELS.items():
        model = make_pipeline(factory(), numeric, categorical).fit(X_train, y_train)
        results[name] = metrics(y_test, model.predict(X_test))
    return {"n_train": len(X_train), "numeric": numeric, "categorical": categorical, "results": results}


# ---------------------------------------------------------------------------
# Modelo final: GradientBoostingRegressor con variables importantes
# ---------------------------------------------------------------------------
GBR_SEARCH_SPACE = {
    "regressor__n_estimators": randint(50, 300),
    "regressor__max_depth": randint(3, 21),
    "regressor__learning_rate": uniform(0.01, 0.2),
    "regressor__subsample": uniform(0.5, 0.5),
    "regressor__min_samples_split": randint(2, 20),
    "regressor__min_samples_leaf": randint(1, 20),
}


def final_pipeline(params: dict | None = None) -> Pipeline:
    gbr = GradientBoostingRegressor(random_state=config.RANDOM_STATE, **(params or {}))
    return make_pipeline(gbr, config.FINAL_NUMERIC, config.FINAL_CATEGORICAL, smoothing=None)


def train_final(data: pd.DataFrame, search: bool = False, n_iter: int = 50):
    """Entrena el modelo final del TFM.

    search=False usa los hiperparámetros publicados en el TFM (rápido).
    search=True repite la RandomizedSearchCV original (n_iter=50, cv=5).
    Devuelve (modelo_eval, modelo_produccion, params, métricas, predicciones_test).
    """
    X_train, X_test, y_train, y_test = split(data, config.FINAL_FEATURES)

    if search:
        rs = RandomizedSearchCV(final_pipeline(), GBR_SEARCH_SPACE, n_iter=n_iter, cv=5,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=config.RANDOM_STATE)
        rs.fit(X_train, y_train)
        params = {k.replace("regressor__", ""): v for k, v in rs.best_params_.items()}
    else:
        params = dict(config.FINAL_GBR_PARAMS)

    model_eval = final_pipeline(params).fit(X_train, y_train)
    y_pred = model_eval.predict(X_test)
    scores = metrics(y_test, y_pred)

    preds = data.loc[X_test.index, [config.ID_COL, "district", "neighborhood", "propertyType",
                                    "size", "rooms", "bathrooms", "priceByArea", "latitude",
                                    "longitude", config.TARGET]].copy()
    preds["Predicted_Price"] = y_pred
    # Igual que en el TFM: Difference = precio real - precio predicho
    # (negativo => la oferta está por debajo del valor estimado => infravalorada)
    preds["Difference"] = preds[config.TARGET] - preds["Predicted_Price"]

    # Como en el Business Case 2 del TFM, se reentrena con todos los datos
    model_prod = final_pipeline(params).fit(data[config.FINAL_FEATURES], data[config.TARGET])

    importances = dict(zip(config.FINAL_FEATURES,
                           model_eval.named_steps["regressor"].feature_importances_.round(4).tolist()))
    return model_eval, model_prod, params, {**scores, "feature_importance": importances}, preds
