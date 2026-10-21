"""Tests de regresión: la réplica debe reproducir los resultados del TFM."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfm import config, data, features, models  # noqa: E402
from tfm.predict import predict_dataframe, predict_price  # noqa: E402


@pytest.fixture(scope="module")
def fe():
    return features.feature_engineering(data.procesamiento_datos(data.load_raw()))


@pytest.fixture(scope="module")
def trained(fe):
    return models.train_final(data.filter_scope(fe))


def test_merge_shape():
    assert data.load_raw().shape == (14993, 19)


def test_processing_shape(fe):
    assert fe.shape == (14936, 27)
    assert len(data.filter_scope(fe)) == 10766


def test_floor_and_flags_clean(fe):
    assert fe["floor"].dtype.kind == "i"
    for c in ["exterior", "hasLift", "hasParkingSpace", "isParkingSpaceIncludedInPrice"]:
        assert set(fe[c].unique()) <= {0, 1}
    assert (fe["bathrooms"] > 0).all()
    assert fe["neighborhood"].notna().all()


def test_final_model_matches_tfm(trained):
    _, _, _, scores, preds = trained
    # TFM: R2 0.9985, MAPE 0.013
    assert scores["R2"] >= 0.998
    assert scores["MAPE"] <= 0.015
    assert len(preds) == 2154


def test_business_case_1(trained):
    preds = trained[4]
    row = preds[preds[config.ID_COL] == 104475383].iloc[0]
    assert row["price"] == 210000
    assert row["Difference"] < -50000  # infravalorada según el modelo


def test_predict_price_and_batch(trained):
    model = trained[1]
    p = predict_price("Almagro", 7335, 63, 2, model=model)
    assert 300_000 < p < 600_000
    raw = pd.read_csv(config.RAW_DIR / "idealista_05.csv", nrows=50, low_memory=False)
    out = predict_dataframe(raw, model=model, raw=True)
    assert "Predicted_Price" in out and out["Predicted_Price"].notna().all()
