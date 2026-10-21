"""Ejecuta el pipeline completo del TFM de principio a fin.

    python scripts/run_pipeline.py              # hiperparámetros del TFM (≈1 min)
    python scripts/run_pipeline.py --search     # repite RandomizedSearchCV (lento)
    python scripts/run_pipeline.py --skip-benchmark

Genera:
    data/processed/datos_procesados_final.csv
    models/gbr_tfm.joblib
    reports/metrics.json, reports/predicciones_test.csv, reports/figures/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tfm import config, data, features, models  # noqa: E402


def plot_real_vs_pred(preds, path: Path) -> None:
    diff = preds["Difference"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(preds["price"], preds["Predicted_Price"], alpha=0.4, s=10)
    lims = [preds["price"].min(), preds["price"].max()]
    ax[0].plot(lims, lims, "r--", label="Ajuste ideal")
    ax[0].set(title="Precio real vs predicho", xlabel="Real (€)", ylabel="Predicho (€)")
    ax[0].legend()
    ax[1].hist(diff, bins=30, color="gray", edgecolor="black")
    m, s = diff.mean(), diff.std()
    ax[1].axvline(m, color="red", ls="--", label=f"Media: {m:,.0f}")
    ax[1].axvline(m + s, color="blue", ls="--", label=f"+1σ: {m + s:,.0f}")
    ax[1].axvline(m - s, color="blue", ls="--", label=f"-1σ: {m - s:,.0f}")
    ax[1].set(title="Histograma de diferencias (real - predicho)", xlabel="€")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="repetir la búsqueda de hiperparámetros")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("1/4 Cargando y fusionando CSV de Idealista...")
    merged = data.load_raw()
    print(f"    {merged.shape[0]:,} ofertas únicas, {merged.shape[1]} columnas")

    print("2/4 Procesamiento y feature engineering...")
    processed = data.procesamiento_datos(merged)
    fe = features.feature_engineering(processed)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fe.to_csv(config.PROCESSED_CSV, index=False)
    scoped = data.filter_scope(fe)
    print(f"    {len(fe):,} filas procesadas -> {len(scoped):,} en alcance (Madrid, <= 800k €)")

    report = {"n_rows": {"merged": len(merged), "processed": len(fe), "in_scope": len(scoped)}}

    if not args.skip_benchmark:
        print("3/4 Benchmark de modelos (Tabla 6 del TFM)...")
        report["benchmark"] = {
            "1_datos_sin_procesar_sin_nan": models.benchmark(data.filter_scope(merged), dropna=True),
            "2_datos_procesados": models.benchmark(data.filter_scope(processed)),
            "3_procesados_y_feature_engineering": models.benchmark(scoped),
        }
        for k, v in report["benchmark"].items():
            print(f"    {k}:")
            for name, m in v["results"].items():
                print(f"      {name:<28} R2={m['R2']:.4f}  MAPE={m['MAPE']:.4f}  MAE={m['MAE']:,.0f}")
    else:
        print("3/4 Benchmark omitido")

    print("4/4 Modelo final GradientBoostingRegressor...")
    _, model_prod, params, scores, preds = models.train_final(scoped, search=args.search)
    report["final_model"] = {"params": params, "features": config.FINAL_FEATURES, **scores}
    print(f"    R2={scores['R2']}  MAPE={scores['MAPE']}  RMSE={scores['RMSE']:,}  MAE={scores['MAE']:,}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(model_prod, config.MODEL_PATH)
    preds.to_csv(config.PREDICTIONS_PATH, index=False)
    features.neighborhood_reference(scoped).to_csv(config.PROCESSED_DIR / "referencia_barrios.csv", index=False)
    plot_real_vs_pred(preds, config.REPORTS_DIR / "figures" / "real_vs_predicho.png")
    config.METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))

    print(f"\nListo en {time.time() - t0:.0f}s. Modelo: {config.MODEL_PATH.relative_to(config.ROOT)}")
    print("Lanza la app con:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
