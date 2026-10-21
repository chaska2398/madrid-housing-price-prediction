"""App del TFM: Predicción de precios de vivienda en Madrid.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm import config, data, features, models  # noqa: E402
from tfm.predict import predict_dataframe, predict_price  # noqa: E402

# Paleta (dataviz reference palette): serie única azul, rampa secuencial azul
BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#8a8985"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Precios vivienda Madrid · TFM", page_icon="🏠", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos y modelo (se entrena automáticamente si no existe)
# ---------------------------------------------------------------------------
def _model_loads() -> bool:
    """El .joblib puede no cargar si la versión de scikit-learn es distinta."""
    import joblib
    try:
        joblib.load(config.MODEL_PATH)
        return True
    except Exception:  # noqa: BLE001
        return False


@st.cache_resource(show_spinner="Preparando datos y entrenando el modelo (solo la primera vez)...")
def load_assets():
    if (not config.PROCESSED_CSV.exists() or not config.MODEL_PATH.exists()
            or not config.PREDICTIONS_PATH.exists() or not _model_loads()):
        import joblib
        fe = features.feature_engineering(data.procesamiento_datos(data.load_raw()))
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        fe.to_csv(config.PROCESSED_CSV, index=False)
        scoped = data.filter_scope(fe)
        _, model_prod, _, _, preds = models.train_final(scoped)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_prod, config.MODEL_PATH)
        preds.to_csv(config.PREDICTIONS_PATH, index=False)
    import joblib
    model = joblib.load(config.MODEL_PATH)
    df = data.filter_scope(pd.read_csv(config.PROCESSED_CSV))
    preds = pd.read_csv(config.PREDICTIONS_PATH)
    metrics = json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {}
    geo = json.loads(config.GEO_PATH.read_text(encoding="utf-8"))
    return model, df, preds, metrics, geo


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    aliases = {"barrio de salamanca": "salamanca", "fuencarral": "fuencarral-el pardo",
               "moncloa": "moncloa-aravaca"}
    return aliases.get(s, s)


def district_map(geo: dict, agg: pd.DataFrame, col: str, label: str) -> go.Figure:
    """Coropleta dibujada como polígonos (funciona sin conexión a internet)."""
    from plotly.colors import sample_colorscale
    vals = agg.set_index("key")
    lo, hi = vals[col].min(), vals[col].max()
    fig = go.Figure()
    for f in geo["features"]:
        key = norm(f["properties"]["name"])
        if key not in vals.index:
            continue
        r = vals.loc[key]
        color = sample_colorscale(SEQ_BLUE, [(r[col] - lo) / (hi - lo) if hi > lo else 0.5])[0]
        polys = f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]]
        text = (f"<b>{r['Distrito']}</b><br>€/m² medio: {r['priceByArea']:,.0f}<br>"
                f"Ofertas: {int(r['ofertas'])}<br>Precio mediano: {r['precio_mediano']:,.0f} €").replace(",", ".")
        for poly in polys:
            ring = poly[0]
            fig.add_trace(go.Scatter(x=[p[0] for p in ring], y=[p[1] for p in ring], fill="toself", mode="lines",
                                     fillcolor=color, line=dict(color="white", width=1.5), hoveron="fills",
                                     hoverinfo="text", text=text, showlegend=False))
        xs = [p[0] for poly in polys for p in poly[0]]
        ys = [p[1] for poly in polys for p in poly[0]]
        fig.add_annotation(x=sum(xs) / len(xs), y=sum(ys) / len(ys), text=r["Distrito"], showarrow=False,
                           font=dict(size=10, color="#0b0b0b"), bgcolor="rgba(255,255,255,0.7)")
    # Barra de color
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=SEQ_BLUE, cmin=lo, cmax=hi, color=[lo], showscale=True,
                                         colorbar=dict(title=label, thickness=14))))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1.27)  # corrección de latitud (~40°N)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


model, df, preds, metrics, geo = load_assets()
ref = features.neighborhood_reference(df)

st.title("Predicción de precios de vivienda en Madrid")
st.caption("Réplica funcional del TFM · Máster en Data Science for Finance (EAE Business School, 2024) · "
           "Datos: API de Idealista, feb-may 2024 · Modelo: GradientBoostingRegressor + TargetEncoder")

tab_tas, tab_opp, tab_up, tab_map, tab_mod, tab_data = st.tabs(
    ["Tasador", "Oportunidades", "Cargar datos", "Mapa", "Modelo", "Datos"])

# ---------------------------------------------------------------------------
# 1. Tasador (Business Case 2)
# ---------------------------------------------------------------------------
with tab_tas:
    st.subheader("Business Case 2 · Estimación del valor de un inmueble")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        districts = sorted(ref["district"].unique())
        district = st.selectbox("Distrito", districts, index=districts.index("Chamberí") if "Chamberí" in districts else 0)
        sub = ref[ref["district"] == district]
        neighborhood = st.selectbox("Barrio", sub["neighborhood"].tolist())
        row = sub[sub["neighborhood"] == neighborhood].iloc[0]
        price_by_area = st.number_input(
            "Precio por m² (€/m²)", min_value=100, max_value=40000, step=50,
            value=int(round(row["priceByArea_media"])),
            help="Por defecto se rellena con la media del barrio en los datos del TFM. "
                 "El modelo del TFM usa esta variable como entrada.")
        size = st.number_input("Superficie (m²)", min_value=10, max_value=1000, value=63, step=1)
        bathrooms = st.number_input("Baños", min_value=1, max_value=10, value=2, step=1)
        st.caption(f"bathroomSize = superficie / baños = **{size / bathrooms:.1f} m²** (calculado)")

    with c2:
        price = predict_price(neighborhood, price_by_area, size, bathrooms, model=model)
        st.metric("Precio estimado", eur(price))
        st.write(f"**{eur(price / size)}/m²** estimado · referencia del barrio: "
                 f"{eur(row['priceByArea_media'])}/m² · mediana de ofertas: {eur(row['precio_mediano'])} "
                 f"({int(row['ofertas'])} ofertas)")
        comps = df[df["neighborhood"] == neighborhood][["propertyCode", "price", "size", "rooms", "bathrooms", "priceByArea", "propertyType"]]
        fig = px.scatter(comps, x="size", y="price", hover_data=["propertyCode", "rooms", "bathrooms", "priceByArea"],
                         labels={"size": "Superficie (m²)", "price": "Precio (€)"}, color_discrete_sequence=[BLUE])
        fig.update_traces(marker=dict(size=9, opacity=0.6, line=dict(width=1, color="white")), name="Ofertas del barrio", showlegend=True)
        fig.add_trace(go.Scatter(x=[size], y=[price], mode="markers", name="Tu inmueble",
                                 marker=dict(size=16, color=RED, symbol="star", line=dict(width=2, color="white")),
                                 hovertemplate="Tu inmueble<br>%{x} m²<br>%{y:,.0f} €<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=60), title=f"Ofertas en {neighborhood}",
                          legend=dict(orientation="h", y=-0.22, x=0))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Ejemplo del TFM"):
        st.write("En la memoria se estima un inmueble en Chamberí de 63 m², 2 baños y 6.427 €/m². "
                 "En el notebook se usó `neighborhood='Chamberi'` (nombre de distrito, no de barrio), "
                 "por lo que el TargetEncoder aplicó la media global. Aquí el barrio se elige de una lista "
                 "para evitar categorías desconocidas.")
        st.write(f"Resultado con la réplica (`'Chamberi'`, 63 m², 2 baños, 6.427 €/m²): "
                 f"**{eur(predict_price('Chamberi', 6427, 63, 2, model=model))}**")

# ---------------------------------------------------------------------------
# 2. Oportunidades (Business Case 1)
# ---------------------------------------------------------------------------
with tab_opp:
    st.subheader("Business Case 1 · Identificación de ofertas infravaloradas")
    st.caption("Ofertas del conjunto de test (20%, no vistas en el entrenamiento). "
               "Diferencia = precio publicado − precio estimado. Negativa ⇒ la oferta está por debajo del valor del modelo.")
    p = preds.copy()
    p["Descuento_%"] = -p["Difference"] / p["Predicted_Price"] * 100
    f1, f2, f3 = st.columns([2, 1, 1])
    dsel = f1.multiselect("Distritos", sorted(p["district"].unique()), placeholder="Todos")
    min_disc = f2.slider("Descuento mínimo vs. modelo (%)", 0, 40, 5)
    max_price = f3.number_input("Precio máximo (€)", 50_000, 800_000, 800_000, step=25_000)
    q = p[(p["Descuento_%"] >= min_disc) & (p["price"] <= max_price)]
    if dsel:
        q = q[q["district"].isin(dsel)]
    q = q.sort_values("Difference")

    k1, k2, k3 = st.columns(3)
    k1.metric("Oportunidades", f"{len(q):,}".replace(",", "."))
    k2.metric("Ahorro medio vs. modelo", eur(-q["Difference"].mean()) if len(q) else "—")
    k3.metric("Descuento mediano", f"{q['Descuento_%'].median():.1f} %" if len(q) else "—")

    show = q[["propertyCode", "district", "neighborhood", "propertyType", "size", "rooms", "bathrooms",
              "price", "Predicted_Price", "Difference", "Descuento_%"]].rename(columns={
        "price": "Precio publicado", "Predicted_Price": "Precio estimado", "Difference": "Diferencia",
        "district": "Distrito", "neighborhood": "Barrio", "propertyType": "Tipo", "size": "m²",
        "rooms": "Habitaciones", "bathrooms": "Baños"})
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "propertyCode": st.column_config.NumberColumn("Código", format="%d"),
        "Precio publicado": st.column_config.NumberColumn(format="%.0f €"),
        "Precio estimado": st.column_config.NumberColumn(format="%.0f €"),
        "Diferencia": st.column_config.NumberColumn(format="%.0f €"),
        "Descuento_%": st.column_config.ProgressColumn("Descuento", format="%.1f %%", min_value=0, max_value=40),
    })
    st.download_button("Descargar oportunidades (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "oportunidades.csv", "text/csv")
    st.caption("Ejemplo de la memoria: oferta **104475383** publicada a 210.000 € → "
               + (f"estimada en {eur(p.loc[p.propertyCode == 104475383, 'Predicted_Price'].iloc[0])}."
                  if (p.propertyCode == 104475383).any() else "no está en el test."))

# ---------------------------------------------------------------------------
# 3. Cargar datos nuevos
# ---------------------------------------------------------------------------
with tab_up:
    st.subheader("Valorar un lote de inmuebles")
    st.write("Sube un CSV en uno de estos dos formatos:")
    st.markdown("- **Formato Idealista** (descarga de la API, como `data/raw/idealista_0X.csv`): "
                "se aplica la misma limpieza y feature engineering del TFM.\n"
                "- **Formato simple**: columnas `neighborhood, priceByArea, size, bathrooms` "
                "(opcional `price` para calcular la diferencia).")
    template = pd.DataFrame({"neighborhood": ["Almagro", "Lavapiés-Embajadores"], "priceByArea": [7500, 5200],
                             "size": [80, 55], "bathrooms": [2, 1], "price": [620000, 260000]})
    st.download_button("Descargar plantilla (formato simple)", template.to_csv(index=False).encode("utf-8"),
                       "plantilla_inmuebles.csv", "text/csv")
    up = st.file_uploader("CSV de inmuebles", type=["csv"])
    if up is not None:
        try:
            new = pd.read_csv(io.BytesIO(up.getvalue()), low_memory=False)
            is_raw = {"propertyCode", "parkingSpace", "floor"}.issubset(new.columns)
            out = predict_dataframe(new, model=model, raw=is_raw)
            if is_raw:
                out = data.filter_scope(out)
            unknown = sorted(set(out["neighborhood"]) - set(ref["neighborhood"]))
            st.success(f"{len(out)} inmuebles valorados · formato {'Idealista' if is_raw else 'simple'}")
            if unknown:
                st.warning(f"Barrios no vistos en el entrenamiento (se usa la media global): {', '.join(map(str, unknown[:10]))}")
            cols = [c for c in ["propertyCode", "district", "neighborhood", "size", "bathrooms", "priceByArea",
                                "price", "Predicted_Price", "Difference"] if c in out.columns]
            st.dataframe(out[cols], width="stretch", hide_index=True)
            st.download_button("Descargar resultados", out[cols].to_csv(index=False).encode("utf-8"),
                               "valoraciones.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo procesar el fichero: {e}")

# ---------------------------------------------------------------------------
# 4. Mapa
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader("Precio medio por m² y densidad de oferta por distrito")
    agg = df.assign(key=df["district"].map(norm)).groupby("key").agg(
        priceByArea=("priceByArea", "mean"), ofertas=("price", "size"), precio_mediano=("price", "median")).reset_index()
    for f in geo["features"]:
        f["id"] = norm(f["properties"]["name"])
    names = {norm(f["properties"]["name"]): f["properties"]["name"] for f in geo["features"]}
    agg = agg[agg["key"].isin(names)].assign(Distrito=lambda x: x["key"].map(names))
    metric = st.radio("Variable", ["€/m² medio", "Nº de ofertas"], horizontal=True)
    col = "priceByArea" if metric == "€/m² medio" else "ofertas"
    st.plotly_chart(district_map(geo, agg, col, metric), width="stretch")
    st.caption("Solo se muestran los 21 distritos del municipio de Madrid (el geojson del TFM). "
               "Las ofertas de otros municipios de la provincia se incluyen en el modelo pero no en el mapa.")

# ---------------------------------------------------------------------------
# 5. Modelo
# ---------------------------------------------------------------------------
with tab_mod:
    st.subheader("Rendimiento del modelo final")
    fm = metrics.get("final_model", {})
    if fm:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{fm['R2']:.4f}")
        m2.metric("MAPE", f"{fm['MAPE'] * 100:.2f} %")
        m3.metric("RMSE", eur(fm["RMSE"]))
        m4.metric("MAE", eur(fm["MAE"]))
        st.caption(f"Variables: {', '.join(fm['features'])} · Hiperparámetros: {fm['params']}")

    a, b = st.columns(2)
    with a:
        fig = px.scatter(preds, x="price", y="Predicted_Price", hover_data=["propertyCode", "neighborhood"],
                         labels={"price": "Precio real (€)", "Predicted_Price": "Precio predicho (€)"},
                         color_discrete_sequence=[BLUE], title="Real vs predicho (test)")
        fig.update_traces(marker=dict(size=7, opacity=0.5))
        lo, hi = preds["price"].min(), preds["price"].max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GRAY, dash="dash", width=2),
                                 name="Ajuste ideal", hoverinfo="skip"))
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with b:
        if fm:
            imp = pd.Series(fm["feature_importance"]).sort_values()
            fig = px.bar(imp, orientation="h", labels={"value": "Importancia", "index": ""},
                         color_discrete_sequence=[BLUE], title="Importancia de variables")
            fig.update_traces(hovertemplate="%{y}: %{x:.3f}<extra></extra>")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")

    if "benchmark" in metrics:
        st.markdown("**Benchmark de modelos (Tabla 6 del TFM)**")
        rows = []
        for ds, v in metrics["benchmark"].items():
            for name, m in v["results"].items():
                rows.append({"Dataset": ds.split("_", 1)[1].replace("_", " "), "Modelo": name, **m})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.info("**Nota metodológica.** En los datos de Idealista `price = priceByArea × size`, y el modelo usa "
            "`priceByArea` como entrada; esto explica el R² cercano a 1. Esta versión replica el TFM tal cual "
            "se entregó; la corrección se abordará en una siguiente versión (ver README).")

# ---------------------------------------------------------------------------
# 6. Datos
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset procesado")
    st.caption(f"{len(df):,} ofertas en alcance (provincia de Madrid, ≤ 800.000 €) · {df['neighborhood'].nunique()} barrios".replace(",", "."))
    st.dataframe(df, width="stretch", hide_index=True, height=420)
    st.download_button("Descargar dataset procesado", df.to_csv(index=False).encode("utf-8"),
                       "datos_procesados_final.csv", "text/csv")
    st.markdown("**Referencia por barrio**")
    st.dataframe(ref, width="stretch", hide_index=True)
