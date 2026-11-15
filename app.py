"""Aplicación real del ACA 2. Ejecutar: python -m streamlit run app.py"""
from pathlib import Path
import logging
import pandas as pd
import streamlit as st

from modelo_distribucion import asegurar_pesos, ModeloDescargaError

from predictor import CLASES, ModeloNoDisponibleError, PredictorSuperstore, estado_artefactos, ruta_modelo
from ui_componentes import (MAX_CSV_BYTES, aplicar_estilo, csv_seguro, leer_lote_csv,
                           mostrar_datos, mostrar_metodologia, mostrar_metricas,
                           mostrar_resultado)

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Catálogo inteligente · ACA 2", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
aplicar_estilo()


@st.cache_resource(show_spinner="Cargando pesos entrenados de DistilBERT…")
def cargar_predictor(ruta: str, firma: tuple):
    return PredictorSuperstore(ruta)


with st.sidebar:
    logo = BASE_DIR / "materiales/assets/cun_logo.png"
    if logo.is_file():
        st.image(str(logo), width=160)
    st.markdown("### Catálogo inteligente")
    st.caption("ACA 2 · Aprendizaje Automático\n\nEspecialización en Inteligencia Artificial")
    st.divider()
    st.markdown("**Modelo**\n\nDistilBERT · clasificación de texto")
    st.markdown("**Dominio**\n\nVentas · Retail Superstore")
    st.markdown("**Idioma de entrada**\n\nInglés")
    st.divider()
    umbral = st.slider("Umbral de revisión manual", 0.3, 0.95, 0.6, 0.05,
                       help="Regla operativa configurable, no umbral estadísticamente calibrado.")
    st.caption("Las predicciones son sugerencias. Revisa productos fuera del catálogo o con descripciones ambiguas.")

st.markdown('<div class="aca-kicker">INTELIGENCIA ARTIFICIAL APLICADA A VENTAS</div>', unsafe_allow_html=True)
st.title("Clasificación de productos con DistilBERT")
st.markdown("Asigna una categoría comercial a partir del nombre del producto y consulta la evidencia del modelo.")
st.markdown('<div class="aca-summary">Escribe el nombre en inglés. El modelo distingue <strong>mobiliario</strong>, <strong>suministros de oficina</strong> y <strong>tecnología</strong>.</div>', unsafe_allow_html=True)

try:
    with st.spinner("Preparando el modelo entrenado para la aplicación…"):
        asegurar_pesos(ruta_modelo())
except ModeloDescargaError as exc:
    st.error(str(exc))
    st.info("El administrador debe publicar model.safetensors en la Release v1.0.0 del repositorio. No necesitas una clave de API ni volver a entrenar.")
    st.stop()

estado = estado_artefactos(ruta_modelo())
predictor = None
if estado["disponible"]:
    try:
        predictor = cargar_predictor(estado["ruta"], estado["firma"])
    except ModeloNoDisponibleError as exc:
        st.error(str(exc))
        logging.getLogger(__name__).warning("No se pudo cargar el modelo: %s", exc)
else:
    st.warning("No se han cargado los pesos entrenados. " + " ".join(estado["errores"]))
    st.caption("Ejecuta las celdas de entrenamiento/exportación y copia modelo_final junto a app.py. Los datos, métricas y metodología siguen disponibles; no se sustituyen las predicciones por datos simulados.")

tab1, tab2, tab3, tab4 = st.tabs(["Clasificar", "Evaluación", "Datos SQL", "Metodología"])
with tab1:
    individual, lote = st.tabs(["Un producto", "Archivo CSV"])
    with individual:
        st.subheader("Consulta individual")
        st.caption("Ejemplos: Ergonomic office chair · Stapler and paper clips · Wireless keyboard")
        with st.form("clasificacion_individual", clear_on_submit=False):
            texto = st.text_input("Nombre del producto en inglés", placeholder="Por ejemplo: Wireless keyboard", max_chars=2000, key="nombre_producto")
            enviar = st.form_submit_button("Clasificar producto", type="primary", disabled=predictor is None)
        if enviar and predictor is not None:
            try:
                with st.spinner("Analizando el producto…"):
                    resultado = predictor.predecir(texto)
                st.session_state["ultima_prediccion"] = {"texto": texto.strip(), "resultado": resultado, "firma": estado["firma"]}
            except ValueError as exc:
                st.session_state.pop("ultima_prediccion", None)
                st.error(str(exc))
            except (RuntimeError, OSError) as exc:
                st.session_state.pop("ultima_prediccion", None)
                st.error("No se pudo ejecutar la predicción. Revisa el modelo y las dependencias.")
                logging.getLogger(__name__).exception("Error de inferencia")
        ultimo = st.session_state.get("ultima_prediccion")
        if ultimo and ultimo["firma"] == estado["firma"] and predictor is not None:
            st.caption(f"Resultado para: {ultimo['texto']}")
            mostrar_resultado(ultimo["resultado"], umbral)
    with lote:
        st.subheader("Clasificación por lotes")
        st.caption("CSV UTF-8, separado por comas. Columna obligatoria: Product Name. Máximo 200 filas y 2 MB. Solo se procesará esa columna.")
        st.download_button("Descargar plantilla CSV", b"Product Name\nErgonomic office chair\nStapler and paper clips\nWireless keyboard\n", "plantilla_productos.csv", "text/csv")
        archivo = st.file_uploader("Cargar productos", type=["csv"], key="archivo_productos")
        frame = None
        contenido = None
        if archivo is not None:
            try:
                if archivo.size > MAX_CSV_BYTES:
                    raise ValueError("El CSV supera el límite de 2 MB.")
                contenido = archivo.getvalue()
                frame = leer_lote_csv(contenido)
                st.dataframe(frame.head(8), hide_index=True, width="stretch")
                st.caption(f"{len(frame)} productos validados. No se han clasificado todavía.")
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Clasificar archivo", type="primary", disabled=predictor is None or frame is None):
            try:
                with st.spinner("Clasificando el archivo…"):
                    predicciones = predictor.predecir_lote(frame["Product Name"].tolist())
                salida = frame.copy()
                salida["Categoría"] = [r["categoria"] for r in predicciones]
                salida["Probabilidad"] = [r["confianza"] for r in predicciones]
                salida["Texto truncado"] = [r.get("truncado", False) for r in predicciones]
                for clase in CLASES:
                    salida[f"P({clase})"] = [r["probabilidades"][clase] for r in predicciones]
                st.session_state["ultimo_lote"] = {"contenido": contenido, "salida": salida, "firma": estado["firma"]}
            except (ValueError, RuntimeError, OSError) as exc:
                st.session_state.pop("ultimo_lote", None)
                st.error(f"No se pudo clasificar el lote: {exc}")
        ultimo_lote = st.session_state.get("ultimo_lote")
        if ultimo_lote and contenido == ultimo_lote["contenido"] and ultimo_lote["firma"] == estado["firma"] and predictor is not None:
            salida = ultimo_lote["salida"].copy()
            salida["Revisión manual"] = salida["Probabilidad"] < umbral
            st.dataframe(salida, hide_index=True, width="stretch", column_config={"Probabilidad": st.column_config.NumberColumn(format="%.3f")})
            st.download_button("Descargar predicciones", csv_seguro(salida), "predicciones_productos.csv", "text/csv")
            st.caption("Probabilidades softmax no calibradas. Las etiquetas desconocidas o fuera de dominio no se detectan automáticamente.")
with tab2:
    mostrar_metricas(BASE_DIR)
with tab3:
    mostrar_datos(BASE_DIR)
with tab4:
    mostrar_metodologia(BASE_DIR)
st.markdown('<div class="aca-footer">CUN · Aprendizaje Automático · Prototipo académico de clasificación de catálogo</div>', unsafe_allow_html=True)
