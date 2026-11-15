"""Componentes visuales y lecturas seguras para la aplicación académica."""
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import CLASES, MAX_FILAS, validar_texto

MAX_CSV_BYTES = 2 * 1024 * 1024
TRADUCCIONES = {"Furniture": "Mobiliario", "Office Supplies": "Suministros de oficina", "Technology": "Tecnología"}
CONSULTAS = {
    "Catálogo por categoría": 'SELECT category AS categoria, COUNT(*) AS productos FROM catalogo GROUP BY category ORDER BY category',
    "Particiones por categoría": 'SELECT split AS particion, category AS categoria, COUNT(*) AS productos FROM catalogo GROUP BY split, category ORDER BY split, category',
    "Muestra del catálogo": 'SELECT sample_id AS id, product_name AS producto, category AS categoria, split AS particion FROM catalogo ORDER BY sample_id LIMIT 30',
    "Registros fuente por categoría": 'SELECT category AS categoria, COUNT(*) AS registros FROM registros GROUP BY category ORDER BY category',
}


def leer_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    contenido = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contenido, dict):
        raise ValueError(f"{path.name} debe contener un objeto JSON.")
    return contenido


def leer_lote_csv(contenido: bytes) -> pd.DataFrame:
    if not contenido:
        raise ValueError("El CSV está vacío.")
    if len(contenido) > MAX_CSV_BYTES:
        raise ValueError("El CSV supera el límite de 2 MB.")
    try:
        texto = contenido.decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False, nrows=MAX_FILAS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Carga un CSV UTF-8 separado por comas, con encabezado Product Name.") from exc
    if "Product Name" not in df.columns:
        raise ValueError("El CSV debe tener una columna llamada Product Name.")
    if not 1 <= len(df) <= MAX_FILAS:
        raise ValueError(f"El CSV debe contener entre 1 y {MAX_FILAS} filas.")
    nombres = []
    for i, texto in enumerate(df["Product Name"], start=2):
        try:
            nombres.append(validar_texto(texto))
        except ValueError as exc:
            raise ValueError(f"Fila {i}: {exc}") from exc
    # No se retienen columnas adicionales (pueden contener datos personales).
    return pd.DataFrame({"Product Name": nombres})


def csv_seguro(df: pd.DataFrame) -> bytes:
    salida = df.copy()
    for col in salida.columns:
        if pd.api.types.is_string_dtype(salida[col]) or salida[col].dtype == object:
            salida[col] = salida[col].map(lambda x: "'" + x if isinstance(x, str) and x.startswith(("=", "+", "-", "@", "\t", "\r")) else x)
    return salida.to_csv(index=False).encode("utf-8-sig")


def consulta_predefinida(db_path: Path, nombre: str) -> pd.DataFrame:
    if nombre not in CONSULTAS:
        raise ValueError("Consulta no permitida.")
    if not db_path.is_file():
        raise FileNotFoundError("Aún no se ha generado la base SQLite.")
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only = ON")
        return pd.read_sql_query(CONSULTAS[nombre], conn)


def aplicar_estilo() -> None:
    # Solo HTML/CSS estático. Ningún texto del usuario se interpola aquí.
    st.markdown("""
    <style>
    .block-container{max-width:1220px;padding-top:2.2rem;padding-bottom:2rem;}
    [data-testid="stSidebar"]{background:#f0f4f8;border-right:1px solid #dce5ed;}
    h1,h2,h3{color:#18334c;letter-spacing:-.025em;}
    [data-testid="stMetric"]{border:1px solid #e2e8ef;border-radius:12px;padding:16px;background:#fff;}
    [data-testid="stMetricLabel"]{color:#496176;}
    [data-testid="stMetricValue"]{color:#18334c;font-weight:700;}
    .aca-kicker{color:#147c71;font-size:.78rem;font-weight:750;letter-spacing:.12em;margin-bottom:.5rem;}
    .aca-summary{border-left:4px solid #179c8e;background:#edf8f6;padding:16px 20px;border-radius:0 10px 10px 0;color:#294d5b;margin:1rem 0 1.2rem;}
    .aca-footer{font-size:.8rem;color:#6b7d8d;border-top:1px solid #e1e8ed;margin-top:2rem;padding-top:.8rem;}
    </style>
    """, unsafe_allow_html=True)


def mostrar_resultado(resultado: dict, umbral: float) -> None:
    categoria = resultado["categoria"]
    izquierda, derecha = st.columns([1, 1.35], gap="large")
    with izquierda:
        st.markdown("#### Categoría sugerida")
        st.success(f"{TRADUCCIONES.get(categoria, categoria)} · {categoria}")
        st.metric("Probabilidad del modelo", f"{resultado['confianza']:.1%}")
        if resultado["confianza"] < umbral:
            st.warning("Revisar manualmente: la probabilidad está por debajo del umbral seleccionado.")
        if resultado.get("truncado"):
            st.warning("El texto excede la longitud del modelo. Se clasificó el comienzo del nombre.")
    with derecha:
        st.markdown("#### Distribución por clase")
        probabilidades = pd.DataFrame({"Categoría": list(resultado["probabilidades"]),
                                       "Probabilidad": list(resultado["probabilidades"].values())})
        st.bar_chart(probabilidades.set_index("Categoría"), horizontal=True, color="#179c8e", height=200)
    st.caption("La salida softmax no es una garantía de acierto ni una probabilidad calibrada. El modelo solo compara tres clases del catálogo.")


def _metrica(datos: dict, *nombres):
    for nombre in nombres:
        if nombre in datos and isinstance(datos[nombre], (int, float)):
            return datos[nombre]
    return None


def mostrar_metricas(base: Path) -> None:
    st.subheader("Evaluación del modelo")
    st.caption("Resultados del conjunto de prueba reservado. La validación se utiliza para elegir el checkpoint; el test no participa en esa selección.")
    try:
        metricas = leer_json(base / "resultados/metricas_test.json")
        metadata = leer_json(base / "modelo_final/metadata.json")
        marcador = leer_json(base / "modelo_final/estado.json")
    except (ValueError, OSError) as exc:
        st.error(f"No se pudieron leer las métricas: {exc}")
        return
    if marcador is not None and marcador.get("estado") != "completado":
        st.info("El entrenamiento/exportación está en curso. La evaluación se mostrará cuando el paquete esté completado.")
        return
    if metricas and metadata and metricas.get("model_sha256") and metadata.get("model_sha256"):
        if metricas["model_sha256"] != metadata["model_sha256"]:
            st.error("Las métricas no corresponden a los pesos de este modelo. Restaura el paquete completo de la misma ejecución.")
            return
    if metricas is None:
        st.info("Las métricas aún no están exportadas. Ejecuta la evaluación del notebook para generar resultados/metricas_test.json.")
    else:
        valores = metricas.get("metrics", metricas)
        accuracy = _metrica(valores, "accuracy", "test_accuracy")
        f1 = _metrica(valores, "f1_macro", "f1", "test_f1", "test_f1_macro")
        loss = _metrica(valores, "loss", "test_loss")
        n = _metrica(metricas, "n_samples", "n_test", "support")
        cols = st.columns(4)
        for col, etiqueta, valor, porcentaje in zip(cols, ["Accuracy", "F1 macro", "Pérdida test", "Productos test"], [accuracy, f1, loss, n], [True, True, False, False]):
            col.metric(etiqueta, "No disponible" if valor is None else f"{valor:.1%}" if porcentaje else f"{int(valor):,}" if etiqueta == "Productos test" else f"{valor:.4f}")
        reporte = metricas.get("classification_report", metricas.get("reporte_clasificacion"))
        if isinstance(reporte, dict):
            filas = []
            for clase in CLASES:
                item = reporte.get(clase)
                if isinstance(item, dict):
                    filas.append({"Clase": clase, "Precisión": item.get("precision"), "Recall": item.get("recall"), "F1": item.get("f1-score"), "Soporte": item.get("support")})
            if filas:
                st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Precisión", "Recall", "F1"]})
        with st.expander("Evidencia exportada en JSON"):
            st.json(metricas, expanded=False)
        st.download_button("Descargar métricas", json.dumps(metricas, ensure_ascii=False, indent=2), "metricas_test.json", "application/json")
    figs = base / "figuras"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Matriz de confusión")
        p = figs / "matriz_confusion.png"
        if p.is_file():
            st.image(str(p), width="stretch")
        else:
            st.caption("Pendiente de generar en la evaluación.")
    with c2:
        st.markdown("#### Curvas de aprendizaje")
        candidatos = [figs / n for n in ["curvas_entrenamiento.png", "curvas_aprendizaje.png"]]
        p = next((p for p in candidatos if p.is_file()), None)
        if p:
            st.image(str(p), width="stretch")
        else:
            st.caption("Pendiente de exportar el historial del entrenamiento.")
    historial = base / "resultados/historial_entrenamiento.csv"
    if historial.is_file():
        with st.expander("Historial por época"):
            try:
                tabla = pd.read_csv(historial)
                st.dataframe(tabla, hide_index=True, width="stretch")
                st.download_button("Descargar historial", historial.read_bytes(), "historial_entrenamiento.csv", "text/csv")
            except (OSError, ValueError) as exc:
                st.error(f"No se puede leer el historial: {exc}")


def mostrar_datos(base: Path) -> None:
    st.subheader("Datos y gestión SQL")
    st.caption("Se muestran atributos del producto, sin nombres, direcciones ni identificadores de clientes. Consultas predefinidas de solo lectura.")
    try:
        resumen = leer_json(base / "resultados/eda_summary.json")
    except (OSError, ValueError) as exc:
        st.error(f"Resumen EDA no válido: {exc}")
        resumen = None
    if resumen:
        a, b, c = st.columns(3)
        a.metric("Filas de la fuente", f"{resumen.get('source_rows', 0):,}")
        b.metric("Nombres de catálogo", f"{resumen.get('catalog_rows', 0):,}")
        c.metric("ID de producto", f"{resumen.get('unique_product_ids', 0):,}")
        with st.expander("Auditoría y particiones"):
            st.json(resumen, expanded=False)
    nombre = st.selectbox("Consulta de la base SQLite", list(CONSULTAS), key="consulta_sql")
    try:
        df = consulta_predefinida(base / "scripts_sql/datos_superstore.sqlite", nombre)
        st.dataframe(df, hide_index=True, width="stretch")
        if nombre == "Catálogo por categoría":
            st.bar_chart(df.set_index("categoria"), color="#179c8e", height=240)
    except (FileNotFoundError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        st.info(f"La consulta no está disponible todavía: {exc}")
    with st.expander("Ver consulta SQL"):
        st.code(CONSULTAS[nombre], language="sql")


def mostrar_metodologia(base: Path) -> None:
    st.subheader("Metodología y alcance")
    st.markdown("""
    **Problema.** Asignar una categoría comercial a partir del nombre de un producto, como apoyo a la organización de un catálogo de ventas.

    **Entrada y salida.** Nombre del producto en inglés → Furniture, Office Supplies o Technology. No se predicen ventas, precios ni subcategorías.

    **Red neuronal.** DistilBERT preentrenado en inglés, adaptado a clasificación de tres clases. El tokenizador transforma el texto en tokens, el Transformer contextualiza la secuencia y una cabeza de clasificación produce puntuaciones por clase.

    **Flujo reproducible.** Datos originales → gestión SQLite y EDA → limpieza y particiones documentadas → tokenización → fine-tuning → selección por validación → prueba reservada → inferencia local en Streamlit.

    **Uso responsable.** Es un prototipo académico, no un sistema productivo validado. La taxonomía es cerrada: incluso un producto fuera del dominio recibe una de las tres clases. La puntuación softmax no detecta por sí sola entradas ajenas al catálogo. No introduce productos nuevos en la base ni reentrena con lo escrito en la interfaz.

    **Privacidad.** Los productos ingresados se procesan en el equipo o servidor que ejecuta esta app. La aplicación no llama a una API de IA externa, no guarda los nombres ingresados en disco y no registra clientes.
    """)
    st.markdown("#### Fuentes")
    st.markdown("- [Sales Forecasting - Retail Superstore, Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)\n- [DistilBERT: artículo original](https://arxiv.org/abs/1910.01108)\n- [Modelo DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased)\n- [Documentación de Streamlit](https://docs.streamlit.io/)")
    p = base / "modelo_final/inferencia_config.json"
    if not p.is_file():
        p = base / "modelo_final/metadata.json"
    if p.is_file():
        with st.expander("Configuración de inferencia"):
            try:
                st.json(leer_json(p))
            except (OSError, ValueError) as exc:
                st.error(str(exc))
