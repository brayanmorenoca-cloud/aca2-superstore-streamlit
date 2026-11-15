# Clasificación de productos con DistilBERT

ACA 2 - Aprendizaje Automático, CUN.

**Integrantes:** Brayan Guillermo Moreno Castro y José Daniel Vargas Otalvaro.

Aplicación de clasificación de nombres de productos en inglés en tres categorías:
Furniture, Office Supplies y Technology. Integra DistilBERT, SQLite y Streamlit.

## Aplicación publicada

- [Abrir la aplicación](https://aca2-superstore-brayan-jose.streamlit.app/)
- [Repositorio](https://github.com/brayanmorenoca-cloud/aca2-superstore-streamlit)
- [Modelo entrenado - Release v1.0.0](https://github.com/brayanmorenoca-cloud/aca2-superstore-streamlit/releases/tag/v1.0.0)

La publicación usa los mismos pesos CPU del paquete académico; no reentrena ni
sustituye predicciones por resultados simulados. Evaluación: accuracy **0.975**,
F1 macro **0.9712572464**, pérdida **0.108757927**, sobre **280 productos**.
Prueba comprobada: `Wireless keyboard` -> `Technology`, confianza mostrada **99.0 %**.
La disponibilidad depende de Community Cloud; no se garantiza acceso continuo.

## Modelo e integridad

`model.safetensors` se distribuye como asset de la Release, no dentro de Git.
La app lo descarga cuando falta y verifica tamaño y SHA-256 antes de cargarlo.
Si la descarga falla o no coincide, muestra un error y no fabrica predicciones.

- Tamaño: **267835644 bytes**.
- SHA-256: `9acdd4816d4ad9ed5d74912e07f8f31e138c880f97784186df974ea264a5135a`.
- [Descargar pesos](https://github.com/brayanmorenoca-cloud/aca2-superstore-streamlit/releases/download/v1.0.0/model.safetensors)

## Ejecutar y desplegar

Con Python **3.12**, desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Community Cloud: repositorio `brayanmorenoca-cloud/aca2-superstore-streamlit`,
rama `main`, archivo `app.py`, Python `3.12`, sin secretos de API.
La primera descarga del modelo añade tiempo al arranque. Si ya existe el modelo
correcto en `modelo_final/model.safetensors`, se valida y reutiliza sin descargarlo.
`ACA2_MODEL_DIR` permite indicar otra carpeta completa del modelo.

## Alcance y privacidad

- Clasificación individual y CSV UTF-8 con columna `Product Name`, hasta 200 filas / 2 MB.
- Softmax no es una probabilidad calibrada ni garantiza acierto fuera del dominio.
- SQLite de solo lectura, reducido a atributos de productos; sin datos de clientes.
- CSV de entrada procesados en memoria; sin API externa de IA ni credenciales.
- Notebook, entrenamiento y evidencias completas se conservan en el ZIP académico.

## Atribuciones

Consultar [ATRIBUCIONES.md](ATRIBUCIONES.md), [NOTICE.txt](NOTICE.txt) y `licencias/`.
Modelo base: Apache-2.0. La documentación del dataset original registra GPL 2.
No se asigna una licencia nueva al código de los integrantes.
