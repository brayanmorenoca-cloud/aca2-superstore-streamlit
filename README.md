# Clasificación de productos con DistilBERT

ACA 2 - Aprendizaje Automático, CUN.

**Integrantes:** Brayan Guillermo Moreno Castro y José Daniel Vargas Otalvaro.

Esta es la versión de inferencia para Streamlit Community Cloud. Conserva el
modelo CPU, los datos SQL y las métricas del paquete académico; no reentrena y
no reemplaza predicciones por datos simulados. La aplicación recibe nombres
de productos en inglés y clasifica Furniture, Office Supplies o Technology.

## 1. Subir el código a GitHub

Repositorio: https://github.com/brayanmorenoca-cloud/aca2-superstore-streamlit

Descomprimir `ACA2_Streamlit_GitHub.zip`. En **Add file > Upload files**, arrastrar
TODO el contenido de la carpeta descomprimida, conservando subcarpetas. `app.py`
y `requirements.txt` deben quedar en la raíz del repositorio, no dentro de
otra carpeta. Confirmar con **Commit changes** en la rama `main`.

No subir el ZIP como un único archivo ni los pesos mediante Upload files.

## 2. Publicar el modelo entrenado como Release

El modelo está dentro del ZIP académico completo:
`ACA2_Superstore/modelo_final/model.safetensors`.

En el repositorio, abrir **Releases > Create a new release**:

- Etiqueta nueva: `v1.0.0`.
- Target: `main`.
- Título: `Modelo DistilBERT Superstore - ACA2`.
- Adjuntar el archivo **model.safetensors**, sin comprimirlo ni renombrarlo.
- Esperar a que termine la carga. Publicar con **Publish release**.

El archivo debe tener 267835644 bytes. SHA-256 esperado:

`9acdd4816d4ad9ed5d74912e07f8f31e138c880f97784186df974ea264a5135a`

La aplicación descarga ese asset en su primer inicio y comprueba su tamaño y
su huella antes de cargarlo. Si falta la Release o el archivo es incorrecto,
muestra un error explicativo y no habilita una clasificación falsa.

URL prevista del asset (solo existirá después de publicarlo):
https://github.com/brayanmorenoca-cloud/aca2-superstore-streamlit/releases/download/v1.0.0/model.safetensors

## 3. Crear la aplicación en Streamlit

En https://share.streamlit.io/ seleccionar **Create app**, publicación desde GitHub:

- Repository: `brayanmorenoca-cloud/aca2-superstore-streamlit`.
- Branch: `main`.
- Main file path: `app.py`.
- Advanced settings / Python version: `3.12`.
- Secrets: vacío; esta aplicación no necesita claves de una API de IA.

Publicar y esperar la instalación. La primera descarga del modelo añade tiempo
al arranque. Usar la URL que Streamlit asigne después de desplegar: todavía no
se conoce ni se presenta como una aplicación publicada en este paquete.

## 4. Verificación antes de entregar el enlace

1. Abrir la URL pública sin la sesión de administrador (por ejemplo, incógnito).
2. Probar `Wireless keyboard`: debe producir Technology con este modelo.
3. Revisar las pestañas Evaluación, Datos SQL y Metodología.
4. Confirmar las métricas del modelo CPU: accuracy 0.975 y F1 macro 0.9712572464.
5. Copiar la URL real al póster. No usar el enlace del repositorio como si fuera la app.

Una URL de Community Cloud no depende de mantener abierta la sesión de Colab.
El servicio gratuito puede suspender aplicaciones inactivas; el visitante puede
reactivarlas. No se promete disponibilidad continua ni tiempos de arranque fijos.

## Ejecución local

Python 3.12. Desde la raíz de esta carpeta:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Si se copia el modelo correcto a `modelo_final/model.safetensors`, se valida y
reutiliza localmente sin descargarlo. La variable opcional `ACA2_MODEL_DIR`
permite señalar una carpeta completa de modelo ya disponible.

## Alcance y privacidad

- Inferencia CPU con pesos locales, sin API de IA externa.
- SQLite de solo lectura; atributos de productos, no datos de clientes.
- Los CSV de entrada se procesan en memoria; máximo 200 filas y 2 MB.
- Softmax no es una probabilidad calibrada. Revisar productos fuera del dominio.
- La evidencia del entrenamiento completo se conserva en el notebook y ZIP académico.

## Atribuciones y fuentes

Ver `ATRIBUCIONES.md`, `NOTICE.txt` y la carpeta `licencias/`.

- GitHub, distribución de archivos grandes: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- GitHub Releases: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- Streamlit, despliegue: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit, administración y suspensión: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
