"""Descarga verificada de los pesos CPU del ACA 2, sin claves ni reentrenamiento.

El archivo grande se distribuye como asset de GitHub Releases; el código y el
tokenizador permanecen en el repositorio. Importar este módulo no usa la red.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MODEL_URL = (
    "https://github.com/brayanmorenoca-cloud/aca2-superstore-streamlit/"
    "releases/download/v1.0.0/model.safetensors"
)
MODEL_FILENAME = "model.safetensors"
EXPECTED_SIZE = 267_835_644
EXPECTED_SHA256 = "9acdd4816d4ad9ed5d74912e07f8f31e138c880f97784186df974ea264a5135a"
CHUNK_SIZE = 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 30
TOTAL_TIMEOUT_SECONDS = 300

_LOCK = threading.RLock()
_VALIDATED: dict[str, tuple] = {}


class ModeloDescargaError(RuntimeError):
    """Los pesos no están disponibles o no pasan la verificación de integridad."""


def _firma(estado: os.stat_result) -> tuple:
    return (
        estado.st_dev,
        estado.st_ino,
        estado.st_size,
        estado.st_mtime_ns,
        estado.st_ctime_ns,
        EXPECTED_SIZE,
        EXPECTED_SHA256,
    )


def _hash_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(CHUNK_SIZE), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _verificar_existente(ruta: Path) -> bool:
    """Devuelve False únicamente si no existe; nunca sustituye archivos inválidos."""
    try:
        antes = ruta.lstat()
    except FileNotFoundError:
        return False
    clave = str(ruta.absolute())
    if not stat.S_ISREG(antes.st_mode):
        raise ModeloDescargaError(
            "La ruta de los pesos existe, pero no es un archivo regular. "
            "No se modificó. Revisa modelo_final/model.safetensors."
        )
    if antes.st_size != EXPECTED_SIZE:
        _VALIDATED.pop(clave, None)
        raise ModeloDescargaError(
            "El archivo de pesos existente tiene un tamaño incorrecto. "
            "No se reemplazó: conserva una copia y revisa el archivo cargado."
        )
    firma = _firma(antes)
    if _VALIDATED.get(clave) == firma:
        return True
    digest = _hash_archivo(ruta)
    despues = ruta.lstat()
    if _firma(despues) != firma or not stat.S_ISREG(despues.st_mode):
        _VALIDATED.pop(clave, None)
        raise ModeloDescargaError(
            "Los pesos cambiaron durante la verificación. Intenta cargar la "
            "aplicación de nuevo cuando termine la copia del archivo."
        )
    if digest != EXPECTED_SHA256:
        _VALIDATED.pop(clave, None)
        raise ModeloDescargaError(
            "El SHA-256 del archivo de pesos existente no coincide con el modelo "
            "CPU de la entrega. No se eliminó ni se reemplazó el archivo."
        )
    _VALIDATED[clave] = firma
    return True


def _descargar(ruta: Path) -> Path:
    temporal: Path | None = None
    inicio = time.monotonic()
    try:
        peticion = Request(
            MODEL_URL,
            headers={
                "User-Agent": "ACA2-Superstore/1.0",
                "Accept": "application/octet-stream",
                "Accept-Encoding": "identity",
            },
        )
        with urlopen(peticion, timeout=NETWORK_TIMEOUT_SECONDS) as respuesta:
            if urlsplit(respuesta.geturl()).scheme != "https":
                raise ModeloDescargaError(
                    "La descarga no terminó en una conexión HTTPS segura. "
                    "Los pesos no se instalaron."
                )
            longitud = respuesta.headers.get("Content-Length")
            if longitud is not None:
                try:
                    longitud_numero = int(longitud)
                except (TypeError, ValueError) as exc:
                    raise ModeloDescargaError(
                        "GitHub devolvió un tamaño de descarga no válido."
                    ) from exc
                if longitud_numero != EXPECTED_SIZE:
                    raise ModeloDescargaError(
                        "El tamaño anunciado de la descarga no coincide con los "
                        "pesos esperados. Revisa el asset model.safetensors."
                    )
            digest = hashlib.sha256()
            total = 0
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".model-", suffix=".part", dir=ruta.parent,
                delete=False,
            ) as salida:
                temporal = Path(salida.name)
                while True:
                    if time.monotonic() - inicio > TOTAL_TIMEOUT_SECONDS:
                        raise TimeoutError("Tiempo total de descarga agotado")
                    bloque = respuesta.read(CHUNK_SIZE)
                    if not bloque:
                        break
                    total += len(bloque)
                    if total > EXPECTED_SIZE:
                        raise ModeloDescargaError(
                            "La descarga supera el tamaño permitido del modelo. "
                            "No se instalaron los pesos."
                        )
                    digest.update(bloque)
                    salida.write(bloque)
                salida.flush()
                os.fsync(salida.fileno())
            if total != EXPECTED_SIZE:
                raise ModeloDescargaError(
                    "La descarga del modelo quedó incompleta: el tamaño no "
                    "coincide. Vuelve a cargar la aplicación para reintentar."
                )
            if digest.hexdigest() != EXPECTED_SHA256:
                raise ModeloDescargaError(
                    "El SHA-256 de la descarga no coincide con el modelo CPU "
                    "de la entrega. No se instalaron pesos distintos."
                )
        # Puede haber aparecido un archivo mientras se recibía la descarga.
        # Si existe y es inválido, esta comprobación falla sin reemplazarlo.
        if _verificar_existente(ruta):
            return ruta
        os.replace(temporal, ruta)
        temporal = None
        _VALIDATED[str(ruta.absolute())] = _firma(ruta.lstat())
        return ruta
    except HTTPError as exc:
        if exc.code == 404:
            mensaje = (
                "No se encontró el modelo publicado. Falta publicar la Release "
                "v1.0.0 y adjuntar model.safetensors en GitHub. "
                "No es un fallo de entrenamiento ni requiere volver a entrenar."
            )
        elif exc.code in (401, 403):
            mensaje = (
                "GitHub no permite descargar el modelo. Comprueba que el "
                "repositorio y su Release sean públicos y estén accesibles."
            )
        else:
            mensaje = (
                f"GitHub devolvió el error HTTP {exc.code} al descargar los pesos. "
                "Intenta de nuevo cuando el servicio esté disponible."
            )
        raise ModeloDescargaError(mensaje) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ModeloDescargaError(
            "Se agotó el tiempo para descargar el modelo. Vuelve a cargar la "
            "aplicación; no se conservará una descarga incompleta."
        ) from exc
    except URLError as exc:
        raise ModeloDescargaError(
            "No se pudo conectar con GitHub para descargar el modelo. "
            "Comprueba la conexión y vuelve a cargar la aplicación."
        ) from exc
    except OSError as exc:
        raise ModeloDescargaError(
            "No se pudieron guardar los pesos descargados. Revisa el espacio "
            "disponible y los permisos de la carpeta modelo_final."
        ) from exc
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def asegurar_pesos(model_dir: Path) -> Path:
    """Reutiliza el modelo validado o descarga exactamente la versión de entrega.

    La caché usa identidad, tamaño y fechas del archivo: una modificación fuerza
    una nueva comprobación SHA-256. El bloqueo evita descargas simultáneas entre
    las sesiones del mismo proceso de Streamlit.
    """
    ruta = Path(model_dir) / MODEL_FILENAME
    with _LOCK:
        try:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            if _verificar_existente(ruta):
                return ruta
        except OSError as exc:
            raise ModeloDescargaError(
                "No se pudo acceder a modelo_final. Revisa sus permisos de "
                "lectura y escritura. No se modificaron los pesos existentes."
            ) from exc
        return _descargar(ruta)
