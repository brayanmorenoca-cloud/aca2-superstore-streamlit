"""Pruebas aisladas: no usan GitHub ni ninguna conexión de red."""
import hashlib
import io
import os
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import modelo_distribucion as modulo


PAYLOAD = b"pesos-safetensors-simulados-para-test" * 17


class Respuesta(io.BytesIO):
    def __init__(self, body, *, length=None, url=None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.url = url or modulo.MODEL_URL

    def geturl(self):
        return self.url


@pytest.fixture(autouse=True)
def configuracion_pequena(monkeypatch):
    modulo._VALIDATED.clear()
    monkeypatch.setattr(modulo, "EXPECTED_SIZE", len(PAYLOAD))
    monkeypatch.setattr(modulo, "EXPECTED_SHA256", hashlib.sha256(PAYLOAD).hexdigest())

    def sin_red(*args, **kwargs):
        raise AssertionError("La prueba no debe acceder a la red")

    monkeypatch.setattr(modulo, "urlopen", sin_red)
    yield
    modulo._VALIDATED.clear()


def respuesta(monkeypatch, body=PAYLOAD, **kwargs):
    monkeypatch.setattr(modulo, "urlopen", lambda *a, **k: Respuesta(body, **kwargs))


def sin_temporales(directorio: Path):
    assert not list(directorio.glob(".model-*.part"))


def test_reutiliza_existente_sin_red(tmp_path):
    destino = tmp_path / modulo.MODEL_FILENAME
    destino.write_bytes(PAYLOAD)
    assert modulo.asegurar_pesos(tmp_path) == destino
    assert destino.read_bytes() == PAYLOAD


def test_existente_tamano_incorrecto_no_se_reemplaza(tmp_path):
    destino = tmp_path / modulo.MODEL_FILENAME
    destino.write_bytes(b"archivo-original")
    with pytest.raises(modulo.ModeloDescargaError, match="tamaño incorrecto"):
        modulo.asegurar_pesos(tmp_path)
    assert destino.read_bytes() == b"archivo-original"
    sin_temporales(tmp_path)


def test_existente_hash_incorrecto_no_se_reemplaza(tmp_path):
    destino = tmp_path / modulo.MODEL_FILENAME
    destino.write_bytes(b"x" * len(PAYLOAD))
    with pytest.raises(modulo.ModeloDescargaError, match="SHA-256"):
        modulo.asegurar_pesos(tmp_path)
    assert destino.read_bytes() == b"x" * len(PAYLOAD)


def test_descarga_url_fija_con_timeout_y_reemplazo_atomico(tmp_path, monkeypatch):
    llamadas = []

    def abrir(peticion, timeout):
        assert peticion.full_url == (
            "https://github.com/brayanmorenoca-cloud/aca2-superstore-streamlit/"
            "releases/download/v1.0.0/model.safetensors"
        )
        assert timeout == 30
        llamadas.append(peticion.full_url)
        return Respuesta(PAYLOAD, length=len(PAYLOAD))

    reemplazo = modulo.os.replace

    def reemplazar(origen, destino):
        assert Path(origen).parent == Path(destino).parent
        assert Path(origen).read_bytes() == PAYLOAD
        assert not Path(destino).exists()
        reemplazo(origen, destino)

    monkeypatch.setattr(modulo, "urlopen", abrir)
    monkeypatch.setattr(modulo.os, "replace", reemplazar)
    destino = modulo.asegurar_pesos(tmp_path / "modelo_final")
    assert destino.read_bytes() == PAYLOAD
    assert len(llamadas) == 1
    assert modulo.asegurar_pesos(tmp_path / "modelo_final") == destino
    assert len(llamadas) == 1
    sin_temporales(destino.parent)


@pytest.mark.parametrize("body", [PAYLOAD[:-1], PAYLOAD + b"x"])
def test_tamano_descargado_incorrecto_limpia_tmp(tmp_path, monkeypatch, body):
    respuesta(monkeypatch, body)
    with pytest.raises(modulo.ModeloDescargaError, match="tamaño"):
        modulo.asegurar_pesos(tmp_path)
    assert not (tmp_path / modulo.MODEL_FILENAME).exists()
    sin_temporales(tmp_path)


def test_tamano_anunciado_incorrecto(tmp_path, monkeypatch):
    respuesta(monkeypatch, length=len(PAYLOAD) + 10)
    with pytest.raises(modulo.ModeloDescargaError, match="tamaño anunciado"):
        modulo.asegurar_pesos(tmp_path)
    sin_temporales(tmp_path)


def test_hash_descargado_incorrecto_limpia_tmp(tmp_path, monkeypatch):
    respuesta(monkeypatch, b"x" * len(PAYLOAD))
    with pytest.raises(modulo.ModeloDescargaError, match="SHA-256"):
        modulo.asegurar_pesos(tmp_path)
    assert not (tmp_path / modulo.MODEL_FILENAME).exists()
    sin_temporales(tmp_path)


def test_404_explica_release_pendiente(tmp_path, monkeypatch):
    def abrir(*args, **kwargs):
        raise HTTPError(modulo.MODEL_URL, 404, "Not Found", {}, None)

    monkeypatch.setattr(modulo, "urlopen", abrir)
    with pytest.raises(modulo.ModeloDescargaError, match="Release v1.0.0"):
        modulo.asegurar_pesos(tmp_path)
    assert not (tmp_path / modulo.MODEL_FILENAME).exists()
    sin_temporales(tmp_path)


def test_timeout_limpia_descarga_parcial(tmp_path, monkeypatch):
    class RespuestaTimeout(Respuesta):
        def read(self, size):
            if self.tell():
                raise TimeoutError("timeout de lectura")
            return super().read(8)

    monkeypatch.setattr(modulo, "urlopen", lambda *a, **k: RespuestaTimeout(PAYLOAD))
    with pytest.raises(modulo.ModeloDescargaError, match="agotó el tiempo"):
        modulo.asegurar_pesos(tmp_path)
    assert not (tmp_path / modulo.MODEL_FILENAME).exists()
    sin_temporales(tmp_path)


def test_error_de_red_sin_fallback(tmp_path, monkeypatch):
    def abrir(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(modulo, "urlopen", abrir)
    with pytest.raises(modulo.ModeloDescargaError, match="conectar con GitHub"):
        modulo.asegurar_pesos(tmp_path)
    assert not (tmp_path / modulo.MODEL_FILENAME).exists()


def test_rechaza_redireccion_no_https(tmp_path, monkeypatch):
    respuesta(monkeypatch, url="http://example.invalid/model.safetensors")
    with pytest.raises(modulo.ModeloDescargaError, match="HTTPS"):
        modulo.asegurar_pesos(tmp_path)
    sin_temporales(tmp_path)


def test_cache_no_repite_hash_pero_detecta_modificacion(tmp_path, monkeypatch):
    destino = tmp_path / modulo.MODEL_FILENAME
    destino.write_bytes(PAYLOAD)
    llamadas = []
    hash_original = modulo._hash_archivo

    def contar(ruta):
        llamadas.append(ruta)
        return hash_original(ruta)

    monkeypatch.setattr(modulo, "_hash_archivo", contar)
    modulo.asegurar_pesos(tmp_path)
    modulo.asegurar_pesos(tmp_path)
    assert len(llamadas) == 1
    estado = destino.stat()
    destino.write_bytes(b"x" * len(PAYLOAD))
    os.utime(destino, ns=(estado.st_atime_ns, estado.st_mtime_ns + 100_000))
    with pytest.raises(modulo.ModeloDescargaError, match="SHA-256"):
        modulo.asegurar_pesos(tmp_path)
    assert len(llamadas) == 2


def test_archivo_invalido_aparece_durante_descarga_no_se_pisa(tmp_path, monkeypatch):
    destino = tmp_path / modulo.MODEL_FILENAME

    class RespuestaConcurrente(Respuesta):
        def read(self, size):
            bloque = super().read(size)
            if not bloque:
                destino.write_bytes(b"archivo-que-no-debe-sobrescribirse")
            return bloque

    monkeypatch.setattr(modulo, "urlopen", lambda *a, **k: RespuestaConcurrente(PAYLOAD))
    with pytest.raises(modulo.ModeloDescargaError, match="tamaño incorrecto"):
        modulo.asegurar_pesos(tmp_path)
    assert destino.read_bytes() == b"archivo-que-no-debe-sobrescribirse"
    sin_temporales(tmp_path)


def test_no_sigue_symlink(tmp_path):
    otro = tmp_path / "otro"
    otro.write_bytes(PAYLOAD)
    (tmp_path / modulo.MODEL_FILENAME).symlink_to(otro)
    with pytest.raises(modulo.ModeloDescargaError, match="archivo regular"):
        modulo.asegurar_pesos(tmp_path)
    assert otro.read_bytes() == PAYLOAD
