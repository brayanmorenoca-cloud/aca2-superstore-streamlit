"""Inferencia local de DistilBERT. No descarga pesos ni genera resultados de muestra."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Iterable

CLASES = ("Furniture", "Office Supplies", "Technology")
MAX_CARACTERES = 2000
MAX_FILAS = 200
BASE_DIR = Path(__file__).resolve().parent


class ModeloNoDisponibleError(RuntimeError):
    """El paquete local del modelo está incompleto o es incompatible."""


def ruta_modelo() -> Path:
    valor = os.environ.get("ACA2_MODEL_DIR")
    return Path(valor).expanduser().resolve() if valor else BASE_DIR / "modelo_final"


def validar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        raise ValueError("El nombre del producto debe ser texto.")
    texto = " ".join(texto.split())
    if len(texto) < 2 or not any(c.isalpha() for c in texto):
        raise ValueError("Escribe un nombre de producto con al menos dos caracteres y alguna letra.")
    if len(texto) > MAX_CARACTERES:
        raise ValueError(f"El nombre supera el límite de {MAX_CARACTERES} caracteres.")
    return texto


def estado_artefactos(model_path: str | Path | None = None) -> dict:
    """Comprueba archivos antes de importar librerías pesadas. Nunca confunde tokenizer con pesos."""
    ruta = Path(model_path).resolve() if model_path else ruta_modelo()
    errores = []
    archivos = []
    for name in ("config.json", "tokenizer_config.json"):
        p = ruta / name
        if not p.is_file() or not p.stat().st_size:
            errores.append(f"Falta {name}.")
        else:
            archivos.append(p)
    tokenizer_files = [p for p in (ruta / "tokenizer.json", ruta / "vocab.txt") if p.is_file() and p.stat().st_size]
    if not tokenizer_files:
        errores.append("Falta tokenizer.json o vocab.txt.")
    archivos.extend(tokenizer_files)
    for name in ("inferencia_config.json", "metadata.json", "estado.json", "special_tokens_map.json", "added_tokens.json"):
        p = ruta / name
        if p.is_file() and p.stat().st_size:
            archivos.append(p)
    marcador = ruta / "estado.json"
    if marcador.is_file():
        try:
            contenido = json.loads(marcador.read_text(encoding="utf-8"))
            if not isinstance(contenido, dict) or contenido.get("estado") != "completado":
                errores.append("El entrenamiento/exportación aún no está completado; espera su finalización.")
        except (OSError, ValueError):
            errores.append("El marcador de estado del modelo no es válido.")
    pesos = ruta / "model.safetensors"
    indice = ruta / "model.safetensors.index.json"
    if pesos.is_file() and pesos.stat().st_size:
        archivos.append(pesos)
    elif indice.is_file():
        try:
            contenido = json.loads(indice.read_text(encoding="utf-8"))
            nombres = set(contenido["weight_map"].values())
            if not nombres:
                raise ValueError("índice vacío")
            for name in nombres:
                archivo = (ruta / name).resolve()
                if archivo.parent != ruta or not archivo.is_file() or not archivo.stat().st_size:
                    raise ValueError("fragmento inexistente o ruta no válida")
                archivos.append(archivo)
            archivos.append(indice)
        except (OSError, ValueError, KeyError, TypeError):
            errores.append("El índice de pesos safetensors está incompleto o no es válido.")
    else:
        errores.append("Faltan los pesos entrenados model.safetensors.")
    # El fingerprint invalida la caché al reemplazar pesos o configuración.
    firma = tuple(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in archivos))
    return {"disponible": not errores, "errores": errores, "ruta": str(ruta), "firma": firma}


def resultado_probabilidades(valores: Iterable[float], clases=CLASES, truncado=False) -> dict:
    valores = [float(x) for x in valores]
    if len(valores) != len(clases) or any(not math.isfinite(x) or x < 0 or x > 1 for x in valores):
        raise ModeloNoDisponibleError("El modelo produjo probabilidades inválidas.")
    if abs(sum(valores) - 1.0) > 1e-4:
        raise ModeloNoDisponibleError("Las probabilidades del modelo no suman uno.")
    indice = max(range(len(valores)), key=valores.__getitem__)
    return {"categoria": clases[indice], "confianza": valores[indice],
            "probabilidades": dict(zip(clases, valores)), "truncado": bool(truncado)}


def configuracion_local(ruta: Path) -> tuple[dict, int]:
    """Verifica la integridad una vez al construir el predictor, no en cada inferencia."""
    metadata_path = ruta / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        raise ModeloNoDisponibleError("metadata.json no contiene una configuración válida.")
    esperado = metadata.get("model_sha256")
    if esperado is not None:
        if not isinstance(esperado, str) or len(esperado) != 64 or any(c not in "0123456789abcdef" for c in esperado.lower()):
            raise ModeloNoDisponibleError("La huella SHA-256 del modelo no es válida.")
        pesos = ruta / "model.safetensors"
        if not pesos.is_file():
            raise ModeloNoDisponibleError("metadata.json declara una huella para model.safetensors, pero el archivo no existe.")
        with pesos.open("rb") as archivo:
            obtenido = hashlib.file_digest(archivo, "sha256").hexdigest()
        if obtenido != esperado.lower():
            raise ModeloNoDisponibleError("La huella SHA-256 de los pesos no coincide con metadata.json. Restaura el paquete del modelo completo.")
    config = ruta / "inferencia_config.json"
    parametros = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else metadata
    if not isinstance(parametros, dict):
        raise ModeloNoDisponibleError("La configuración de inferencia no es válida.")
    max_length = int(parametros.get("max_length", metadata.get("max_length", 64)))
    if not 8 <= max_length <= 512:
        raise ModeloNoDisponibleError("max_length debe estar entre 8 y 512.")
    return metadata, max_length


class PredictorSuperstore:
    """Carga una sola vez un modelo local y comparte su uso con un bloqueo de inferencia."""

    def __init__(self, model_path: str | Path | None = None, device: str = "cpu"):
        estado = estado_artefactos(model_path)
        if not estado["disponible"]:
            raise ModeloNoDisponibleError(" ".join(estado["errores"]))
        self.ruta = Path(estado["ruta"])
        try:
            config = json.loads((self.ruta / "config.json").read_text(encoding="utf-8"))
            etiquetas = config.get("id2label", {})
            self.clases = tuple(etiquetas.get(str(i), "") for i in range(3))
            if self.clases != CLASES or config.get("model_type") != "distilbert":
                raise ModeloNoDisponibleError("La configuración no coincide con DistilBERT y las tres categorías acordadas.")
            self.metadata, self.max_length = configuracion_local(self.ruta)
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.torch = torch
            self.device = torch.device(device)
            if self.device.type == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            self.tokenizer = AutoTokenizer.from_pretrained(self.ruta, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.ruta, local_files_only=True, trust_remote_code=False, use_safetensors=True
            ).to(self.device)
            self.model.eval()
            self._lock = threading.RLock()
        except ModeloNoDisponibleError:
            raise
        except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise ModeloNoDisponibleError(
                "No se pudo cargar el modelo local. Reinstala requirements.txt y restaura la carpeta modelo_final completa "
                "generada por el notebook. Detalle: " + str(exc)
            ) from exc

    def predecir(self, texto: str) -> dict:
        return self.predecir_lote([texto])[0]

    def predecir_lote(self, textos: Iterable[str], batch_size: int = 16) -> list[dict]:
        if isinstance(textos, (str, bytes)):
            raise ValueError("Para un lote se requiere una lista de nombres de productos.")
        textos = list(textos)
        if not 1 <= len(textos) <= MAX_FILAS:
            raise ValueError(f"El lote debe contener entre 1 y {MAX_FILAS} productos.")
        if not 1 <= batch_size <= 64:
            raise ValueError("El tamaño de lote debe estar entre 1 y 64.")
        textos = [validar_texto(t) for t in textos]
        resultados = []
        with self._lock, self.torch.inference_mode():
            for inicio in range(0, len(textos), batch_size):
                fragmento = textos[inicio:inicio + batch_size]
                longitudes = [len(self.tokenizer(t, truncation=False)["input_ids"]) for t in fragmento]
                inputs = self.tokenizer(fragmento, return_tensors="pt", truncation=True,
                                        padding=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                probs = self.torch.softmax(logits, dim=-1).cpu().tolist()
                resultados.extend(resultado_probabilidades(p, self.clases, n > self.max_length)
                                  for p, n in zip(probs, longitudes))
        return resultados
