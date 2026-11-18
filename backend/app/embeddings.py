"""Generación de embeddings con sentence-transformers (consultas e ingesta)."""

from functools import lru_cache

from app.config import VECTOR_DIMENSIONS


class EmbeddingError(RuntimeError):
    """El modelo de embeddings no se pudo cargar o produjo un vector inválido."""


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    # Import diferido: cargar torch es lento y solo hace falta al generar embeddings.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> list[list[float]]:
    """Embeddings normalizados (norma 1) de varios textos, como listas de floats."""
    if not texts:
        return []
    try:
        model = _load_model(model_name)
        matrix = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    except Exception as exc:  # Descarga fallida, modelo inexistente, falta de memoria...
        raise EmbeddingError(
            f"No se pudo generar el embedding con el modelo '{model_name}': {exc}"
        ) from exc

    vectors = [[float(value) for value in row] for row in matrix]
    if vectors and len(vectors[0]) != VECTOR_DIMENSIONS:
        raise EmbeddingError(
            f"El modelo '{model_name}' produce vectores de {len(vectors[0])} dimensiones, "
            f"pero el índice vectorial espera {VECTOR_DIMENSIONS}."
        )
    return vectors


def embed_query(text: str, model_name: str) -> list[float]:
    """Embedding normalizado de una consulta."""
    return embed_texts([text], model_name)[0]


# --- Imágenes (CLIP) -------------------------------------------------------------------

IMAGE_DOWNLOAD_TIMEOUT = 20  # segundos


class ImageLoadError(RuntimeError):
    """No se pudo descargar o abrir una imagen."""


def load_image(source: str):
    """Abre una imagen desde una URL http(s) o una ruta relativa a la raíz del repositorio."""
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    from app.config import PROJECT_ROOT

    try:
        if source.startswith(("http://", "https://")):
            import httpx

            response = httpx.get(source, timeout=IMAGE_DOWNLOAD_TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            path = (PROJECT_ROOT / source).resolve()
            image = Image.open(path)
        return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageLoadError(f"No se pudo abrir la imagen '{source}': {exc}") from exc
    except Exception as exc:  # Errores de red de httpx (timeout, 404, DNS...).
        raise ImageLoadError(f"No se pudo descargar la imagen '{source}': {exc}") from exc


def _check_image_dimensions(vectors: list[list[float]], model_name: str) -> list[list[float]]:
    from app.config import IMAGE_VECTOR_DIMENSIONS

    if vectors and len(vectors[0]) != IMAGE_VECTOR_DIMENSIONS:
        raise EmbeddingError(
            f"El modelo '{model_name}' produce vectores de {len(vectors[0])} dimensiones, "
            f"pero el índice de imágenes espera {IMAGE_VECTOR_DIMENSIONS}."
        )
    return vectors


def embed_images(images: list, model_name: str) -> list[list[float]]:
    """Embeddings CLIP normalizados de imágenes PIL."""
    if not images:
        return []
    try:
        matrix = _load_model(model_name).encode(images, normalize_embeddings=True)
    except Exception as exc:
        raise EmbeddingError(f"No se pudo generar el embedding de imagen con '{model_name}': {exc}") from exc
    return _check_image_dimensions([[float(v) for v in row] for row in matrix], model_name)


def embed_image_query(text: str, model_name: str) -> list[float]:
    """Embedding CLIP de un texto, en el mismo espacio que las imágenes."""
    try:
        matrix = _load_model(model_name).encode([text], normalize_embeddings=True)
    except Exception as exc:
        raise EmbeddingError(f"No se pudo generar el embedding con el modelo '{model_name}': {exc}") from exc
    return _check_image_dimensions([[float(v) for v in row] for row in matrix], model_name)[0]
