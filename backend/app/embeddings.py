"""Generación de embeddings de consultas con sentence-transformers."""

from functools import lru_cache

from app.config import VECTOR_DIMENSIONS


class EmbeddingError(RuntimeError):
    """El modelo de embeddings no se pudo cargar o produjo un vector inválido."""


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    # Import diferido: cargar torch es lento y solo hace falta en búsquedas vectoriales.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_query(text: str, model_name: str) -> list[float]:
    """Devuelve el embedding normalizado de la consulta como lista de floats."""
    try:
        model = _load_model(model_name)
        vector = model.encode(text, normalize_embeddings=True)
    except Exception as exc:  # Descarga fallida, modelo inexistente, falta de memoria...
        raise EmbeddingError(
            f"No se pudo generar el embedding con el modelo '{model_name}': {exc}"
        ) from exc

    values = [float(value) for value in vector]
    if len(values) != VECTOR_DIMENSIONS:
        raise EmbeddingError(
            f"El modelo '{model_name}' produce vectores de {len(values)} dimensiones, "
            f"pero el índice vectorial espera {VECTOR_DIMENSIONS}."
        )
    return values
