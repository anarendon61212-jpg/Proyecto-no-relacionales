"""Generación de embeddings con sentence-transformers (consultas e ingesta)."""

from functools import lru_cache

from app.config import VECTOR_DIMENSIONS


class EmbeddingError(RuntimeError):
    """El modelo de embeddings no se pudo cargar o produjo un vector inválido."""


@lru_cache(maxsize=2)
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
