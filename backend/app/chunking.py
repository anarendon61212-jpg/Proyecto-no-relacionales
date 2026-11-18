"""Estrategias de chunking: fixed-size, sentence-aware y semantic.

Los tamaños se miden en palabras. all-MiniLM-L6-v2 trunca la entrada en 256 tokens
(unas 180 palabras en español), así que los chunks se mantienen por debajo de eso.
"""

import re
from collections.abc import Callable

MAX_WORDS = 150
OVERLAP_WORDS = 30
SEMANTIC_PERCENTILE = 0.25  # Se corta donde la similitud entre oraciones cae al 25 % más bajo.
SEMANTIC_MIN_WORDS = 30  # Un cambio de tema no corta un chunk que aún es más corto que esto.

ESTRATEGIAS = ("fixed-size", "sentence-aware", "semantic")

# Fin de oración: . ! ? … seguidos de espacio y de una mayúscula, signo de apertura o dígito.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[\"'«“¿¡(\[A-ZÁÉÍÓÚÑÜ0-9])")


def fixed_size_chunks(text: str, size: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Ventanas de `size` palabras que se solapan `overlap` palabras."""
    if overlap >= size:
        raise ValueError("El solapamiento debe ser menor que el tamaño del chunk.")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def split_sentences(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Divide en oraciones; una oración más larga que `max_words` se parte en ventanas."""
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in _SENTENCE_BOUNDARY.split(paragraph.strip()):
            sentence = " ".join(sentence.split())
            if not sentence:
                continue
            if len(sentence.split()) > max_words:
                sentences.extend(fixed_size_chunks(sentence, max_words, overlap=max_words // 5))
            else:
                sentences.append(sentence)
    return sentences


def _group_sentences(
    sentences: list[str], breaks: set[int], max_words: int, overlap_last: bool, min_words: int = 0
) -> list[str]:
    """Agrupa oraciones consecutivas; empieza un chunk nuevo al llenarse o en `breaks`
    (estos últimos solo si el chunk actual ya tiene `min_words`)."""
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for index, sentence in enumerate(sentences):
        length = len(sentence.split())
        topic_break = index in breaks and words >= min_words
        if current and (topic_break or words + length > max_words):
            chunks.append(" ".join(current))
            # Repetir la última oración da continuidad entre chunks (solo si cabe).
            keep = current[-1:] if overlap_last and len(current) > 1 else []
            if keep and len(keep[0].split()) + length > max_words:
                keep = []
            current, words = list(keep), sum(len(s.split()) for s in keep)
        current.append(sentence)
        words += length
    if current:
        chunks.append(" ".join(current))
    return chunks


def sentence_aware_chunks(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Oraciones completas hasta `max_words`, repitiendo la última oración como solapamiento."""
    return _group_sentences(split_sentences(text, max_words), set(), max_words, overlap_last=True)


def semantic_chunks(
    text: str,
    embed: Callable[[list[str]], list[list[float]]],
    max_words: int = MAX_WORDS,
    percentile: float = SEMANTIC_PERCENTILE,
    min_words: int = SEMANTIC_MIN_WORDS,
) -> list[str]:
    """Corta donde cambia el tema: baja similitud entre oraciones consecutivas.

    `embed` debe devolver vectores normalizados, así el producto punto es la similitud coseno.
    """
    sentences = split_sentences(text, max_words)
    if len(sentences) <= 2:
        return _group_sentences(sentences, set(), max_words, overlap_last=False)

    vectors = embed(sentences)
    similarities = [
        sum(a * b for a, b in zip(vectors[i], vectors[i + 1])) for i in range(len(sentences) - 1)
    ]
    ordered = sorted(similarities)
    threshold = ordered[int(percentile * (len(ordered) - 1))]
    # El corte va antes de la oración i+1 cuando su similitud con la i queda por debajo del
    # percentil. Es estricto: si muchas similitudes son iguales, no se corta entre ellas.
    breaks = {i + 1 for i, similarity in enumerate(similarities) if similarity < threshold}
    return _group_sentences(sentences, breaks, max_words, overlap_last=False, min_words=min_words)


def chunk_text(
    text: str, estrategia: str, embed: Callable[[list[str]], list[list[float]]] | None = None
) -> list[str]:
    if estrategia == "fixed-size":
        return fixed_size_chunks(text)
    if estrategia == "sentence-aware":
        return sentence_aware_chunks(text)
    if estrategia == "semantic":
        if embed is None:
            raise ValueError("La estrategia semantic necesita una función de embeddings.")
        return semantic_chunks(text, embed)
    raise ValueError(f"Estrategia desconocida: {estrategia}. Usa una de {', '.join(ESTRATEGIAS)}.")
