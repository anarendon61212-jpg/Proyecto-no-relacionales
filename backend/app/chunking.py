"""Estrategias de chunking: sentence-aware y semantic (comparadas en el experimento) y fixed-size.

Parámetros según el plan de la Persona C (docs/persona-c-dataset-y-chunking.md):
- sentence-aware: máximo 5 oraciones por chunk, 1 oración de solapamiento.
- semantic: corta cuando la similitud coseno entre oraciones consecutivas cae por
  debajo de 0.80, con límites mínimo y máximo para evitar chunks diminutos o gigantes.

Todas comparten un tope de palabras: all-MiniLM-L6-v2 trunca la entrada en 256 tokens
(unas 180 palabras en español), así que ningún chunk supera MAX_WORDS.

Tratamiento del texto:
- Oraciones: terminan en . ! ? … seguidos de espacio y mayúscula, signo de apertura o dígito
  (así "3.5" no corta). Abreviaturas como "Sr." o "EE. UU." seguidas de mayúscula sí cortan.
- Saltos de línea: cada línea es una unidad aparte, así títulos y elementos de listas sin
  punto final no se pegan a la oración siguiente.
- Texto sin puntuación fiable: un fragmento más largo que MAX_WORDS se parte en ventanas.
"""

import re
from collections.abc import Callable

CHUNKING_VERSION = "2.0"
MAX_WORDS = 150

FIXED_SIZE = {"palabras": 150, "solapamiento_palabras": 30}
SENTENCE_AWARE = {"max_oraciones": 5, "solapamiento_oraciones": 1, "max_palabras": MAX_WORDS}
SEMANTIC = {
    "umbral_similitud": 0.80,
    "comparacion": "oraciones consecutivas (coseno)",
    "min_oraciones": 2,
    "max_oraciones": 8,
    "max_palabras": MAX_WORDS,
}

ESTRATEGIAS = ("sentence-aware", "semantic", "fixed-size")
ESTRATEGIAS_EXPERIMENTO = ("sentence-aware", "semantic")

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[\"'«“¿¡(\[A-ZÁÉÍÓÚÑÜ0-9])")


def parametros(estrategia: str) -> dict:
    """Parámetros efectivos de una estrategia, para guardarlos junto a cada chunk."""
    valores = {"fixed-size": FIXED_SIZE, "sentence-aware": SENTENCE_AWARE, "semantic": SEMANTIC}
    if estrategia not in valores:
        raise ValueError(f"Estrategia desconocida: {estrategia}. Usa una de {', '.join(ESTRATEGIAS)}.")
    return {"version": CHUNKING_VERSION, **valores[estrategia]}


def fixed_size_chunks(
    text: str, size: int = FIXED_SIZE["palabras"], overlap: int = FIXED_SIZE["solapamiento_palabras"]
) -> list[str]:
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
    """Divide en oraciones (y líneas); una unidad más larga que `max_words` se parte en ventanas."""
    sentences = []
    for line in text.splitlines():
        for sentence in _SENTENCE_BOUNDARY.split(line.strip()):
            sentence = " ".join(sentence.split())
            if not sentence:
                continue
            if len(sentence.split()) > max_words:
                sentences.extend(fixed_size_chunks(sentence, max_words, overlap=max_words // 5))
            else:
                sentences.append(sentence)
    return sentences


def _group(
    sentences: list[str],
    max_sentences: int,
    max_words: int,
    overlap: int = 0,
    breaks: set[int] | None = None,
    min_sentences: int = 1,
) -> list[str]:
    """Agrupa oraciones consecutivas en chunks.

    Cierra el chunk al llegar a `max_sentences`, si la siguiente oración excede `max_words`,
    o en un índice de `breaks` (cambio de tema) cuando el chunk ya tiene `min_sentences`.
    Las últimas `overlap` oraciones se repiten al inicio del chunk siguiente.
    """
    breaks = breaks or set()
    chunks: list[str] = []
    current: list[str] = []
    for index, sentence in enumerate(sentences):
        words = sum(len(s.split()) for s in current)
        full = len(current) >= max_sentences or words + len(sentence.split()) > max_words
        topic_change = index in breaks and len(current) >= min_sentences
        if current and (full or topic_change):
            chunks.append(" ".join(current))
            # El solapamiento solo se aplica si deja espacio para avanzar y cabe en el tope.
            keep = current[-overlap:] if overlap and len(current) > overlap else []
            while keep and sum(len(s.split()) for s in keep) + len(sentence.split()) > max_words:
                keep = keep[1:]
            current = list(keep)
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def sentence_aware_chunks(text: str, params: dict = SENTENCE_AWARE) -> list[str]:
    sentences = split_sentences(text, params["max_palabras"])
    return _group(
        sentences,
        max_sentences=params["max_oraciones"],
        max_words=params["max_palabras"],
        overlap=params["solapamiento_oraciones"],
    )


def semantic_chunks(
    text: str, embed: Callable[[list[str]], list[list[float]]], params: dict = SEMANTIC
) -> list[str]:
    """Corta antes de la oración i+1 si su similitud con la oración i es menor que el umbral.

    `embed` debe devolver vectores normalizados: así el producto punto es la similitud coseno.
    """
    sentences = split_sentences(text, params["max_palabras"])
    breaks: set[int] = set()
    if len(sentences) > 1:
        vectors = embed(sentences)
        for i in range(len(sentences) - 1):
            similarity = sum(a * b for a, b in zip(vectors[i], vectors[i + 1]))
            if similarity < params["umbral_similitud"]:
                breaks.add(i + 1)
    return _group(
        sentences,
        max_sentences=params["max_oraciones"],
        max_words=params["max_palabras"],
        breaks=breaks,
        min_sentences=params["min_oraciones"],
    )


def chunk_text(
    text: str, estrategia: str, embed: Callable[[list[str]], list[list[float]]] | None = None
) -> list[str]:
    """Interfaz común: devuelve los chunks del texto, en orden, con la estrategia indicada."""
    if estrategia == "fixed-size":
        return fixed_size_chunks(text)
    if estrategia == "sentence-aware":
        return sentence_aware_chunks(text)
    if estrategia == "semantic":
        if embed is None:
            raise ValueError("La estrategia semantic necesita una función de embeddings.")
        return semantic_chunks(text, embed)
    raise ValueError(f"Estrategia desconocida: {estrategia}. Usa una de {', '.join(ESTRATEGIAS)}.")
