import math

import pytest

from app.chunking import (
    chunk_text,
    fixed_size_chunks,
    semantic_chunks,
    sentence_aware_chunks,
    split_sentences,
)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_fixed_size_respeta_tamano_y_solapamiento():
    chunks = fixed_size_chunks(words(25), size=10, overlap=3)
    assert [len(c.split()) for c in chunks] == [10, 10, 10, 4]
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "w24"  # no se pierde el final
    assert fixed_size_chunks("") == []


def test_fixed_size_texto_corto_es_un_solo_chunk():
    assert fixed_size_chunks(words(5), size=10, overlap=3) == [words(5)]


def test_split_sentences_en_espanol():
    text = "Primera oración. ¿Segunda pregunta? ¡Tercera! El valor es 3.5 por ciento.\n\nNuevo párrafo"
    assert split_sentences(text) == [
        "Primera oración.",
        "¿Segunda pregunta?",
        "¡Tercera!",
        "El valor es 3.5 por ciento.",
        "Nuevo párrafo",
    ]


def test_split_sentences_parte_oraciones_demasiado_largas():
    parts = split_sentences(words(40) + ".", max_words=10)
    assert all(len(p.split()) <= 10 for p in parts)
    assert len(parts) > 1


def test_sentence_aware_no_corta_oraciones_y_solapa_la_ultima():
    sentences = [f"Oración número {i} con algo de texto." for i in range(12)]  # 6 palabras c/u
    chunks = sentence_aware_chunks(" ".join(sentences), max_words=20)
    for chunk in chunks:
        assert len(chunk.split()) <= 20
        assert chunk.endswith(".")
    # La última oración de un chunk se repite al inicio del siguiente.
    assert chunks[1].startswith(chunks[0].split(". ")[-1].rstrip("."))
    assert sentences[-1] in chunks[-1]


def fake_embed(sentences):
    """Dos temas: oraciones con 'agua' apuntan a un eje y el resto a otro."""
    vectors = []
    for sentence in sentences:
        vector = [1.0, 0.1] if "agua" in sentence else [0.1, 1.0]
        norm = math.sqrt(sum(v * v for v in vector))
        vectors.append([v / norm for v in vector])
    return vectors


def test_semantic_corta_donde_cambia_el_tema():
    text = (
        "El agua lluvia se recoge. El agua se filtra. El agua llega al tanque. "
        "Las huertas crecen al sol. Las huertas usan compost. Las huertas dan tomates."
    )
    chunks = semantic_chunks(text, fake_embed, max_words=100, min_words=5)
    assert chunks == [
        "El agua lluvia se recoge. El agua se filtra. El agua llega al tanque.",
        "Las huertas crecen al sol. Las huertas usan compost. Las huertas dan tomates.",
    ]


def test_semantic_no_deja_chunks_diminutos():
    text = (
        "El agua se recoge. Las huertas crecen. "
        "Las huertas usan compost y dan tomates en las terrazas de la ciudad."
    )
    # Con mínimo de 10 palabras, el corte tras "El agua se recoge." (4 palabras) se ignora.
    assert len(semantic_chunks(text, fake_embed, max_words=100, min_words=10)) == 1


def test_semantic_respeta_el_maximo_de_palabras():
    text = " ".join(f"El agua número {i} se mide." for i in range(30))
    chunks = semantic_chunks(text, fake_embed, max_words=15)
    assert all(len(c.split()) <= 15 for c in chunks)


def test_chunk_text_valida_estrategia_y_embed():
    with pytest.raises(ValueError):
        chunk_text("hola", "otra")
    with pytest.raises(ValueError):
        chunk_text("hola", "semantic")
