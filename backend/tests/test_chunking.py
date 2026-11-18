import math

import pytest

from app.chunking import (
    SEMANTIC,
    SENTENCE_AWARE,
    chunk_text,
    fixed_size_chunks,
    parametros,
    semantic_chunks,
    sentence_aware_chunks,
    split_sentences,
)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def sentences(n, template="Oración número {i} con algo de texto."):
    return [template.format(i=i) for i in range(n)]


def test_fixed_size_respeta_tamano_y_solapamiento():
    chunks = fixed_size_chunks(words(25), size=10, overlap=3)
    assert [len(c.split()) for c in chunks] == [10, 10, 10, 4]
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "w24"
    assert fixed_size_chunks("") == []


def test_split_sentences_en_espanol_e_ingles():
    text = "Primera oración. ¿Segunda pregunta? ¡Tercera! El valor es 3.5 por ciento. Then English. Done"
    assert split_sentences(text) == [
        "Primera oración.",
        "¿Segunda pregunta?",
        "¡Tercera!",
        "El valor es 3.5 por ciento.",
        "Then English.",
        "Done",
    ]


def test_split_sentences_separa_titulos_y_listas_por_linea():
    text = "Energía solar\n- Paneles en techos\n- Parques solares\nLa demanda crece. Mucho."
    assert split_sentences(text) == [
        "Energía solar", "- Paneles en techos", "- Parques solares", "La demanda crece.", "Mucho."
    ]


def test_split_sentences_parte_texto_sin_puntuacion():
    parts = split_sentences(words(40), max_words=10)
    assert all(len(p.split()) <= 10 for p in parts) and len(parts) > 1


def test_sentence_aware_cinco_oraciones_con_una_de_solapamiento():
    s = sentences(12)
    chunks = sentence_aware_chunks(" ".join(s))
    # Avanza de 4 en 4: [0-4], [4-8], [8-11].
    assert chunks[0] == " ".join(s[0:5])
    assert chunks[1] == " ".join(s[4:9])
    assert chunks[2] == " ".join(s[8:12])
    assert len(chunks) == 3


def test_sentence_aware_respeta_el_tope_de_palabras():
    s = sentences(5, template=" ".join(["palabra"] * 40) + " {i}.")  # 41 palabras c/u
    chunks = sentence_aware_chunks(" ".join(s))
    assert all(len(c.split()) <= SENTENCE_AWARE["max_palabras"] for c in chunks)
    assert len(chunks) >= 2


def unit(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def topic_embed(texts):
    """Oraciones con 'agua' en un eje, el resto en otro (similitud entre temas ≈ 0.2)."""
    return [unit([1.0, 0.1]) if "agua" in t else unit([0.1, 1.0]) for t in texts]


def test_semantic_corta_bajo_el_umbral():
    text = (
        "El agua lluvia se recoge. El agua se filtra. El agua llega al tanque. "
        "Las huertas crecen al sol. Las huertas usan compost. Las huertas dan tomates."
    )
    assert semantic_chunks(text, topic_embed) == [
        "El agua lluvia se recoge. El agua se filtra. El agua llega al tanque.",
        "Las huertas crecen al sol. Las huertas usan compost. Las huertas dan tomates.",
    ]


def test_semantic_respeta_minimo_de_oraciones():
    # El tema cambia tras la primera oración, pero un chunk no puede tener solo una.
    text = "El agua se recoge. Las huertas crecen. Las huertas dan tomates."
    assert semantic_chunks(text, topic_embed) == [
        "El agua se recoge. Las huertas crecen. Las huertas dan tomates."
    ]


def test_semantic_respeta_maximo_de_oraciones():
    text = " ".join(f"El agua número {i} se mide." for i in range(20))
    chunks = semantic_chunks(text, topic_embed)
    assert [len(split_sentences(c)) for c in chunks] == [8, 8, 4]


def test_parametros_registran_version_y_umbral():
    assert parametros("semantic")["umbral_similitud"] == SEMANTIC["umbral_similitud"] == 0.80
    assert parametros("sentence-aware")["max_oraciones"] == 5
    assert "version" in parametros("fixed-size")
    with pytest.raises(ValueError):
        parametros("otra")


def test_chunk_text_valida_estrategia_y_embed():
    with pytest.raises(ValueError):
        chunk_text("hola", "otra")
    with pytest.raises(ValueError):
        chunk_text("hola", "semantic")
