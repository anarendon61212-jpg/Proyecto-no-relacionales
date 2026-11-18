import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest
from bson import ObjectId

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ingest  # noqa: E402

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "articulos_ejemplo.json"


def fake_embed(texts):
    return [[1.0] + [0.0] * 383 for _ in texts]


def test_parse_fecha_acepta_formatos_comunes():
    utc = timezone.utc
    assert ingest.parse_fecha("2024-05-01") == datetime(2024, 5, 1, tzinfo=utc)
    assert ingest.parse_fecha("2024-05-01T10:00:00Z") == datetime(2024, 5, 1, 10, tzinfo=utc)
    assert ingest.parse_fecha("2024-05-01T10:00:00-05:00").utcoffset().total_seconds() == -5 * 3600
    with pytest.raises(ValueError):
        ingest.parse_fecha("ayer")


def test_normalize_article_valida_y_separa_imagenes():
    raw = {
        "titulo": " Título ", "contenido_texto": "Texto.", "idioma": "es",
        "fecha": "2024-01-02", "fuente": "F", "extra": "se ignora",
        "imagenes": [{"url": "https://x/y.jpg", "categoria": "c"}],
    }
    article, images = ingest.normalize_article(raw)
    assert article["titulo"] == "Título" and "extra" not in article
    assert isinstance(article["fecha"], datetime)
    assert images == [{"url": "https://x/y.jpg", "categoria": "c"}]

    with pytest.raises(ValueError, match="contenido_texto"):
        ingest.normalize_article({**raw, "contenido_texto": "  "})


def test_datos_de_ejemplo_son_validos():
    articles = ingest.load_articles(DATA_FILE)
    assert len(articles) >= 3
    for raw in articles:
        ingest.normalize_article(raw)


def test_build_chunk_documents_cumple_el_validador():
    doc_id = ObjectId()
    article, _ = ingest.normalize_article(ingest.load_articles(DATA_FILE)[0])
    now = datetime.now(timezone.utc)
    for estrategia in ("fixed-size", "sentence-aware", "semantic"):
        docs = ingest.build_chunk_documents(doc_id, article, estrategia, fake_embed, "modelo-x", now)
        assert docs, estrategia
        assert [d["chunk_index"] for d in docs] == list(range(len(docs)))
        for doc in docs:
            assert doc["doc_id"] == doc_id and doc["estrategia_chunking"] == estrategia
            assert len(doc["embedding"]) == 384 and all(isinstance(v, float) for v in doc["embedding"])
            assert doc["idioma"] == "es" and doc["modelo"] == "modelo-x" and doc["chunk_texto"]


def test_main_carga_articulos_y_reemplaza_chunks():
    database = mock.MagicMock()
    database.imagenes.find_one_and_update.return_value = {"_id": ObjectId()}
    database.articulos.find_one_and_update.side_effect = lambda *a, **k: {"_id": ObjectId()}
    client = mock.MagicMock()
    client.__getitem__.return_value = database

    with mock.patch.object(ingest, "MongoClient", return_value=client), mock.patch.object(
        ingest, "embed_texts", side_effect=lambda texts, model: fake_embed(texts)
    ):
        code = ingest.main([str(DATA_FILE), "--estrategias", "fixed-size", "sentence-aware"])

    articles = ingest.load_articles(DATA_FILE)
    assert code == 0
    assert database.articulos.find_one_and_update.call_count == len(articles)
    # Por cada artículo y estrategia se borran los chunks anteriores antes de insertar.
    assert database.chunks.delete_many.call_count == len(articles) * 2
    assert database.chunks.insert_many.call_count == len(articles) * 2
    first_filter = database.articulos.find_one_and_update.call_args_list[0].args[0]
    assert set(first_filter) == {"titulo", "fuente"}


def test_main_exige_archivo_o_desde_mongo():
    with pytest.raises(SystemExit):
        ingest.main([])
