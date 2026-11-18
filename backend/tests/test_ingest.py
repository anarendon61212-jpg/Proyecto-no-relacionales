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
            assert doc["parametros_chunking"]["version"]


def test_main_carga_articulos_y_reemplaza_chunks():
    database = mock.MagicMock()
    database.imagenes.find_one_and_update.return_value = {"_id": ObjectId()}
    database.articulos.find_one_and_update.side_effect = lambda *a, **k: {"_id": ObjectId()}
    client = mock.MagicMock()
    client.__getitem__.return_value = database

    with mock.patch.object(ingest, "MongoClient", return_value=client), mock.patch.object(
        ingest, "embed_texts", side_effect=lambda texts, model: fake_embed(texts)
    ):
        code = ingest.main([str(DATA_FILE), "--estrategias", "fixed-size", "sentence-aware", "--sin-imagenes"])

    articles = ingest.load_articles(DATA_FILE)
    assert code == 0
    assert database.articulos.find_one_and_update.call_count == len(articles)
    # Por cada artículo y estrategia se borran los chunks anteriores antes de insertar.
    assert database.chunks.delete_many.call_count == len(articles) * 2
    assert database.chunks.insert_many.call_count == len(articles) * 2
    first_filter = database.articulos.find_one_and_update.call_args_list[0].args[0]
    assert set(first_filter) == {"titulo", "fuente"}


def test_dataset_con_catalogo_de_imagenes(tmp_path):
    dataset = {
        "articulos": [
            {
                "titulo": "A", "contenido_texto": "Texto uno.", "idioma": "es", "fecha": "2024-01-01",
                "fuente": "F", "url": "https://fuente/a", "licencia": "CC BY 4.0",
                "imagenes": ["panel-1", {"url": "https://img/2.jpg", "categoria": "eolica"}],
            },
            {
                "titulo": "a", "contenido_texto": "Duplicado.", "idioma": "es", "fecha": "2024-01-02",
                "fuente": "f",
            },
            {
                "titulo": "B", "contenido_texto": "Texto dos.", "idioma": "en", "fecha": "2024-02-01",
                "fuente": "F", "imagenes": ["no-existe"],
            },
        ],
        "imagenes": [
            {"clave": "panel-1", "url": "data/imagenes/panel.jpg", "categoria": "solar",
             "licencia": "CC0", "atribucion": "Autor X"},
            {"clave": "sin-articulo", "url": "data/imagenes/eolica.jpg", "categoria": "eolica"},
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(__import__("json").dumps(dataset), encoding="utf-8")

    database = mock.MagicMock()
    image_ids = {}
    def upsert_image(filter_, update, **kwargs):
        return {"_id": image_ids.setdefault(filter_["url"], ObjectId())}
    database.imagenes.find_one_and_update.side_effect = upsert_image
    database.articulos.find_one_and_update.side_effect = lambda *a, **k: {"_id": ObjectId()}
    client = mock.MagicMock()
    client.__getitem__.return_value = database

    with mock.patch.object(ingest, "MongoClient", return_value=client), mock.patch.object(
        ingest, "embed_texts", side_effect=lambda texts, model: fake_embed(texts)
    ):
        code = ingest.main([str(path), "--estrategias", "sentence-aware", "--sin-imagenes"])

    assert code == 0
    # Las dos imágenes del catálogo + la imagen embebida en el artículo A.
    assert set(image_ids) == {"data/imagenes/panel.jpg", "data/imagenes/eolica.jpg", "https://img/2.jpg"}
    catalog_update = database.imagenes.find_one_and_update.call_args_list[0].args[1]["$set"]
    assert catalog_update["licencia"] == "CC0" and catalog_update["atribucion"] == "Autor X"
    # Solo A se guarda: el duplicado y el artículo con clave inexistente se omiten.
    assert database.articulos.find_one_and_update.call_count == 1
    stored = database.articulos.find_one_and_update.call_args.args[1]["$set"]
    assert stored["licencia"] == "CC BY 4.0" and stored["url"] == "https://fuente/a"
    assert stored["imagenes_relacionadas"] == [
        image_ids["data/imagenes/panel.jpg"], image_ids["https://img/2.jpg"]
    ]


def test_embed_pending_images_omite_las_que_fallan():
    database = mock.MagicMock()
    ok_id, bad_id = ObjectId(), ObjectId()
    database.imagenes.find.return_value = [{"_id": ok_id, "url": "ok.jpg"}, {"_id": bad_id, "url": "rota.jpg"}]

    def load(url):
        if url == "rota.jpg":
            raise ingest.ImageLoadError("404")
        return "imagen"

    with mock.patch.object(ingest, "load_image", side_effect=load), mock.patch.object(
        ingest, "embed_images", return_value=[[0.5] * 512]
    ):
        done, failed = ingest.embed_pending_images(database, "clip", reprocess=False)

    assert done == 1 and failed == ["404"]
    assert database.imagenes.find.call_args.args[0] == {"modelo_embedding": {"$ne": "clip"}}
    update = database.imagenes.update_one.call_args
    assert update.args[0] == {"_id": ok_id} and update.args[1]["$set"]["modelo_embedding"] == "clip"


def test_main_exige_archivo_o_desde_mongo():
    with pytest.raises(SystemExit):
        ingest.main([])
