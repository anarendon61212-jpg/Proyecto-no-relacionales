from unittest import mock

from bson import ObjectId
from pymongo.errors import OperationFailure

from app import main
from app.embeddings import ImageLoadError
from conftest import ART_2, chunk

IMG_1, IMG_2 = ObjectId(), ObjectId()


def image_results(database):
    database.imagenes.aggregate.return_value = [
        {"_id": IMG_2, "url": "data/imagenes/eolica.jpg", "categoria": "eolica", "score": 0.83,
         "articulos": [{"_id": ART_2, "titulo": "Dos"}]},
    ]


def test_buscar_imagenes_por_texto(client, database):
    image_results(database)
    with mock.patch.object(main, "embed_image_query", return_value=[0.1] * 512) as embed:
        response = client.post("/imagenes/buscar", json={"texto": "turbinas eólicas", "categoria": "eolica"})
    body = response.json()
    assert response.status_code == 200
    assert body["results"][0]["_id"] == str(IMG_2)
    assert body["results"][0]["articulos"][0]["_id"] == str(ART_2)
    assert embed.call_args.args[1] == main.get_settings()["image_text_model"]
    stage = database.imagenes.aggregate.call_args.args[0][0]["$vectorSearch"]
    assert stage["index"] == "imagenes_vector_index"
    assert stage["filter"] == {"categoria": {"$eq": "eolica"}}


def test_similares_por_id_excluye_la_propia_imagen(client, database):
    image_results(database)
    database.imagenes.find_one.return_value = {"_id": IMG_1, "embedding": [0.2] * 512}
    response = client.post("/imagenes/similares", json={"imagen_id": str(IMG_1), "limit": 3})
    assert response.status_code == 200
    pipeline = database.imagenes.aggregate.call_args.args[0]
    assert pipeline[0]["$vectorSearch"]["limit"] == 4  # uno extra para descartar la propia imagen
    assert pipeline[1] == {"$match": {"_id": {"$ne": IMG_1}}}
    assert pipeline[2] == {"$limit": 3}


def test_similares_errores_de_id(client, database):
    assert client.post("/imagenes/similares", json={"imagen_id": "123"}).status_code == 422
    database.imagenes.find_one.return_value = None
    assert client.post("/imagenes/similares", json={"imagen_id": str(IMG_1)}).status_code == 404
    database.imagenes.find_one.return_value = {"_id": IMG_1}
    assert client.post("/imagenes/similares", json={"imagen_id": str(IMG_1)}).status_code == 409


def test_similares_valida_la_fuente(client):
    assert client.post("/imagenes/similares", json={}).status_code == 422
    both = {"imagen_id": str(IMG_1), "url": "https://x/y.jpg"}
    assert client.post("/imagenes/similares", json=both).status_code == 422
    assert client.post("/imagenes/similares", json={"url": "/etc/passwd"}).status_code == 422


def test_similares_por_url(client, database):
    image_results(database)
    with mock.patch.object(main, "load_image", return_value="img"), mock.patch.object(
        main, "embed_images", return_value=[[0.3] * 512]
    ):
        response = client.post("/imagenes/similares", json={"url": "https://x/y.jpg"})
    assert response.status_code == 200
    assert "$match" not in str(database.imagenes.aggregate.call_args.args[0][1])

    with mock.patch.object(main, "load_image", side_effect=ImageLoadError("404")):
        assert client.post("/imagenes/similares", json={"url": "https://x/rota.jpg"}).status_code == 422


def test_buscar_imagenes_sin_indice_responde_503(client, database):
    database.imagenes.aggregate.side_effect = OperationFailure("index not found")
    with mock.patch.object(main, "embed_image_query", return_value=[0.1] * 512):
        response = client.post("/imagenes/buscar", json={"texto": "paneles"})
    assert response.status_code == 503
    assert "imagenes_vector_index" in response.json()["detail"]


def test_rag_incluye_imagenes_de_los_articulos_fuente(client, database):
    con_imagen = chunk(ART_2, "Dos", 0.9)
    con_imagen["articulo"]["imagenes_relacionadas"] = [IMG_1]
    database.chunks.aggregate.return_value = [con_imagen]
    database.imagenes.find.return_value = [{"_id": IMG_1, "url": "data/imagenes/panel.jpg", "categoria": "solar"}]
    with mock.patch.object(main, "generate_answer", return_value="Respuesta [1]"):
        response = client.post("/rag", json={"question": "¿Paneles?"})
    assert response.status_code == 200
    assert response.json()["imagenes"] == [
        {"_id": str(IMG_1), "url": "data/imagenes/panel.jpg", "categoria": "solar"}
    ]
    assert database.imagenes.find.call_args.args == ({"_id": {"$in": [IMG_1]}}, {"embedding": 0})
