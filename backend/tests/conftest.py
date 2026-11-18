"""Configuración común: las pruebas no necesitan Atlas, el modelo de embeddings ni Gemini."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest
from bson import ObjectId

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ["GEMINI_API_KEY"] = "clave-de-prueba"

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.db import get_database  # noqa: E402

ART_1 = ObjectId()
ART_2 = ObjectId()
ART_3 = ObjectId()
FECHA = datetime(2024, 5, 1, tzinfo=timezone.utc)


def articulo(_id, titulo, score=None):
    doc = {"_id": _id, "titulo": titulo, "idioma": "es", "fecha": FECHA, "fuente": "Fuente"}
    if score is not None:
        doc["score"] = score
    return doc


def chunk(doc_id, titulo, score, index=0):
    return {
        "_id": ObjectId(),
        "doc_id": doc_id,
        "chunk_index": index,
        "estrategia_chunking": "fixed-size",
        "chunk_texto": f"Texto del chunk {index} de {titulo}",
        "score": score,
        "articulo": articulo(doc_id, titulo),
    }


@pytest.fixture
def database():
    db = mock.MagicMock()
    db.name = "rag_nosql"
    # Texto: artículos 1 y 2.  Vector: chunks de los artículos 2, 2 (repetido) y 3.
    db.articulos.find.return_value.sort.return_value.limit.return_value = [
        articulo(ART_1, "Uno", 5.0),
        articulo(ART_2, "Dos", 3.0),
    ]
    db.chunks.aggregate.return_value = [
        chunk(ART_2, "Dos", 0.91, 0),
        chunk(ART_2, "Dos", 0.88, 1),
        chunk(ART_3, "Tres", 0.80, 0),
    ]
    return db


@pytest.fixture
def client(database):
    main.app.dependency_overrides[get_database] = lambda: database
    with mock.patch.object(main, "embed_query", return_value=[0.1] * 384):
        yield TestClient(main.app)
    main.app.dependency_overrides.clear()
