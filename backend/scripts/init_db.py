"""Crea colecciones, validadores e índices requeridos para la Entrega 1."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from app.config import get_settings


VECTOR_DIMENSIONS = 384  # all-MiniLM-L6-v2, según el modelo sugerido en la guía.

VALIDATORS = {
    "articulos": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["titulo", "contenido_texto", "idioma", "fecha", "fuente"],
            "properties": {
                "titulo": {"bsonType": "string", "minLength": 1},
                "contenido_texto": {"bsonType": "string", "minLength": 1},
                "idioma": {"bsonType": "string", "minLength": 2},
                "fecha": {"bsonType": "date"},
                "categoria": {"bsonType": "string"},
                "fuente": {"bsonType": "string", "minLength": 1},
                "imagenes_relacionadas": {
                    "bsonType": "array",
                    "items": {"bsonType": "objectId"},
                },
            },
        }
    },
    "imagenes": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["url", "categoria"],
            "properties": {
                "url": {"bsonType": "string", "minLength": 1},
                "categoria": {"bsonType": "string", "minLength": 1},
                "descripcion": {"bsonType": "string"},
                "fuente": {"bsonType": "string"},
                "embedding": {"bsonType": "array", "items": {"bsonType": "double"}},
            },
        }
    },
    "chunks": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "doc_id",
                "chunk_index",
                "estrategia_chunking",
                "chunk_texto",
                "embedding",
                "modelo",
                "fecha_ingesta",
            ],
            "properties": {
                "doc_id": {"bsonType": "objectId"},
                "chunk_index": {"bsonType": "int", "minimum": 0},
                "estrategia_chunking": {
                    "enum": ["fixed-size", "sentence-aware", "semantic"]
                },
                "chunk_texto": {"bsonType": "string", "minLength": 1},
                "embedding": {
                    "bsonType": "array",
                    "minItems": VECTOR_DIMENSIONS,
                    "maxItems": VECTOR_DIMENSIONS,
                    "items": {"bsonType": "double"},
                },
                "modelo": {"bsonType": "string"},
                "fecha_ingesta": {"bsonType": "date"},
            },
        }
    },
    "usuarios": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["usuario_id", "historial"],
            "properties": {
                "usuario_id": {"bsonType": "string", "minLength": 1},
                "historial": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["consulta", "fecha"],
                        "properties": {
                            "consulta": {"bsonType": "string"},
                            "fecha": {"bsonType": "date"},
                        },
                    },
                },
            },
        }
    },
}


def main() -> None:
    settings = get_settings()
    client = MongoClient(settings["mongodb_uri"], serverSelectionTimeoutMS=5000)
    database = client[settings["database_name"]]
    try:
        client.admin.command("ping")
        existing = set(database.list_collection_names())
        for name, validator in VALIDATORS.items():
            if name not in existing:
                database.create_collection(
                    name,
                    validator=validator,
                    validationLevel="strict",
                    validationAction="error",
                )
            else:
                database.command(
                    {
                        "collMod": name,
                        "validator": validator,
                        "validationLevel": "strict",
                        "validationAction": "error",
                    }
                )
            print(f"Colección lista: {name}")

        database.articulos.create_index(
            [("fecha", ASCENDING), ("idioma", ASCENDING)],
            name="fecha_idioma_idx",
        )
        database.articulos.create_index(
            [("titulo", "text"), ("contenido_texto", "text")],
            name="articulos_text_idx",
        )
        database.chunks.create_index(
            [("doc_id", ASCENDING), ("estrategia_chunking", ASCENDING), ("chunk_index", ASCENDING)],
            name="chunks_documento_estrategia_idx",
        )
        database.usuarios.create_index([("usuario_id", ASCENDING)], unique=True, name="usuario_id_unique_idx")
        print("Índices convencionales listos.")

        vector_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": VECTOR_DIMENSIONS,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "estrategia_chunking"},
            ]
        }
        try:
            existing_vector_indexes = list(
                database.chunks.list_search_indexes(settings["vector_index_name"])
            )
            if existing_vector_indexes:
                print(f"Índice vectorial ya existente: {settings['vector_index_name']}")
            else:
                search_index = SearchIndexModel(
                    definition=vector_definition,
                    name=settings["vector_index_name"],
                    type="vectorSearch",
                )
                database.chunks.create_search_index(model=search_index)
                print(f"Índice vectorial solicitado: {settings['vector_index_name']}")
        except PyMongoError as exc:
            print("No se pudo crear el índice vectorial desde el driver:", exc)
            print("Si el cluster es Atlas M0, créalo desde Atlas > Search & Vector Search.")
            print(f"Nombre del índice: {settings['vector_index_name']}")
            print("Pega esta definición en el editor JSON de Vector Search:")
            print(json.dumps(vector_definition, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
