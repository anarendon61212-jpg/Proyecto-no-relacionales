"""Crea colecciones, validadores e índices requeridos para la Entrega 1."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import ASCENDING, MongoClient
from pymongo.errors import OperationFailure, PyMongoError
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
                "idioma": {"bsonType": "string", "minLength": 2},
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


UNAUTHORIZED_CODE = 13  # Código de MongoDB para operaciones sin permiso.


def vector_fields_key(definition: dict) -> list:
    """Normaliza los campos de un índice vectorial para compararlos sin importar el orden."""
    return sorted(
        json.dumps(field, sort_keys=True) for field in definition.get("fields", [])
    )


TEXT_INDEX_NAME = "articulos_text_idx"
TEXT_INDEX_LANGUAGE = "spanish"  # Stemming y stop words en español para $text.


def ensure_text_index(collection) -> None:
    """Crea el índice de texto en español; si existía con otro idioma, lo recrea."""
    existing = collection.index_information().get(TEXT_INDEX_NAME)
    if existing is not None and existing.get("default_language") != TEXT_INDEX_LANGUAGE:
        collection.drop_index(TEXT_INDEX_NAME)
        print(f"Índice de texto recreado con idioma '{TEXT_INDEX_LANGUAGE}'.")
    collection.create_index(
        [("titulo", "text"), ("contenido_texto", "text")],
        name=TEXT_INDEX_NAME,
        default_language=TEXT_INDEX_LANGUAGE,
    )


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
                try:
                    database.command(
                        {
                            "collMod": name,
                            "validator": validator,
                            "validationLevel": "strict",
                            "validationAction": "error",
                        }
                    )
                except OperationFailure as exc:
                    if exc.code != UNAUTHORIZED_CODE:
                        raise
                    print(f"Sin permiso para actualizar el validador de '{name}' (collMod).")
                    print(
                        "  Asigna al usuario de Atlas el rol 'Atlas admin' o 'dbAdmin' sobre "
                        f"'{settings['database_name']}' y vuelve a ejecutar el script."
                    )
                    print("  Se continúa con el validador que la colección ya tenía.")
                    continue
            print(f"Colección lista: {name}")

        database.articulos.create_index(
            [("fecha", ASCENDING), ("idioma", ASCENDING)],
            name="fecha_idioma_idx",
        )
        ensure_text_index(database.articulos)
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
                {"type": "filter", "path": "idioma"},
            ]
        }
        try:
            existing_vector_indexes = list(
                database.chunks.list_search_indexes(settings["vector_index_name"])
            )
            if existing_vector_indexes:
                current = existing_vector_indexes[0].get("latestDefinition", {})
                if vector_fields_key(current) != vector_fields_key(vector_definition):
                    database.chunks.update_search_index(
                        settings["vector_index_name"], vector_definition
                    )
                    print(f"Índice vectorial actualizado: {settings['vector_index_name']}")
                else:
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
