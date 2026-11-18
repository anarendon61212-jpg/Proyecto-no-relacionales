"""Ingesta: carga artículos, los divide en chunks con cada estrategia y guarda sus embeddings.

Ejemplos (desde la raíz del repositorio):
    python backend/scripts/ingest.py data/articulos_ejemplo.json
    python backend/scripts/ingest.py data/articulos.jsonl --estrategias fixed-size semantic
    python backend/scripts/ingest.py --desde-mongo          # re-procesa los artículos ya cargados

Es idempotente: un artículo se identifica por (titulo, fuente) y se actualiza si ya existe;
sus chunks de cada estrategia se reemplazan por completo en cada ejecución.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from app.chunking import ESTRATEGIAS, chunk_text
from app.config import get_settings
from app.embeddings import EmbeddingError, embed_texts

REQUIRED_FIELDS = ("titulo", "contenido_texto", "idioma", "fecha", "fuente")
OPTIONAL_FIELDS = ("categoria",)


def parse_fecha(value) -> datetime:
    """Acepta '2024-05-01', '2024-05-01T10:00:00' o con zona ('Z', '-05:00'). Sin zona = UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise ValueError(f"fecha inválida: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_articles(path: Path) -> list[dict]:
    """Lee una lista JSON (.json) o un objeto por línea (.jsonl)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("El archivo JSON debe contener una lista de artículos.")
    return data


def normalize_article(raw: dict) -> tuple[dict, list[dict]]:
    """Valida campos obligatorios y separa las imágenes. Lanza ValueError si falta algo."""
    missing = [
        field for field in REQUIRED_FIELDS if not isinstance(raw.get(field), str) or not raw[field].strip()
    ]
    if missing:
        raise ValueError(f"faltan campos obligatorios o están vacíos: {', '.join(missing)}")

    article = {field: raw[field].strip() for field in REQUIRED_FIELDS if field != "fecha"}
    article["fecha"] = parse_fecha(raw["fecha"])
    for field in OPTIONAL_FIELDS:
        if isinstance(raw.get(field), str) and raw[field].strip():
            article[field] = raw[field].strip()

    images = []
    for image in raw.get("imagenes", []) or []:
        if not isinstance(image, dict) or not image.get("url") or not image.get("categoria"):
            raise ValueError("cada imagen necesita al menos 'url' y 'categoria'")
        images.append(
            {key: str(image[key]) for key in ("url", "categoria", "descripcion", "fuente") if image.get(key)}
        )
    return article, images


def upsert_images(database, images: list[dict]) -> list:
    ids = []
    for image in images:
        stored = database.imagenes.find_one_and_update(
            {"url": image["url"]},
            {"$set": image},
            upsert=True,
            projection={"_id": 1},
            return_document=ReturnDocument.AFTER,
        )
        ids.append(stored["_id"])
    return ids


def upsert_article(database, article: dict, image_ids: list) -> object:
    document = dict(article)
    if image_ids:
        document["imagenes_relacionadas"] = image_ids
    stored = database.articulos.find_one_and_update(
        {"titulo": article["titulo"], "fuente": article["fuente"]},
        {"$set": document},
        upsert=True,
        projection={"_id": 1},
        return_document=ReturnDocument.AFTER,
    )
    return stored["_id"]


def build_chunk_documents(
    doc_id, article: dict, estrategia: str, embed, model_name: str, fecha_ingesta: datetime
) -> list[dict]:
    """Chunks de un artículo con una estrategia, listos para insertar en `chunks`."""
    texts = chunk_text(article["contenido_texto"], estrategia, embed=embed)
    vectors = embed(texts)
    return [
        {
            "doc_id": doc_id,
            "chunk_index": index,
            "estrategia_chunking": estrategia,
            "chunk_texto": text,
            "embedding": vector,
            "idioma": article["idioma"],
            "modelo": model_name,
            "fecha_ingesta": fecha_ingesta,
        }
        for index, (text, vector) in enumerate(zip(texts, vectors))
    ]


def replace_chunks(database, doc_id, estrategia: str, documents: list[dict]) -> None:
    database.chunks.delete_many({"doc_id": doc_id, "estrategia_chunking": estrategia})
    if documents:
        database.chunks.insert_many(documents)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivo", nargs="?", type=Path, help="Archivo .json (lista) o .jsonl con artículos.")
    parser.add_argument(
        "--desde-mongo", action="store_true", help="Re-procesa los artículos que ya están en la colección."
    )
    parser.add_argument(
        "--estrategias", nargs="+", choices=ESTRATEGIAS, default=list(ESTRATEGIAS),
        help="Estrategias de chunking a generar (por defecto, las tres).",
    )
    parser.add_argument("--limite", type=int, default=None, help="Procesa solo los primeros N artículos.")
    args = parser.parse_args(argv)
    if bool(args.archivo) == args.desde_mongo:
        parser.error("indica un archivo de artículos o --desde-mongo (solo uno de los dos).")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    model_name = settings["embedding_model"]

    def embed(texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, model_name)

    client = MongoClient(settings["mongodb_uri"], serverSelectionTimeoutMS=5000)
    database = client[settings["database_name"]]
    fecha_ingesta = datetime.now(timezone.utc)
    procesados, fallidos, total_chunks = 0, 0, 0
    try:
        client.admin.command("ping")
        if args.desde_mongo:
            raw_articles = list(database.articulos.find({}, {"imagenes_relacionadas": 0}))
        else:
            raw_articles = load_articles(args.archivo)
        if args.limite is not None:
            raw_articles = raw_articles[: args.limite]
        print(f"Artículos a procesar: {len(raw_articles)} | estrategias: {', '.join(args.estrategias)}")
        print(f"Modelo de embeddings: {model_name} (la primera vez se descarga)")

        for position, raw in enumerate(raw_articles, start=1):
            etiqueta = f"[{position}/{len(raw_articles)}] {str(raw.get('titulo', '(sin título)'))[:60]}"
            try:
                if args.desde_mongo:
                    article, doc_id = raw, raw["_id"]
                else:
                    article, images = normalize_article(raw)
                    doc_id = upsert_article(database, article, upsert_images(database, images))
                counts = []
                for estrategia in args.estrategias:
                    documents = build_chunk_documents(
                        doc_id, article, estrategia, embed, model_name, fecha_ingesta
                    )
                    replace_chunks(database, doc_id, estrategia, documents)
                    counts.append(f"{estrategia}={len(documents)}")
                    total_chunks += len(documents)
                procesados += 1
                print(f"{etiqueta}: {', '.join(counts)}")
            except (ValueError, PyMongoError) as exc:
                # Un artículo inválido (o rechazado por el validador) no detiene la ingesta.
                fallidos += 1
                print(f"{etiqueta}: OMITIDO ({exc})")
    except EmbeddingError as exc:
        print(f"Error con el modelo de embeddings: {exc}")
        return 1
    except PyMongoError as exc:
        print(f"No se pudo conectar o leer MongoDB: {exc}")
        print("Revisa MONGODB_URI en .env y que tu IP esté autorizada en Atlas.")
        return 1
    finally:
        client.close()

    print(f"\nListo: {procesados} artículos procesados, {fallidos} omitidos, {total_chunks} chunks guardados.")
    return 1 if fallidos and not procesados else 0


if __name__ == "__main__":
    sys.exit(main())
