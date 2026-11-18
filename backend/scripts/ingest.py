"""Ingesta: carga artículos e imágenes, divide los artículos en chunks y genera los embeddings.

Ejemplos (desde la raíz del repositorio):
    python backend/scripts/ingest.py data/articulos_ejemplo.json
    python backend/scripts/ingest.py data/dataset.json --estrategias sentence-aware fixed-size
    python backend/scripts/ingest.py --desde-mongo          # re-procesa los artículos ya cargados

Formatos de entrada:
- .json con una lista de artículos, o con un objeto {"articulos": [...], "imagenes": [...]}.
- .jsonl con un artículo por línea.
Cada artículo puede traer "imagenes" como objetos completos o como claves ("clave") de la
lista "imagenes" del archivo. Las imágenes de esa lista se cargan aunque ningún artículo
las referencie.

Es idempotente: un artículo se identifica por (titulo, fuente) y una imagen por su url;
los chunks de cada estrategia se reemplazan por completo en cada ejecución.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from app.chunking import ESTRATEGIAS, ESTRATEGIAS_EXPERIMENTO, chunk_text, parametros
from app.config import get_settings
from app.embeddings import EmbeddingError, ImageLoadError, embed_images, embed_texts, load_image

REQUIRED_FIELDS = ("titulo", "contenido_texto", "idioma", "fecha", "fuente")
OPTIONAL_FIELDS = ("categoria", "url", "licencia")
IMAGE_REQUIRED_FIELDS = ("url", "categoria")
IMAGE_OPTIONAL_FIELDS = ("clave", "descripcion", "fuente", "licencia", "atribucion")


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


def load_dataset(path: Path) -> tuple[list[dict], list[dict]]:
    """Devuelve (artículos, imágenes del catálogo) desde .json o .jsonl."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()], []
    data = json.loads(text)
    if isinstance(data, list):
        return data, []
    if isinstance(data, dict) and isinstance(data.get("articulos"), list):
        return data["articulos"], list(data.get("imagenes") or [])
    raise ValueError('El JSON debe ser una lista de artículos o un objeto con "articulos" (e "imagenes").')


def load_articles(path: Path) -> list[dict]:
    return load_dataset(path)[0]


def _text_fields(raw: dict, required: tuple, optional: tuple, what: str) -> dict:
    missing = [f for f in required if not isinstance(raw.get(f), str) or not raw[f].strip()]
    if missing:
        raise ValueError(f"{what}: faltan campos obligatorios o están vacíos: {', '.join(missing)}")
    values = {f: raw[f].strip() for f in required}
    for field in optional:
        if isinstance(raw.get(field), str) and raw[field].strip():
            values[field] = raw[field].strip()
    return values


def normalize_image(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("cada imagen debe ser un objeto o la clave de una imagen del catálogo")
    return _text_fields(raw, IMAGE_REQUIRED_FIELDS, IMAGE_OPTIONAL_FIELDS, "imagen")


def normalize_article(raw: dict) -> tuple[dict, list]:
    """Valida el artículo. Devuelve el documento y sus imágenes (objetos o claves del catálogo)."""
    fields = _text_fields(
        raw, tuple(f for f in REQUIRED_FIELDS if f != "fecha"), OPTIONAL_FIELDS, "artículo"
    )
    if raw.get("fecha") in (None, ""):
        raise ValueError("artículo: faltan campos obligatorios o están vacíos: fecha")
    fields["fecha"] = parse_fecha(raw["fecha"])
    images = []
    for image in raw.get("imagenes", []) or []:
        images.append(image.strip() if isinstance(image, str) else normalize_image(image))
    return fields, images


def upsert_image(database, image: dict):
    stored = database.imagenes.find_one_and_update(
        {"url": image["url"]},
        {"$set": image},
        upsert=True,
        projection={"_id": 1},
        return_document=ReturnDocument.AFTER,
    )
    return stored["_id"]


def resolve_images(database, images: list, catalog_ids: dict) -> list:
    """Convierte las imágenes de un artículo en ObjectIds (guardando las que vienen completas)."""
    ids = []
    for image in images:
        if isinstance(image, str):
            if image not in catalog_ids:
                raise ValueError(f"la imagen con clave '{image}' no está en la lista de imágenes")
            ids.append(catalog_ids[image])
        else:
            ids.append(upsert_image(database, image))
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
            "parametros_chunking": parametros(estrategia),
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


def embed_pending_images(database, model_name: str, reprocess: bool) -> tuple[int, list[str]]:
    """Genera el embedding CLIP de las imágenes que no lo tienen (o de todas con `reprocess`)."""
    # $ne también coincide con las imágenes que aún no tienen modelo_embedding.
    query = {} if reprocess else {"modelo_embedding": {"$ne": model_name}}
    done, failed = 0, []
    for image in database.imagenes.find(query, {"url": 1}):
        try:
            vector = embed_images([load_image(image["url"])], model_name)[0]
        except ImageLoadError as exc:
            failed.append(str(exc))
            continue
        database.imagenes.update_one(
            {"_id": image["_id"]}, {"$set": {"embedding": vector, "modelo_embedding": model_name}}
        )
        done += 1
    return done, failed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivo", nargs="?", type=Path, help="Archivo .json o .jsonl con el dataset.")
    parser.add_argument(
        "--desde-mongo", action="store_true", help="Re-procesa los artículos que ya están en la colección."
    )
    parser.add_argument(
        "--estrategias", nargs="+", choices=ESTRATEGIAS, default=list(ESTRATEGIAS_EXPERIMENTO),
        help="Estrategias de chunking a generar (por defecto, sentence-aware y semantic).",
    )
    parser.add_argument("--limite", type=int, default=None, help="Procesa solo los primeros N artículos.")
    parser.add_argument(
        "--sin-imagenes", action="store_true", help="No genera embeddings de imágenes (más rápido)."
    )
    parser.add_argument(
        "--reprocesar-imagenes", action="store_true",
        help="Regenera el embedding de todas las imágenes, no solo de las que no lo tienen.",
    )
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
        catalog_ids: dict = {}
        if args.desde_mongo:
            raw_articles = list(database.articulos.find({}, {"imagenes_relacionadas": 0}))
        else:
            raw_articles, raw_images = load_dataset(args.archivo)
            for position, raw in enumerate(raw_images, start=1):
                try:
                    image = normalize_image(raw)
                    image_id = upsert_image(database, image)
                    if "clave" in image:
                        catalog_ids[image["clave"]] = image_id
                except (ValueError, PyMongoError) as exc:
                    print(f"Imagen {position}: OMITIDA ({exc})")
            if raw_images:
                print(f"Imágenes del catálogo guardadas: {len(raw_images)}")
        if args.limite is not None:
            raw_articles = raw_articles[: args.limite]
        print(f"Artículos a procesar: {len(raw_articles)} | estrategias: {', '.join(args.estrategias)}")
        print(f"Modelo de embeddings: {model_name} (la primera vez se descarga)")

        vistos: set = set()
        for position, raw in enumerate(raw_articles, start=1):
            etiqueta = f"[{position}/{len(raw_articles)}] {str(raw.get('titulo', '(sin título)'))[:60]}"
            try:
                if args.desde_mongo:
                    article, doc_id = raw, raw["_id"]
                else:
                    article, images = normalize_article(raw)
                    key = (article["titulo"].casefold(), article["fuente"].casefold())
                    if key in vistos:
                        raise ValueError("artículo duplicado en el archivo (mismo título y fuente)")
                    vistos.add(key)
                    doc_id = upsert_article(database, article, resolve_images(database, images, catalog_ids))
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

        if not args.sin_imagenes:
            print(f"Generando embeddings de imágenes con {settings['image_model']}...")
            done, failed = embed_pending_images(database, settings["image_model"], args.reprocesar_imagenes)
            print(f"Imágenes con embedding nuevo: {done}")
            for message in failed:
                print(f"  OMITIDA: {message}")
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
