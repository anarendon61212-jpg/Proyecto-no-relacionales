"""API del sistema RAG: búsqueda textual, vectorial e híbrida, y preguntas con Gemini."""

from datetime import datetime, timezone
from typing import Literal

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from pymongo.errors import PyMongoError

from app.config import get_settings
from app.db import get_database, lifespan
from app.embeddings import (
    EmbeddingError,
    ImageLoadError,
    embed_image_query,
    embed_images,
    embed_query,
    load_image,
)
from app.llm import LLMError, generate_answer
from app.retrieval import (
    article_filters,
    group_chunks_by_article,
    image_search_pipeline,
    reciprocal_rank_fusion,
    related_images,
    text_search,
    vector_search_chunks,
)

HISTORIAL_MAXIMO = 20  # Consultas recientes que se guardan embebidas por usuario.

app = FastAPI(
    title="API RAG NoSQL con MongoDB",
    description=(
        "Búsqueda textual, vectorial e híbrida sobre artículos, búsqueda de imágenes "
        "por texto o por similitud visual (CLIP) y respuestas RAG generadas con Google Gemini."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

EstrategiaChunking = Literal["fixed-size", "sentence-aware", "semantic"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    modo: Literal["texto", "vector", "hibrido"] = Field(
        default="hibrido",
        description="texto: índice $text; vector: Atlas Vector Search; hibrido: ambos combinados con RRF.",
    )
    idioma: str | None = Field(default=None, min_length=2, max_length=10)
    fecha_desde: datetime | None = None
    fecha_hasta: datetime | None = None
    estrategia_chunking: EstrategiaChunking | None = Field(
        default=None, description="Solo aplica a los modos vector e hibrido."
    )
    limit: int = Field(default=10, ge=1, le=50)


class RagRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    idioma: str | None = Field(default=None, min_length=2, max_length=10)
    fecha_desde: datetime | None = None
    fecha_hasta: datetime | None = None
    estrategia_chunking: EstrategiaChunking = Field(
        default="sentence-aware",
        description=(
            "Estrategia cuyos chunks se usan como contexto. Se usa una sola para no enviar a "
            "Gemini fragmentos repetidos de varias estrategias."
        ),
    )
    limit: int = Field(default=5, ge=1, le=20, description="Chunks que se envían como contexto.")
    usuario_id: str | None = Field(
        default=None, min_length=1, max_length=100,
        description="Si se envía, la pregunta se guarda en el historial del usuario.",
    )


class ImageTextSearchRequest(BaseModel):
    texto: str = Field(min_length=2, max_length=300, description="Descripción de lo que se busca, en español o inglés.")
    categoria: str | None = Field(default=None, max_length=60)
    limit: int = Field(default=8, ge=1, le=50)


class SimilarImagesRequest(BaseModel):
    imagen_id: str | None = Field(default=None, description="_id de una imagen ya guardada.")
    url: str | None = Field(
        default=None, max_length=2000, description="URL http(s) de una imagen externa."
    )
    categoria: str | None = Field(default=None, max_length=60)
    limit: int = Field(default=8, ge=1, le=50)

    @model_validator(mode="after")
    def exactly_one_source(self):
        if bool(self.imagen_id) == bool(self.url):
            raise ValueError("Envía imagen_id o url (solo uno de los dos).")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError("La url debe empezar por http:// o https://.")
        return self


def serialize_document(document):
    """Convierte tipos BSON comunes a valores serializables como JSON."""
    if isinstance(document, ObjectId):
        return str(document)
    if isinstance(document, datetime):
        return document.isoformat()
    if isinstance(document, list):
        return [serialize_document(item) for item in document]
    if isinstance(document, dict):
        return {key: serialize_document(item) for key, item in document.items()}
    return document


def query_embedding(text: str) -> list[float]:
    try:
        return embed_query(text, get_settings()["embedding_model"])
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def mongo_unavailable(exc: PyMongoError, vector: bool) -> HTTPException:
    detail = "No se pudo consultar MongoDB. Revisa la conexión y los índices inicializados."
    if vector:
        detail += (
            " Para búsquedas vectoriales, confirma en Atlas que el índice "
            f"'{get_settings()['vector_index_name']}' existe y está en estado READY."
        )
    return HTTPException(status_code=503, detail=detail)


@app.get("/health")
def health(database=Depends(get_database)):
    try:
        database.command("ping")
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB no está disponible.") from exc
    return {"status": "ok", "database": database.name}


@app.post("/search")
def search(payload: SearchRequest, database=Depends(get_database)):
    """Busca artículos por texto, por similitud semántica o combinando ambos (RRF)."""
    query = payload.query.strip()
    filters = article_filters(payload.idioma, payload.fecha_desde, payload.fecha_hasta)
    use_text = payload.modo in ("texto", "hibrido")
    use_vector = payload.modo in ("vector", "hibrido")
    # En modo híbrido se piden más candidatos a cada lado para que la fusión tenga margen.
    per_source = payload.limit * 2 if payload.modo == "hibrido" else payload.limit

    query_vector = query_embedding(query) if use_vector else None
    try:
        text_results = text_search(database, query, filters, per_source) if use_text else []
        vector_results = []
        if use_vector:
            chunks = vector_search_chunks(
                database,
                get_settings()["vector_index_name"],
                query_vector,
                limit=min(per_source * 5, 200),
                estrategia=payload.estrategia_chunking,
                filters=article_filters(
                    payload.idioma, payload.fecha_desde, payload.fecha_hasta, prefix="articulo."
                ),
            )
            vector_results = group_chunks_by_article(chunks, per_source)
    except PyMongoError as exc:
        raise mongo_unavailable(exc, vector=use_vector) from exc

    if payload.modo == "texto":
        results = text_results
    elif payload.modo == "vector":
        results = vector_results
    else:
        results = reciprocal_rank_fusion(text_results, vector_results, payload.limit)

    return {
        "query": payload.query,
        "modo": payload.modo,
        "count": len(results),
        "results": serialize_document(results),
    }


@app.post("/rag")
def rag(payload: RagRequest, database=Depends(get_database)):
    """Recupera los chunks más relevantes y genera una respuesta citada con Gemini."""
    settings = get_settings()
    question = payload.question.strip()
    query_vector = query_embedding(question)
    try:
        chunks = vector_search_chunks(
            database,
            settings["vector_index_name"],
            query_vector,
            limit=payload.limit,
            estrategia=payload.estrategia_chunking,
            filters=article_filters(
                payload.idioma, payload.fecha_desde, payload.fecha_hasta, prefix="articulo."
            ),
        )
    except PyMongoError as exc:
        raise mongo_unavailable(exc, vector=True) from exc

    if chunks:
        try:
            answer = generate_answer(
                question, chunks, settings["gemini_api_key"], settings["gemini_model"]
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:
        answer = (
            "No encontré fragmentos relevantes con los filtros indicados, "
            "así que no puedo responder con base en los artículos."
        )

    try:
        imagenes = related_images(database, chunks)
    except PyMongoError:
        imagenes = []  # Las imágenes complementan la respuesta; no deben impedirla.

    fuentes = [
        {
            "numero": number,
            "doc_id": chunk["doc_id"],
            "titulo": chunk["articulo"].get("titulo"),
            "fuente": chunk["articulo"].get("fuente"),
            "fecha": chunk["articulo"].get("fecha"),
            "idioma": chunk["articulo"].get("idioma"),
            "chunk_index": chunk.get("chunk_index"),
            "estrategia_chunking": chunk.get("estrategia_chunking"),
            "score": chunk.get("score"),
            "fragmento": chunk.get("chunk_texto"),
        }
        for number, chunk in enumerate(chunks, start=1)
    ]

    if payload.usuario_id:
        try:
            database.usuarios.update_one(
                {"usuario_id": payload.usuario_id},
                {
                    "$push": {
                        "historial": {
                            "$each": [{"consulta": question, "fecha": datetime.now(timezone.utc)}],
                            "$slice": -HISTORIAL_MAXIMO,
                        }
                    }
                },
                upsert=True,
            )
        except PyMongoError:
            pass  # El historial es secundario: no debe impedir entregar la respuesta.

    return serialize_document(
        {
            "question": payload.question,
            "respuesta": answer,
            "modelo": settings["gemini_model"] if chunks else None,
            "fuentes": fuentes,
            "imagenes": imagenes,
        }
    )


def run_image_search(database, query_vector, payload, exclude_id=None) -> dict:
    pipeline = image_search_pipeline(
        get_settings()["image_vector_index_name"],
        query_vector,
        payload.limit,
        categoria=payload.categoria,
        exclude_id=exclude_id,
    )
    try:
        results = list(database.imagenes.aggregate(pipeline))
    except PyMongoError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "No se pudo consultar las imágenes. Confirma en Atlas que el índice "
                f"'{get_settings()['image_vector_index_name']}' existe y está en estado READY."
            ),
        ) from exc
    return {"count": len(results), "results": serialize_document(results)}


@app.post("/imagenes/buscar")
def search_images_by_text(payload: ImageTextSearchRequest, database=Depends(get_database)):
    """Busca imágenes que coincidan con una descripción en texto (CLIP multilingüe)."""
    try:
        vector = embed_image_query(payload.texto.strip(), get_settings()["image_text_model"])
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"texto": payload.texto, **run_image_search(database, vector, payload)}


@app.post("/imagenes/similares")
def similar_images(payload: SimilarImagesRequest, database=Depends(get_database)):
    """Busca imágenes visualmente parecidas a una imagen guardada o a una URL."""
    exclude_id = None
    if payload.imagen_id:
        try:
            exclude_id = ObjectId(payload.imagen_id)
        except InvalidId as exc:
            raise HTTPException(status_code=422, detail="imagen_id no es un ObjectId válido.") from exc
        try:
            image = database.imagenes.find_one({"_id": exclude_id}, {"embedding": 1})
        except PyMongoError as exc:
            raise mongo_unavailable(exc, vector=False) from exc
        if image is None:
            raise HTTPException(status_code=404, detail="No existe una imagen con ese imagen_id.")
        if not image.get("embedding"):
            raise HTTPException(
                status_code=409,
                detail="La imagen aún no tiene embedding. Ejecuta la ingesta sin --sin-imagenes.",
            )
        vector = image["embedding"]
    else:
        try:
            vector = embed_images([load_image(payload.url)], get_settings()["image_model"])[0]
        except ImageLoadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return run_image_search(database, vector, payload, exclude_id=exclude_id)


@app.get("/")
def root():
    return {"name": "API RAG NoSQL con MongoDB", "docs": "/docs", "health": "/health"}
