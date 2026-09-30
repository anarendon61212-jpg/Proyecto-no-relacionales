"""API inicial para búsquedas y preguntas RAG."""

from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from app.db import get_database, lifespan

app = FastAPI(
    title="API RAG NoSQL con MongoDB",
    description="Backend inicial para buscar artículos y preparar el flujo RAG.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    idioma: str | None = Field(default=None, min_length=2, max_length=10)
    fecha_desde: datetime | None = None
    fecha_hasta: datetime | None = None
    limit: int = Field(default=10, ge=1, le=50)


class RagRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    idioma: str | None = Field(default=None, min_length=2, max_length=10)
    estrategia_chunking: str | None = Field(default=None, max_length=40)
    limit: int = Field(default=5, ge=1, le=20)


def serialize_document(document: dict) -> dict:
    """Convierte tipos BSON comunes a valores serializables como JSON."""
    from bson import ObjectId

    def convert(value):
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(document)


@app.get("/health")
def health(database=Depends(get_database)):
    try:
        database.command("ping")
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB no está disponible.") from exc
    return {"status": "ok", "database": database.name}


@app.post("/search")
def search(payload: SearchRequest, database=Depends(get_database)):
    """Búsqueda textual inicial; los filtros se combinan con la consulta."""
    filters: dict = {"$text": {"$search": payload.query.strip()}}
    if payload.idioma:
        filters["idioma"] = payload.idioma

    date_filter = {}
    if payload.fecha_desde:
        date_filter["$gte"] = payload.fecha_desde
    if payload.fecha_hasta:
        date_filter["$lte"] = payload.fecha_hasta
    if date_filter:
        filters["fecha"] = date_filter

    try:
        documents = database.articulos.find(
            filters,
            {"score": {"$meta": "textScore"}},
        ).sort([("score", {"$meta": "textScore"})]).limit(payload.limit)
        results = []
        for document in documents:
            item = serialize_document(document)
            results.append(item)
    except PyMongoError as exc:
        raise HTTPException(
            status_code=503,
            detail="No se pudo consultar MongoDB. Revisa la conexión y los índices inicializados.",
        ) from exc

    return {"query": payload.query, "count": len(results), "results": results}


@app.post("/rag")
def rag(payload: RagRequest):
    """Punto de entrada reservado para recuperación de chunks e integración de un LLM."""
    raise HTTPException(
        status_code=501,
        detail=(
            "Endpoint preparado. En la siguiente etapa se conectarán la generación de "
            "embeddings, la recuperación vectorial de chunks y el proveedor LLM."
        ),
    )


@app.get("/")
def root():
    return {"name": "API RAG NoSQL con MongoDB", "docs": "/docs", "health": "/health"}
