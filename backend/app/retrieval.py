"""Consultas de recuperación: texto, vectorial (Atlas Vector Search) e híbrida."""

from datetime import datetime

RRF_K = 60  # Constante habitual de Reciprocal Rank Fusion.
MAX_CHUNK_CANDIDATES = 200


def article_filters(
    idioma: str | None, fecha_desde: datetime | None, fecha_hasta: datetime | None, prefix: str = ""
) -> dict:
    """Filtros de idioma y fecha sobre artículos (prefix='articulo.' tras un $lookup)."""
    filters: dict = {}
    if idioma:
        filters[f"{prefix}idioma"] = idioma
    date_filter = {}
    if fecha_desde:
        date_filter["$gte"] = fecha_desde
    if fecha_hasta:
        date_filter["$lte"] = fecha_hasta
    if date_filter:
        filters[f"{prefix}fecha"] = date_filter
    return filters


def text_search(database, query: str, filters: dict, limit: int) -> list[dict]:
    """Búsqueda $text sobre artículos, ordenada por relevancia textual."""
    cursor = (
        database.articulos.find(
            {"$text": {"$search": query}, **filters},
            {"score": {"$meta": "textScore"}},
        )
        .sort([("score", {"$meta": "textScore"})])
        .limit(limit)
    )
    return list(cursor)


def chunk_search_pipeline(
    index_name: str,
    query_vector: list[float],
    candidates: int,
    estrategia: str | None,
    filters: dict,
) -> list[dict]:
    """$vectorSearch sobre chunks + datos del artículo de origen.

    La estrategia se filtra dentro de $vectorSearch (es un campo filter del índice).
    Idioma y fechas se filtran después sobre el artículo, porque ahí son obligatorios;
    por eso se piden más candidatos de los que se devuelven.
    """
    vector_stage: dict = {
        "index": index_name,
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": min(max(candidates * 10, 100), 10_000),
        "limit": candidates,
    }
    if estrategia:
        vector_stage["filter"] = {"estrategia_chunking": {"$eq": estrategia}}

    pipeline: list[dict] = [
        {"$vectorSearch": vector_stage},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
        {
            "$lookup": {
                "from": "articulos",
                "localField": "doc_id",
                "foreignField": "_id",
                "as": "articulo",
                "pipeline": [{"$project": {"contenido_texto": 0}}],
            }
        },
        {"$unwind": "$articulo"},
    ]
    if filters:
        pipeline.append({"$match": filters})
    return pipeline


def vector_search_chunks(
    database,
    index_name: str,
    query_vector: list[float],
    limit: int,
    estrategia: str | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Devuelve hasta `limit` chunks ordenados por similitud, con su artículo."""
    candidates = min(limit * 10, MAX_CHUNK_CANDIDATES) if filters else limit
    pipeline = chunk_search_pipeline(index_name, query_vector, candidates, estrategia, filters or {})
    return list(database.chunks.aggregate(pipeline))[:limit]


def group_chunks_by_article(chunks: list[dict], limit: int) -> list[dict]:
    """Un resultado por artículo, con el chunk más parecido como fragmento."""
    articles: dict = {}
    for chunk in chunks:  # Ya vienen ordenados por score descendente.
        doc_id = chunk["doc_id"]
        if doc_id in articles:
            continue
        articulo = dict(chunk["articulo"])
        articulo["score"] = chunk["score"]
        articulo["fragmento"] = {
            "chunk_index": chunk.get("chunk_index"),
            "estrategia_chunking": chunk.get("estrategia_chunking"),
            "texto": chunk.get("chunk_texto"),
        }
        articles[doc_id] = articulo
        if len(articles) == limit:
            break
    return list(articles.values())


def reciprocal_rank_fusion(text_results: list[dict], vector_results: list[dict], limit: int) -> list[dict]:
    """Combina rankings de texto y vectorial con RRF: score = Σ 1 / (k + posición)."""
    merged: dict = {}
    for source, results in (("texto", text_results), ("vector", vector_results)):
        for rank, document in enumerate(results, start=1):
            entry = merged.setdefault(document["_id"], {"document": {}, "scores": {}, "rrf": 0.0})
            # El resultado textual trae el contenido completo; el vectorial, el fragmento.
            entry["document"] = {**document, **entry["document"]}
            entry["scores"][source] = document.get("score")
            entry["rrf"] += 1 / (RRF_K + rank)

    ranked = sorted(merged.values(), key=lambda entry: entry["rrf"], reverse=True)[:limit]
    results = []
    for entry in ranked:
        document = entry["document"]
        document["score"] = round(entry["rrf"], 6)
        document["score_texto"] = entry["scores"].get("texto")
        document["score_vector"] = entry["scores"].get("vector")
        results.append(document)
    return results
