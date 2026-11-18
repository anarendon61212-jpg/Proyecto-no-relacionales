"""Comparación controlada de estrategias de chunking (entregable C5 del plan de la Persona C).

Ejecuta las mismas consultas, con los mismos filtros, límite y modelo, contra los chunks de
cada estrategia y guarda lo recuperado para evaluarlo con la rúbrica acordada con la Persona A.

Ejemplos (desde la raíz del repositorio):
    python backend/scripts/comparar_estrategias.py
    python backend/scripts/comparar_estrategias.py --limite 5 --con-respuesta

Genera en resultados/:
- comparacion_<fecha>.json: consulta, filtros, chunks recuperados por estrategia (y respuesta de
  Gemini con --con-respuesta).
- comparacion_<fecha>.csv: una fila por chunk recuperado, con columnas vacías de la rúbrica
  (relevancia, cobertura, precision) para calificar a mano.
- estadisticas_<fecha>.csv: chunks por estrategia y longitud promedio en palabras.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, time, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.chunking import ESTRATEGIAS_EXPERIMENTO, parametros
from app.config import PROJECT_ROOT, get_settings
from app.embeddings import EmbeddingError, embed_query
from app.llm import LLMError, generate_answer
from app.retrieval import article_filters, vector_search_chunks

DEFAULT_QUERIES = PROJECT_ROOT / "data" / "consultas_experimentales.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "resultados"
RUBRICA = ("relevancia", "cobertura", "precision")


def parse_date(value: str | None, end_of_day: bool = False) -> datetime | None:
    """'2024-12-31' como fecha_hasta incluye todo ese día."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if len(value) == 10 and end_of_day:
        parsed = datetime.combine(parsed.date(), time.max)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_queries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data["consultas"] if isinstance(data, dict) else data
    for query in queries:
        if not query.get("id") or not query.get("consulta"):
            raise ValueError("cada consulta necesita 'id' y 'consulta'")
    return queries


def query_filters(query: dict) -> dict:
    filtros = query.get("filtros") or {}
    return article_filters(
        filtros.get("idioma"),
        parse_date(filtros.get("fecha_desde")),
        parse_date(filtros.get("fecha_hasta"), end_of_day=True),
        prefix="articulo.",
    )


def chunk_statistics(database, estrategias) -> list[dict]:
    """Cantidad de chunks y longitud promedio (en palabras) por estrategia."""
    pipeline = [
        {"$match": {"estrategia_chunking": {"$in": list(estrategias)}}},
        {"$project": {
            "estrategia_chunking": 1,
            "doc_id": 1,
            "palabras": {"$size": {"$split": [{"$trim": {"input": "$chunk_texto"}}, " "]}},
        }},
        {"$group": {
            "_id": "$estrategia_chunking",
            "chunks": {"$sum": 1},
            "articulos": {"$addToSet": "$doc_id"},
            "palabras_promedio": {"$avg": "$palabras"},
            "palabras_min": {"$min": "$palabras"},
            "palabras_max": {"$max": "$palabras"},
        }},
    ]
    stats = {row["_id"]: row for row in database.chunks.aggregate(pipeline)}
    rows = []
    for estrategia in estrategias:
        row = stats.get(estrategia, {})
        chunks = row.get("chunks", 0)
        articulos = len(row.get("articulos", []))
        rows.append({
            "estrategia": estrategia,
            "chunks": chunks,
            "articulos": articulos,
            "chunks_por_articulo": round(chunks / articulos, 2) if articulos else 0,
            "palabras_promedio": round(row.get("palabras_promedio") or 0, 1),
            "palabras_min": row.get("palabras_min", 0),
            "palabras_max": row.get("palabras_max", 0),
            "parametros": json.dumps(parametros(estrategia), ensure_ascii=False),
        })
    return rows


def run_comparison(database, queries, estrategias, limit, settings, with_answer) -> list[dict]:
    results = []
    for query in queries:
        vector = embed_query(query["consulta"], settings["embedding_model"])  # Mismo vector para ambas.
        entry = {**query, "por_estrategia": {}}
        relevantes = {title.casefold() for title in query.get("relevantes", [])}
        for estrategia in estrategias:
            chunks = vector_search_chunks(
                database, settings["vector_index_name"], vector,
                limit=limit, estrategia=estrategia, filters=query_filters(query),
            )
            recuperados = [
                {
                    "rank": rank,
                    "titulo": chunk["articulo"].get("titulo"),
                    "idioma": chunk["articulo"].get("idioma"),
                    "fecha": chunk["articulo"].get("fecha"),
                    "chunk_index": chunk.get("chunk_index"),
                    "score": round(chunk.get("score", 0), 4),
                    "palabras": len(chunk.get("chunk_texto", "").split()),
                    "relevante_referencia": (
                        chunk["articulo"].get("titulo", "").casefold() in relevantes if relevantes else None
                    ),
                    "texto": chunk.get("chunk_texto"),
                }
                for rank, chunk in enumerate(chunks, start=1)
            ]
            outcome = {"recuperados": recuperados}
            if relevantes:
                hits = sum(1 for r in recuperados if r["relevante_referencia"])
                found = {r["titulo"].casefold() for r in recuperados if r["relevante_referencia"]}
                outcome["precision_en_k"] = round(hits / limit, 3)
                outcome["articulos_relevantes_encontrados"] = f"{len(found)}/{len(relevantes)}"
            if not chunks:
                outcome["nota"] = "Sin evidencia: ningún chunk cumple los filtros."
            elif with_answer:
                try:
                    outcome["respuesta"] = generate_answer(
                        query["consulta"], chunks, settings["gemini_api_key"], settings["gemini_model"]
                    )
                except LLMError as exc:
                    outcome["error_respuesta"] = str(exc)
            entry["por_estrategia"][estrategia] = outcome
            print(f"{query['id']} [{estrategia}]: {len(chunks)} chunks", outcome.get("nota", ""))
        results.append(entry)
    return results


def write_outputs(output_dir: Path, results, stats, metadata) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = metadata["fecha"].strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"comparacion_{stamp}.json"
    csv_path = output_dir / f"comparacion_{stamp}.csv"
    stats_path = output_dir / f"estadisticas_{stamp}.csv"

    json_path.write_text(
        json.dumps({**metadata, "estadisticas": stats, "resultados": results},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    columns = ["consulta_id", "consulta", "estrategia", "rank", "titulo", "chunk_index", "score",
               "palabras", "relevante_referencia", "texto", *RUBRICA]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:  # utf-8-sig: Excel lee bien las tildes.
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for entry in results:
            for estrategia, outcome in entry["por_estrategia"].items():
                if not outcome["recuperados"]:
                    writer.writerow({"consulta_id": entry["id"], "consulta": entry["consulta"],
                                     "estrategia": estrategia, "texto": outcome.get("nota", "")})
                for row in outcome["recuperados"]:
                    writer.writerow({"consulta_id": entry["id"], "consulta": entry["consulta"],
                                     "estrategia": estrategia, **row})
    with stats_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    return [json_path, csv_path, stats_path]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--consultas", type=Path, default=DEFAULT_QUERIES, help="JSON con las consultas.")
    parser.add_argument("--limite", type=int, default=5, help="Chunks recuperados por consulta (igual para todas).")
    parser.add_argument("--con-respuesta", action="store_true", help="Genera también la respuesta con Gemini.")
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT, help="Carpeta de resultados.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    queries = load_queries(args.consultas)
    estrategias = ESTRATEGIAS_EXPERIMENTO
    metadata = {
        "fecha": datetime.now(timezone.utc),
        "modelo_embeddings": settings["embedding_model"],
        "indice_vectorial": settings["vector_index_name"],
        "limite": args.limite,
        "estrategias": list(estrategias),
        "modelo_respuesta": settings["gemini_model"] if args.con_respuesta else None,
    }
    client = MongoClient(settings["mongodb_uri"], serverSelectionTimeoutMS=5000)
    try:
        database = client[settings["database_name"]]
        client.admin.command("ping")
        stats = chunk_statistics(database, estrategias)
        results = run_comparison(database, queries, estrategias, args.limite, settings, args.con_respuesta)
    except EmbeddingError as exc:
        print(f"Error con el modelo de embeddings: {exc}")
        return 1
    except PyMongoError as exc:
        print(f"No se pudo consultar MongoDB (¿índice vectorial en estado READY?): {exc}")
        return 1
    finally:
        client.close()

    for path in write_outputs(args.salida, results, stats, metadata):
        print(f"Guardado: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    for row in stats:
        print(f"{row['estrategia']}: {row['chunks']} chunks, {row['palabras_promedio']} palabras en promedio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
