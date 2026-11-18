import csv
import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comparar_estrategias as comp  # noqa: E402
from conftest import ART_1, ART_2, chunk  # noqa: E402


def test_fecha_hasta_incluye_todo_el_dia():
    assert comp.parse_date("2024-12-31", end_of_day=True).hour == 23
    assert comp.parse_date("2024-12-31").hour == 0
    assert comp.parse_date(None) is None


def test_plantilla_tiene_diez_consultas_validas():
    queries = comp.load_queries(comp.DEFAULT_QUERIES)
    assert len(queries) == 10 and len({q["id"] for q in queries}) == 10
    assert comp.query_filters(queries[1]) == {
        "articulo.idioma": "en",
        "articulo.fecha": {
            "$gte": comp.parse_date("2024-01-01"),
            "$lte": comp.parse_date("2024-12-31", end_of_day=True),
        },
    }


def test_main_compara_con_las_mismas_condiciones(tmp_path):
    queries = tmp_path / "consultas.json"
    queries.write_text(json.dumps({"consultas": [
        {"id": "C01", "consulta": "agua", "filtros": {"idioma": "es"}, "relevantes": ["Uno"]},
        {"id": "C02", "consulta": "nada", "filtros": {}},
    ]}), encoding="utf-8")

    database = mock.MagicMock()
    stats = [{"_id": "sentence-aware", "chunks": 10, "articulos": [ART_1, ART_2],
              "palabras_promedio": 80.0, "palabras_min": 20, "palabras_max": 150}]

    def aggregate(pipeline):
        if "$group" in pipeline[-1]:
            return stats
        query_is_empty = pipeline[0]["$vectorSearch"]["queryVector"] == [0.0] * 384
        return [] if query_is_empty else [chunk(ART_1, "Uno", 0.9), chunk(ART_2, "Dos", 0.7)]

    database.chunks.aggregate.side_effect = aggregate
    client = mock.MagicMock()
    client.__getitem__.return_value = database
    vectors = {"agua": [0.1] * 384, "nada": [0.0] * 384}

    with mock.patch.object(comp, "MongoClient", return_value=client), mock.patch.object(
        comp, "embed_query", side_effect=lambda text, model: vectors[text]
    ) as embed:
        code = comp.main(["--consultas", str(queries), "--limite", "2", "--salida", str(tmp_path / "out")])

    assert code == 0
    assert embed.call_count == 2  # un embedding por consulta, compartido por ambas estrategias
    search_calls = [c.args[0] for c in database.chunks.aggregate.call_args_list if "$vectorSearch" in c.args[0][0]]
    assert len(search_calls) == 4
    assert {c[0]["$vectorSearch"]["filter"]["estrategia_chunking"]["$eq"] for c in search_calls} == {
        "sentence-aware", "semantic"
    }

    output = sorted((tmp_path / "out").iterdir())
    data = json.loads(next(p for p in output if p.suffix == ".json").read_text(encoding="utf-8"))
    sa = data["resultados"][0]["por_estrategia"]["sentence-aware"]
    assert sa["precision_en_k"] == 0.5 and sa["articulos_relevantes_encontrados"] == "1/1"
    assert "Sin evidencia" in data["resultados"][1]["por_estrategia"]["semantic"]["nota"]

    comparacion = next(p for p in output if p.name.startswith("comparacion_") and p.suffix == ".csv")
    rows = list(csv.DictReader(comparacion.open(encoding="utf-8-sig")))
    assert {"relevancia", "cobertura", "precision"} <= set(rows[0])
    assert len(rows) == 2 * 2 + 2  # C01: 2 chunks x 2 estrategias; C02: una fila "sin evidencia" por estrategia

    estadisticas = next(p for p in output if p.name.startswith("estadisticas_"))
    stat_rows = list(csv.DictReader(estadisticas.open(encoding="utf-8-sig")))
    assert stat_rows[0]["chunks_por_articulo"] == "5.0"
    assert stat_rows[1]["estrategia"] == "semantic" and stat_rows[1]["chunks"] == "0"
