from unittest import mock

from pymongo.errors import OperationFailure

from app import main
from app.llm import LLMError, build_prompt
from conftest import ART_1, ART_2, ART_3, chunk


def test_search_texto_no_usa_embeddings(client, database):
    with mock.patch.object(main, "embed_query") as embed:
        response = client.post("/search", json={"query": "agua", "modo": "texto"})
    assert response.status_code == 200
    assert [r["titulo"] for r in response.json()["results"]] == ["Uno", "Dos"]
    embed.assert_not_called()
    database.chunks.aggregate.assert_not_called()


def test_search_vector_agrupa_por_articulo(client, database):
    response = client.post(
        "/search",
        json={"query": "agua", "modo": "vector", "idioma": "es", "estrategia_chunking": "fixed-size"},
    )
    body = response.json()
    assert response.status_code == 200
    assert [r["_id"] for r in body["results"]] == [str(ART_2), str(ART_3)]
    assert body["results"][0]["fragmento"]["chunk_index"] == 0  # el chunk con mejor score

    pipeline = database.chunks.aggregate.call_args.args[0]
    vector_stage = pipeline[0]["$vectorSearch"]
    assert vector_stage["filter"] == {"estrategia_chunking": {"$eq": "fixed-size"}}
    assert vector_stage["numCandidates"] >= vector_stage["limit"]
    assert pipeline[-1] == {"$match": {"articulo.idioma": "es"}}
    assert {"$project": {"embedding": 0}} in pipeline


def test_search_hibrido_combina_con_rrf(client):
    response = client.post("/search", json={"query": "agua"})
    results = response.json()["results"]
    assert response.json()["modo"] == "hibrido"
    # El artículo 2 aparece en ambos rankings, así que debe quedar primero.
    assert results[0]["_id"] == str(ART_2)
    assert results[0]["score_texto"] == 3.0 and results[0]["score_vector"] == 0.91
    assert {r["_id"] for r in results} == {str(ART_1), str(ART_2), str(ART_3)}


def test_search_sin_indice_vectorial_responde_503(client, database):
    database.chunks.aggregate.side_effect = OperationFailure("index not found")
    response = client.post("/search", json={"query": "agua", "modo": "vector"})
    assert response.status_code == 503
    assert "chunks_vector_index" in response.json()["detail"]


def test_rag_responde_con_fuentes_y_guarda_historial(client, database):
    with mock.patch.object(main, "generate_answer", return_value="Respuesta [1]") as llm:
        response = client.post(
            "/rag", json={"question": "¿Qué pasó?", "limit": 2, "usuario_id": "u1"}
        )
    body = response.json()
    assert response.status_code == 200
    assert body["respuesta"] == "Respuesta [1]"
    assert [f["numero"] for f in body["fuentes"]] == [1, 2]
    assert len(llm.call_args.args[1]) == 2  # solo `limit` chunks van como contexto
    vector_stage = database.chunks.aggregate.call_args.args[0][0]["$vectorSearch"]
    assert vector_stage["filter"] == {"estrategia_chunking": {"$eq": "sentence-aware"}}

    update = database.usuarios.update_one.call_args
    assert update.kwargs["upsert"] is True
    assert update.args[1]["$push"]["historial"]["$slice"] == -main.HISTORIAL_MAXIMO


def test_rag_sin_chunks_no_llama_a_gemini(client, database):
    database.chunks.aggregate.return_value = []
    with mock.patch.object(main, "generate_answer") as llm:
        response = client.post("/rag", json={"question": "¿Qué pasó?"})
    assert response.status_code == 200
    assert response.json()["fuentes"] == []
    llm.assert_not_called()


def test_rag_error_de_gemini_responde_502(client):
    with mock.patch.object(main, "generate_answer", side_effect=LLMError("cuota agotada")):
        response = client.post("/rag", json={"question": "¿Qué pasó?"})
    assert response.status_code == 502
    assert "cuota agotada" in response.json()["detail"]


def test_prompt_numera_los_fragmentos():
    prompt = build_prompt("¿Pregunta?", [chunk(ART_1, "Uno", 0.9), chunk(ART_2, "Dos", 0.8, 3)])
    assert "[1] Uno · Fuente · 2024-05-01" in prompt
    assert "[2] Dos" in prompt and "Texto del chunk 3 de Dos" in prompt
    assert prompt.endswith("Pregunta: ¿Pregunta?")
