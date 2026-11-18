"""Generación de respuestas con Google Gemini a partir de los chunks recuperados."""

from functools import lru_cache

SYSTEM_INSTRUCTION = (
    "Eres un asistente que responde preguntas usando únicamente los fragmentos de "
    "artículos que se te entregan como contexto. Reglas:\n"
    "- Responde en el mismo idioma de la pregunta.\n"
    "- Usa solo información presente en el contexto; no inventes datos.\n"
    "- Cita los fragmentos que uses con su número entre corchetes, por ejemplo [1] o [2][3].\n"
    "- Si el contexto no contiene la respuesta, dilo claramente en lugar de adivinar."
)


class LLMError(RuntimeError):
    """Gemini no está configurado o la llamada falló."""


@lru_cache(maxsize=1)
def _client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Arma el contexto numerado que Gemini debe citar."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        articulo = chunk.get("articulo", {})
        fecha = articulo.get("fecha")
        fecha_texto = fecha.date().isoformat() if hasattr(fecha, "date") else ""
        encabezado = " · ".join(
            part for part in (articulo.get("titulo"), articulo.get("fuente"), fecha_texto) if part
        )
        blocks.append(f"[{number}] {encabezado}\n{chunk['chunk_texto']}")
    contexto = "\n\n".join(blocks)
    return f"Contexto:\n\n{contexto}\n\nPregunta: {question}"


def generate_answer(question: str, chunks: list[dict], api_key: str, model: str) -> str:
    if not api_key:
        raise LLMError("Falta GEMINI_API_KEY en el archivo .env; /rag la necesita para responder.")

    from google.genai import errors, types

    try:
        response = _client(api_key).models.generate_content(
            model=model,
            contents=build_prompt(question, chunks),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
    except errors.APIError as exc:
        raise LLMError(f"Gemini devolvió un error ({exc.code}): {exc.message}") from exc

    text = (response.text or "").strip()
    if not text:
        raise LLMError("Gemini no devolvió texto (posible bloqueo por filtros de seguridad).")
    return text
