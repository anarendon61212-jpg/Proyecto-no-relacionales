"""Configuración cargada desde variables de entorno o desde .env."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Dimensión de all-MiniLM-L6-v2, el modelo sugerido en la guía. Si el equipo cambia
# de modelo, esta constante, el validador de chunks y el índice vectorial deben coincidir.
VECTOR_DIMENSIONS = 384

# Imágenes: CLIP ViT-B/32 produce vectores de 512 dimensiones. El codificador de texto
# multilingüe está alineado con el de imágenes, así una consulta en español encuentra fotos.
IMAGE_VECTOR_DIMENSIONS = 512


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Falta la variable {name}. Copia .env.example a .env y completa su valor."
        )
    return value


@lru_cache(maxsize=1)
def get_settings() -> dict[str, str]:
    return {
        "mongodb_uri": required_env("MONGODB_URI"),
        "database_name": os.getenv("DATABASE_NAME", "rag_nosql").strip(),
        "vector_index_name": os.getenv("VECTOR_INDEX_NAME", "chunks_vector_index").strip(),
        "image_vector_index_name": os.getenv(
            "IMAGE_VECTOR_INDEX_NAME", "imagenes_vector_index"
        ).strip(),
        "image_model": os.getenv("IMAGE_MODEL", "clip-ViT-B-32").strip(),
        "image_text_model": os.getenv(
            "IMAGE_TEXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        ).strip(),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        # La API key de Gemini es opcional para arrancar: solo /rag la necesita.
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
    }
