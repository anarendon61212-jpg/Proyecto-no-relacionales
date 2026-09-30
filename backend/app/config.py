"""Configuración cargada desde variables de entorno o desde .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Falta la variable {name}. Copia .env.example a .env y completa su valor."
        )
    return value


def get_settings() -> dict[str, str]:
    return {
        "mongodb_uri": required_env("MONGODB_URI"),
        "database_name": os.getenv("DATABASE_NAME", "rag_nosql").strip(),
        "vector_index_name": os.getenv("VECTOR_INDEX_NAME", "chunks_vector_index").strip(),
    }
