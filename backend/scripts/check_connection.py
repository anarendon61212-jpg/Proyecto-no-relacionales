"""Comprueba la conexión a MongoDB Atlas y muestra el nombre de la base."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import MongoClient

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    client = MongoClient(settings["mongodb_uri"], serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        print("Conexión exitosa con MongoDB.")
        print(f"Base de datos configurada: {settings['database_name']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
