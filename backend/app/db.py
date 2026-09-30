"""Conexión compartida a MongoDB para las rutas de FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pymongo import MongoClient

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = MongoClient(settings["mongodb_uri"], serverSelectionTimeoutMS=5000)
    # Verifica credenciales, DNS y acceso al cluster al iniciar la aplicación.
    client.admin.command("ping")
    app.state.mongo_client = client
    app.state.database = client[settings["database_name"]]
    app.state.vector_index_name = settings["vector_index_name"]
    try:
        yield
    finally:
        client.close()


def get_database(request: Request):
    return request.app.state.database
