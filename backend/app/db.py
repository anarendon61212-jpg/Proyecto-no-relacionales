"""Conexión compartida a MongoDB para las rutas de FastAPI."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, PyMongoError

from app.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.mongo_client = None
    app.state.database = None
    app.state.vector_index_name = settings["vector_index_name"]
    try:
        client = MongoClient(settings["mongodb_uri"], serverSelectionTimeoutMS=5000)
    except ConfigurationError as exc:
        # Por ejemplo, URI mal formada o DNS SRV no resoluble. La API arranca igual
        # y /health informa el problema con 503.
        logger.error("No se pudo configurar el cliente de MongoDB: %s", exc)
        client = None
    else:
        app.state.mongo_client = client
        app.state.database = client[settings["database_name"]]
        try:
            client.admin.command("ping")
        except PyMongoError as exc:
            # El driver reintenta en cada consulta, así que la API se recupera sola
            # cuando MongoDB vuelve a estar accesible.
            logger.warning("MongoDB no respondió al iniciar la API: %s", exc)
    try:
        yield
    finally:
        if client is not None:
            client.close()


def get_database(request: Request):
    database = request.app.state.database
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB no está configurado. Revisa MONGODB_URI en el archivo .env.",
        )
    return database
