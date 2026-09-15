from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.feedback import router as feedback_router
from api.routes.health import router as health_router
from api.routes.prediction import router as prediction_router
from api.routes.monitoring import router as monitoring_router
from api.services.model_service import model_service

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Charger le modèle au démarrage.

    Une erreur de chargement ne fait pas planter
    complètement FastAPI. La route /health indiquera
    alors que le service est dégradé.
    """

    try:
        model_service.load()

        LOGGER.info(
            "Modèle chargé au démarrage : "
            "%s version=%s alias=%s",
            model_service.model_name,
            model_service.model_version,
            model_service.model_alias,
        )

    except Exception as exc:
        model_service.loading_error = str(exc)

        LOGGER.exception(
            "Impossible de charger le modèle "
            "au démarrage. L'API reste accessible "
            "en mode dégradé."
        )

    yield

    LOGGER.info(
        "Arrêt de l'API LANALA."
    )


app = FastAPI(
    title="LANALA Banking Recommendation API",
    description=(
        "API de recommandation bancaire avec "
        "MLflow Model Registry, monitoring "
        "et feedback."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(feedback_router)
app.include_router(monitoring_router)



@app.get(
    "/",
    tags=["General"],
)
def root() -> dict[str, str]:
    return {
        "name": "LANALA Banking Recommendation API",
        "version": "0.2.0",
        "documentation": "/docs",
        "health": "/health",
        "prediction": "/predict",
        "feedback": "/feedback",
    }