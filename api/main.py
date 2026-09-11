import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.prediction import router as prediction_router
from api.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    skip_loading = (
        os.getenv("SKIP_MODEL_LOADING", "false")
        .lower()
        == "true"
    )

    if not skip_loading:
        try:
            model_service.load()
        except Exception as exception:
            model_service.loading_error = str(exception)
            print(
                "Erreur pendant le chargement du modèle : "
                f"{exception}"
            )

    yield


app = FastAPI(
    title="LANALA Banking Recommendation API",
    description=(
        "API FastAPI servant le modèle Champion "
        "enregistré dans MLflow Model Registry."
    ),
    version="0.5.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(prediction_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "LANALA Banking Recommendation API",
        "documentation": "/docs",
        "health": "/health",
    }