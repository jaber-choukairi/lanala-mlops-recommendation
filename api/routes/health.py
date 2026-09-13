from fastapi import APIRouter

from api.services.model_service import model_service


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
def health() -> dict:
    return {
        "status": (
            "healthy"
            if model_service.is_loaded
            else "degraded"
        ),
        "api": "running",
        "model_loaded": model_service.is_loaded,
        "model_name": model_service.model_name,
        "model_alias": model_service.model_alias,
        "model_version": model_service.model_version,
        "loading_error": model_service.loading_error,
    }
