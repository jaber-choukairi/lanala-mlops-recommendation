from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from api.schemas.prediction import ClientInput
from api.schemas.prediction import PredictionResponse
from api.services.model_service import model_service


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post(
    "",
    response_model=PredictionResponse,
)
def predict(
    client: ClientInput,
) -> PredictionResponse:
    if client.commercial_consent == 0:
        return PredictionResponse(
            client_id=client.client_id,
            status="blocked",
            reason="Consentement commercial absent.",
            recommendations=[],
            model_name=model_service.model_name,
            model_alias=model_service.model_alias,
            model_version=(
                model_service.model_version
                or "not-loaded"
            ),
        )

    if client.complaint_open == 1:
        return PredictionResponse(
            client_id=client.client_id,
            status="blocked",
            reason="Une réclamation est actuellement ouverte.",
            recommendations=[],
            model_name=model_service.model_name,
            model_alias=model_service.model_alias,
            model_version=(
                model_service.model_version
                or "not-loaded"
            ),
        )

    try:
        recommendations = model_service.predict(
            client.model_dump()
        )
    except Exception as exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Le modèle de recommandation est "
                f"indisponible : {exception}"
            ),
        ) from exception

    return PredictionResponse(
        client_id=client.client_id,
        status="success",
        reason=None,
        recommendations=recommendations,
        model_name=model_service.model_name,
        model_alias=model_service.model_alias,
        model_version=model_service.model_version or "unknown",
    )