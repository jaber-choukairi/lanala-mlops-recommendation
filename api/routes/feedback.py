from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from api.schemas.feedback import (
    FeedbackRequest,
    FeedbackResponse,
)
from api.services.supabase_service import (
    SupabaseConfigurationError,
    is_supabase_configured,
    prediction_exists,
    save_feedback,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback(
    feedback: FeedbackRequest,
) -> FeedbackResponse:
    if not is_supabase_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Supabase n'est pas configuré. "
                "Définissez SUPABASE_URL et SUPABASE_KEY."
            ),
        )

    prediction_id = str(
        feedback.prediction_id
    )

    try:
        if not prediction_exists(prediction_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "La prédiction demandée "
                    "n'existe pas."
                ),
            )

        saved_feedback = save_feedback(
            {
                "prediction_id": prediction_id,
                "client_id": feedback.client_id,
                "recommended_product": (
                    feedback.recommended_product
                ),
                "feedback_status": (
                    feedback.feedback_status.value
                ),
                "comment": feedback.comment,
            }
        )

        return FeedbackResponse(
            **saved_feedback
        )

    except HTTPException:
        raise

    except SupabaseConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        LOGGER.exception(
            "Erreur pendant l'enregistrement "
            "du feedback."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Impossible d'enregistrer "
                f"le feedback : {exc}"
            ),
        ) from exc