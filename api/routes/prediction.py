from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas.prediction import (
    ClientInput,
    PredictionResponse,
)
from api.services.model_service import model_service
from api.services.supabase_service import (
    is_supabase_configured,
    save_prediction,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(
    tags=["Prediction"],
)


def save_prediction_safely(
    *,
    client_data: dict[str, Any],
    recommendations: list[dict[str, Any]],
    prediction_status: str,
    latency_ms: float,
) -> tuple[str | None, str | None]:
    """
    Sauvegarder sans empêcher la prédiction de fonctionner.

    Retour :
    - identifiant de la prédiction ;
    - message d'avertissement éventuel.
    """

    if not is_supabase_configured():
        return (
            None,
            "Supabase n'est pas configuré. "
            "La prédiction n'a pas été sauvegardée.",
        )

    try:
        saved_prediction = save_prediction(
            client_id=str(
                client_data["client_id"]
            ),
            input_features=client_data,
            recommendations=recommendations,
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            model_alias=model_service.model_alias,
            prediction_status=prediction_status,
            latency_ms=round(latency_ms, 3),
        )

        return (
            str(saved_prediction["id"]),
            None,
        )

    except Exception as exc:
        LOGGER.exception(
            "Impossible d'enregistrer la prédiction."
        )

        return (
            None,
            f"Sauvegarde Supabase impossible : {exc}",
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    client: ClientInput,
) -> PredictionResponse:
    started_at = perf_counter()
    client_data = client.model_dump()

    # Règle métier 1 :
    # pas de recommandation sans consentement.
    if client.commercial_consent == 0:
        latency_ms = (
            perf_counter() - started_at
        ) * 1000

        prediction_id, warning = (
            save_prediction_safely(
                client_data=client_data,
                recommendations=[],
                prediction_status="blocked",
                latency_ms=latency_ms,
            )
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            client_id=client.client_id,
            status="blocked",
            reason=(
                "La recommandation est bloquée car "
                "le client n'a pas donné son "
                "consentement commercial."
            ),
            recommendations=[],
            model_name=model_service.model_name,
            model_alias=model_service.model_alias,
            model_version=model_service.model_version,
            monitoring_saved=prediction_id is not None,
            monitoring_warning=warning,
        )

    # Règle métier 2 :
    # pas de recommandation si une plainte est ouverte.
    if client.complaint_open == 1:
        latency_ms = (
            perf_counter() - started_at
        ) * 1000

        prediction_id, warning = (
            save_prediction_safely(
                client_data=client_data,
                recommendations=[],
                prediction_status="blocked",
                latency_ms=latency_ms,
            )
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            client_id=client.client_id,
            status="blocked",
            reason=(
                "La recommandation est bloquée car "
                "une réclamation client est ouverte."
            ),
            recommendations=[],
            model_name=model_service.model_name,
            model_alias=model_service.model_alias,
            model_version=model_service.model_version,
            monitoring_saved=prediction_id is not None,
            monitoring_warning=warning,
        )

    try:
        recommendations = model_service.predict(
            client_data
        )

    except Exception as exc:
        LOGGER.exception(
            "Erreur pendant la prédiction."
        )

        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Le modèle de recommandation "
                    "n'est pas disponible."
                ),
                "error": str(exc),
            },
        ) from exc

    latency_ms = (
        perf_counter() - started_at
    ) * 1000

    prediction_id, warning = save_prediction_safely(
        client_data=client_data,
        recommendations=recommendations,
        prediction_status="completed",
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        client_id=client.client_id,
        status="success",
        reason=None,
        recommendations=recommendations,
        model_name=model_service.model_name,
        model_alias=model_service.model_alias,
        model_version=model_service.model_version,
        monitoring_saved=prediction_id is not None,
        monitoring_warning=warning,
    )