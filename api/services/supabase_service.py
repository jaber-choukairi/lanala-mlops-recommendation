from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any
from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)

try:
    from supabase import Client, create_client
except ImportError:
    Client = Any
    create_client = None


class SupabaseConfigurationError(RuntimeError):
    """Erreur de configuration du service Supabase."""


def is_supabase_configured() -> bool:
    """Vérifier que Supabase peut être utilisé."""

    return bool(
        create_client is not None
        and os.getenv("SUPABASE_URL")
        and os.getenv("SUPABASE_KEY")
    )


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Créer et mettre en cache le client Supabase."""

    if create_client is None:
        raise SupabaseConfigurationError(
            "Le paquet 'supabase' n'est pas installé. "
            "Exécutez : python -m pip install -r requirements.txt"
        )

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise SupabaseConfigurationError(
            "La variable SUPABASE_URL est absente."
        )

    if not supabase_key:
        raise SupabaseConfigurationError(
            "La variable SUPABASE_KEY est absente."
        )

    return create_client(
        supabase_url,
        supabase_key,
    )


def save_prediction(
    *,
    client_id: str,
    input_features: dict[str, Any],
    recommendations: list[dict[str, Any]],
    model_name: str,
    model_version: str | None,
    model_alias: str,
    prediction_status: str = "completed",
    latency_ms: float | None = None,
) -> dict[str, Any]:
    """Enregistrer une prédiction dans Supabase."""

    if prediction_status not in {
        "completed",
        "blocked",
        "failed",
    }:
        raise ValueError(
            "prediction_status doit être "
            "'completed', 'blocked' ou 'failed'."
        )

    payload = {
        "client_id": client_id,
        "input_features": input_features,
        "recommendations": recommendations,
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "top_k": len(recommendations),
        "prediction_status": prediction_status,
        "prediction_latency_ms": latency_ms,
    }

    response = (
        get_supabase_client()
        .table("prediction_logs")
        .insert(payload)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Supabase n'a retourné aucune ligne "
            "après l'enregistrement de la prédiction."
        )

    return response.data[0]


def save_feedback(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Enregistrer un feedback client."""

    response = (
        get_supabase_client()
        .table("prediction_feedback")
        .insert(payload)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Supabase n'a retourné aucune ligne "
            "après l'enregistrement du feedback."
        )

    return response.data[0]


def prediction_exists(
    prediction_id: str,
) -> bool:
    """Vérifier qu'une prédiction existe."""

    response = (
        get_supabase_client()
        .table("prediction_logs")
        .select("id")
        .eq("id", prediction_id)
        .limit(1)
        .execute()
    )

    return bool(response.data)


def check_supabase_connection() -> dict[str, Any]:
    """Retourner l'état du stockage Supabase."""

    if not is_supabase_configured():
        return {
            "configured": False,
            "connected": False,
            "error": (
                "Supabase n'est pas configuré. "
                "Les prédictions fonctionneront sans sauvegarde."
            ),
        }

    try:
        get_supabase_client()

        return {
            "configured": True,
            "connected": True,
            "error": None,
        }

    except Exception as exc:
        LOGGER.exception(
            "Connexion Supabase impossible."
        )

        return {
            "configured": True,
            "connected": False,
            "error": str(exc),
        }