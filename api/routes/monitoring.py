from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter


router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"],
)

DRIFT_SUMMARY_PATH = Path(
    "reports/monitoring/latest_drift_summary.json"
)

FEEDBACK_SUMMARY_PATH = Path(
    "reports/monitoring/feedback_summary.json"
)


def read_json_file(
    path: Path,
) -> dict[str, Any]:
    """Lire un rapport JSON s'il existe."""

    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "error": "Rapport non généré.",
        }

    if not path.is_file():
        return {
            "available": False,
            "path": str(path),
            "error": "Le chemin trouvé n'est pas un fichier.",
        }

    try:
        content = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return {
            "available": True,
            "path": str(path),
            "data": content,
            "error": None,
        }

    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "path": str(path),
            "error": (
                "Le rapport JSON est invalide : "
                f"{exc}"
            ),
        }


@router.get(
    "/summary",
    summary="Afficher le résumé du monitoring",
)
def get_monitoring_summary() -> dict[str, Any]:
    """Retourner les derniers résultats de drift et feedback."""

    drift = read_json_file(
        DRIFT_SUMMARY_PATH
    )

    feedback = read_json_file(
        FEEDBACK_SUMMARY_PATH
    )

    return {
        "status": (
            "available"
            if drift["available"]
            or feedback["available"]
            else "unavailable"
        ),
        "drift": drift,
        "feedback": feedback,
    }