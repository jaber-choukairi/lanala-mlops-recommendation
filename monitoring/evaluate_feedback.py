from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from api.services.supabase_service import (
    get_supabase_client,
    is_supabase_configured,
)


load_dotenv()

OUTPUT_PATH = Path(
    "reports/monitoring/feedback_summary.json"
)


def safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    """Calculer un taux sans division par zéro."""

    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator,
        6,
    )


def load_feedbacks() -> list[dict[str, Any]]:
    """Récupérer les feedbacks depuis Supabase."""

    if not is_supabase_configured():
        raise RuntimeError(
            "Supabase n'est pas configuré dans "
            "l'environnement Python local. "
            "Vérifiez le fichier .env."
        )

    response = (
        get_supabase_client()
        .table("prediction_feedback")
        .select(
            "id,"
            "prediction_id,"
            "client_id,"
            "recommended_product,"
            "feedback_status,"
            "created_at"
        )
        .order(
            "created_at",
            desc=True,
        )
        .limit(1000)
        .execute()
    )

    return response.data or []


def calculate_product_statistics(
    feedbacks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculer les résultats pour chaque produit."""

    product_statistics: dict[
        str,
        dict[str, Any],
    ] = {}

    products = {
        row.get("recommended_product")
        for row in feedbacks
        if row.get("recommended_product")
    }

    for product in sorted(products):
        product_feedbacks = [
            row
            for row in feedbacks
            if row.get("recommended_product")
            == product
        ]

        statuses = Counter(
            row.get("feedback_status")
            for row in product_feedbacks
        )

        total = len(product_feedbacks)
        accepted = statuses.get(
            "accepted",
            0,
        )
        rejected = statuses.get(
            "rejected",
            0,
        )
        no_action = statuses.get(
            "no_action",
            0,
        )

        product_statistics[product] = {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "no_action": no_action,
            "acceptance_rate": safe_rate(
                accepted,
                total,
            ),
        }

    return product_statistics


def build_summary(
    feedbacks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construire le rapport global de feedback."""

    statuses = Counter(
        row.get("feedback_status")
        for row in feedbacks
    )

    total = len(feedbacks)

    accepted = statuses.get(
        "accepted",
        0,
    )
    rejected = statuses.get(
        "rejected",
        0,
    )
    no_action = statuses.get(
        "no_action",
        0,
    )

    return {
        "total_feedbacks": total,
        "accepted": accepted,
        "rejected": rejected,
        "no_action": no_action,
        "acceptance_rate": safe_rate(
            accepted,
            total,
        ),
        "rejection_rate": safe_rate(
            rejected,
            total,
        ),
        "no_action_rate": safe_rate(
            no_action,
            total,
        ),
        "feedbacks_by_product": (
            calculate_product_statistics(
                feedbacks
            )
        ),
    }


def save_summary(
    summary: dict[str, Any],
) -> None:
    """Enregistrer le rapport au format JSON."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def print_summary(
    summary: dict[str, Any],
) -> None:
    """Afficher le résultat dans le terminal."""

    print()
    print("RÉSUMÉ DES FEEDBACKS")
    print("=" * 60)

    print(
        f"Nombre total   : "
        f"{summary['total_feedbacks']}"
    )

    print(
        f"Acceptés       : "
        f"{summary['accepted']}"
    )

    print(
        f"Refusés        : "
        f"{summary['rejected']}"
    )

    print(
        f"Sans suite     : "
        f"{summary['no_action']}"
    )

    print(
        f"Taux accepté   : "
        f"{summary['acceptance_rate']:.2%}"
    )

    print(
        f"Taux refusé    : "
        f"{summary['rejection_rate']:.2%}"
    )

    print(
        f"Taux sans suite: "
        f"{summary['no_action_rate']:.2%}"
    )

    print()
    print(
        f"Rapport généré : {OUTPUT_PATH}"
    )


def main() -> None:
    try:
        feedbacks = load_feedbacks()

        summary = build_summary(
            feedbacks
        )

        save_summary(
            summary
        )

        print_summary(
            summary
        )

    except Exception as exc:
        print()
        print("ERREUR PENDANT L'ÉVALUATION")
        print("=" * 60)
        print(str(exc))
        raise


if __name__ == "__main__":
    main()