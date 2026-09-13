"""Tests d'intégration de l'API LANALA déployée en staging."""

from __future__ import annotations

import os
from typing import Any

import requests


API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30


def get_error_message(response: requests.Response) -> str:
    """Afficher le détail renvoyé par FastAPI en cas d'échec."""

    try:
        response_body: Any = response.json()
    except requests.exceptions.JSONDecodeError:
        response_body = response.text

    return (
        "\nLa requête vers l'API a échoué."
        f"\nURL : {response.request.url}"
        f"\nMéthode : {response.request.method}"
        f"\nStatus HTTP : {response.status_code}"
        f"\nRéponse FastAPI : {response_body}"
    )


def build_valid_client_payload() -> dict[str, object]:
    """Créer un profil client valide pour le test de staging."""

    return {
        "client_id": "CI-STAGING-001",
        "age": 35,
        "gender": "female",
        "marital_status": "married",
        "number_of_children": 2,
        "employment_status": "employed",
        "monthly_income": 6500.0,
        "account_balance": 18000.0,
        "credit_score": 760,
        "customer_tenure_months": 48,
        "number_of_transactions": 25,
        "average_transaction_amount": 420.0,
        "digital_activity_score": 0.80,
        "commercial_consent": True,
        "complaint_open": False,
        "has_savings_account": True,
        "has_premium_card": False,
        "has_personal_loan": False,
        "has_home_loan": False,
        "has_life_insurance": False,
        "has_investment_plan": False,
    }


def test_staging_health() -> None:
    """Vérifier que FastAPI et le modèle staging sont disponibles."""

    response = requests.get(
        f"{API_URL}/health",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200, get_error_message(response)

    body = response.json()

    assert body["status"] == "healthy"
    assert body["api"] == "running"
    assert body["model_loaded"] is True
    assert body["model_name"] == "lanala-banking-recommender"
    assert body["model_alias"] == "staging"
    assert body["model_version"] is not None
    assert body["loading_error"] is None


def test_staging_prediction() -> None:
    """Vérifier que le modèle staging produit trois recommandations."""

    payload = build_valid_client_payload()

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200, get_error_message(response)

    body = response.json()

    assert body["client_id"] == "CI-STAGING-001"
    assert body["model_name"] == "lanala-banking-recommender"
    assert body["model_alias"] == "staging"
    assert body["model_version"] is not None

    recommendations = body["recommendations"]

    assert isinstance(recommendations, list)
    assert len(recommendations) == 3

    for recommendation in recommendations:
        assert "product" in recommendation
        assert "probability" in recommendation
        assert isinstance(recommendation["product"], str)
        assert 0.0 <= recommendation["probability"] <= 1.0

    recommended_products = [
        recommendation["product"]
        for recommendation in recommendations
    ]

    # Le client possède déjà ce produit : il ne doit pas être recommandé.
    assert "savings_account" not in recommended_products
