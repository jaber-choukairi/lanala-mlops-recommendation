from __future__ import annotations

import os
from typing import Any

import pytest
import requests
from requests import Response


API_URL = os.getenv(
    "TEST_API_URL",
    "http://localhost:8000",
)

EXPECTED_MODEL_NAME = os.getenv(
    "EXPECTED_MODEL_NAME",
    "lanala-banking-recommender",
)

EXPECTED_MODEL_ALIAS = os.getenv(
    "EXPECTED_MODEL_ALIAS",
    "production",
)

REQUEST_TIMEOUT_SECONDS = int(
    os.getenv(
        "TEST_REQUEST_TIMEOUT_SECONDS",
        "30",
    )
)


def build_valid_client_payload() -> dict[str, Any]:
    """Construire une entrée client valide."""

    return {
        "client_id": "CI-STAGING-001",
        "age": 35,
        "gender": "female",
        "marital_status": "married",
        "number_of_children": 2,
        "employment_status": "employed",
        "monthly_income": 6500,
        "account_balance": 18000,
        "credit_score": 760,
        "customer_tenure_months": 48,
        "number_of_transactions": 25,
        "average_transaction_amount": 420,

        # La valeur doit être entre 0 et 1.
        "digital_activity_score": 0.8,

        "commercial_consent": 1,
        "complaint_open": 0,
        "has_savings_account": 1,
        "has_premium_card": 0,
        "has_personal_loan": 0,
        "has_home_loan": 0,
        "has_life_insurance": 0,
        "has_investment_plan": 0,
    }


def get_error_message(
    response: Response,
) -> str:
    """Construire un message d'erreur détaillé."""

    try:
        response_content = response.json()
    except ValueError:
        response_content = response.text

    return (
        "\nLa requête vers l'API a échoué.\n"
        f"URL : {response.url}\n"
        f"Méthode : {response.request.method}\n"
        f"Status HTTP : {response.status_code}\n"
        f"Réponse FastAPI : {response_content}"
    )


def get_api_health() -> Response:
    """Interroger le healthcheck de FastAPI."""

    try:
        return requests.get(
            f"{API_URL}/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.RequestException as exc:
        pytest.fail(
            "FastAPI n'est pas accessible.\n"
            f"URL testée : {API_URL}/health\n"
            "Démarrez les services avec : "
            "docker compose up -d\n"
            f"Erreur : {exc}"
        )


def test_deployed_api_health() -> None:
    """Vérifier l'API et le modèle déployé."""

    response = get_api_health()

    assert response.status_code == 200, (
        get_error_message(response)
    )

    body = response.json()

    assert body["status"] == "healthy"
    assert body["api"] == "running"
    assert body["model_loaded"] is True

    assert (
        body["model_name"]
        == EXPECTED_MODEL_NAME
    )

    assert (
        body["model_alias"]
        == EXPECTED_MODEL_ALIAS
    ), (
        "Mauvais alias MLflow servi.\n"
        f"Alias attendu : {EXPECTED_MODEL_ALIAS}\n"
        f"Alias reçu : {body['model_alias']}\n"
        "Définissez EXPECTED_MODEL_ALIAS "
        "selon l'environnement testé."
    )

    assert body["model_version"] is not None
    assert body["loading_error"] is None


def test_deployed_api_prediction() -> None:
    """Vérifier que l'API produit trois recommandations."""

    payload = build_valid_client_payload()

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.RequestException as exc:
        pytest.fail(
            "Impossible de contacter POST /predict.\n"
            f"URL testée : {API_URL}/predict\n"
            f"Erreur : {exc}"
        )

    assert response.status_code == 200, (
        get_error_message(response)
    )

    body = response.json()

    assert (
        body["client_id"]
        == "CI-STAGING-001"
    )

    assert body["status"] == "success"

    assert (
        body["model_name"]
        == EXPECTED_MODEL_NAME
    )

    assert (
        body["model_alias"]
        == EXPECTED_MODEL_ALIAS
    ), (
        "Mauvais alias MLflow servi.\n"
        f"Alias attendu : {EXPECTED_MODEL_ALIAS}\n"
        f"Alias reçu : {body['model_alias']}"
    )

    assert body["model_version"] is not None

    recommendations = body["recommendations"]

    assert len(recommendations) == 3

    ranks = [
        recommendation["rank"]
        for recommendation in recommendations
    ]

    assert ranks == [1, 2, 3]

    probabilities = [
        recommendation["probability"]
        for recommendation in recommendations
    ]

    assert probabilities == sorted(
        probabilities,
        reverse=True,
    )

    recommended_products = [
        recommendation["product"]
        for recommendation in recommendations
    ]

    assert len(
        set(recommended_products)
    ) == len(recommended_products)

    # Le client possède déjà ce produit.
    assert (
        "savings_account"
        not in recommended_products
    )

    # Vérifier les nouveaux champs de monitoring.
    assert "monitoring_saved" in body
    assert "monitoring_warning" in body

    if body["monitoring_saved"]:
        assert body["prediction_id"] is not None
        assert body["monitoring_warning"] is None