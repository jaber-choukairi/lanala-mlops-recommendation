import os

os.environ["SKIP_MODEL_LOADING"] = "true"

from fastapi.testclient import TestClient

from api.main import app
from api.routes import prediction


class FakeModelService:
    model_name = "lanala-banking-recommender"
    model_alias = "test"
    model_version = "99"

    def predict(self, client_data: dict) -> list[dict]:
        return [
            {
                "rank": 1,
                "product": "life_insurance",
                "probability": 0.50,
            },
            {
                "rank": 2,
                "product": "premium_card",
                "probability": 0.30,
            },
            {
                "rank": 3,
                "product": "personal_loan",
                "probability": 0.20,
            },
        ]


VALID_CLIENT = {
    "client_id": "CLI_TEST",
    "age": 35,
    "gender": "male",
    "marital_status": "married",
    "number_of_children": 2,
    "employment_status": "employed",
    "monthly_income": 6000,
    "account_balance": 30000,
    "credit_score": 700,
    "customer_tenure_months": 72,
    "number_of_transactions": 30,
    "average_transaction_amount": 400,
    "digital_activity_score": 0.80,
    "has_savings_account": 1,
    "has_premium_card": 0,
    "has_personal_loan": 0,
    "has_home_loan": 0,
    "has_life_insurance": 0,
    "has_investment_plan": 0,
    "complaint_open": 0,
    "commercial_consent": 1,
}


def test_predict_returns_recommendations(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        prediction,
        "model_service",
        FakeModelService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=VALID_CLIENT,
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["recommendations"]) == 3


def test_predict_is_blocked_without_consent(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        prediction,
        "model_service",
        FakeModelService(),
    )

    request_data = VALID_CLIENT.copy()
    request_data["commercial_consent"] = 0

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=request_data,
        )

    assert response.status_code == 200
    assert response.json()["status"] == "blocked"
    assert response.json()["recommendations"] == []


def test_predict_is_blocked_with_open_complaint(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        prediction,
        "model_service",
        FakeModelService(),
    )

    request_data = VALID_CLIENT.copy()
    request_data["complaint_open"] = 1

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=request_data,
        )

    assert response.status_code == 200
    assert response.json()["status"] == "blocked"