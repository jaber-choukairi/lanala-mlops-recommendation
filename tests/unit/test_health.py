import os

os.environ["SKIP_MODEL_LOADING"] = "true"

from fastapi.testclient import TestClient

from api.main import app
from api.routes import health


class FakeHealthyModelService:
    is_loaded = True
    model_name = "lanala-banking-recommender"
    model_alias = "staging"
    model_version = "1"
    loading_error = None


class FakeDegradedModelService:
    is_loaded = False
    model_name = "lanala-banking-recommender"
    model_alias = "staging"
    model_version = None
    loading_error = "MLflow indisponible"


def test_health_endpoint_when_model_is_loaded(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        health,
        "model_service",
        FakeHealthyModelService(),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "api": "running",
        "model_loaded": True,
        "model_name": "lanala-banking-recommender",
        "model_alias": "staging",
        "model_version": "1",
        "loading_error": None,
    }


def test_health_endpoint_when_model_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        health,
        "model_service",
        FakeDegradedModelService(),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "degraded"
    assert result["api"] == "running"
    assert result["model_loaded"] is False
    assert result["model_version"] is None
    assert result["loading_error"] == "MLflow indisponible"