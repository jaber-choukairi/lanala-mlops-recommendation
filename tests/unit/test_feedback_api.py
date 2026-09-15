from fastapi.testclient import TestClient

from api.main import app
from api.routes import feedback as feedback_route


client = TestClient(app)


def valid_payload() -> dict:
    return {
        "prediction_id": (
            "12345678-1234-5678-1234-567812345678"
        ),
        "client_id": "CLIENT-001",
        "recommended_product": "premium_card",
        "feedback_status": "accepted",
        "comment": "Accepté pendant le test.",
    }


def test_feedback_is_saved(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        feedback_route,
        "prediction_exists",
        lambda prediction_id: True,
    )

    monkeypatch.setattr(
        feedback_route,
        "save_feedback",
        lambda payload: {
            "id": (
                "87654321-4321-8765-4321-876543218765"
            ),
            **payload,
            "created_at": "2026-09-14T12:00:00+00:00",
        },
    )

    response = client.post(
        "/feedback",
        json=valid_payload(),
    )

    assert response.status_code == 201
    assert (
        response.json()["feedback_status"]
        == "accepted"
    )


def test_unknown_prediction_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        feedback_route,
        "prediction_exists",
        lambda prediction_id: False,
    )

    response = client.post(
        "/feedback",
        json=valid_payload(),
    )

    assert response.status_code == 404