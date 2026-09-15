import pytest
from pydantic import ValidationError

from api.schemas.feedback import FeedbackRequest


def valid_payload() -> dict:
    return {
        "prediction_id": (
            "12345678-1234-5678-1234-567812345678"
        ),
        "client_id": "CLIENT-001",
        "recommended_product": "premium_card",
        "feedback_status": "accepted",
        "comment": "Le client accepte.",
    }


@pytest.mark.parametrize(
    "feedback_status",
    [
        "accepted",
        "rejected",
        "no_action",
    ],
)
def test_feedback_status_is_valid(
    feedback_status: str,
) -> None:
    payload = valid_payload()
    payload["feedback_status"] = feedback_status

    feedback = FeedbackRequest(**payload)

    assert (
        feedback.feedback_status.value
        == feedback_status
    )


def test_unknown_feedback_status_is_rejected() -> None:
    payload = valid_payload()
    payload["feedback_status"] = "maybe"

    with pytest.raises(ValidationError):
        FeedbackRequest(**payload)