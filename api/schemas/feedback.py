from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FeedbackStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NO_ACTION = "no_action"


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": (
                    "12345678-1234-5678-1234-567812345678"
                ),
                "client_id": "CLIENT-001",
                "recommended_product": "premium_card",
                "feedback_status": "accepted",
                "comment": (
                    "Le client a accepté la proposition."
                ),
            }
        }
    )

    prediction_id: UUID

    client_id: str = Field(
        min_length=1,
        max_length=100,
    )

    recommended_product: str | None = Field(
        default=None,
        max_length=100,
    )

    feedback_status: FeedbackStatus

    comment: str | None = Field(
        default=None,
        max_length=500,
    )


class FeedbackResponse(BaseModel):
    id: UUID
    prediction_id: UUID
    client_id: str
    recommended_product: str | None
    feedback_status: FeedbackStatus
    comment: str | None = None
    created_at: datetime