from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class ClientInput(BaseModel):
    client_id: str = Field(min_length=1, max_length=100)

    age: int = Field(ge=18, le=85)

    gender: Literal[
        "female",
        "male",
    ]

    marital_status: Literal[
        "single",
        "married",
        "divorced",
        "widowed",
    ]

    number_of_children: int = Field(ge=0, le=10)

    employment_status: Literal[
        "employed",
        "self_employed",
        "student",
        "retired",
        "unemployed",
    ]

    monthly_income: float = Field(ge=0, le=50000)
    account_balance: float = Field(ge=0, le=300000)
    credit_score: int = Field(ge=300, le=850)

    customer_tenure_months: int = Field(
        ge=0,
        le=1000,
    )

    number_of_transactions: int = Field(
        ge=0,
        le=100000,
    )

    average_transaction_amount: float = Field(
        ge=0,
        le=100000,
    )

    digital_activity_score: float = Field(
        ge=0,
        le=1,
    )

    has_savings_account: int = Field(ge=0, le=1)
    has_premium_card: int = Field(ge=0, le=1)
    has_personal_loan: int = Field(ge=0, le=1)
    has_home_loan: int = Field(ge=0, le=1)
    has_life_insurance: int = Field(ge=0, le=1)
    has_investment_plan: int = Field(ge=0, le=1)

    complaint_open: int = Field(ge=0, le=1)
    commercial_consent: int = Field(ge=0, le=1)


class ProductRecommendation(BaseModel):
    rank: int
    product: str
    probability: float


class PredictionResponse(BaseModel):
    client_id: str
    status: Literal["success", "blocked"]
    reason: str | None = None
    recommendations: list[ProductRecommendation]
    model_name: str
    model_alias: str
    model_version: str