"""Request/response contracts for the API (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH = 1000


class Customer(BaseModel):
    """Features for one customer. Unknown fields are rejected, not silently ignored."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "tenure_months": 3,
                "contract": "month_to_month",
                "monthly_charges": 89.9,
                "internet_service": "fiber",
                "payment_method": "electronic_check",
                "senior_citizen": False,
                "paperless_billing": True,
                "support_tickets_90d": 2,
            }
        },
    )

    tenure_months: int = Field(ge=0, le=72, description="Months as a customer")
    contract: Literal["month_to_month", "one_year", "two_year"]
    monthly_charges: float = Field(gt=0, le=500, description="Monthly bill in EUR")
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["electronic_check", "bank_transfer", "credit_card", "mailed_check"]
    senior_citizen: bool
    paperless_billing: bool
    support_tickets_90d: int = Field(ge=0, le=50, description="Support tickets, last 90 days")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    will_churn: bool
    risk_band: Literal["low", "medium", "high"]


class PredictionResponse(Prediction):
    model_version: str
    threshold: float


class BatchResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    threshold: float
    features: list[str]
    test_metrics: dict[str, float]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
