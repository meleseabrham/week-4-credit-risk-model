"""Pydantic models for the FastAPI layer."""

from pydantic import BaseModel, Field


class CreditRiskRequest(BaseModel):
    """Request schema mirroring training-time raw features."""

    Amount: float = Field(..., description="Transaction amount")
    Value: float = Field(..., description="Absolute transaction value")
    TransactionStartTime: str = Field(
        ..., description="Transaction timestamp in ISO-8601"
    )

    CustomerId: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str


class CreditRiskResponse(BaseModel):
    """Model prediction response."""

    probability: float
    is_high_risk: int
