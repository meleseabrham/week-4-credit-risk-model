from pydantic import BaseModel, Field
from typing import Optional

class CreditRiskRequest(BaseModel):
    # Required features as determined by our data processing and training pipeline
    Amount: float = Field(..., description="Transaction Amount")
    Value: float = Field(..., description="Transaction Value")
    TransactionStartTime: str = Field(..., description="Transaction Start Time (ISO 8601)")
    
    # Optional fields or fields we might calculate internally if raw data is passed
    # In a real scenario, we might receive raw IDs and query a Feature Store
    # For this simplified API, we accept key numeric inputs + raw features for transformations
    CustomerId: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str

    # Pre-calculated aggregates if the client sends them? 
    # Or should the API compute them?
    # Ideally, the API should mirror the training input.
    # Our `preprocess_data` function takes a DataFrame with RAW columns.
    # So we should accept RAW columns.

class CreditRiskResponse(BaseModel):
    probability: float
    is_high_risk: int
