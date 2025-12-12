from pydantic import BaseModel

class CreditRiskRequest(BaseModel):
    # Define fields here based on data
    feature1: float
    feature2: float

class CreditRiskResponse(BaseModel):
    prediction: float
