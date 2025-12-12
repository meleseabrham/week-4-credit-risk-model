from fastapi import FastAPI
from .pydantic_models import CreditRiskRequest, CreditRiskResponse

app = FastAPI()

@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    # Dummy prediction logic
    return {"prediction": 0.5}
