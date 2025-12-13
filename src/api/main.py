from fastapi import FastAPI, HTTPException
from .pydantic_models import CreditRiskRequest, CreditRiskResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI()

@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    try:
        logger.info(f"Received prediction request: {request}")
        # Dummy prediction logic
        prediction = 0.5
        
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
