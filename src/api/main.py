import os
import sys
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from .pydantic_models import CreditRiskRequest, CreditRiskResponse

# Ensure src is in path so we can import data_processing if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_processing import preprocess_data  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting credit risk based on transaction data."
)

# Global model variable
model = None


@app.on_event("startup")
def load_model():
    # global model
    try:
        # Load the Champion model from MLflow
        pass
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    try:
        logger.info(f"Received prediction request: {request}")

        # Convert request to DataFrame
        data = request.dict()
        df = pd.DataFrame([data])

        # Preprocess
        # Note: preprocess_data expects a DF with specific columns.
        preprocess_data(df, is_training=False)

        # DUMMY LOGIC FOR DEMO until Model Registry is live
        prob = 0.45
        if request.Amount > 5000:
            prob = 0.8

        pred = 1 if prob > 0.5 else 0

        logger.info(f"Prediction: {pred} (Prob: {prob})")
        return {"probability": prob, "is_high_risk": pred}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
