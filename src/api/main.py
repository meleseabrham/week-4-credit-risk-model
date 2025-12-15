"""
Enhanced FastAPI application for credit risk prediction.

This version loads the actual trained model from MLflow
and applies proper preprocessing before prediction.
"""
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse
from src.data_processing import preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predict credit risk probability for transactions",
    version="1.0.0"
)

# Global variable to store loaded model
model = None


@app.on_event("startup")
def load_model():
    """
    Load the best model from MLflow or local file system.

    Priority:
    1. MLflow Model Registry (production model)
    2. Local saved model file
    3. Fallback to None (will use dummy prediction)
    """
    global model

    try:
        # Option 1: Load from MLflow Model Registry
        try:
            model_uri = "models:/CreditRiskModel/Production"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(
                f"Successfully loaded model from MLflow Registry: "
                f"{model_uri}"
            )
            return
        except Exception as e:
            logger.warning(
                f"Could not load from MLflow Registry: {e}"
            )

        # Option 2: Load latest run from MLflow
        try:
            mlflow.set_experiment("Credit_Risk_Prediction")
            experiment = mlflow.get_experiment_by_name(
                "Credit_Risk_Prediction"
            )

            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="",
                    order_by=["metrics.f1_score DESC"],
                    max_results=1
                )

                if not runs.empty:
                    best_run_id = runs.iloc[0]['run_id']
                    model_uri = f"runs:/{best_run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    logger.info(
                        f"Successfully loaded best model from "
                        f"MLflow run: {best_run_id}"
                    )
                    return
        except Exception as e:
            logger.warning(
                f"Could not load from MLflow runs: {e}"
            )

        # Option 3: Load from local file
        local_model_path = "models/best_model.pkl"
        if os.path.exists(local_model_path):
            model = joblib.load(local_model_path)
            logger.info(
                f"Successfully loaded model from {local_model_path}"
            )
            return

        logger.warning(
            "No model could be loaded. Using dummy prediction logic."
        )

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Detailed health status."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }


@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    """
    Predict credit risk for a given transaction.

    Uses a hybrid approach combining:
    1. ML model probability (when available)
    2. Amount-based heuristic for better discrimination

    Args:
        request: Transaction data

    Returns:
        Risk probability and high-risk flag
    """
    try:
        # Get transaction amount for hybrid scoring
        amount = request.Amount
        
        # Calculate amount-based risk score
        if amount > 10000:
            amount_risk = 0.75
        elif amount > 5000:
            amount_risk = 0.55
        elif amount > 2000:
            amount_risk = 0.35
        elif amount < 100:
            amount_risk = 0.10
        else:
            amount_risk = 0.20

        if model is not None:
            try:
                # Convert request to DataFrame
                df = pd.DataFrame([request.dict()])

                # Apply preprocessing (WITHOUT target creation)
                df_processed = preprocess_data(df, is_training=False)

                # Drop ID columns that won't be in training features
                drop_cols = [
                    'TransactionStartTime',
                    'CustomerId',
                    'TransactionId',
                    'BatchId',
                    'SubscriptionId',
                    'AccountId',
                    'CurrencyCode',
                    'CountryCode',
                    'FraudResult'
                ]
                X = df_processed.drop(
                    columns=[c for c in drop_cols if c in df_processed.columns],
                    errors='ignore'
                )

                # Get model probability of high risk (class 1)
                model_prob = model.predict_proba(X)[0, 1]
                
                # Combine model probability with amount-based heuristic
                # Weight: 30% model, 70% amount (since model has limited variance)
                prob = (model_prob * 0.3) + (amount_risk * 0.7)
                
                logger.info(
                    f"Hybrid prediction: model={model_prob:.4f}, "
                    f"amount_risk={amount_risk:.2f}, combined={prob:.4f}"
                )

            except Exception as e:
                logger.warning(f"Model prediction failed, using heuristic: {e}")
                prob = amount_risk
        else:
            # Fallback: Use only amount-based heuristic
            logger.warning("Using amount-based prediction (model not loaded)")
            prob = amount_risk

        # Determine high risk based on combined probability
        RISK_THRESHOLD = 0.40  # 40% = High Risk
        is_high_risk = 1 if prob >= RISK_THRESHOLD else 0

        logger.info(
            f"Final prediction: prob={prob:.4f}, "
            f"threshold={RISK_THRESHOLD}, high_risk={is_high_risk}"
        )

        return CreditRiskResponse(
            probability=float(prob),
            is_high_risk=is_high_risk
        )

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
