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

    Args:
        request: Transaction data

    Returns:
        Risk probability and high-risk flag
    """
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
            'CountryCode'
        ]
        X = df_processed.drop(
            columns=[c for c in drop_cols if c in df_processed.columns],
            errors='ignore'
        )

        if model is not None:
            # Use actual model prediction
            try:
                # Get probability of high risk (class 1)
                prob = model.predict_proba(X)[0, 1]
                pred = int(model.predict(X)[0])

                logger.info(
                    f"Prediction made: probability={prob:.4f}, "
                    f"is_high_risk={pred}"
                )

                return CreditRiskResponse(
                    probability=float(prob),
                    is_high_risk=pred
                )

            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {str(e)}"
                )
        else:
            # Fallback: Dummy prediction logic
            logger.warning("Using dummy prediction (model not loaded)")

            # Simple heuristic based on amount
            amount = request.Amount
            if amount > 5000:
                prob = 0.65  # High amount = higher risk
                pred = 1
            elif amount < 500:
                prob = 0.25  # Low amount = lower risk
                pred = 0
            else:
                prob = 0.45  # Medium risk
                pred = 0

            return CreditRiskResponse(
                probability=float(prob),
                is_high_risk=pred
            )

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
