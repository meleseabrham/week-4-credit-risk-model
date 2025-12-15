"""
Enhanced model training with MLflow tracking and model registry.

This script trains multiple models, logs comprehensive metrics,
and registers the best model for deployment.
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import sys
import os
import joblib
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from src.data_processing import load_data, preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred, pred_proba=None):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        actual: True labels
        pred: Predicted labels
        pred_proba: Predicted probabilities (optional, for ROC AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(actual, pred),
        'precision': precision_score(actual, pred, zero_division=0),
        'recall': recall_score(actual, pred, zero_division=0),
        'f1_score': f1_score(actual, pred, zero_division=0)
    }

    if pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(actual, pred_proba)
        except ValueError:
            metrics['roc_auc'] = 0

    return metrics


def train_models(data_path):
    """
    Main training function with enhanced MLflow tracking.
    """
    logger.info("Loading and processing data...")
    try:
        df = load_data(data_path)
        df_processed = preprocess_data(df, is_training=True)
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        return

    # Prepare features and target
    target_col = 'is_high_risk'
    if target_col not in df_processed.columns:
        logger.error(
            f"Target column '{target_col}' not found in processed data."
        )
        return

    # Drop ID columns and other non-feature columns
    drop_cols = [
        target_col,
        'TransactionStartTime',
        'CustomerId',
        'TransactionId',
        'BatchId',
        'SubscriptionId',
        'AccountId',
        'CurrencyCode',
        'CountryCode'
    ]

    # Filter features
    X = df_processed.drop(
        columns=[c for c in drop_cols if c in df_processed.columns]
    )

    # Ensure all data is numeric
    X = X.select_dtypes(include=[np.number])

    y = df_processed[target_col]

    logger.info(f"Training features: {X.columns.tolist()}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set MLflow Experiment
    experiment_name = "Credit_Risk_Prediction"
    mlflow.set_experiment(experiment_name)

    # Define models and hyperparameter grids
    models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        }
    }

    best_overall_model = None
    best_overall_f1 = -1
    best_model_name = ""
    best_run_id = None

    for model_name, config in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            logger.info(f"Training {model_name}...")

            # Grid Search with CV
            clf = GridSearchCV(
                config["model"],
                config["params"],
                cv=5,  # 5-fold cross-validation
                scoring='f1',
                n_jobs=-1,
                return_train_score=True
            )
            clf.fit(X_train, y_train)

            # Log cross-validation results
            cv_results = clf.cv_results_
            for i in range(len(cv_results['mean_test_score'])):
                mlflow.log_metric(
                    f"cv_fold_f1_mean",
                    cv_results['mean_test_score'][i]
                )

            best_model = clf.best_estimator_
            best_params = clf.best_params_

            # Predict
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = eval_metrics(y_test, y_pred, y_pred_proba)

            # Log Parameters
            mlflow.log_params(best_params)

            # Log Metrics
            mlflow.log_metrics(metrics)

            # Log feature importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Save as artifact
                importance_path = f"feature_importance_{model_name}.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)

            elif hasattr(best_model, 'coef_'):
                # For logistic regression, log coefficients
                coef_df = pd.DataFrame({
                    'feature': X.columns,
                    'coefficient': best_model.coef_[0]
                }).sort_values('coefficient', key=abs, ascending=False)

                coef_path = f"coefficients_{model_name}.csv"
                coef_df.to_csv(coef_path, index=False)
                mlflow.log_artifact(coef_path)
                os.remove(coef_path)

            # Log Model
            mlflow.sklearn.log_model(best_model, model_name)

            logger.info(
                f"{model_name} Results - "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1_score']:.4f}, "
                f"AUC: {metrics['roc_auc']:.4f}"
            )

            # Track best model
            if metrics['f1_score'] > best_overall_f1:
                best_overall_f1 = metrics['f1_score']
                best_overall_model = best_model
                best_model_name = model_name
                best_run_id = run.info.run_id

    # Register best model in MLflow Model Registry
    if best_overall_model and best_run_id:
        logger.info(
            f"Best model was {best_model_name} "
            f"with F1: {best_overall_f1:.4f}"
        )

        with mlflow.start_run(run_name="Champion_Model") as run:
            mlflow.log_param("original_model", best_model_name)
            mlflow.log_metric("f1_score", best_overall_f1)
            mlflow.sklearn.log_model(best_overall_model, "model")

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                model_details = mlflow.register_model(
                    model_uri,
                    "CreditRiskModel"
                )
                logger.info(
                    f"Model registered: {model_details.name}, "
                    f"version {model_details.version}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not register model in registry: {e}"
                )

        # Also save locally for easy loading
        model_save_path = "models/best_model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_overall_model, model_save_path)
        logger.info(f"Model saved locally to {model_save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default path relative to project root
        data_path = "data/raw/data.csv"

    if os.path.exists(data_path):
        train_models(data_path)
    else:
        # Try absolute path fallback if relative fails
        abs_path = os.path.join(os.getcwd(), "data/raw/data.csv")
        if os.path.exists(abs_path):
            train_models(abs_path)
        else:
            logger.warning(
                f"Data file not found at {data_path} or {abs_path}."
            )
