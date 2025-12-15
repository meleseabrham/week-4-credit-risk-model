import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import sys
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_processing import load_data, preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(actual, pred)
    except ValueError:
        roc_auc = 0
    return accuracy, precision, recall, f1, roc_auc


def train_models(data_path):
    logger.info("Loading and processing data...")
    try:
        df = load_data(data_path)
        df_processed = preprocess_data(df)
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        return

    # Prepare features and target
    target_col = 'is_high_risk'
    if target_col not in df_processed.columns:
        logger.error(f"Target column '{target_col}' not found in processed data.")
        return

    # Drop ID columns and other non-feature columns
    # Adjust this list based on what is actually in your processed dataframe
    drop_cols = [target_col, 'TransactionStartTime', 'CustomerId', 'TransactionId', 'BatchId', 'SubscriptionId', 'AccountId', 'CurrencyCode', 'CountryCode']
    
    # Filter features: keep only numeric (already scaled) and boolean (one-hot)
    X = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])
    
    # Ensure all data is numeric (drop any lingering strings)
    X = X.select_dtypes(include=[np.number])
    
    y = df_processed[target_col]

    logger.info(f"Training features: {X.columns.tolist()}")
    logger.info(f"Target distribution: \n{y.value_counts()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

    for model_name, config in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")
            
            clf = GridSearchCV(config["model"], config["params"], cv=3, scoring='f1', n_jobs=-1)
            clf.fit(X_train, y_train)
            
            best_model = clf.best_estimator_
            best_params = clf.best_params_
            
            # Predict
            y_pred = best_model.predict(X_test)
            acc, prec, rec, f1, auc = eval_metrics(y_test, y_pred)
            
            # Log Params
            mlflow.log_params(best_params)
            
            # Log Metrics
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc
            })
            
            # Log Model
            mlflow.sklearn.log_model(best_model, model_name)
            
            logger.info(f"{model_name} Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_overall_model = best_model
                best_model_name = model_name

    # Register best model (Hypothetically, normally done via UI or explicitly)
    if best_overall_model:
        logger.info(f"Best model was {best_model_name} with F1: {best_overall_f1:.4f}")
        # We could log this separately as the "champion"
        with mlflow.start_run(run_name="Champion_Model"):
            mlflow.log_param("original_model", best_model_name)
            mlflow.log_metric("f1_score", best_overall_f1)
            mlflow.sklearn.log_model(best_overall_model, "model")
            
            # Registering model programmatically
            # result = mlflow.register_model(
            #     "runs:/{}/model".format(mlflow.active_run().info.run_id),
            #     "CreditRiskModel"
            # )

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
