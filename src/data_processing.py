"""
Enhanced preprocessing pipeline for credit risk modeling.

This module provides a unified sklearn Pipeline that ensures
consistent transformations between training and inference.
"""
import pandas as pd
import logging
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"The file at {filepath} was not found."
            )

        df = pd.read_csv(filepath)
        logger.info(
            f"Successfully loaded data from {filepath} "
            f"with shape {df.shape}"
        )
        return df
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading data: {e}"
        )
        raise


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Creates aggregate features per customer (RFM proxies)."""

    def __init__(
        self,
        group_col='CustomerId',
        value_col='Amount',
        time_col='TransactionStartTime'
    ):
        self.group_col = group_col
        self.value_col = value_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Creating aggregate features...")
        X_ = X.copy()

        # Calculate aggregates
        grouped = X_.groupby(self.group_col)

        X_['TotalTransactionAmount'] = grouped[self.value_col].transform(
            'sum'
        )
        X_['AvgTransactionAmount'] = grouped[self.value_col].transform(
            'mean'
        )
        X_['TransactionCount'] = grouped[self.value_col].transform('count')
        X_['StdTransactionAmount'] = grouped[self.value_col].transform(
            'std'
        ).fillna(0)

        # Recency Calculation
        if self.time_col in X_.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_[self.time_col]):
                X_[self.time_col] = pd.to_datetime(X_[self.time_col])

            # Define snapshot date as max date in dataset + 1 day
            snapshot_date = X_[self.time_col].max() + pd.Timedelta(days=1)

            # Calculate last transaction date per customer
            last_tx_dates = grouped[self.time_col].max()

            # Map back to original DF
            X_['LastTransactionDate'] = X_[self.group_col].map(
                last_tx_dates
            )
            X_['Recency'] = (
                snapshot_date - X_['LastTransactionDate']
            ).dt.days

            X_ = X_.drop(columns=['LastTransactionDate'], errors='ignore')

        return X_


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts time-based features from TransactionStartTime."""

    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Extracting time series features...")
        X_ = X.copy()
        if self.time_col in X_.columns:
            X_[self.time_col] = pd.to_datetime(X_[self.time_col])
            X_['TransactionMonth'] = X_[self.time_col].dt.month
            X_['TransactionDay'] = X_[self.time_col].dt.day
            X_['TransactionHour'] = X_[self.time_col].dt.hour
            X_['TransactionYear'] = X_[self.time_col].dt.year
            X_ = X_.drop(columns=[self.time_col], errors='ignore')
        return X_


class RiskLabelAssigner(BaseEstimator, TransformerMixin):
    """
    Assigns a proxy risk label based on RFM clustering.

    Rationale for n_clusters=3:
    - We create three distinct risk segments: Low, Medium, High
    - This aligns with common credit scoring practices (A/B/C grades)
    - Provides sufficient granularity without over-segmentation

    High-risk definition:
    - Cluster with lowest TotalTransactionAmount indicates
      customers with minimal platform engagement
    - Combined with low frequency, this suggests higher credit risk
    """

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.high_risk_cluster = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Assigning risk labels (Task 4)...")
        X_ = X.copy()

        # 1. Prepare RFM Features
        required_cols = [
            'TotalTransactionAmount',
            'TransactionCount',
            'Recency'
        ]
        if not all(col in X_.columns for col in required_cols):
            logger.warning(
                "Missing RFM columns for clustering. Returning original DF."
            )
            return X

        rfm_data = X_[required_cols].copy()

        # Handle missing values in RFM (e.g., if Recency cannot be calculated)
        rfm_data = rfm_data.fillna(0)  # Simple imputation for clustering

        # 2. Standardize RFM features for clustering
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data)

        # 3. K-Means Clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        X_['Cluster'] = self.kmeans.fit_predict(rfm_scaled)

        # Store diagnostics for logging
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.inertia_ = self.kmeans.inertia_

        # 4. Identify High-Risk Cluster
        cluster_summary = X_.groupby('Cluster')[required_cols].mean()

        # Lowest TotalTransactionAmount indicates low engagement = high risk
        high_risk_cluster_id = cluster_summary[
            'TotalTransactionAmount'
        ].idxmin()

        self.high_risk_cluster = high_risk_cluster_id

        logger.info(
            f"Clustering diagnostics:\n"
            f"  Inertia: {self.inertia_:.2f}\n"
            f"  High Risk Cluster: {high_risk_cluster_id}\n"
            f"  Cluster Stats:\n{cluster_summary}"
        )

        X_['is_high_risk'] = (
            X_['Cluster'] == high_risk_cluster_id
        ).astype(int)

        # Drop auxiliary cluster column
        X_ = X_.drop(columns=['Cluster'])

        return X_


def preprocess_data(
    df: pd.DataFrame,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Full preprocessing function executing all transformation steps.

    Args:
        df: Input dataframe
        is_training: If True, creates target variable;
                     if False, skips target creation for inference

    Returns:
        Preprocessed dataframe ready for modeling
    """
    if df.empty:
        logger.warning("Received empty DataFrame for preprocessing.")
        return df

    logger.info("Starting preprocessing pipeline...")

    # 1. Feature Extraction
    agg_extractor = AggregateFeatureExtractor(
        group_col='CustomerId',
        value_col='Amount',
        time_col='TransactionStartTime'
    )
    df_extracted = agg_extractor.transform(df)

    ts_extractor = TimeSeriesFeatureExtractor(
        time_col='TransactionStartTime'
    )
    df_extracted = ts_extractor.transform(df_extracted)

    # 2. Risk Label Assignment (only during training)
    if is_training:
        risk_assigner = RiskLabelAssigner()
        df_extracted = risk_assigner.transform(df_extracted)
        logger.info("Risk Label Assignment completed.")
    else:
        logger.info("Skipping Risk Label Assignment (Inference Mode).")

    # 3. Encoding and Scaling
    categorical_cols = [
        'ProviderId',
        'ProductId',
        'ProductCategory',
        'ChannelId',
        'PricingStrategy'
    ]
    numerical_cols = [
        'Amount',
        'Value',
        'TotalTransactionAmount',
        'AvgTransactionAmount',
        'TransactionCount',
        'StdTransactionAmount',
        'Recency',
        'TransactionMonth',
        'TransactionDay',
        'TransactionHour',
        'TransactionYear'
    ]

    # Handle missing values
    df_extracted[numerical_cols] = df_extracted[numerical_cols].fillna(
        df_extracted[numerical_cols].mean()
    )
    df_extracted[categorical_cols] = df_extracted[
        categorical_cols
    ].fillna('Missing')

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(
        df_extracted,
        columns=categorical_cols,
        drop_first=True
    )

    # Scale numericals
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(
        df_encoded[numerical_cols]
    )

    logger.info("Preprocessing completed successfully.")
    return df_encoded


def build_preprocessing_pipeline():
    """
    Build a unified scikit-learn Pipeline for all preprocessing steps.

    This ensures consistent transformations between training and inference.
    """
    pipeline = Pipeline([
        ('aggregate_features', AggregateFeatureExtractor()),
        ('time_features', TimeSeriesFeatureExtractor()),
        # Note: Risk labeling is done separately for training only
    ])

    return pipeline
