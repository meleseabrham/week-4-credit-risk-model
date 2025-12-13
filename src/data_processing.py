import pandas as pd
import numpy as np
import logging
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
try:
    from xverse.transformer import WOE
except ImportError:
    WOE = None # Handle case if installation failed, though we tried to install it

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at {filepath} was not found.")
        
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from TransactionStartTime.
    """
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
            # Drop original time col if needed, keeping for now or dropping in next steps
            X_ = X_.drop(columns=[self.time_col], errors='ignore')
        return X_

from sklearn.cluster import KMeans

class RiskLabelAssigner(BaseEstimator, TransformerMixin):
    """
    Assigns a proxy risk label based on RFM clustering.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.high_risk_cluster = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Assigning risk labels (Task 4)...")
        X_ = X.copy()
        
        # 1. Prepare RFM Features
        # We assume X_ already has 'TotalTransactionAmount' (Monetary), 'TransactionCount' (Frequency)
        # We need 'Recency'. 
        
        # Ensure we have the necessary columns
        required_cols = ['TotalTransactionAmount', 'TransactionCount', 'Recency']
        if not all(col in X_.columns for col in required_cols):
            logger.warning("Missing RFM columns for clustering. Returning original DF.")
            return X
            
        rfm_data = X_[required_cols].copy()
        
        # 2. Scale RFM before clustering
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data)
        
        # 3. K-Means Clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(rfm_scaled)
        
        X_['Cluster'] = clusters
        
        # 4. Identify High Risk Cluster
        # Logic: Lowest mean Frequency and Monetary, High Recency often indicates engagement drop which acts as a risk proxy here
        # Or specifically "least engaged".
        
        cluster_summary = X_.groupby('Cluster')[['TotalTransactionAmount', 'TransactionCount', 'Recency']].mean()
        
        # We define high risk as the cluster with LOWEST (Monetary + Frequency) and HIGHEST Recency potentially.
        # A simple heuristic: Sort by Monetary ascending. The lowest monetary group is likely the "worst" / least engaged.
        # Let's verify with Frequency too.
        
        # Normalize stats to Pick the 'worst'
        # We want min(Monetary) and min(Frequency). Recency might be high (churned) or just low engagement.
        
        # Let's pick the cluster with the lowest 'TotalTransactionAmount' as the primary proxy for "Low Value/High Risk"
        high_risk_cluster_id = cluster_summary['TotalTransactionAmount'].idxmin()
        
        logger.info(f"Identified Cluster {high_risk_cluster_id} as High Risk (Proxy). Stats:\n{cluster_summary.loc[high_risk_cluster_id]}")
        
        X_['is_high_risk'] = (X_['Cluster'] == high_risk_cluster_id).astype(int)
        
        # Drop aux cluster column
        X_ = X_.drop(columns=['Cluster'])
        
        return X_

class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Creates aggregate features per customer (RFM proxies).
    """
    def __init__(self, group_col='CustomerId', value_col='Amount', time_col='TransactionStartTime'):
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
        
        X_['TotalTransactionAmount'] = grouped[self.value_col].transform('sum')
        X_['AvgTransactionAmount'] = grouped[self.value_col].transform('mean')
        X_['TransactionCount'] = grouped[self.value_col].transform('count')
        X_['StdTransactionAmount'] = grouped[self.value_col].transform('std').fillna(0)
        
        # Recency Calculation
        if self.time_col in X_.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_[self.time_col]):
                X_[self.time_col] = pd.to_datetime(X_[self.time_col])
                
            # Define snapshot date as max date in dataset + 1 day
            snapshot_date = X_[self.time_col].max() + pd.Timedelta(days=1)
            
            # Calculate last transaction date per customer (broadcast transform is tricky for dates, using map)
            last_tx_dates = grouped[self.time_col].max()
            
            # Map back to original DF
            X_['LastTransactionDate'] = X_[self.group_col].map(last_tx_dates)
            X_['Recency'] = (snapshot_date - X_['LastTransactionDate']).dt.days
            
            X_ = X_.drop(columns=['LastTransactionDate'], errors='ignore')
            
        return X_

# ... (WoETransformerProp remains same) ...

# ... (get_data_processing_pipeline remains same) ...

def preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Full preprocessing function executing the steps.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for preprocessing.")
        return df

    logger.info("Starting preprocessing pipeline...")
    
    # 1. Feature Extraction
    # Note: Aggregate now needs time_col for Recency
    agg_extractor = AggregateFeatureExtractor(group_col='CustomerId', value_col='Amount', time_col='TransactionStartTime')
    df_extracted = agg_extractor.transform(df)
    
    time_extractor = TimeSeriesFeatureExtractor()
    df_extracted = time_extractor.transform(df_extracted)

    # 2. Risk Label Assignment (Task 4) - ONLY DURING TRAINING
    if is_training:
        risk_assigner = RiskLabelAssigner()
        df_extracted = risk_assigner.transform(df_extracted)
        logger.info("Risk Label Assignment completed.")
    else:
        logger.info("Skipping Risk Label Assignment (Inference Mode).")

    # 3. Define column groups (Post-extraction)
    numerical_cols = ['Amount', 'Value', 'TotalTransactionAmount', 'AvgTransactionAmount', 
                      'TransactionCount', 'StdTransactionAmount', 'TransactionHour', 
                      'TransactionDay', 'TransactionMonth', 'TransactionYear', 'Recency']
    
    # Filter only those that exist
    numerical_cols = [c for c in numerical_cols if c in df_extracted.columns]
    
    categorical_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    # Ensure they exist
    categorical_cols = [c for c in categorical_cols if c in df_extracted.columns]

    # 4. Standard Preprocessing (Encoding/Scaling)
    # pipeline = get_data_processing_pipeline(categorical_cols, numerical_cols)
    
    # Manual application again to keep DF structure
    df_extracted[numerical_cols] = df_extracted[numerical_cols].fillna(df_extracted[numerical_cols].mean())
    df_extracted[categorical_cols] = df_extracted[categorical_cols].fillna('Missing')
    
    df_encoded = pd.get_dummies(df_extracted, columns=categorical_cols, drop_first=True)
    
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    logger.info("Preprocessing completed successfully.")
    return df_encoded
