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

class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Creates aggregate features per customer (RFM proxies).
    """
    def __init__(self, group_col='CustomerId', value_col='Amount'):
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Creating aggregate features...")
        X_ = X.copy()
        
        # Calculate aggregates
        grouped = X_.groupby(self.group_col)[self.value_col]
        
        X_['TotalTransactionAmount'] = grouped.transform('sum')
        X_['AvgTransactionAmount'] = grouped.transform('mean')
        X_['TransactionCount'] = grouped.transform('count')
        X_['StdTransactionAmount'] = grouped.transform('std').fillna(0) # Std is NaN if count=1
        
        return X_

class WoETransformerProp(BaseEstimator, TransformerMixin):
    """
    Wrapper for xverse WOE or custom implementation.
    """
    def __init__(self, target_col='FraudResult', feature_cols=None):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.woe_model = None

    def fit(self, X, y=None):
        if WOE and self.feature_cols:
            self.woe_model = WOE()
            # X must contain target for fit
            if y is not None:
                # If y is passed separately (pipeline standard)
                combined = X.copy()
                combined[self.target_col] = y
                self.woe_model.fit(combined[self.feature_cols], combined[self.target_col])
            elif self.target_col in X.columns:
                 self.woe_model.fit(X[self.feature_cols], X[self.target_col])
        return self

    def transform(self, X):
        if self.woe_model:
            logger.info("Applying WoE transformation...")
            return self.woe_model.transform(X[self.feature_cols])
        return X

def get_data_processing_pipeline(categorical_cols, numerical_cols):
    """
    Returns a scikit-learn pipeline for data processing.
    """
    
    # Numerical Steps: Impute -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical Steps: Impute -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Full Pipeline
    # Note: We apply feature extractors BEFORE ColumnTransformer because they generate new columns
    # But sklearn ColumnTransformer requires fixed columns. 
    # For simplicity, we define the extraction steps as separate manageable functions to be called before pipeline, 
    # or wrapping them in a big pipeline that doesn't use ColumnTransformer immediately.
    
    # Alternative: A robust function that applies the custom transformers, then the standard preprocessing
    return preprocessor

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing function executing the steps.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for preprocessing.")
        return df

    logger.info("Starting preprocessing pipeline...")
    
    # 1. Feature Extraction
    time_extractor = TimeSeriesFeatureExtractor()
    df_extracted = time_extractor.transform(df)
    
    agg_extractor = AggregateFeatureExtractor(group_col='CustomerId', value_col='Amount')
    df_extracted = agg_extractor.transform(df_extracted)
    
    # 2. Define column groups (Post-extraction)
    numerical_cols = ['Amount', 'Value', 'TotalTransactionAmount', 'AvgTransactionAmount', 
                      'TransactionCount', 'StdTransactionAmount', 'TransactionHour', 
                      'TransactionDay', 'TransactionMonth', 'TransactionYear']
    
    # Filter only those that exist
    numerical_cols = [c for c in numerical_cols if c in df_extracted.columns]
    
    categorical_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    # Ensure they exist
    categorical_cols = [c for c in categorical_cols if c in df_extracted.columns]

    # 3. Standard Preprocessing (Encoding/Scaling)
    pipeline = get_data_processing_pipeline(categorical_cols, numerical_cols)
    
    # We strip the target and ID columns for the transformation
    # Keeping raw dataframe structure for return, or returning numpy array? 
    # Usually returning DataFrame is clearer for the user.
    
    # For now, let's execute the transformers "in place" conceptually or return the transformed df
    # Since ColumnTransformer returns an array/sparse matrix, we might lose column names.
    # To keep names, we can use set_output(transform="pandas") in newer sklearn versions.
    
    # Let's apply simple manual transformation for readability or re-construct DF.
    
    # Apply Encoding
    # Handle Missing
    df_extracted[numerical_cols] = df_extracted[numerical_cols].fillna(df_extracted[numerical_cols].mean())
    df_extracted[categorical_cols] = df_extracted[categorical_cols].fillna('Missing')
    
    # One Hot Encoding
    df_encoded = pd.get_dummies(df_extracted, columns=categorical_cols, drop_first=True)
    
    # Scaling - only on numerical features
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    logger.info("Preprocessing completed successfully.")
    return df_encoded
