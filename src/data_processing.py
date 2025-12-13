import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the training data.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for preprocessing.")
        return df
        
    # Add preprocessing steps here
    # Example: Simple imputation
    df_clean = df.fillna(0)
    logger.info("Preprocessing completed.")
    return df_clean
