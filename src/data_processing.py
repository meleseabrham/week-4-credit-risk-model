import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the training data."""
    # Add preprocessing steps here
    return df.fillna(0)
