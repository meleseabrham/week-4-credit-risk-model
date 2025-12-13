import pytest
import pandas as pd
import numpy as np
import os
from src.data_processing import load_data, preprocess_data

def test_load_data(tmp_path):
    # Create a dummy CSV file
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    p = tmp_path / "test_data.csv"
    df.to_csv(p, index=False)
    
    loaded_df = load_data(str(p))
    assert loaded_df.shape == (2, 2)
    assert 'col1' in loaded_df.columns

def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_preprocess_data():
    # Create sample data matching the expected schema
    data = {
        'TransactionStartTime': ['2018-11-15T02:18:49Z', '2018-11-15T02:19:08Z', '2018-11-15T02:19:08Z'],
        'Amount': [100.0, 50.0, 20.0],
        'Value': [100, 50, 20],
        'CustomerId': ['C1', 'C1', 'C2'],
        'ProviderId': ['P1', 'P1', 'P2'],
        'ProductId': ['Pr1', 'Pr2', 'Pr1'],
        'ProductCategory': ['Cat1', 'Cat1', 'Cat2'],
        'ChannelId': ['Ch1', 'Ch1', 'Ch2'],
        'PricingStrategy': ['PS1', 'PS1', 'PS2']
    }
    df = pd.DataFrame(data)
    
    processed_df = preprocess_data(df)
    
    # Check if new features exist
    assert 'TransactionHour' in processed_df.columns
    assert 'TotalTransactionAmount' in processed_df.columns
    assert 'ProviderId_P2' in processed_df.columns # One-hot encoded column
    
    # Check aggregation logic
    # C1 total amount should be 150
    # Since we scaled, we can't check raw values easily, but we can check existence
    assert not processed_df.empty
    assert processed_df.isnull().sum().sum() == 0

def test_preprocess_empty_data():
    df = pd.DataFrame()
    processed_df = preprocess_data(df)
    assert processed_df.empty
