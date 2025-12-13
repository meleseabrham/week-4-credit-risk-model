import pytest
import pandas as pd
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
    # Test with NaN values
    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
    processed_df = preprocess_data(df)
    
    # Check if NaNs are filled
    assert processed_df.isnull().sum().sum() == 0
    assert processed_df.shape == df.shape

def test_preprocess_empty_data():
    df = pd.DataFrame()
    processed_df = preprocess_data(df)
    assert processed_df.empty
