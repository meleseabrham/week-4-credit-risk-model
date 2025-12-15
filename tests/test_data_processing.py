"""
Tests for the enhanced data processing pipeline.
"""
import pandas as pd
import pytest
from src.data_processing import preprocess_data, AggregateFeatureExtractor


def make_sample_df():
    return pd.DataFrame(
        {
            "TransactionStartTime": [
                "2023-01-01T10:00:00Z",
                "2023-01-02T11:00:00Z",
                None,
            ],
            "Amount": [100.0, 50.0, 25.0],
            "Value": [100, 50, 25],
            "CustomerId": ["C1", "C1", "C2"],
            "ProviderId": ["P1", "P1", "P2"],
            "ProductId": ["PR1", "PR2", "PR3"],
            "ProductCategory": ["Cat1", "Cat1", "Cat2"],
            "ChannelId": ["Web", "Web", "iOS"],
            "PricingStrategy": ["S1", "S1", "S2"],
        }
    )


def test_preprocess_data_training_mode():
    """Test preprocessing in training mode (creates target)."""
    df = make_sample_df()
    # Mock RiskLabelAssigner logic by ensuring we have RFM columns or handled gracefully
    # The pipeline handles missingness/imputation, so we expect a dataframe back
    
    processed_df = preprocess_data(df, is_training=True)
    
    assert not processed_df.empty
    assert "is_high_risk" in processed_df.columns
    assert "Amount" in processed_df.columns  # Numerical feature
    assert "ProviderId_P2" in processed_df.columns  # OHE feature (if drop_first=True) or present


def test_preprocess_data_inference_mode():
    """Test preprocessing in inference mode (skips target creation)."""
    df = make_sample_df()
    
    processed_df = preprocess_data(df, is_training=False)
    
    assert not processed_df.empty
    assert "is_high_risk" not in processed_df.columns
    # Check for encoded features
    assert any(col.startswith("ProviderId_") for col in processed_df.columns)


def test_preprocess_empty_dataframe():
    """Test handling of empty dataframe."""
    df = pd.DataFrame()
    processed_df = preprocess_data(df)
    assert processed_df.empty


def test_aggregate_feature_extractor():
    """Test the AggregateFeatureExtractor specifically."""
    df = make_sample_df()
    extractor = AggregateFeatureExtractor(
        group_col='CustomerId',
        value_col='Amount',
        time_col='TransactionStartTime'
    )
    transformed = extractor.transform(df)
    
    # Check for expected columns
    expected_cols = [
        'TotalTransactionAmount', 
        'AvgTransactionAmount', 
        'Recency'
    ]
    for col in expected_cols:
        assert col in transformed.columns
    
    # Check logic: C1 has 2 transactions (100+50=150)
    c1_total = transformed[transformed['CustomerId'] == 'C1']['TotalTransactionAmount'].iloc[0]
    assert c1_total == 150.0


def test_missing_timestamp_handling():
    """Test that missing timestamps are handled without crash."""
    df = make_sample_df()
    # Ensure one row has valid timestamp for snapshot calculation
    processed_df = preprocess_data(df, is_training=False)
    assert not processed_df.empty
