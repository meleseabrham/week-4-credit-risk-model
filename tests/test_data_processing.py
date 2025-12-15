"""
Tests for the enhanced data processing pipeline.
"""
import pandas as pd

from src.data_processing import AggregateFeatureExtractor, preprocess_data


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
    """Training mode should create proxy target and encoded features."""
    df = make_sample_df()
    processed_df = preprocess_data(df, is_training=True)

    assert not processed_df.empty
    assert "is_high_risk" in processed_df.columns
    assert "TotalTransactionAmount" in processed_df.columns
    assert any(col.startswith("ProviderId_") for col in processed_df.columns)


def test_preprocess_data_inference_mode():
    """Inference mode skips proxy target creation."""
    df = make_sample_df()
    processed_df = preprocess_data(df, is_training=False)

    assert not processed_df.empty
    assert "is_high_risk" not in processed_df.columns
    assert any(col.startswith("ProviderId_") for col in processed_df.columns)


def test_preprocess_empty_dataframe():
    """Test handling of empty dataframe."""
    df = pd.DataFrame()
    processed_df = preprocess_data(df)
    assert processed_df.empty


def test_aggregate_feature_extractor():
    """Aggregate extractor should compute customer-level totals."""
    df = make_sample_df()
    extractor = AggregateFeatureExtractor()
    transformed = extractor.transform(df)

    for col in ["TotalTransactionAmount", "AvgTransactionAmount", "Recency"]:
        assert col in transformed.columns

    c1_total = transformed.loc[
        transformed["CustomerId"] == "C1", "TotalTransactionAmount"
    ].iloc[0]
    assert c1_total == 150.0


def test_missing_timestamp_handling():
    """Test that missing timestamps are handled without crash."""
    df = make_sample_df()
    # Ensure one row has valid timestamp for snapshot calculation
    processed_df = preprocess_data(df, is_training=False)
    assert not processed_df.empty
