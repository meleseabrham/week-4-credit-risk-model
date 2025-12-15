import pandas as pd
import pytest

from src.data_processing import (
    load_data,
    preprocess_data,
    DataPreprocessor,
)


def test_load_data(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    loaded = load_data(str(path))
    assert loaded.shape == (2, 2)
    assert list(loaded.columns) == ["a", "b"]


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("no_such_file.csv")


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


def test_preprocess_handles_missing_timestamp():
    df = make_sample_df()
    processed, preproc = preprocess_data(df, fit=True)
    assert processed.isnull().sum().sum() == 0
    assert processed.shape[0] == df["CustomerId"].nunique()
    # Temporal features should be present after aggregation/scaling
    expected_cols = [col for col in processed.columns if "Transaction" in col or "Recency" in col]
    assert expected_cols
    assert isinstance(preproc, DataPreprocessor)


def test_preprocess_rare_categories_consistency():
    # Fit on train
    train = make_sample_df()
    processed_train, preproc = preprocess_data(train, fit=True)

    # Transform with unseen categories
    test = make_sample_df()
    test.loc[0, "ProviderId"] = "NEW_PROVIDER"
    processed_test, _ = preprocess_data(test, fit=False, preprocessor=preproc)

    # Column alignment preserved
    assert list(processed_train.columns) == list(processed_test.columns)
    assert processed_test.isnull().sum().sum() == 0


def test_woe_encoder_applies_when_requested():
    df = make_sample_df()
    y = pd.Series([0, 1, 0])
    processed, preproc = preprocess_data(df, y=y, apply_woe=True, woe_features=["ChannelId"], fit=True)
    # WoE should create encoded column
    woe_cols = [c for c in processed.columns if "ChannelId" in c]
    assert woe_cols, "WoE columns missing"
    assert preproc.woe_encoder_ is not None


def test_preprocess_empty_dataframe():
    df = pd.DataFrame()
    processed, _ = preprocess_data(df)
    assert processed.empty

