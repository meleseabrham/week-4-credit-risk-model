import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xverse.transformer import WOE as XverseWOE
except ImportError:  # pragma: no cover
    XverseWOE = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEBUG_LOG_PATH = r"c:\project\kifya\Week 4\.cursor\debug.log"


# region agent log
def _agent_log(location: str, message: str, data: dict, hypothesis_id: str = "H1", run_id: str = "run1"):
    """Append NDJSON debug log."""
    payload = {
        "sessionId": "debug-session",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(datetime.utcnow().timestamp() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Debug log failed: {exc}")
# endregion


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at {filepath} was not found.")
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except Exception as e:  # pragma: no cover
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract time-based features from a timestamp column."""

    def __init__(self, time_col: str = "TransactionStartTime"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        if self.time_col in X_:
            X_[self.time_col] = pd.to_datetime(X_[self.time_col], errors="coerce")
            X_["TransactionHour"] = X_[self.time_col].dt.hour
            X_["TransactionDay"] = X_[self.time_col].dt.day
            X_["TransactionMonth"] = X_[self.time_col].dt.month
            X_["TransactionYear"] = X_[self.time_col].dt.year
            X_["TransactionDayOfWeek"] = X_[self.time_col].dt.dayofweek
            X_["TransactionIsWeekend"] = (X_["TransactionDayOfWeek"] >= 5).astype(int)
        return X_


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level data to customer level, preserving key categoricals via mode."""

    def __init__(
        self,
        group_col: str = "CustomerId",
        value_col: str = "Amount",
        time_col: str = "TransactionStartTime",
        categorical_cols: Optional[List[str]] = None,
    ):
        self.group_col = group_col
        self.value_col = value_col
        self.time_col = time_col
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        if self.group_col not in X_ or self.value_col not in X_:
            raise ValueError(f"Missing required columns: {self.group_col}, {self.value_col}")

        if self.time_col in X_:
            X_[self.time_col] = pd.to_datetime(X_[self.time_col], errors="coerce")

        grouped = X_.groupby(self.group_col)
        agg_df = grouped[self.value_col].agg(
            TotalTransactionAmount="sum",
            AvgTransactionAmount="mean",
            StdTransactionAmount="std",
            TransactionCount="count",
        ).reset_index()
        agg_df["StdTransactionAmount"] = agg_df["StdTransactionAmount"].fillna(0)

        # Aggregate categorical columns (mode) to retain for WoE/encoding
        cat_cols = self.categorical_cols
        if cat_cols is None:
            cat_cols = X_.select_dtypes(include=["object"]).columns.tolist()
            cat_cols = [c for c in cat_cols if c not in [self.group_col, self.time_col, self.value_col]]
        for col in cat_cols:
            if col in X_:
                mode_series = grouped[col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "Missing")
                agg_df[col] = agg_df[self.group_col].map(mode_series)

        if self.time_col in X_:
            snapshot_date = X_[self.time_col].max() + timedelta(days=1)
            last_tx = grouped[self.time_col].max()
            agg_df["Recency"] = agg_df[self.group_col].map(lambda cid: (snapshot_date - last_tx.loc[cid]).days)
        else:
            agg_df["Recency"] = np.nan
        return agg_df


class WoEIVEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence and Information Value encoder for categorical features."""

    def __init__(self, features: Optional[List[str]] = None, target_col: Optional[str] = None, epsilon: float = 1e-4):
        self.features = features
        self.target_col = target_col
        self.epsilon = epsilon
        self.iv_scores_ = {}
        self.mapping_ = {}
        self._uses_xverse = XverseWOE is not None
        self._xverse_encoder = None

    def fit(self, X, y=None):
        X_ = pd.DataFrame(X).copy()
        if y is None:
            if self.target_col and self.target_col in X_:
                y = X_[self.target_col]
                X_ = X_.drop(columns=[self.target_col])
            else:
                raise ValueError("Target is required for WoE/IV computation.")

        cat_cols = self.features or X_.select_dtypes(include=["object"]).columns.tolist()
        if not cat_cols:
            return self

        if self._uses_xverse:
            # xverse WOE expects fit/transform on the same column set; we fit on cat_cols only
            self._xverse_encoder = XverseWOE()
            self._xverse_encoder.fit(X_[cat_cols], y)
            try:
                self.iv_scores_ = {var: self._xverse_encoder.woe_dict[var]["iv"] for var in cat_cols}
            except Exception:  # pragma: no cover
                self.iv_scores_ = {}
        else:
            y_series = pd.Series(y).astype(int)
            total_good = (y_series == 0).sum()
            total_bad = (y_series == 1).sum()
            for col in cat_cols:
                iv = 0.0
                self.mapping_[col] = {}
                for category, subset in X_[col].fillna("Missing").groupby(X_[col].fillna("Missing")):
                    good = (y_series.loc[subset.index] == 0).sum()
                    bad = (y_series.loc[subset.index] == 1).sum()
                    dist_good = max(good / total_good, self.epsilon)
                    dist_bad = max(bad / total_bad, self.epsilon)
                    woe = np.log(dist_bad / dist_good)
                    iv += (dist_bad - dist_good) * woe
                    self.mapping_[col][category] = woe
                self.iv_scores_[col] = iv
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        if self._uses_xverse and self._xverse_encoder is not None:
            # Apply WOE to categorical subset and merge back
            cat_cols = list(self._xverse_encoder.woe_dict.keys())
            transformed_subset = self._xverse_encoder.transform(X_[cat_cols])
            X_[cat_cols] = transformed_subset[cat_cols]
            return X_

        for col, mapping in self.mapping_.items():
            if col not in X_:
                continue
            X_[f"{col}_woe"] = X_[col].fillna("Missing").map(mapping).fillna(0)
            X_.drop(columns=[col], inplace=True)
        return X_


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing:
    - Temporal features
    - Aggregations (RFM-like)
    - Missing value handling
    - Optional WoE/IV on categorical features
    - One-hot encoding for remaining categoricals
    - Scaling for numeric features
    """

    def __init__(
        self,
        apply_woe: bool = False,
        woe_features: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        run_id: str = "run1",
    ):
        self.apply_woe = apply_woe
        self.woe_features = woe_features
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.run_id = run_id
        self.imputer_num_ = None
        self.imputer_cat_ = None
        self.woe_encoder_ = None
        self.ohe_ = None
        self.scaler_ = None
        self.feature_names_ = []

    def _select_columns(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        num_cols = self.numeric_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.categorical_cols or X.select_dtypes(include=["object"]).columns.tolist()
        return num_cols, cat_cols

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        df = TimeSeriesFeatureExtractor().transform(df)
        df = AggregateFeatureExtractor().transform(df)

        num_cols, cat_cols = self._select_columns(df)

        self.imputer_num_ = SimpleImputer(strategy="median")
        self.imputer_cat_ = SimpleImputer(strategy="most_frequent")
        if num_cols:
            df[num_cols] = self.imputer_num_.fit_transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.imputer_cat_.fit_transform(df[cat_cols])

        if self.apply_woe and (y is not None):
            self.woe_encoder_ = WoEIVEncoder(features=self.woe_features or cat_cols)
            df = self.woe_encoder_.fit(df, y).transform(df)
            cat_cols = [c for c in df.columns if df[c].dtype == "object"]

        if cat_cols:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
            encoded = self.ohe_.fit_transform(df[cat_cols])
            encoded_cols = self.ohe_.get_feature_names_out(cat_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df.drop(columns=cat_cols), df_encoded], axis=1)

        self.scaler_ = StandardScaler()
        num_cols_final = df.select_dtypes(include=[np.number]).columns.tolist()
        df[num_cols_final] = self.scaler_.fit_transform(df[num_cols_final])

        self.feature_names_ = df.columns.tolist()

        _agent_log(
            location="data_processing.py:DataPreprocessor.fit",
            message="Fitted preprocessing",
            data={"num_cols": num_cols, "cat_cols": cat_cols, "woe": self.apply_woe, "features_out": len(self.feature_names_)},
            hypothesis_id="H1",
            run_id=self.run_id,
        )
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        df = TimeSeriesFeatureExtractor().transform(df)
        df = AggregateFeatureExtractor().transform(df)

        num_cols, cat_cols = self._select_columns(df)

        if num_cols:
            df[num_cols] = self.imputer_num_.transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.imputer_cat_.transform(df[cat_cols])

        if self.apply_woe and self.woe_encoder_ is not None:
            df = self.woe_encoder_.transform(df)
            cat_cols = [c for c in df.columns if df[c].dtype == "object"]

        if cat_cols and self.ohe_ is not None:
            encoded = self.ohe_.transform(df[cat_cols])
            encoded_cols = self.ohe_.get_feature_names_out(cat_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df.drop(columns=cat_cols), df_encoded], axis=1)

        for col in self.feature_names_:
            if col not in df:
                df[col] = 0
        df = df[self.feature_names_]

        num_cols_final = df.select_dtypes(include=[np.number]).columns.tolist()
        df[num_cols_final] = self.scaler_.transform(df[num_cols_final])

        _agent_log(
            location="data_processing.py:DataPreprocessor.transform",
            message="Transformed data",
            data={"shape": df.shape},
            hypothesis_id="H2",
            run_id=self.run_id,
        )
        return df


def preprocess_data(
    df: pd.DataFrame,
    y: Optional[pd.Series] = None,
    apply_woe: bool = False,
    woe_features: Optional[List[str]] = None,
    run_id: str = "run1",
    preprocessor: Optional[DataPreprocessor] = None,
    fit: bool = True,
):
    """Preprocess data and return (processed_df, preprocessor)."""
    if df.empty:
        logger.warning("Received empty DataFrame for preprocessing.")
        return df, preprocessor or DataPreprocessor()

    if preprocessor is None:
        preprocessor = DataPreprocessor(apply_woe=apply_woe, woe_features=woe_features, run_id=run_id)

    if fit:
        preprocessor = preprocessor.fit(df, y)
    processed = preprocessor.transform(df)

    _agent_log(
        location="data_processing.py:preprocess_data",
        message="Preprocess completed",
        data={"input_shape": df.shape, "output_shape": processed.shape, "apply_woe": apply_woe},
        hypothesis_id="H3",
        run_id=run_id,
    )
    return processed, preprocessor

