"""Enhanced preprocessing pipeline with instrumentation and WoE support."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xverse.transformer import WOE as XverseWOE
except ImportError:  # pragma: no cover
    XverseWOE = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEBUG_LOG_PATH = r"c:\project\kifya\Week 4\.cursor\debug.log"
CATEGORICAL_FEATURES = [
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "PricingStrategy",
]
NUMERIC_FEATURES = [
    "TotalTransactionAmount",
    "AvgTransactionAmount",
    "StdTransactionAmount",
    "TransactionCount",
    "TotalTransactionValue",
    "AvgTransactionValue",
    "StdTransactionValue",
    "Recency",
]


def _agent_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str):
    """Append NDJSON debug log."""
    payload = {
        "sessionId": "debug-session",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError as exc:  # pragma: no cover
        logger.warning("Failed to write debug log: %s", exc)


def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file with logging."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    try:
        df = pd.read_csv(filepath)
        logger.info("Loaded data from %s with shape %s", filepath, df.shape)
        return df
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected error loading %s: %s", filepath, exc)
        raise


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Add per-transaction temporal features."""

    def __init__(self, datetime_col: str = "TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.datetime_col not in df:
            return df
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors="coerce")
        df["transaction_hour"] = df[self.datetime_col].dt.hour
        df["transaction_day"] = df[self.datetime_col].dt.day
        df["transaction_month"] = df[self.datetime_col].dt.month
        df["transaction_year"] = df[self.datetime_col].dt.year
        df["transaction_dayofweek"] = df[self.datetime_col].dt.dayofweek
        df["transaction_is_weekend"] = (df["transaction_dayofweek"] >= 5).astype(int)
        return df


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level features to customer level."""

    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        amount_col: str = "Amount",
        value_col: str = "Value",
        datetime_col: str = "TransactionStartTime",
        categorical_cols: Optional[List[str]] = None,
        **legacy_kwargs,
    ):
        if "group_col" in legacy_kwargs:
            customer_id_col = legacy_kwargs["group_col"]
        if "value_col" in legacy_kwargs:
            amount_col = legacy_kwargs["value_col"]
        if "time_col" in legacy_kwargs:
            datetime_col = legacy_kwargs["time_col"]

        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.datetime_col = datetime_col
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.customer_id_col not in df:
            raise ValueError("CustomerId column missing from dataframe.")

        if self.datetime_col in df:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors="coerce")

        grouped = df.groupby(self.customer_id_col)

        agg_spec = {
            self.amount_col: ["sum", "mean", "std", "count"],
        }
        if self.value_col in df:
            agg_spec[self.value_col] = ["sum", "mean", "std"]

        other_numeric = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in {self.amount_col, self.value_col}
        ]
        for col in other_numeric:
            agg_spec[col] = "mean"

        customer_df = grouped.agg(agg_spec)
        customer_df.columns = [
            f"{col}_{stat}" if isinstance(stat, str) else f"{col}_{stat}"
            for col, stats in agg_spec.items()
            for stat in ([stats] if isinstance(stats, str) else stats)
        ]

        rename_map = {
            f"{self.amount_col}_sum": "TotalTransactionAmount",
            f"{self.amount_col}_mean": "AvgTransactionAmount",
            f"{self.amount_col}_std": "StdTransactionAmount",
            f"{self.amount_col}_count": "TransactionCount",
            f"{self.value_col}_sum": "TotalTransactionValue",
            f"{self.value_col}_mean": "AvgTransactionValue",
            f"{self.value_col}_std": "StdTransactionValue",
        }
        customer_df = customer_df.rename(columns=rename_map)
        std_cols = [col for col in customer_df.columns if col.endswith("_std")]
        for col in std_cols:
            customer_df[col] = customer_df[col].fillna(0)

        if self.datetime_col in df:
            snapshot_date = df[self.datetime_col].max() + timedelta(days=1)
            last_tx = grouped[self.datetime_col].max()
            customer_df["Recency"] = (snapshot_date - last_tx).dt.days.values
        else:
            customer_df["Recency"] = np.nan

        cat_cols = self.categorical_cols
        if cat_cols is None:
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
            cat_cols = [c for c in cat_cols if c != self.customer_id_col]
        for col in cat_cols:
            if col in df:
                mode_series = grouped[col].agg(
                    lambda s: s.mode().iat[0] if not s.mode().empty else "Missing"
                )
                customer_df[col] = customer_df.index.map(mode_series)

        customer_df = customer_df.reset_index()
        logger.info(
            "Aggregated %s transactions into %s customers", len(df), len(customer_df)
        )
        return customer_df


class WoEIVEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoder with xverse fallback."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        epsilon: float = 1e-4,
    ):
        self.features = features
        self.target_col = target_col
        self.epsilon = epsilon
        self.mapping_: dict = {}
        self.iv_scores_: dict = {}
        self._xverse_encoder = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        if y is None:
            if self.target_col and self.target_col in df:
                y = df[self.target_col]
                df = df.drop(columns=[self.target_col])
            else:
                raise ValueError("Target is required for WoE.")

        cat_cols = self.features or df.select_dtypes(include=["object"]).columns
        if not len(cat_cols):
            return self

        if XverseWOE is not None:
            self._xverse_encoder = XverseWOE()
            self._xverse_encoder.fit(df[cat_cols], y)
            self.iv_scores_ = {
                var: self._xverse_encoder.woe_dict[var]["iv"]
                for var in self._xverse_encoder.woe_dict
            }
        else:
            y_series = pd.Series(y).astype(int)
            total_good = (y_series == 0).sum()
            total_bad = (y_series == 1).sum()
            for col in cat_cols:
                mapping = {}
                iv = 0.0
                for value, subset in df[col].fillna("Missing").groupby(
                    df[col].fillna("Missing")
                ):
                    good = (y_series.loc[subset.index] == 0).sum()
                    bad = (y_series.loc[subset.index] == 1).sum()
                    dist_good = max(good / total_good, self.epsilon)
                    dist_bad = max(bad / total_bad, self.epsilon)
                    woe = np.log(dist_bad / dist_good)
                    iv += (dist_bad - dist_good) * woe
                    mapping[value] = woe
                self.mapping_[col] = mapping
                self.iv_scores_[col] = iv
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self._xverse_encoder is not None:
            cat_cols = list(self._xverse_encoder.woe_dict.keys())
            transformed = self._xverse_encoder.transform(df[cat_cols])
            df[cat_cols] = transformed[cat_cols]
            return df

        for col, mapping in self.mapping_.items():
            if col in df:
                df[f"{col}_woe"] = df[col].fillna("Missing").map(mapping).fillna(0)
                df = df.drop(columns=[col])
        return df


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Imputation, encoding, scaling, with optional WoE."""

    def __init__(
        self,
        apply_woe: bool = False,
        woe_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        run_id: str = "train",
    ):
        self.apply_woe = apply_woe
        self.woe_cols = woe_cols
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.run_id = run_id
        self.imputer_num_: Optional[SimpleImputer] = None
        self.imputer_cat_: Optional[SimpleImputer] = None
        self.woe_encoder_: Optional[WoEIVEncoder] = None
        self.ohe_: Optional[OneHotEncoder] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []

    def _split_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        candidate_num = self.numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_cat = self.categorical_cols or df.select_dtypes(include=["object"]).columns.tolist()
        num_cols = [col for col in candidate_num if col in df.columns]
        cat_cols = [col for col in candidate_cat if col in df.columns]
        return num_cols, cat_cols

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        num_cols, cat_cols = self._split_columns(df)

        self.imputer_num_ = SimpleImputer(strategy="median")
        self.imputer_cat_ = SimpleImputer(strategy="most_frequent")
        if num_cols:
            df[num_cols] = self.imputer_num_.fit_transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.imputer_cat_.fit_transform(df[cat_cols])

        if self.apply_woe and (y is not None):
            self.woe_encoder_ = WoEIVEncoder(features=self.woe_cols or cat_cols)
            df = self.woe_encoder_.fit(df, y).transform(df)
            cat_cols = [col for col in df.columns if df[col].dtype == "object"]

        if cat_cols:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
            encoded = self.ohe_.fit_transform(df[cat_cols])
            encoded_cols = self.ohe_.get_feature_names_out(cat_cols)
            df = pd.concat(
                [df.drop(columns=cat_cols), pd.DataFrame(encoded, columns=encoded_cols, index=df.index)],
                axis=1,
            )

        self.scaler_ = StandardScaler()
        numeric_final = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_final] = self.scaler_.fit_transform(df[numeric_final])
        self.feature_names_ = df.columns.tolist()

        _agent_log(
            "data_processing.py:DataPreprocessor.fit",
            "Fitted preprocessing pipeline",
            {"num_cols": len(num_cols), "cat_cols": len(cat_cols), "woe": self.apply_woe},
            hypothesis_id="H1",
            run_id=self.run_id,
        )
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        num_cols, cat_cols = self._split_columns(df)

        if self.imputer_num_ and num_cols:
            df[num_cols] = self.imputer_num_.transform(df[num_cols])
        if self.imputer_cat_ and cat_cols:
            df[cat_cols] = self.imputer_cat_.transform(df[cat_cols])

        if self.apply_woe and self.woe_encoder_ is not None:
            df = self.woe_encoder_.transform(df)
            cat_cols = [col for col in df.columns if df[col].dtype == "object"]

        if cat_cols and self.ohe_ is not None:
            encoded = self.ohe_.transform(df[cat_cols])
            encoded_cols = self.ohe_.get_feature_names_out(cat_cols)
            df = pd.concat(
                [df.drop(columns=cat_cols), pd.DataFrame(encoded, columns=encoded_cols, index=df.index)],
                axis=1,
            )

        for col in self.feature_names_:
            if col not in df:
                df[col] = 0
        df = df[self.feature_names_]

        numeric_final = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.scaler_ and numeric_final:
            df[numeric_final] = self.scaler_.transform(df[numeric_final])

        _agent_log(
            "data_processing.py:DataPreprocessor.transform",
            "Transformed dataset batch",
            {"shape": df.shape},
            hypothesis_id="H2",
            run_id=self.run_id,
        )
        return df


class RiskLabelAssigner(BaseEstimator, TransformerMixin):
    """Cluster-based proxy target generator."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_: Optional[KMeans] = None
        self.high_risk_cluster_: Optional[int] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        required_cols = ["TotalTransactionAmount", "TransactionCount", "Recency"]
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing RFM columns, skipping risk labeling.")
            return df

        rfm = df[required_cols].fillna(0)
        sample_count = len(rfm)
        if sample_count == 0:
            df["is_high_risk"] = 0
            return df

        cluster_count = min(self.n_clusters, sample_count)
        if cluster_count < 2:
            df["is_high_risk"] = 0
            high_idx = df["TotalTransactionAmount"].idxmin()
            df.loc[high_idx, "is_high_risk"] = 1
            _agent_log(
                "data_processing.py:RiskLabelAssigner.transform",
                "Fallback proxy target (insufficient samples)",
                {"customers": sample_count},
                hypothesis_id="H3",
                run_id="risk",
            )
            return df

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        self.kmeans_ = KMeans(n_clusters=cluster_count, random_state=self.random_state, n_init=10)
        df["cluster"] = self.kmeans_.fit_predict(rfm_scaled)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.inertia_ = self.kmeans_.inertia_

        summary = df.groupby("cluster")[required_cols].mean()
        self.high_risk_cluster_ = summary["TotalTransactionAmount"].idxmin()
        df["is_high_risk"] = (df["cluster"] == self.high_risk_cluster_).astype(int)
        df = df.drop(columns=["cluster"])

        _agent_log(
            "data_processing.py:RiskLabelAssigner.transform",
            "Proxy target assignment complete",
            {"high_risk_cluster": int(self.high_risk_cluster_), "inertia": float(self.inertia_)},
            hypothesis_id="H3",
            run_id="risk",
        )
        return df


def preprocess_data(
    df: pd.DataFrame,
    is_training: bool = True,
    apply_woe: bool = False,
    run_id: str = "preprocess",
    preprocessor: Optional[DataPreprocessor] = None,
    fit: bool = True,
):
    """Complete preprocessing entry point for training and inference."""
    if df.empty:
        logger.warning("Received empty DataFrame for preprocessing.")
        return df
        
    df = pd.DataFrame(df).copy()
    temporal = TemporalFeatureExtractor().transform(df)
    aggregated = AggregateFeatureExtractor(
        categorical_cols=CATEGORICAL_FEATURES
    ).transform(temporal)

    if is_training:
        aggregated = RiskLabelAssigner().transform(aggregated)

    target = aggregated["is_high_risk"] if "is_high_risk" in aggregated else None
    features_df = aggregated.drop(columns=["CustomerId"], errors="ignore")

    if preprocessor is None:
        preprocessor = DataPreprocessor(
            apply_woe=apply_woe,
            woe_cols=CATEGORICAL_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
            numeric_cols=NUMERIC_FEATURES,
            run_id=run_id,
        )

    if fit:
        preprocessor = preprocessor.fit(features_df, target)
    processed = preprocessor.transform(features_df)

    _agent_log(
        "data_processing.py:preprocess_data",
        "Preprocessing complete",
        {
            "input_shape": df.shape,
            "output_shape": processed.shape,
            "training": is_training,
            "apply_woe": apply_woe,
        },
        hypothesis_id="H4",
        run_id=run_id,
    )
    return processed
