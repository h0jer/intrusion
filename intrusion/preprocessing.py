"""Preprocessing that was applied in the training notebook.

The notebook performs exactly one preprocessing step before training:

    df['dest_port'] = df['dest_port'].fillna(-1)
    df['src_port']  = df['src_port'].fillna(-1)

The model was then fit on all columns except ``label`` in the original column
order. ``FeatureCleaner`` reproduces that step and enforces the same column
order, and is saved *inside* the pipeline together with the model so that raw
test data goes through identical preprocessing at prediction time.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Column order the model was trained on (notebook: df.drop(columns=['label'])).
FEATURE_NAMES: list[str] = [
    "avg_ipt",
    "bytes_in",
    "bytes_out",
    "dest_ip",
    "dest_port",
    "entropy",
    "num_pkts_out",
    "num_pkts_in",
    "proto",
    "src_ip",
    "src_port",
    "time_end",
    "time_start",
    "total_entropy",
    "duration",
]

# Columns imputed with -1 in the notebook.
PORT_COLUMNS: list[str] = ["dest_port", "src_port"]
PORT_FILL_VALUE = -1


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Select/order the training features and impute missing ports with -1."""

    def __init__(self, feature_names=None, port_columns=None, fill_value=PORT_FILL_VALUE):
        self.feature_names = feature_names
        self.port_columns = port_columns
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = list(self.feature_names or FEATURE_NAMES)
        ports = list(self.port_columns or PORT_COLUMNS)

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature column(s): {missing}")

        out = df[features].copy()  # also fixes column order
        try:
            out = out.apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"All features must be numeric: {exc}") from exc

        out[ports] = out[ports].fillna(self.fill_value)

        bad = [c for c in features if out[c].isna().any()]
        if bad:
            raise ValueError(
                f"Null values found in column(s) {bad}. Only {ports} may be "
                "missing (they are imputed with -1, as in training)."
            )
        return out

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names or FEATURE_NAMES)
