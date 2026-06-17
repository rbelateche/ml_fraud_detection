"""Shared preprocessing — one definition used by every model.

Building the feature transform once and reusing it across all candidates is how
we guarantee an apples-to-apples bake-off and, later, training/serving parity.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.data import schema


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Return a ColumnTransformer for the canonical features.

    Parameters
    ----------
    scale_numeric:
        Standardise numeric features. Needed for linear models; harmless but
        unnecessary for trees (set False to skip for tree models).
    """
    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, schema.NUMERIC_FEATURES),
            ("cat", categorical_pipe, schema.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into feature matrix X and label vector y."""
    x = df[schema.feature_columns()].copy()
    y = df[schema.LABEL].astype(int).copy()
    return x, y
