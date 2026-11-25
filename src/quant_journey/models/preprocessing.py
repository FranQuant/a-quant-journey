"""Train-only feature preprocessing for benchmark models."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "return_1d",
    "momentum_5d",
    "volatility_5d",
)


def fit_training_scaler(train: pd.DataFrame) -> StandardScaler:
    """Fit a scaler exclusively on training feature columns."""
    features = train.loc[:, FEATURE_COLUMNS]

    if features.empty or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Invalid training features")

    return StandardScaler().fit(features)


def transform_features(
    dataset: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply an already-fitted scaler without updating it."""
    features = dataset.loc[:, FEATURE_COLUMNS]

    if features.empty or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Invalid features")

    transformed = scaler.transform(features)

    return pd.DataFrame(
        transformed,
        index=features.index,
        columns=features.columns,
    )
