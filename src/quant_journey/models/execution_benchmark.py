"""Explicit logistic benchmark for execution-aligned return labels."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import FEATURE_COLUMNS


def fit_execution_benchmark(
    train: pd.DataFrame,
    training_cutoff: pd.Timestamp,
) -> Pipeline:
    """Fit only execution labels known before the training cutoff.

    Label availability is strictly earlier than training_cutoff.
    The caller must provide an independently established cutoff.
    """
    required = list(FEATURE_COLUMNS) + [
        "execution_target_up",
        "label_available_at",
    ]

    if train.empty or not set(required).issubset(train.columns):
        raise ValueError("Missing execution training fields")

    cutoff = pd.Timestamp(training_cutoff)

    if pd.isna(cutoff):
        raise ValueError("Invalid training cutoff")

    if not isinstance(train.index, pd.DatetimeIndex):
        raise ValueError("Expected feature-date DatetimeIndex")

    if (
        train.index.hasnans
        or not train.index.is_unique
        or not train.index.is_monotonic_increasing
    ):
        raise ValueError("Invalid feature-date index")

    availability = train["label_available_at"]

    if (
        not pd.api.types.is_datetime64_any_dtype(availability)
        or availability.isna().any()
    ):
        raise ValueError("Invalid label availability")

    if not (availability > train.index).all():
        raise ValueError("Labels must mature after feature dates")

    if not (availability < cutoff).all():
        raise ValueError("Training labels unavailable at cutoff")

    features = train.loc[:, list(FEATURE_COLUMNS)]
    labels = train["execution_target_up"]

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite training features")

    if not labels.isin([0, 1]).all():
        raise ValueError("Invalid execution target")

    if labels.nunique() != 2:
        raise ValueError("Training labels must contain both classes")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver="lbfgs",
        )),
    ])

    model.fit(features, labels)
    return model


def predict_execution_benchmark(
    model: Pipeline,
    dataset: pd.DataFrame,
) -> pd.Series:
    """Predict execution-aligned direction without refitting."""
    features = dataset.loc[:, list(FEATURE_COLUMNS)]

    if features.empty or not np.isfinite(
        features.to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid prediction features")

    predictions = model.predict(features)

    return pd.Series(
        predictions.astype(int),
        index=features.index,
        name="execution_prediction",
    )
