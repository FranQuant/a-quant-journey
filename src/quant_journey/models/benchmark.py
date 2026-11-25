"""Deterministic classification benchmark."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import FEATURE_COLUMNS


def fit_logistic_benchmark(train: pd.DataFrame) -> Pipeline:
    """Fit a train-only logistic regression pipeline."""
    required = list(FEATURE_COLUMNS) + ["target_up"]

    if train.empty or not set(required).issubset(train.columns):
        raise ValueError("Missing training features or target")
    if not np.isfinite(train[required].to_numpy()).all():
        raise ValueError("Nonfinite training values")
    if not train["target_up"].isin([0, 1]).all():
        raise ValueError("Target must be binary")
    if train["target_up"].nunique() != 2:
        raise ValueError("Training data must contain both classes")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver="lbfgs",
        )),
    ])

    model.fit(train[list(FEATURE_COLUMNS)], train["target_up"])
    return model


def predict_benchmark(
    model: Pipeline,
    dataset: pd.DataFrame,
) -> pd.Series:
    """Predict without fitting or modifying the supplied model."""
    features = dataset.loc[:, list(FEATURE_COLUMNS)]

    if features.empty or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Invalid prediction features")

    return pd.Series(
        model.predict(features),
        index=features.index,
        name="prediction",
    )
