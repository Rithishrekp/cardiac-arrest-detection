from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _score(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute a standard set of classification metrics."""
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics


def train_xgboost(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "CardiacRisk_Encoded",
    test_size: float = 0.2,
    random_state: int = 42,
    **xgboost_kwargs: Any,
) -> dict:
    """Train an XGBoost classifier on static features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-engineered dataset.
    feature_cols : list[str] | None
        Columns to use as features.  If ``None``, all numeric columns
        except *target_col* are used.
    target_col : str
        Name of the target column (default ``"CardiacRisk_Encoded"``).
    test_size : float
        Fraction for the test set (default ``0.2``).
    random_state : int
        Random seed (default ``42``).
    **xgboost_kwargs
        Additional keyword arguments forwarded to ``XGBClassifier``.

    Returns
    -------
    dict
        Keys include ``"model"``, ``"metrics"``, ``"feature_importance"``,
        ``"confusion_matrix"``, and the train/test splits.
    """
    from xgboost import XGBClassifier

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss", random_state=random_state, **xgboost_kwargs
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _score(y_test, y_pred, prefix="test_")
    y_train_pred = model.predict(X_train)
    metrics.update(_score(y_train, y_train_pred, prefix="train_"))

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


def train_lightgbm(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "CardiacRisk_Encoded",
    test_size: float = 0.2,
    random_state: int = 42,
    **lgbm_kwargs: Any,
) -> dict:
    """Train a LightGBM classifier on static features.

    API is identical to :func:`train_xgboost`; see its docstring for
    parameter details.
    """
    import lightgbm as lgb

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = lgb.LGBMClassifier(random_state=random_state, **lgbm_kwargs)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _score(y_test, y_pred, prefix="test_")
    y_train_pred = model.predict(X_train)
    metrics.update(_score(y_train, y_train_pred, prefix="train_"))

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }
