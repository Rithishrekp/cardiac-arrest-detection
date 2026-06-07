from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

_MODULE_DIR = os.path.dirname(__file__)
_SAVED_MODELS_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "saved_models"))
_PROCESSED_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "datasets", "processed"))


def _score(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    metrics = {
        f"{prefix}precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    return metrics


def derive_binary_target(
    feature_df: pd.DataFrame | None = None,
    static_csv: str | None = None,
    qt_threshold: float = 450.0,
    return_combined: bool = True,
) -> pd.DataFrame:
    """Derive a binary cardiac-risk target from QTInterval.

    High Risk = QTInterval >= *qt_threshold* → label 1.
    All others → label 0.
    """
    if static_csv is None:
        static_csv = os.path.join(_PROCESSED_DIR, "static_features_clean.csv")
    static = pd.read_csv(static_csv)

    qt = pd.to_numeric(static.get("QTInterval", static.get("QTInterval (ms)")), errors="coerce")
    y = (qt >= qt_threshold).astype(int).values

    if feature_df is None:
        predict_csv = os.path.join(_PROCESSED_DIR, "predict_features.csv")
        feature_df = pd.read_csv(predict_csv)

    if return_combined:
        out = feature_df.copy()
        out["CardiacRisk_Encoded"] = y
        return out
    return feature_df, y


def train_xgboost_gridsearch(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "CardiacRisk_Encoded",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    param_grid: dict[str, list[Any]] | None = None,
    scoring: str = "f1",
    save_model: bool = True,
    model_name: str = "best_xgboost",
    verbose: int = 1,
) -> dict[str, Any]:
    """Train an XGBoost classifier with GridSearchCV hyperparameter sweep.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both features and the target column.
    feature_cols : list[str] | None
        Columns to use as features.  If ``None``, all numeric columns
        except *target_col* are used.
    target_col : str
        Name of the binary target column (default ``"CardiacRisk_Encoded"``).
    test_size : float
        Fraction held out for final evaluation (default ``0.2``).
    random_state : int
        Seed for reproducibility (default ``42``).
    cv_folds : int
        Number of cross-validation folds (default ``5``).
    param_grid : dict | None
        Hyperparameter grid.  Default sweeps ``max_depth`` (3, 5, 7),
        ``learning_rate`` (0.01, 0.1, 0.2), ``n_estimators`` (50, 100, 200).
    scoring : str
        GridSearchCV scoring metric (default ``"f1"``).
    save_model : bool
        Persist the best model + scaler via ``joblib`` (default ``True``).
    model_name : str
        Stem for the saved model file (default ``"best_xgboost"``).
    verbose : int
        GridSearchCV verbosity (default ``1``).

    Returns
    -------
    dict
        Keys: ``"model"``, ``"scaler"``, ``"metrics"``,
        ``"grid_search_results"``, ``"feature_importance"``,
        ``"confusion_matrix"``, ``"classification_report"``,
        ``"X_train"``, ``"X_test"``, ``"y_train"``, ``"y_test"``,
        ``"y_pred"``, ``"y_prob"``, ``"feature_cols"``.
    """
    from xgboost import XGBClassifier

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if param_grid is None:
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
        }

    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
    )
    grid.fit(X_train_s, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test_s)
    y_prob = best.predict_proba(X_test_s)[:, 1]

    metrics = _score(y_test, y_pred, prefix="test_")
    metrics["best_params"] = grid.best_params_
    metrics["best_cv_score"] = grid.best_score_

    result: dict[str, Any] = {
        "model": best,
        "scaler": scaler,
        "metrics": metrics,
        "grid_search_results": grid.cv_results_,
        "feature_importance": dict(zip(feature_cols, best.feature_importances_)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_cols": feature_cols,
    }

    if save_model:
        os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
        import joblib

        joblib.dump(
            {
                "model": best,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "best_params": grid.best_params_,
            },
            os.path.join(_SAVED_MODELS_DIR, f"{model_name}.pkl"),
        )

    return result


def train_lightgbm_gridsearch(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "CardiacRisk_Encoded",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    param_grid: dict[str, list[Any]] | None = None,
    scoring: str = "f1",
    save_model: bool = True,
    model_name: str = "best_lightgbm",
    verbose: int = -1,
) -> dict[str, Any]:
    """Train a LightGBM classifier with GridSearchCV hyperparameter sweep.

    API mirrors :func:`train_xgboost_gridsearch`; see its docstring.
    """
    import lightgbm as lgb

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if param_grid is None:
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
        }

    base = lgb.LGBMClassifier(random_state=random_state, verbose=verbose)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=max(0, verbose),
    )
    grid.fit(X_train_s, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test_s)
    y_prob = best.predict_proba(X_test_s)[:, 1]

    metrics = _score(y_test, y_pred, prefix="test_")
    metrics["best_params"] = grid.best_params_
    metrics["best_cv_score"] = grid.best_score_

    result: dict[str, Any] = {
        "model": best,
        "scaler": scaler,
        "metrics": metrics,
        "grid_search_results": grid.cv_results_,
        "feature_importance": dict(zip(feature_cols, best.feature_importances_)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_cols": feature_cols,
    }

    if save_model:
        os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
        import joblib

        joblib.dump(
            {
                "model": best,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "best_params": grid.best_params_,
            },
            os.path.join(_SAVED_MODELS_DIR, f"{model_name}.pkl"),
        )

    return result
