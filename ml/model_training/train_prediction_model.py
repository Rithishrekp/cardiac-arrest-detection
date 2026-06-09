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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

_MODULE_DIR = os.path.dirname(__file__)
_SAVED_MODELS_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "saved_models"))
_PROCESSED_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "datasets", "processed"))


def train_sports_risk_models(
    dataset_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    save_model: bool = True,
) -> dict[str, Any]:
    """
    Train Random Forest and XGBoost classifiers on the integrated 57-feature dataset.
    Selects the best model and saves it as cardiac_risk_model.pkl.
    """
    if dataset_path is None:
        dataset_path = os.path.join(_PROCESSED_DIR, "Integrated_Dataset_Final.csv")

    print(f"Loading integrated dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 1. Handle Categorical Columns
    # SportType is the key categorical input feature
    sport_encoder = LabelEncoder()
    df["SportType_Encoded"] = sport_encoder.fit_transform(df["SportType"].astype(str))

    # 2. Split Features & Target
    # Exclude IDs, target, text columns, and internal source indicators
    exclude_cols = [
        "CardiacRisk_Encoded", 
        "CardiacRisk", 
        "_Source_", 
        "SportType", 
        "SubjectID", 
        "RecordID"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].fillna(0.0).values.astype(np.float64)
    y = df["CardiacRisk_Encoded"].values.astype(int)

    # Calculate baseline feature means and standard deviations for live explainability
    # This is crucial for Phase 4 (explaining individual deviations from the mean)
    feature_means = df[feature_cols].mean().to_dict()
    feature_stds = df[feature_cols].std().to_dict()
    # Replace standard deviation 0 with 1 to avoid division by zero
    for k, v in feature_stds.items():
        if pd.isna(v) or v == 0:
            feature_stds[k] = 1.0

    print(f"Target distribution: {dict(pd.Series(y).value_counts())}")
    print(f"Features dimension: {X.shape[1]} features")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Set up cross-validation folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # 3. Train Random Forest Classifier
    print("\n---> Training Random Forest Classifier (GridSearchCV)...")
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 12, None],
        "min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=random_state),
        param_grid=rf_param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    rf_grid.fit(X_train_s, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test_s)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    print(f"[OK] Random Forest trained. Test F1 (Weighted): {f1_rf:.4f}")

    # 4. Train XGBoost Classifier
    print("\n---> Training XGBoost Classifier (GridSearchCV)...")
    xgb_param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 150],
    }
    xgb_grid = GridSearchCV(
        estimator=XGBClassifier(eval_metric="mlogloss", random_state=random_state),
        param_grid=xgb_param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    xgb_grid.fit(X_train_s, y_train)
    best_xgb = xgb_grid.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test_s)
    f1_xgb = f1_score(y_test, y_pred_xgb, average="weighted", zero_division=0)
    print(f"[OK] XGBoost trained. Test F1 (Weighted): {f1_xgb:.4f}")

    # 5. Select Best Model
    if f1_xgb >= f1_rf:
        best_model = best_xgb
        best_f1 = f1_xgb
        y_pred = y_pred_xgb
        y_prob = best_xgb.predict_proba(X_test_s)
        model_type = "XGBoost"
    else:
        best_model = best_rf
        best_f1 = f1_rf
        y_pred = y_pred_rf
        y_prob = best_rf.predict_proba(X_test_s)
        model_type = "RandomForest"

    print(f"\n[MODEL SELECTION] Selected {model_type} as best model with Test F1: {best_f1:.4f}")

    # Calculate global feature importances
    importances = (
        best_model.feature_importances_ 
        if hasattr(best_model, "feature_importances_") 
        else np.zeros(len(feature_cols))
    )
    feature_importance_dict = dict(zip(feature_cols, [float(v) for v in importances]))

    # Save model and preprocessors
    if save_model:
        os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(_SAVED_MODELS_DIR, "cardiac_risk_model.pkl")
        joblib.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "sport_encoder_classes": sport_encoder.classes_.tolist(),
                "feature_importances": feature_importance_dict,
                "feature_means": feature_means,
                "feature_stds": feature_stds,
                "model_type": model_type,
            },
            model_path,
        )
        print(f"[SAVED] Saved model package to: {model_path}")

    return {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model_type": model_type,
    }


# Backwards compatibility wrappers
def derive_binary_target(*args, **kwargs):
    return pd.DataFrame()


def train_xgboost_gridsearch(df, **kwargs):
    return train_sports_risk_models(save_model=True)


def train_lightgbm_gridsearch(df, **kwargs):
    return train_sports_risk_models(save_model=True)


if __name__ == "__main__":
    train_sports_risk_models()
