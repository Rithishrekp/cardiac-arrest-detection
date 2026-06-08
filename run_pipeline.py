from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

# Set matplotlib backend to Agg before importing anything that might import matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MODULE_DIR = os.path.dirname(__file__)
_PROCESSED_DIR = os.path.join(_MODULE_DIR, "ml", "datasets", "processed")
_SAVED_MODELS_DIR = os.path.join(_MODULE_DIR, "ml", "saved_models")
_OUTPUT_DIR = os.path.join(_MODULE_DIR, "outputs")

def run_ml_pipeline():
    print("\n" + "="*70)
    print("STARTING CARDIAC ARREST DETECTION ML TRAINING & EVALUATION PIPELINE")
    print("="*70)

    # -------------------------------------------------------------------------
    # STEP 1: Preprocessing & Cleaning Clinical Tabular Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1/4] Running Clinical Preprocessing...")
    from ml.preprocessing.predict_preprocess import clean_static_data
    df_clean, encoder_map = clean_static_data()
    print(f"[OK] Cleaned clinical data. Shape: {df_clean.shape}")
    print(f"[OK] Created demographic encoder maps: {list(encoder_map.keys())}")

    # -------------------------------------------------------------------------
    # STEP 2: Clinical Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[STEP 2/4] Running Feature Engineering...")
    from ml.feature_engineering.predict_features import extract_interval_features
    feature_df, info = extract_interval_features()
    print(f"[OK] Extracted sliding-window interval features. Shape: {feature_df.shape}")
    print(f"[OK] Feature set columns: {info.get('raw_columns', [])} and derived statistics.")

    # -------------------------------------------------------------------------
    # STEP 3: Model Training (Tabular Models)
    # -------------------------------------------------------------------------
    print("\n[STEP 3/4] Training Clinical Risk Classification Models (GridSearchCV)...")
    from ml.model_training.train_prediction_model import train_xgboost_gridsearch, train_lightgbm_gridsearch, derive_binary_target
    
    # Derive target: High Risk = QTInterval >= 450 ms
    combined_df = derive_binary_target(feature_df)
    
    print("\n---> Training XGBoost Model...")
    xgb_results = train_xgboost_gridsearch(combined_df, model_name="best_xgboost")
    xgb_metrics = xgb_results["metrics"]
    print("[OK] XGBoost training complete.")
    print(f"  - Best Params: {xgb_metrics['best_params']}")
    print(f"  - Best CV F1 Score: {xgb_metrics['best_cv_score']:.4f}")
    print(f"  - Test F1 Score: {xgb_metrics['test_f1']:.4f}")
    print(f"  - Test Precision: {xgb_metrics['test_precision']:.4f}")
    print(f"  - Test Recall: {xgb_metrics['test_recall']:.4f}")

    print("\n---> Training LightGBM Model...")
    lgb_results = train_lightgbm_gridsearch(combined_df, model_name="best_lightgbm")
    lgb_metrics = lgb_results["metrics"]
    print("[OK] LightGBM training complete.")
    print(f"  - Best Params: {lgb_metrics['best_params']}")
    print(f"  - Best CV F1 Score: {lgb_metrics['best_cv_score']:.4f}")
    print(f"  - Test F1 Score: {lgb_metrics['test_f1']:.4f}")
    print(f"  - Test Precision: {lgb_metrics['test_precision']:.4f}")
    print(f"  - Test Recall: {lgb_metrics['test_recall']:.4f}")

    # -------------------------------------------------------------------------
    # STEP 4: Comprehensive Model Evaluation & Visualizations
    # -------------------------------------------------------------------------
    print("\n[STEP 4/4] Evaluating and Generating Visualizations...")
    from ml.model_evaluation.evaluate_metrics import evaluate_and_visualize
    
    # Evaluate XGBoost
    evaluate_and_visualize(
        y_true=xgb_results["y_test"],
        y_pred=xgb_results["y_pred"],
        y_prob=xgb_results["y_prob"],
        class_names=["Stable/Normal", "High_Cardiac_Risk"],
        prefix="best_xgboost_"
    )
    
    # Evaluate LightGBM
    evaluate_and_visualize(
        y_true=lgb_results["y_test"],
        y_pred=lgb_results["y_pred"],
        y_prob=lgb_results["y_prob"],
        class_names=["Stable/Normal", "High_Cardiac_Risk"],
        prefix="best_lightgbm_"
    )
    
    # Run the exploratory correlation analysis script
    print("\n---> Running Correlation Analysis Visualizer...")
    from ml.visualizations.correlation_analysis import run_correlation_analysis
    run_correlation_analysis(print_report=False)
    
    print("\n" + "="*70)
    print("PIPELINE RUN SUCCESSFUL! ML OUTPUTS & VISUALIZATIONS GENERATED")
    print("="*70)
    print(f"  Cleaned Datasets: ml/datasets/processed/")
    print(f"  Saved Models:     ml/saved_models/ (best_xgboost.pkl, best_lightgbm.pkl)")
    print(f"  Visualizations:   ml/visualizations/ (ROC, PR, and Confusion Matrices)")
    print(f"  Correlation Maps: outputs/ (correlation_heatmap.png, correlation_dashboard.png)")
    print("="*70 + "\n")

    # Determine best model
    best_model_name = "XGBoost" if xgb_metrics["test_f1"] >= lgb_metrics["test_f1"] else "LightGBM"
    best_f1 = max(xgb_metrics["test_f1"], lgb_metrics["test_f1"])
    print(f"[RECOMMENDATION] {best_model_name} is the best performing clinical risk model with a Test F1-Score of {best_f1:.4f}!\n")

if __name__ == "__main__":
    run_ml_pipeline()
