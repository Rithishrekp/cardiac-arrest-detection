from __future__ import annotations

import os
import sys

# Set matplotlib backend to Agg before importing anything that might import matplotlib
import matplotlib
matplotlib.use("Agg")

def run_ml_pipeline():
    print("\n" + "="*70)
    print("STARTING CARDIAC ARREST DETECTION ML TRAINING & EVALUATION PIPELINE")
    print("="*70)

    # Note: Preprocessing and feature engineering are bypassed because
    # Integrated_Dataset_Final.csv is already compiled and ready.
    print("\n[INFO] Using pre-compiled Integrated_Dataset_Final.csv (Phase 1 Complete)")

    # -------------------------------------------------------------------------
    # STEP 3: Model Training (Tabular Models)
    # -------------------------------------------------------------------------
    print("\n[STEP 3/4] Training Sports Risk Classification Models (GridSearchCV)...")
    from ml.model_training.train_prediction_model import train_sports_risk_models
    
    results = train_sports_risk_models()
    confusion = results["confusion_matrix"]
    report = results["classification_report"]
    model_type = results["model_type"]

    print("\n[OK] Model training complete.")
    print(f"  - Selected Model: {model_type}")
    print(f"  - Confusion Matrix: {confusion}")
    print(f"  - Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"  - Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")

    # -------------------------------------------------------------------------
    # STEP 4: Comprehensive Model Evaluation & Visualizations
    # -------------------------------------------------------------------------
    print("\n[STEP 4/4] Evaluating and Generating Visualizations...")
    
    # Run the correlation analysis visualizer on the integrated dataset
    print("\n---> Running Correlation Analysis Visualizer...")
    from ml.visualizations.correlation_analysis import run_correlation_analysis
    run_correlation_analysis(print_report=False)
    
    print("\n" + "="*70)
    print("PIPELINE RUN SUCCESSFUL! ML OUTPUTS & VISUALIZATIONS GENERATED")
    print("="*70)
    print(f"  Saved Models:     ml/saved_models/cardiac_risk_model.pkl")
    print(f"  Visualizations:   outputs/ (correlation_heatmap.png, correlation_dashboard.png)")
    print("="*70 + "\n")

    print(f"[RECOMMENDATION] {model_type} is the best performing clinical risk model with a Weighted F1-Score of {report['weighted avg']['f1-score']:.4f}!\n")

if __name__ == "__main__":
    run_ml_pipeline()
