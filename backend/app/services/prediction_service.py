import os
import sys
import joblib
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

# Inject project root path so FastAPI can load the ml module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..models import PredictionRecord

# Global variable to cache the loaded model bundle
_model_bundle = None

def get_model_bundle() -> dict:
    """Load the trained sports risk model bundle from disk (cached as a singleton)."""
    global _model_bundle
    if _model_bundle is None:
        model_path = os.path.join(project_root, "ml", "saved_models", "cardiac_risk_model.pkl")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Trained model file not found at: {model_path}")
        _model_bundle = joblib.load(model_path)
    return _model_bundle

def get_engine():
    """Mock/backwards compatibility wrapper for main.py checks."""
    class MockEngine:
        is_ready = True
        model_status = {"prediction_model_loaded": True, "detection_model_loaded": False}
    return MockEngine()

def get_health_suggestions(risk_level: str) -> list[str]:
    """Return specific actionable healthcare tips based on risk level."""
    if risk_level == "normal":
        return [
            "Maintain a balanced diet rich in fiber, whole grains, and lean proteins.",
            "Engage in at least 150 minutes of moderate-intensity aerobic exercise weekly.",
            "Ensure regular, quality sleep (7-8 hours per night) and manage daily stress.",
            "Continue scheduled routine medical examinations and blood work."
        ]
    elif risk_level == "medium":
        return [
            "Schedule a clinical consultation with a primary care physician or cardiologist.",
            "Monitor resting blood pressure and heart rate daily, keeping a logged journal.",
            "Adopt a low-sodium, heart-healthy diet and limit intake of saturated fats.",
            "Avoid stimulant substances, excessive caffeine, and tobacco products."
        ]
    else:  # critical
        return [
            "WARNING: High risk of arrhythmia or acute cardiac complications detected.",
            "Seek emergency medical services (EMS) immediately if you experience chest pain, shortness of breath, or palpitations.",
            "Have someone stay with you and do NOT drive yourself to the emergency room.",
            "Keep emergency contacts and a list of current medications easily accessible."
        ]

def calculate_feature_contributions(
    patient_data: dict, 
    feature_cols: list, 
    importances: dict, 
    means: dict, 
    stds: dict
) -> list[dict]:
    """
    Calculate patient-specific feature contributions (explainability).
    Compares patient values against training baseline averages, weighted by global importance.
    """
    raw_contributions = []
    
    # Map raw input keys to feature columns in dataset
    key_mapping = {
        "age": "Age",
        "weight": "Weight",
        "height": "Height",
        "bmi": "BMI",
        "heart_rate": "HeartRate",
        "systolic_bp": "SystolicBP",
        "diastolic_bp": "DiastolicBP",
        "mean_arterial_pressure": "MeanArterialPressure",
        "qt_interval": "QTInterval",
        "qtc_interval": "QTcInterval",
        "qrs_duration": "QRSDuration",
        "pq_interval": "PQInterval",
        "rr_interval": "RRInterval",
        "pp_interval": "PPInterval",
        "family_history_heart_disease": "FamilyHistoryHeartDisease",
        "personal_history_heart_disease": "PersonalHistoryHeartDisease",
        "syncope": "Syncope",
        "pectus_excavatum": "PectusExcavatum"
    }

    for request_key, feat_col in key_mapping.items():
        if feat_col in feature_cols and request_key in patient_data:
            val = float(patient_data[request_key])
            mean_val = float(means.get(feat_col, 0.0))
            std_val = float(stds.get(feat_col, 1.0))
            importance = float(importances.get(feat_col, 0.0))
            
            # Z-score deviation from mean
            deviation = (val - mean_val) / std_val
            abs_deviation = abs(deviation)
            
            # Contribution weight = deviation * feature importance
            weight = abs_deviation * importance
            
            # Human readable label
            feature_name = feat_col
            # Add spaces to camelcase names for UI
            import re
            feature_name_spaced = re.sub(r"([A-Z])", r" \1", feature_name).strip()
            
            direction = "increase" if val > mean_val else "decrease"
            
            raw_contributions.append({
                "feature": feature_name_spaced,
                "weight": weight,
                "direction": direction,
                "value": val
            })
            
    # Sort contributions by weight descending
    raw_contributions.sort(key=lambda x: x["weight"], reverse=True)
    
    # Select top 4 contributing factors
    top_contributions = raw_contributions[:4]
    
    # Normalize weights to sum to 100%
    total_weight = sum(c["weight"] for c in top_contributions)
    if total_weight == 0:
        total_weight = 1.0  # Avoid division by zero
        
    normalized = []
    for c in top_contributions:
        pct = round((c["weight"] / total_weight) * 100.0, 1)
        normalized.append({
            "feature": c["feature"],
            "contribution": pct if pct > 0 else 5.0, # ensure non-zero visual bar
            "direction": c["direction"],
            "value": c["value"]
        })
        
    return normalized

def make_prediction(db: Session, request_data: dict) -> dict:
    """
    Extracts features, runs inference on the sports risk model, and 
    returns predicted risk score, level, recommendations, and feature contributions.
    """
    bundle = get_model_bundle()
    
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    sport_encoder_classes = bundle["sport_encoder_classes"]
    importances = bundle["feature_importances"]
    means = bundle["feature_means"]
    stds = bundle["feature_stds"]
    
    # 1. Compute calculated inputs
    height_m = request_data["height"] / 100.0
    bmi = request_data["weight"] / (height_m ** 2)
    map_val = (request_data["systolic_bp"] + 2.0 * request_data["diastolic_bp"]) / 3.0
    
    # 2. Encode SportType
    # Fallback to first class if sport_type is unrecognized
    sport_str = str(request_data["sport_type"])
    if sport_str in sport_encoder_classes:
        sport_encoded = sport_encoder_classes.index(sport_str)
    else:
        sport_encoded = 0
        
    # 3. Assemble Feature Vector
    # Map request payload fields to database columns
    val_map = {
        "Age": request_data["age"],
        "Weight": request_data["weight"],
        "Height": request_data["height"],
        "BMI": bmi,
        "HeartRate": request_data["heart_rate"],
        "SystolicBP": request_data["systolic_bp"],
        "DiastolicBP": request_data["diastolic_bp"],
        "MeanArterialPressure": map_val,
        "QTInterval": request_data["qt_interval"],
        "QTcInterval": request_data["qtc_interval"],
        "QRSDuration": request_data["qrs_duration"],
        "PQInterval": request_data["pq_interval"],
        "RRInterval": request_data["rr_interval"],
        "PPInterval": request_data["pp_interval"],
        "SportType_Encoded": sport_encoded,
        "FamilyHistoryHeartDisease": request_data["family_history_heart_disease"],
        "PersonalHistoryHeartDisease": request_data["personal_history_heart_disease"],
        "Syncope": request_data["syncope"],
        "PectusExcavatum": request_data["pectus_excavatum"]
    }
    
    # Build complete vector (all other features like one-hot axis/infarction default to 0)
    feat_vector = []
    for col in feature_cols:
        if col in val_map:
            feat_vector.append(val_map[col])
        else:
            feat_vector.append(0.0) # default fallback for ECG leads or Infarction categories
            
    # 4. Standard Scale & Predict
    feat_arr = np.array(feat_vector, dtype=np.float64).reshape(1, -1)
    feat_arr_s = scaler.transform(feat_arr)
    
    # Multi-class prediction: outputs probabilities for classes 0, 1, 2
    probs = model.predict_proba(feat_arr_s)[0]
    
    # Let's map risk probabilities to score:
    # 0 = Normal, 1 = Moderate Risk, 2 = High Risk
    # Risk Score = probability of Moderate (class 1) * 50 + probability of High (class 2) * 100
    risk_score = float(probs[1] * 50.0 + probs[2] * 100.0)
    risk_score = round(np.clip(risk_score, 0.0, 100.0), 1)
    
    # Determine risk category
    if risk_score <= 40.0:
        risk_level = "normal"
        risk_label = "Low Risk / Normal Cardiac Vitality"
        ui_indicator = "white"
    elif risk_score <= 70.0:
        risk_level = "medium"
        risk_label = "Moderate Risk / Primary Alert"
        ui_indicator = "yellow"
    else:
        risk_level = "critical"
        risk_label = "High Risk / Critical Emergency"
        ui_indicator = "red"
        
    # Calculate patient specific feature contributions
    patient_data_full = {**request_data, "bmi": bmi, "mean_arterial_pressure": map_val}
    contributions = calculate_feature_contributions(patient_data_full, feature_cols, importances, means, stds)
    
    suggestions = get_health_suggestions(risk_level)
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_label": risk_label,
        "ui_indicator": ui_indicator,
        "ensemble_method": "multi_feature_xgboost",
        "suggestions": suggestions,
        "contributions": contributions,
        "bmi": round(bmi, 2),
        "mean_arterial_pressure": round(map_val, 2)
    }
