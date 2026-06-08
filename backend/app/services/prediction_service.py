import os
import sys
import joblib
import pandas as pd
from sqlalchemy.orm import Session

# Inject project root path so FastAPI can load the ml module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.inference.realtime_inference_engine import CardiacAssessmentEngine
from ..models import PredictionRecord

# Initialize the assessment engine as a singleton
engine = None

def get_engine() -> CardiacAssessmentEngine:
    """Load the ML inference engine, preferring best_xgboost.pkl when configured."""
    global engine
    if engine is None:
        assessor = CardiacAssessmentEngine()
        model_name = os.getenv("PREDICTION_MODEL", "best_xgboost.pkl")
        models_dir = os.path.join(project_root, "ml", "saved_models")
        model_path = os.path.join(models_dir, model_name)
        if os.path.isfile(model_path):
            bundle = joblib.load(model_path)
            assessor._prediction_model = bundle.get("model")
            assessor._scaler = bundle.get("scaler")
            assessor._feature_cols = bundle.get("feature_cols")
        engine = assessor
    return engine

def get_sliding_window_features(db: Session, patient_id: str, new_rr: float, new_pp: float, new_qt: float) -> dict:
    """
    Retrieve the last 4 records for the patient, add the new record,
    and compute rolling statistics (mean, variance, min, max, delta) over a window of 5.
    """
    # Fetch latest 4 records in descending order
    records = db.query(PredictionRecord)\
        .filter(PredictionRecord.patient_id == patient_id)\
        .order_by(PredictionRecord.created_at.desc())\
        .limit(4)\
        .all()
    
    # Reverse so they are in chronological order (oldest to newest)
    records.reverse()
    
    # Build list of dicts
    history_data = []
    for r in records:
        history_data.append({
            "RRInterval": r.rr_interval,
            "PPInterval": r.pp_interval,
            "QTInterval": r.qt_interval
        })
        
    # Append the new reading
    history_data.append({
        "RRInterval": new_rr,
        "PPInterval": new_pp,
        "QTInterval": new_qt
    })
    
    df = pd.DataFrame(history_data)
    
    features = {}
    features["RRInterval"] = float(new_rr)
    features["PPInterval"] = float(new_pp)
    features["QTInterval"] = float(new_qt)
    
    # Replicate the ML feature engineering pipeline sliding stats
    for col in ["RRInterval", "PPInterval", "QTInterval"]:
        series = df[col]
        roll = series.rolling(window=5, min_periods=1)
        
        features[f"{col}_mean"] = float(roll.mean().iloc[-1])
        features[f"{col}_var"] = float(roll.var(ddof=0).iloc[-1])
        features[f"{col}_min"] = float(roll.min().iloc[-1])
        features[f"{col}_max"] = float(roll.max().iloc[-1])
        features[f"{col}_delta"] = float(series.diff().fillna(0.0).iloc[-1])
        
    return features

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

def make_prediction(db: Session, patient_id: str, new_rr: float, new_pp: float, new_qt: float) -> dict:
    """Orchestrates features generation and runs the model prediction."""
    # Ensure engine is loaded
    assessor = get_engine()
    
    # 1. Compute tabular features
    features = get_sliding_window_features(db, patient_id, new_rr, new_pp, new_qt)
    
    # 2. Assemble packet format
    packet = {
        "patient_id": patient_id,
        "tabular": features
    }
    
    # 3. Assess using the ML engine
    result = assessor.assess(packet)
    
    # 4. Generate recommendations
    suggestions = get_health_suggestions(result["risk_level"])
    
    return {
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "risk_label": result["risk_label"],
        "ui_indicator": result["ui_indicator"],
        "ensemble_method": result["ensemble_method"],
        "suggestions": suggestions
    }
