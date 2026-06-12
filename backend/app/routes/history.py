from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import os
import pandas as pd
import joblib
from ..database import get_db
from ..models import PredictionRecord, User
from ..schemas import PredictionResponse, PredictionResultResponse, DashboardStats
from ..services.prediction_service import get_health_suggestions
from .auth import get_current_user

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

router = APIRouter(prefix="/history", tags=["History & Statistics"])

@router.get("/correlations")
def get_dataset_correlations():
    """Load the pre-computed Pearson correlations with CardiacRisk_Encoded from disk."""
    try:
        csv_path = os.path.abspath(os.path.join(project_root, "outputs", "correlation_percentage_table.csv"))
        if not os.path.exists(csv_path):
            return []
        
        df = pd.read_csv(csv_path)
        
        # Rename the index column (feature name) to "feature"
        df.rename(columns={df.columns[0]: "feature"}, inplace=True)
        
        # Map camelcase to spaced names for UI representation
        import re
        def clean_name(n):
            return re.sub(r"([A-Z])", r" \1", str(n)).strip()
        df["feature"] = df["feature"].apply(clean_name)
        
        # Classify correlation strength
        # Very Strong: >=0.7, Strong: 0.5-0.7, Moderate: 0.3-0.5, Weak: 0.1-0.3, Very Weak: <0.1
        def get_strength(r):
            abs_r = abs(r)
            if abs_r >= 0.70:
                return "Very Strong"
            elif abs_r >= 0.50:
                return "Strong"
            elif abs_r >= 0.30:
                return "Moderate"
            elif abs_r >= 0.10:
                return "Weak"
            else:
                return "Very Weak"
                
        df["strength"] = df["Correlation"].apply(get_strength)
        
        # Fill NaN values for lead metrics that have no variance
        df = df.fillna(0.0)
        
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load correlations: {str(e)}"
        )

@router.get("", response_model=List[PredictionResponse])
def get_prediction_history(
    search: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve previous prediction records, sorted from newest to oldest."""
    query = db.query(PredictionRecord).filter(PredictionRecord.user_id == current_user.id)
    if search:
        # Filter by patient_name or patient_id (case-insensitive)
        query = query.filter(
            PredictionRecord.patient_name.ilike(f"%{search}%") | 
            PredictionRecord.patient_id.ilike(f"%{search}%")
        )
    
    records = query.order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    return records


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Calculate aggregated metrics for the dashboard home page."""
    try:
        total = db.query(PredictionRecord).filter(PredictionRecord.user_id == current_user.id).count()
        normal = db.query(PredictionRecord).filter(
            PredictionRecord.user_id == current_user.id,
            PredictionRecord.risk_level == "normal"
        ).count()
        medium = db.query(PredictionRecord).filter(
            PredictionRecord.user_id == current_user.id,
            PredictionRecord.risk_level == "medium"
        ).count()
        critical = db.query(PredictionRecord).filter(
            PredictionRecord.user_id == current_user.id,
            PredictionRecord.risk_level == "critical"
        ).count()
        
        avg_score_res = db.query(func.avg(PredictionRecord.risk_score)).filter(
            PredictionRecord.user_id == current_user.id
        ).scalar()
        avg_score = round(float(avg_score_res), 1) if avg_score_res is not None else 0.0
        
        # Fetch 5 most recent records for this user
        recent = db.query(PredictionRecord).filter(
            PredictionRecord.user_id == current_user.id
        ).order_by(PredictionRecord.created_at.desc()).limit(5).all()
        
        return {
            "total_assessments": total,
            "normal_count": normal,
            "medium_count": medium,
            "critical_count": critical,
            "average_risk_score": avg_score,
            "recent_assessments": recent
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate stats: {str(e)}"
        )


@router.get("/{record_id}", response_model=PredictionResultResponse)
def get_prediction_by_id(
    record_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetch details of a single historical assessment record, including recommendations."""
    record = db.query(PredictionRecord).filter(
        PredictionRecord.id == record_id,
        PredictionRecord.user_id == current_user.id
    ).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {record_id} not found or unauthorized access."
        )
        
    # Reconstruct the dynamic suggestions from the stored risk level
    suggestions = get_health_suggestions(record.risk_level)
    
    # We load mock contributions for historic metrics if needed or return empty list
    # Reconstruct mock contributions from importances for display
    bundle = joblib.load(os.path.join(project_root, "ml", "saved_models", "cardiac_risk_model.pkl"))
    importances = bundle["feature_importances"]
    means = bundle["feature_means"]
    stds = bundle["feature_stds"]
    feature_cols = bundle["feature_cols"]
    
    # Map record to dict, checking for None values to avoid float conversion errors on old records
    record_dict = {
        "age": record.age if record.age is not None else 25.0,
        "weight": record.weight if record.weight is not None else 70.0,
        "height": record.height if record.height is not None else 170.0,
        "bmi": record.bmi if record.bmi is not None else 24.2,
        "heart_rate": record.heart_rate if record.heart_rate is not None else 70.0,
        "systolic_bp": record.systolic_bp if record.systolic_bp is not None else 120.0,
        "diastolic_bp": record.diastolic_bp if record.diastolic_bp is not None else 80.0,
        "mean_arterial_pressure": record.mean_arterial_pressure if record.mean_arterial_pressure is not None else 93.3,
        "qt_interval": record.qt_interval if record.qt_interval is not None else 400.0,
        "qtc_interval": record.qtc_interval if record.qtc_interval is not None else 410.0,
        "qrs_duration": record.qrs_duration if record.qrs_duration is not None else 95.0,
        "pq_interval": record.pq_interval if record.pq_interval is not None else 150.0,
        "rr_interval": record.rr_interval if record.rr_interval is not None else 800.0,
        "pp_interval": record.pp_interval if record.pp_interval is not None else 800.0,
        "family_history_heart_disease": record.family_history_heart_disease if record.family_history_heart_disease is not None else 0.0,
        "personal_history_heart_disease": record.personal_history_heart_disease if record.personal_history_heart_disease is not None else 0.0,
        "syncope": record.syncope if record.syncope is not None else 0.0,
        "pectus_excavatum": record.pectus_excavatum if record.pectus_excavatum is not None else 0.0
    }

    from ..services.prediction_service import calculate_feature_contributions
    contributions = calculate_feature_contributions(record_dict, feature_cols, importances, means, stds)
    
    return {
        "record": record,
        "suggestions": suggestions,
        "contributions": contributions
    }
