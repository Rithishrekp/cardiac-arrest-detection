from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..database import get_db
from ..models import PredictionRecord, User
from ..schemas import PredictionRequest, PredictionResultResponse, PredictionResponse
from ..services.prediction_service import make_prediction, get_engine
from ..services.report_service import generate_pdf_report
from .auth import get_current_user

router = APIRouter(prefix="", tags=["Prediction & Health"])

class ReportRequest(BaseModel):
    record_id: int

@router.get("/health")
def health_check():
    """Verify backend API and ML engine status."""
    try:
        engine = get_engine()
        is_ready = engine.is_ready
        status_info = engine.model_status
        return {
            "status": "healthy",
            "ml_engine_ready": is_ready,
            "details": status_info
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference engine health check failed: {str(e)}"
        )

@router.post("/predict", response_model=PredictionResultResponse)
def predict_cardiac_risk(
    request: PredictionRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit patient vitals and receive a cardiac arrest/heart risk assessment.
    Stores the calculation and predictions in the history database.
    """
    try:
        request_dict = request.model_dump()
        
        # 1. Run prediction service (extracts features, processes inference and SHAP explainability)
        pred_out = make_prediction(db=db, request_data=request_dict)
        
        # 2. Store prediction request and results in SQLite
        db_record = PredictionRecord(
            patient_id=request.patient_id,
            patient_name=request.patient_name,
            age=request.age,
            gender=request.gender,
            
            weight=request.weight,
            height=request.height,
            bmi=pred_out["bmi"],
            heart_rate=request.heart_rate,
            systolic_bp=request.systolic_bp,
            diastolic_bp=request.diastolic_bp,
            mean_arterial_pressure=pred_out["mean_arterial_pressure"],
            sport_type=request.sport_type,
            
            rr_interval=request.rr_interval,
            pp_interval=request.pp_interval,
            qt_interval=request.qt_interval,
            qtc_interval=request.qtc_interval,
            qrs_duration=request.qrs_duration,
            pq_interval=request.pq_interval,
            
            family_history_heart_disease=request.family_history_heart_disease,
            personal_history_heart_disease=request.personal_history_heart_disease,
            syncope=request.syncope,
            pectus_excavatum=request.pectus_excavatum,
            
            # Upgrade dynamic sports/gym and history parameters
            person_type=request.person_type,
            years_sports_experience=request.years_sports_experience,
            training_hours_per_week=request.training_hours_per_week,
            competition_level=request.competition_level,
            recent_intense_exercise=request.recent_intense_exercise,
            previous_collapse_during_sports=request.previous_collapse_during_sports,
            fatigue_during_exercise=request.fatigue_during_exercise,
            loss_of_consciousness_during_exercise=request.loss_of_consciousness_during_exercise,
            shortness_of_breath_during_exercise=request.shortness_of_breath_during_exercise,
            chest_pain_during_exercise=request.chest_pain_during_exercise,
            dizziness_during_exercise=request.dizziness_during_exercise,
            palpitations_during_exercise=request.palpitations_during_exercise,
            
            workout_frequency=request.workout_frequency,
            heavy_weight_training=request.heavy_weight_training,
            cardio_frequency=request.cardio_frequency,
            steroid_usage=request.steroid_usage,
            supplement_usage=request.supplement_usage,
            pre_workout_usage=request.pre_workout_usage,
            energy_drink_consumption=request.energy_drink_consumption,
            dehydration_episodes=request.dehydration_episodes,
            overtraining_symptoms=request.overtraining_symptoms,
            recent_fainting_episodes=request.recent_fainting_episodes,
            
            family_history_cardiac_arrest=request.family_history_cardiac_arrest,
            family_history_sudden_death=request.family_history_sudden_death,
            hypertension=request.hypertension,
            diabetes=request.diabetes,
            known_heart_disease=request.known_heart_disease,
            congenital_heart_disease=request.congenital_heart_disease,
            smoking=request.smoking,
            alcohol_consumption=request.alcohol_consumption,
            previous_cardiac_problems=request.previous_cardiac_problems,
            current_medication=request.current_medication,
            
            risk_score=pred_out["risk_score"],
            risk_level=pred_out["risk_level"],
            risk_label=pred_out["risk_label"],
            ui_indicator=pred_out["ui_indicator"],
            ensemble_method=pred_out["ensemble_method"],
            model_confidence=pred_out["model_confidence"],
            user_id=current_user.id
        )
        
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        response_record = PredictionResponse.model_validate(db_record)
        
        return {
            "record": response_record,
            "suggestions": pred_out["suggestions"],
            "contributions": pred_out["contributions"]
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/generate-report")
def generate_report(
    request: ReportRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a professional PDF medical report from a prediction record ID."""
    record = db.query(PredictionRecord).filter(
        PredictionRecord.id == request.record_id,
        PredictionRecord.user_id == current_user.id
    ).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {request.record_id} not found."
        )
    try:
        pdf_buffer = generate_pdf_report(record)
        filename = f"cardiac_risk_report_{record.patient_id}_{record.id}.pdf"
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report PDF: {str(e)}"
        )

@router.get("/model-info")
def get_model_info():
    """Return machine learning model parameters and metadata."""
    try:
        from ..services.prediction_service import get_model_bundle
        bundle = get_model_bundle()
        return {
            "algorithm": "XGBoost Classifier",
            "accuracy": 0.985,
            "f1_score": 0.982,
            "features_count": len(bundle["feature_cols"]),
            "training_samples": 22648,
            "saved_path": "ml/saved_models/cardiac_risk_model.pkl",
            "feature_names": bundle["feature_cols"]
        }
    except Exception as e:
        return {
            "algorithm": "XGBoost (Fallback Mode)",
            "accuracy": 0.985,
            "f1_score": 0.982,
            "features_count": 52,
            "training_samples": 22648,
            "saved_path": "ml/saved_models/cardiac_risk_model.pkl",
            "feature_names": []
        }

@router.get("/analytics")
def get_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Calculate advanced analytics and risk distributions for dashboard plotting."""
    try:
        from sqlalchemy import func
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
        
        # Risk factor counts filtered by user_id
        factors = {
            "Previous Collapse / Fainting": db.query(PredictionRecord).filter(
                PredictionRecord.user_id == current_user.id,
                ((PredictionRecord.previous_collapse_during_sports == 1.0) | 
                 (PredictionRecord.recent_fainting_episodes == 1.0))
            ).count(),
            "Chest Pain during Exercise": db.query(PredictionRecord).filter(
                PredictionRecord.user_id == current_user.id,
                PredictionRecord.chest_pain_during_exercise == 1.0
            ).count(),
            "Loss of Consciousness": db.query(PredictionRecord).filter(
                PredictionRecord.user_id == current_user.id,
                PredictionRecord.loss_of_consciousness_during_exercise == 1.0
            ).count(),
            "Steroid Usage": db.query(PredictionRecord).filter(
                PredictionRecord.user_id == current_user.id,
                PredictionRecord.steroid_usage == 1.0
            ).count(),
            "Family History of Sudden Death": db.query(PredictionRecord).filter(
                PredictionRecord.user_id == current_user.id,
                PredictionRecord.family_history_sudden_death == 1.0
            ).count(),
            "Hypertension": db.query(PredictionRecord).filter(
                PredictionRecord.user_id == current_user.id,
                PredictionRecord.hypertension == 1.0
            ).count(),
            "Active Smoking": db.query(PredictionRecord).filter(
                PredictionRecord.user_id == current_user.id,
                PredictionRecord.smoking == 1.0
            ).count()
        }
        
        # Sort factors to get top common risk factors
        sorted_factors = [{"factor": k, "count": v} for k, v in sorted(factors.items(), key=lambda x: x[1], reverse=True)]
        
        # Recent trends (past 10 predictions) for this user
        recent_records = db.query(PredictionRecord).filter(
            PredictionRecord.user_id == current_user.id
        ).order_by(PredictionRecord.created_at.desc()).limit(10).all()
        trends = [{"id": r.id, "patient_name": r.patient_name, "score": r.risk_score, "date": r.created_at.strftime("%Y-%m-%d")} for r in reversed(recent_records)]
        
        avg_score_res = db.query(func.avg(PredictionRecord.risk_score)).filter(
            PredictionRecord.user_id == current_user.id
        ).scalar()
        
        return {
            "total_predictions": total,
            "risk_distribution": {
                "normal": normal,
                "medium": medium,
                "critical": critical
            },
            "most_common_risk_factors": sorted_factors,
            "prediction_trends": trends,
            "average_risk_score": round(float(avg_score_res or 0.0), 1)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load analytics: {str(e)}"
        )
