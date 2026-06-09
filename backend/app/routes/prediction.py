from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import PredictionRecord
from ..schemas import PredictionRequest, PredictionResultResponse, PredictionResponse
from ..services.prediction_service import make_prediction, get_engine

router = APIRouter(prefix="", tags=["Prediction & Health"])

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
def predict_cardiac_risk(request: PredictionRequest, db: Session = Depends(get_db)):
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
            
            risk_score=pred_out["risk_score"],
            risk_level=pred_out["risk_level"],
            risk_label=pred_out["risk_label"],
            ui_indicator=pred_out["ui_indicator"],
            ensemble_method=pred_out["ensemble_method"]
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
