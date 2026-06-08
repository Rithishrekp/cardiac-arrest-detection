from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from ..database import get_db
from ..models import PredictionRecord
from ..schemas import PredictionResponse, PredictionResultResponse, DashboardStats
from ..services.prediction_service import get_health_suggestions

router = APIRouter(prefix="/history", tags=["History & Statistics"])

@router.get("", response_model=List[PredictionResponse])
def get_prediction_history(
    search: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Retrieve previous prediction records, sorted from newest to oldest."""
    query = db.query(PredictionRecord)
    if search:
        # Filter by patient_name or patient_id (case-insensitive)
        query = query.filter(
            PredictionRecord.patient_name.ilike(f"%{search}%") | 
            PredictionRecord.patient_id.ilike(f"%{search}%")
        )
    
    records = query.order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    return records

@router.get("/stats", response_model=DashboardStats)
def get_dashboard_statistics(db: Session = Depends(get_db)):
    """Calculate aggregated metrics for the dashboard home page."""
    try:
        total = db.query(PredictionRecord).count()
        normal = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "normal").count()
        medium = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "medium").count()
        critical = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "critical").count()
        
        avg_score_res = db.query(func.avg(PredictionRecord.risk_score)).scalar()
        avg_score = round(float(avg_score_res), 1) if avg_score_res is not None else 0.0
        
        # Fetch 5 most recent records
        recent = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(5).all()
        
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
def get_prediction_by_id(record_id: int, db: Session = Depends(get_db)):
    """Fetch details of a single historical assessment record, including recommendations."""
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {record_id} not found."
        )
        
    # Reconstruct the dynamic suggestions from the stored risk level
    suggestions = get_health_suggestions(record.risk_level)
    
    return {
        "record": record,
        "suggestions": suggestions
    }
