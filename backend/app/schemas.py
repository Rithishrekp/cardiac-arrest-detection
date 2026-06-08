from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class PredictionRequest(BaseModel):
    patient_id: str = Field(..., description="Unique Patient Identifier/MRN")
    patient_name: str = Field(..., description="Full Name of Patient")
    age: int = Field(..., ge=1, le=120, description="Age of Patient in years")
    gender: str = Field(..., description="Gender (e.g. Male, Female, Other)")

    # ECG inputs (in milliseconds)
    rr_interval: float = Field(..., ge=1.0, description="RR Interval in ms")
    pp_interval: float = Field(..., ge=1.0, description="PP Interval in ms")
    qt_interval: float = Field(..., ge=1.0, description="QT Interval in ms")

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "P-1001",
                "patient_name": "Ramesh Kumar",
                "age": 45,
                "gender": "Male",
                "rr_interval": 1084.0,
                "pp_interval": 1090.0,
                "qt_interval": 448.0
            }
        }
    }


class PredictionResponse(BaseModel):
    id: int
    patient_id: str
    patient_name: str
    age: int
    gender: str

    rr_interval: float
    pp_interval: float
    qt_interval: float

    risk_score: float
    risk_level: str
    risk_label: str
    ui_indicator: str
    ensemble_method: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionResultResponse(BaseModel):
    record: PredictionResponse
    suggestions: List[str]
    disclaimer: str = "This is only an ML-based prediction and not a medical diagnosis."


class DashboardStats(BaseModel):
    total_assessments: int
    normal_count: int
    medium_count: int
    critical_count: int
    average_risk_score: float
    recent_assessments: List[PredictionResponse]
