from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class PredictionRequest(BaseModel):
    patient_id: str = Field(..., description="Unique Patient Identifier/MRN")
    patient_name: str = Field(..., description="Full Name of Patient")
    age: float = Field(..., ge=1, le=120, description="Age of Patient in years")
    gender: str = Field(..., description="Gender (e.g. Male, Female, Other)")

    # Demographics & Sports Vitals
    weight: float = Field(..., ge=5.0, le=300.0, description="Weight in kg")
    height: float = Field(..., ge=50.0, le=250.0, description="Height in cm")
    heart_rate: float = Field(..., ge=30.0, le=220.0, description="Heart rate in bpm")
    systolic_bp: float = Field(..., ge=50.0, le=250.0, description="Systolic Blood Pressure")
    diastolic_bp: float = Field(..., ge=35.0, le=160.0, description="Diastolic Blood Pressure")
    sport_type: str = Field(..., description="Sport category type (e.g. ATH, AMF, VOL)")

    # ECG interval inputs
    rr_interval: float = Field(..., ge=100.0, description="RR Interval in ms")
    pp_interval: float = Field(..., ge=100.0, description="PP Interval in ms")
    qt_interval: float = Field(..., ge=100.0, description="QT Interval in ms")
    qtc_interval: float = Field(..., ge=100.0, description="QTc Interval in ms")
    qrs_duration: float = Field(..., ge=20.0, description="QRS Duration in ms")
    pq_interval: float = Field(..., ge=20.0, description="PQ Interval in ms")

    # Medical & Family History (binary flags)
    family_history_heart_disease: float = Field(0.0, description="Family History of heart disease (0.0 or 1.0)")
    personal_history_heart_disease: float = Field(0.0, description="Personal History of heart disease (0.0 or 1.0)")
    syncope: float = Field(0.0, description="Syncope / fainting episodes history (0.0 or 1.0)")
    pectus_excavatum: float = Field(0.0, description="Pectus Excavatum skeletal issue (0.0 or 1.0)")

    # Upgraded Lifestyle and Sports/Gym dynamic features
    person_type: Optional[str] = "sports"
    years_sports_experience: Optional[float] = 0.0
    training_hours_per_week: Optional[float] = 0.0
    competition_level: Optional[str] = "Amateur"
    recent_intense_exercise: Optional[float] = 0.0
    previous_collapse_during_sports: Optional[float] = 0.0
    fatigue_during_exercise: Optional[float] = 0.0
    loss_of_consciousness_during_exercise: Optional[float] = 0.0
    shortness_of_breath_during_exercise: Optional[float] = 0.0
    chest_pain_during_exercise: Optional[float] = 0.0
    dizziness_during_exercise: Optional[float] = 0.0
    palpitations_during_exercise: Optional[float] = 0.0

    workout_frequency: Optional[float] = 0.0
    heavy_weight_training: Optional[float] = 0.0
    cardio_frequency: Optional[float] = 0.0
    steroid_usage: Optional[float] = 0.0
    supplement_usage: Optional[float] = 0.0
    pre_workout_usage: Optional[float] = 0.0
    energy_drink_consumption: Optional[float] = 0.0
    dehydration_episodes: Optional[float] = 0.0
    overtraining_symptoms: Optional[float] = 0.0
    recent_fainting_episodes: Optional[float] = 0.0

    family_history_cardiac_arrest: Optional[float] = 0.0
    family_history_sudden_death: Optional[float] = 0.0
    hypertension: Optional[float] = 0.0
    diabetes: Optional[float] = 0.0
    known_heart_disease: Optional[float] = 0.0
    congenital_heart_disease: Optional[float] = 0.0
    smoking: Optional[float] = 0.0
    alcohol_consumption: Optional[float] = 0.0
    previous_cardiac_problems: Optional[float] = 0.0
    current_medication: Optional[str] = ""
    user_id: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "P-1001",
                "patient_name": "Ramesh Kumar",
                "age": 45.0,
                "gender": "Male",
                "weight": 70.0,
                "height": 175.0,
                "heart_rate": 72.0,
                "systolic_bp": 120.0,
                "diastolic_bp": 80.0,
                "sport_type": "ATH",
                "rr_interval": 1084.0,
                "pp_interval": 1090.0,
                "qt_interval": 448.0,
                "qtc_interval": 430.0,
                "qrs_duration": 96.0,
                "pq_interval": 160.0,
                "family_history_heart_disease": 0.0,
                "personal_history_heart_disease": 0.0,
                "syncope": 0.0,
                "pectus_excavatum": 0.0,
                "person_type": "sports",
                "years_sports_experience": 5.0,
                "training_hours_per_week": 8.0,
                "competition_level": "Professional",
                "recent_intense_exercise": 1.0,
                "previous_collapse_during_sports": 0.0
            }
        }
    }


class PredictionResponse(BaseModel):
    id: int
    patient_id: str
    patient_name: str
    age: float
    gender: str

    weight: Optional[float] = 0.0
    height: Optional[float] = 0.0
    bmi: Optional[float] = 0.0
    heart_rate: Optional[float] = 0.0
    systolic_bp: Optional[float] = 0.0
    diastolic_bp: Optional[float] = 0.0
    mean_arterial_pressure: Optional[float] = 0.0
    sport_type: Optional[str] = "ATH"

    rr_interval: float
    pp_interval: float
    qt_interval: float
    qtc_interval: Optional[float] = 0.0
    qrs_duration: Optional[float] = 0.0
    pq_interval: Optional[float] = 0.0

    family_history_heart_disease: Optional[float] = 0.0
    personal_history_heart_disease: Optional[float] = 0.0
    syncope: Optional[float] = 0.0
    pectus_excavatum: Optional[float] = 0.0

    person_type: Optional[str] = "sports"
    years_sports_experience: Optional[float] = 0.0
    training_hours_per_week: Optional[float] = 0.0
    competition_level: Optional[str] = "Amateur"
    recent_intense_exercise: Optional[float] = 0.0
    previous_collapse_during_sports: Optional[float] = 0.0
    fatigue_during_exercise: Optional[float] = 0.0
    loss_of_consciousness_during_exercise: Optional[float] = 0.0
    shortness_of_breath_during_exercise: Optional[float] = 0.0
    chest_pain_during_exercise: Optional[float] = 0.0
    dizziness_during_exercise: Optional[float] = 0.0
    palpitations_during_exercise: Optional[float] = 0.0

    workout_frequency: Optional[float] = 0.0
    heavy_weight_training: Optional[float] = 0.0
    cardio_frequency: Optional[float] = 0.0
    steroid_usage: Optional[float] = 0.0
    supplement_usage: Optional[float] = 0.0
    pre_workout_usage: Optional[float] = 0.0
    energy_drink_consumption: Optional[float] = 0.0
    dehydration_episodes: Optional[float] = 0.0
    overtraining_symptoms: Optional[float] = 0.0
    recent_fainting_episodes: Optional[float] = 0.0

    family_history_cardiac_arrest: Optional[float] = 0.0
    family_history_sudden_death: Optional[float] = 0.0
    hypertension: Optional[float] = 0.0
    diabetes: Optional[float] = 0.0
    known_heart_disease: Optional[float] = 0.0
    congenital_heart_disease: Optional[float] = 0.0
    smoking: Optional[float] = 0.0
    alcohol_consumption: Optional[float] = 0.0
    previous_cardiac_problems: Optional[float] = 0.0
    current_medication: Optional[str] = ""

    risk_score: float
    risk_level: str
    risk_label: str
    ui_indicator: str
    model_confidence: Optional[float] = 95.0
    user_id: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}



class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str  # "increase" or "decrease" (how the value shifts risk relative to the mean)
    value: float


class PredictionResultResponse(BaseModel):
    record: PredictionResponse
    suggestions: List[str]
    contributions: List[FeatureContribution] = []
    disclaimer: str = "This is only an ML-based prediction and not a medical diagnosis."


class DashboardStats(BaseModel):
    total_assessments: int
    normal_count: int
    medium_count: int
    critical_count: int
    average_risk_score: float
    recent_assessments: List[PredictionResponse]


# Authentication schemas
class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None

    class Config:
        from_attributes = True

class AuthResponse(BaseModel):
    user: UserResponse
    token: str
