from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from .database import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    patient_name = Column(String, index=True)
    age = Column(Float)
    gender = Column(String)
    
    # Demographics & Sports Vitals
    weight = Column(Float)
    height = Column(Float)
    bmi = Column(Float)
    heart_rate = Column(Float)
    systolic_bp = Column(Float)
    diastolic_bp = Column(Float)
    mean_arterial_pressure = Column(Float)
    sport_type = Column(String)
    
    # ECG Interval Metrics
    rr_interval = Column(Float)
    pp_interval = Column(Float)
    qt_interval = Column(Float)
    qtc_interval = Column(Float)
    qrs_duration = Column(Float)
    pq_interval = Column(Float)
    
    # Medical & Family History
    family_history_heart_disease = Column(Float)  # 0.0 or 1.0
    personal_history_heart_disease = Column(Float)  # 0.0 or 1.0
    syncope = Column(Float)  # 0.0 or 1.0
    pectus_excavatum = Column(Float)  # 0.0 or 1.0
    
    # Extra inputs
    person_type = Column(String, nullable=True)
    years_sports_experience = Column(Float, nullable=True)
    training_hours_per_week = Column(Float, nullable=True)
    competition_level = Column(String, nullable=True)
    recent_intense_exercise = Column(Float, nullable=True)
    previous_collapse_during_sports = Column(Float, nullable=True)
    fatigue_during_exercise = Column(Float, nullable=True)
    loss_of_consciousness_during_exercise = Column(Float, nullable=True)
    shortness_of_breath_during_exercise = Column(Float, nullable=True)
    chest_pain_during_exercise = Column(Float, nullable=True)
    dizziness_during_exercise = Column(Float, nullable=True)
    palpitations_during_exercise = Column(Float, nullable=True)
    
    workout_frequency = Column(Float, nullable=True)
    heavy_weight_training = Column(Float, nullable=True)
    cardio_frequency = Column(Float, nullable=True)
    steroid_usage = Column(Float, nullable=True)
    supplement_usage = Column(Float, nullable=True)
    pre_workout_usage = Column(Float, nullable=True)
    energy_drink_consumption = Column(Float, nullable=True)
    dehydration_episodes = Column(Float, nullable=True)
    overtraining_symptoms = Column(Float, nullable=True)
    recent_fainting_episodes = Column(Float, nullable=True)
    
    family_history_cardiac_arrest = Column(Float, nullable=True)
    family_history_sudden_death = Column(Float, nullable=True)
    hypertension = Column(Float, nullable=True)
    diabetes = Column(Float, nullable=True)
    known_heart_disease = Column(Float, nullable=True)
    congenital_heart_disease = Column(Float, nullable=True)
    smoking = Column(Float, nullable=True)
    alcohol_consumption = Column(Float, nullable=True)
    previous_cardiac_problems = Column(Float, nullable=True)
    current_medication = Column(String, nullable=True)
    
    # Prediction Outputs
    risk_score = Column(Float)  # 0 to 100
    risk_level = Column(String)  # normal, medium, critical
    risk_label = Column(String)  # UX specific descriptive label
    ui_indicator = Column(String)  # white, yellow, red
    ensemble_method = Column(String)  # e.g. tabular_only
    model_confidence = Column(Float, nullable=True)
    user_id = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    password_salt = Column(String)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
