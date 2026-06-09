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
    
    # Prediction Outputs
    risk_score = Column(Float)  # 0 to 100
    risk_level = Column(String)  # normal, medium, critical
    risk_label = Column(String)  # UX specific descriptive label
    ui_indicator = Column(String)  # white, yellow, red
    ensemble_method = Column(String)  # e.g. tabular_only
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
