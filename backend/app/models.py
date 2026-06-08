from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from .database import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    patient_name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    
    # ECG inputs
    rr_interval = Column(Float)
    pp_interval = Column(Float)
    qt_interval = Column(Float)
    
    # Prediction Outputs
    risk_score = Column(Float)  # 0 to 100
    risk_level = Column(String)  # normal, medium, critical
    risk_label = Column(String)  # UX specific descriptive label
    ui_indicator = Column(String)  # white, yellow, red
    ensemble_method = Column(String)  # e.g. tabular_only
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
