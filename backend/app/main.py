from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

from .database import engine, Base
from .routes import prediction, history, auth
from .services.prediction_service import get_engine

# 1. Create SQLite tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Cardiac Risk & Arrest Prediction API",
    description="Backend API powering the ML-based Cardiac Risk prediction system.",
    version="1.0.0"
)

# 2. Configure CORS middleware to connect with React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development ease, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Register Routers
app.include_router(auth.router)
app.include_router(prediction.router)
app.include_router(history.router)

@app.on_event("startup")
def startup_event():
    """Load model early on startup to pre-warm cache for faster response time."""
    print("Initializing Cardiac Risk ML Inference Engine...")
    try:
        engine = get_engine()
        if engine.is_ready:
            print("ML Model loaded successfully. Ready for inference!")
        else:
            print("Warning: ML model failed to load. Checking fallback pathways.")
    except Exception as e:
        print(f"Critical error loading ML model on startup: {str(e)}")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Cardiac Risk Prediction API.",
        "docs": "/docs",
        "health": "/health"
    }
