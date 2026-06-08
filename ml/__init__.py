from ml.preprocessing import predict_preprocess, detect_preprocess
from ml.feature_engineering import predict_features, detect_features

try:
    from ml.model_training import (
        train_prediction_model,
        train_detection_model,
    )
except Exception:
    train_prediction_model = None
    train_detection_model = None

try:
    from ml.model_evaluation import evaluate_metrics
except Exception:
    evaluate_metrics = None

try:
    from ml.inference import realtime_inference_engine
except Exception:
    realtime_inference_engine = None

try:
    from ml.visualizations import run_correlation_analysis
except Exception:
    run_correlation_analysis = None

__all__ = [
    "predict_preprocess",
    "detect_preprocess",
    "predict_features",
    "detect_features",
    "train_prediction_model",
    "train_detection_model",
    "evaluate_metrics",
    "realtime_inference_engine",
    "run_correlation_analysis",
]
