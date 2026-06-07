from ml.preprocessing import predict_preprocess, detect_preprocess
from ml.feature_engineering import predict_features, detect_features

try:
    from ml.model_training import (
        train_prediction_model,
        train_detection_model,
    )
except ImportError:
    train_prediction_model = None
    train_detection_model = None

try:
    from ml.model_evaluation import evaluate_metrics
except ImportError:
    evaluate_metrics = None

try:
    from ml.inference import realtime_inference_engine
except ImportError:
    realtime_inference_engine = None

__all__ = [
    "predict_preprocess",
    "detect_preprocess",
    "predict_features",
    "detect_features",
    "train_prediction_model",
    "train_detection_model",
    "evaluate_metrics",
    "realtime_inference_engine",
]
