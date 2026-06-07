from .train_prediction_model import train_xgboost, train_lightgbm

try:
    from .train_detection_model import train_cnn1d, train_bilstm
except ImportError:
    train_cnn1d = None
    train_bilstm = None

__all__ = ["train_xgboost", "train_lightgbm", "train_cnn1d", "train_bilstm"]
