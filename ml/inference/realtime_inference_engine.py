from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ml.feature_engineering.detect_features import (
    extract_edr,
    extract_st_segment,
    extract_wave_morphology,
)


@dataclass
class RealtimeInferenceEngine:
    """Unified live-streaming classification pipeline.

    The engine accepts raw ECG windows, applies preprocessing,
    extracts real-time features, and runs both a tree-based
    prediction model and a deep-learning detection model.

    Parameters
    ----------
    prediction_model : Any
        A sklearn / XGBoost / LightGBM model with a ``predict``
        (and optionally ``predict_proba``) method.
    detection_model : torch.nn.Module | None
        A PyTorch model (CNN1D or BiLSTM) for waveform classification.
    scaler : Any
        Fitted ``StandardScaler`` or similar.
    feature_cols : list[str]
        Column names expected by *prediction_model*.
    device : str
        ``"auto"``, ``"cuda"``, or ``"cpu"``.
    fs : float
        Sampling frequency in Hz.
    class_names : list[str]
        Human-readable class labels.
    """

    prediction_model: Any
    detection_model: Optional[torch.nn.Module] = None
    scaler: Any = None
    feature_cols: list[str] = field(default_factory=list)
    device: str = "auto"
    fs: float = 1000.0
    class_names: list[str] = field(
        default_factory=lambda: ["Normal", "Moderate Risk", "High Risk"]
    )

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self.device)
        if self.detection_model is not None:
            self.detection_model.to(self._device)
            self.detection_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_static(self, features: np.ndarray) -> dict[str, Any]:
        """Run the tree-based prediction model on static features.

        Parameters
        ----------
        features : np.ndarray
            Feature vector(s), shape ``(1, n_features)`` or
            ``(batch, n_features)``.

        Returns
        -------
        dict
            ``"class_id"``, ``"class_name"``, and optionally
            ``"probabilities"``.
        """
        if features.ndim == 1:
            features = features[np.newaxis, :]

        if self.scaler is not None:
            features = self.scaler.transform(features)

        y_pred = self.prediction_model.predict(features)
        result: dict[str, Any] = {
            "class_id": int(y_pred[0]),
            "class_name": self.class_names[int(y_pred[0])],
        }

        if hasattr(self.prediction_model, "predict_proba"):
            proba = self.prediction_model.predict_proba(features)
            result["probabilities"] = proba[0].tolist()

        return result

    def predict_waveform(self, ecg_window: np.ndarray) -> dict[str, Any]:
        """Run the deep-learning detection model on an ECG window.

        Parameters
        ----------
        ecg_window : np.ndarray
            1-D or 2-D ``(channels, timesteps)`` ECG segment.

        Returns
        -------
        dict
            ``"class_id"``, ``"class_name"``, and ``"probabilities"``.
        """
        if self.detection_model is None:
            return {"class_id": -1, "class_name": "N/A", "probabilities": []}

        if ecg_window.ndim == 1:
            ecg_window = ecg_window[np.newaxis, :]  # (1, T)
        if ecg_window.ndim == 2:
            ecg_window = ecg_window[np.newaxis, :, :]  # (1, C, T)

        tensor = torch.tensor(ecg_window, dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            logits = self.detection_model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        class_id = int(proba.argmax())
        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "probabilities": proba.tolist(),
        }

    def predict_live(
        self,
        ecg_window: np.ndarray,
        static_features: np.ndarray | None = None,
        r_peaks: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run the full live-streaming pipeline on a single ECG window.

        Combines static prediction (if *static_features* provided) with
        waveform-based detection and real-time feature extraction.

        Parameters
        ----------
        ecg_window : np.ndarray
            1-D ECG segment.
        static_features : np.ndarray | None
            Optional vector of static clinical features.
        r_peaks : np.ndarray | None
            Optional R-peak indices within *ecg_window* for morphology
            extraction.

        Returns
        -------
        dict
            ``"prediction"`` (static model output),
            ``"detection"`` (waveform model output),
            ``"ensemble"`` (averaged probabilities),
            and real-time features (EDR, ST-segment, morphology).
        """
        result: dict[str, Any] = {}

        realtime_feats: dict[str, Any] = {}

        edr = extract_edr(ecg_window, fs=self.fs)
        realtime_feats["edr"] = edr[-10:].tolist() if len(edr) > 10 else edr.tolist()

        if r_peaks is not None and len(r_peaks) > 0:
            st = extract_st_segment(ecg_window, r_peaks, fs=self.fs)
            if len(st) > 0:
                realtime_feats["st_segment_mean"] = float(st.mean(axis=0).mean())

            morph = extract_wave_morphology(ecg_window, r_peaks, fs=self.fs)
            if len(morph) > 0:
                realtime_feats["morphology_mean"] = float(morph.mean(axis=0).mean())

        result["realtime_features"] = realtime_feats

        result["detection"] = self.predict_waveform(ecg_window)

        if static_features is not None:
            result["prediction"] = self.predict_static(static_features)
            det_proba = np.array(result["detection"].get("probabilities", []))
            pred_proba = np.array(result["prediction"].get("probabilities", []))
            if len(det_proba) == len(pred_proba) and len(det_proba) > 0:
                ensemble_proba = (det_proba + pred_proba) / 2.0
                result["ensemble"] = {
                    "class_id": int(ensemble_proba.argmax()),
                    "class_name": self.class_names[int(ensemble_proba.argmax())],
                    "probabilities": ensemble_proba.tolist(),
                }
        else:
            result["prediction"] = {"class_id": -1, "class_name": "N/A"}

        return result
