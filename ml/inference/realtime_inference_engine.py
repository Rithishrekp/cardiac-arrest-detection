from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import numpy as np

_MODULE_DIR = os.path.dirname(__file__)
_SAVED_MODELS_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "saved_models"))
_PROCESSED_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "datasets", "processed"))

FS = 1000.0

# ---------------------------------------------------------------------------
# Frontend UX risk threshold constants
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "normal": {"max": 40, "label": "Normal", "ui_indicator": "white"},
    "medium": {"min": 41, "max": 70, "label": "Primary Alert / Medium Risk", "ui_indicator": "yellow"},
    "critical": {"min": 71, "max": 100, "label": "Critical Emergency / Active Arrest", "ui_indicator": "red"},
}


def _classify_risk(score: float) -> dict[str, Any]:
    if score <= RISK_THRESHOLDS["normal"]["max"]:
        return {"level": "normal", "label": RISK_THRESHOLDS["normal"]["label"], "ui": RISK_THRESHOLDS["normal"]["ui_indicator"]}
    if score <= RISK_THRESHOLDS["medium"]["max"]:
        return {"level": "medium", "label": RISK_THRESHOLDS["medium"]["label"], "ui": RISK_THRESHOLDS["medium"]["ui_indicator"]}
    return {"level": "critical", "label": RISK_THRESHOLDS["critical"]["label"], "ui": RISK_THRESHOLDS["critical"]["ui_indicator"]}


# ---------------------------------------------------------------------------
# Telemetry packet schema
# ---------------------------------------------------------------------------

@dataclass
class TelemetryPacket:
    """Simulates a live streaming telemetry packet from a wireless chest patch
    or hospital gateway device.

    Parameters
    ----------
    patient_id : str
        Unique patient identifier.
    session_token : str
        Active monitoring session token.
    timestamp : float
        Unix timestamp of the reading.
    tabular : dict[str, float]
        Key-value pairs of vital-sign metrics (RRInterval, PPInterval,
        QTInterval, etc.) matching the XGBoost feature columns.
    ecg_window : np.ndarray | None
        Raw ECG signal window, shape ``(n_channels, n_samples)`` or
        ``(n_samples,)``.  Sampled at *fs* Hz.
    fs : float
        Sampling frequency in Hz (default 1000).
    meta : dict
        Optional metadata (lead config, device ID, firmware version, etc.).
    """
    patient_id: str = "unknown"
    session_token: str = ""
    timestamp: float = 0.0
    tabular: dict[str, float] = field(default_factory=dict)
    ecg_window: Optional[np.ndarray] = None
    fs: float = FS
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Real-time ECG feature extraction (lightweight, inference-only)
# ---------------------------------------------------------------------------

def _bandpass(ecg: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
    if ecg.ndim == 1:
        return signal.sosfiltfilt(sos, ecg)
    out = np.empty_like(ecg)
    for ch in range(ecg.shape[0]):
        out[ch] = signal.sosfiltfilt(sos, ecg[ch])
    return out


def _detect_r_peaks(ecg: np.ndarray, fs: float) -> list[np.ndarray]:
    from ml.feature_engineering.detect_features import pan_tompkins_detector
    if ecg.ndim == 1:
        signals = [ecg]
    else:
        signals = [ecg[ch] for ch in range(ecg.shape[0])]
    return [pan_tompkins_detector(sig, fs=fs).r_peaks for sig in signals]


def _compute_heart_rate(r_peaks: np.ndarray, fs: float) -> float:
    if len(r_peaks) < 2:
        return 0.0
    rr_intervals = np.diff(r_peaks) / fs
    return float(60.0 / np.mean(rr_intervals)) if np.mean(rr_intervals) > 0 else 0.0


def _extract_edr(ecg: np.ndarray, r_peaks: np.ndarray) -> np.ndarray:
    if len(r_peaks) < 2:
        return np.array([])
    r_vals = np.array([float(np.abs(ecg[p])) for p in r_peaks])
    if np.std(r_vals) == 0:
        return r_vals
    return (r_vals - r_vals.mean()) / r_vals.std()


def _extract_st_segment(ecg: np.ndarray, r_peaks: np.ndarray, fs: float, offset_ms: float = 80.0) -> np.ndarray:
    from ml.feature_engineering.detect_features import _qrs_boundaries
    st_vals = []
    for rp in r_peaks:
        onset, _ = _qrs_boundaries(ecg, rp, fs)
        iso_start = max(0, onset - int(0.04 * fs))
        iso_level = float(np.mean(np.abs(ecg[iso_start:onset]))) if onset > iso_start else 0.0
        st_idx = min(rp + int(offset_ms * fs / 1000.0), len(ecg) - 1)
        st_val = float(np.abs(ecg[st_idx])) - iso_level
        st_vals.append(st_val)
    return np.array(st_vals, dtype=np.float64)


def _extract_morphology_features(ecg: np.ndarray, r_peaks: np.ndarray, fs: float) -> dict[str, Any]:
    from ml.feature_engineering.detect_features import extract_morphology_indicators
    result = extract_morphology_indicators(ecg, r_peaks, fs=fs)
    return {
        "mean_st_deviation": float(np.mean(result.st_elevation_depression)) if len(result.st_elevation_depression) > 0 else 0.0,
        "mean_qrs_duration_ms": float(np.mean(result.qrs_durations_ms)) if len(result.qrs_durations_ms) > 0 else 0.0,
        "t_wave_inversions": int(result.t_wave_inversions),
        "num_beats": len(r_peaks),
    }


def extract_realtime_features(ecg_window: np.ndarray, fs: float) -> dict[str, Any]:
    """Extract real-time features from a live ECG window for downstream scoring."""
    if ecg_window.ndim == 1:
        sig = ecg_window
    else:
        sig = ecg_window[0]  # use lead I for real-time features

    sig_filt = _bandpass(sig, fs)
    r_peaks_list = _detect_r_peaks(sig_filt, fs)
    peaks = r_peaks_list[0] if r_peaks_list else np.array([], dtype=int)

    features: dict[str, Any] = {
        "heart_rate_bpm": _compute_heart_rate(peaks, fs),
        "num_r_peaks": len(peaks),
    }

    if len(peaks) >= 2:
        edr = _extract_edr(sig_filt, peaks)
        features["edr_latest"] = float(edr[-1]) if len(edr) > 0 else 0.0
        features["edr_std"] = float(np.std(edr)) if len(edr) > 0 else 0.0

        st = _extract_st_segment(sig_filt, peaks, fs)
        features["st_deviation_mean"] = float(np.mean(st)) if len(st) > 0 else 0.0

        morph = _extract_morphology_features(sig_filt, peaks, fs)
        features.update(morph)

    return features


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_prediction_model(model_dir: str) -> dict[str, Any]:
    """Load the best available XGBoost model + scaler from *model_dir*."""
    pkl_candidates = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    pkl_candidates.sort(reverse=True)
    if not pkl_candidates:
        return {"model": None, "scaler": None, "feature_cols": None}

    import joblib
    path = os.path.join(model_dir, pkl_candidates[0])
    bundle = joblib.load(path)
    return {
        "model": bundle.get("model"),
        "scaler": bundle.get("scaler"),
        "feature_cols": bundle.get("feature_cols"),
    }


def _load_detection_model(model_dir: str) -> tuple[Any, dict[str, Any]]:
    """Load the best available CNNBiLSTM / BiLSTM PyTorch model.

    Returns ``(model, metadata)``.  Returns ``(None, {})`` if torch
    is unavailable or no checkpoint is found.
    """
    try:
        import torch
    except ImportError:
        return None, {}

    pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    pt_files.sort(reverse=True)
    if not pt_files:
        return None, {}

    from ml.model_training.train_detection_model import CNNBiLSTM, BiLSTM

    path = os.path.join(model_dir, pt_files[0])
    checkpoint = torch.load(path, map_location="cpu")

    model = CNNBiLSTM(input_length=1000, n_channels=2, num_classes=2)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    return model, {"path": path, "name": pt_files[0]}


# ---------------------------------------------------------------------------
# CardiacAssessmentEngine  (enterprise inference pipeline)
# ---------------------------------------------------------------------------

class CardiacAssessmentEngine:
    """Enterprise-grade real-time cardiac risk assessment engine.

    Loads trained models from ``ml/saved_models/`` and exposes a single
    ``assess()`` call that accepts a streaming telemetry packet and
    returns a risk score (0-100) mapped to the frontend UX specification.
    """

    def __init__(self, models_dir: str | None = None):
        self.models_dir = models_dir or _SAVED_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

        pred_bundle = _load_prediction_model(self.models_dir)
        self._prediction_model: Any = pred_bundle["model"]
        self._scaler: Any = pred_bundle["scaler"]
        self._feature_cols: list[str] | None = pred_bundle["feature_cols"]

        self._detection_model: Any
        self._detection_meta: dict[str, Any]
        self._detection_model, self._detection_meta = _load_detection_model(self.models_dir)

        self._device: Any = None
        if self._detection_model is not None:
            try:
                import torch
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._detection_model.to(self._device)
            except ImportError:
                pass

        self._loaded = (
            self._prediction_model is not None,
            self._detection_model is not None,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """At least one model (prediction or detection) must be loaded."""
        return self._prediction_model is not None or self._detection_model is not None

    @property
    def model_status(self) -> dict[str, Any]:
        return {
            "prediction_model_loaded": self._prediction_model is not None,
            "detection_model_loaded": self._detection_model is not None,
            "feature_cols": self._feature_cols,
            "detection_ckpt": self._detection_meta.get("name"),
        }

    # ------------------------------------------------------------------
    # Internal inference
    # ------------------------------------------------------------------

    def _predict_tabular(self, features: np.ndarray) -> dict[str, float]:
        if self._prediction_model is None:
            return {"prob": 0.5, "class": -1, "error": "no_prediction_model"}

        if features.ndim == 1:
            features = features[np.newaxis, :]
        if self._scaler is not None:
            features = self._scaler.transform(features)

        prob = float(self._prediction_model.predict_proba(features)[0, 1])
        cls = int(self._prediction_model.predict(features)[0])
        return {"prob": prob, "class": cls}

    def _predict_waveform(self, ecg: np.ndarray) -> dict[str, float]:
        if self._detection_model is None:
            return {"prob": 0.5, "class": -1, "error": "no_detection_model"}

        try:
            import torch
        except ImportError:
            return {"prob": 0.5, "class": -1, "error": "torch_not_available"}

        x = ecg.astype(np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, np.newaxis, :]
        elif x.ndim == 2:
            x = x[np.newaxis, :, :]

        tensor = torch.from_numpy(x).to(self._device)
        with torch.inference_mode():
            logits = self._detection_model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        cls = int(proba.argmax())
        prob = float(proba[1] if len(proba) == 2 else proba[cls])
        return {"prob": prob, "class": cls}

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def _compute_risk_score(
        self,
        tabular_result: dict[str, float],
        waveform_result: dict[str, float],
        tabular_used: bool = False,
        waveform_used: bool = False,
    ) -> dict[str, Any]:
        """Fuse prediction + detection into a 0-100 risk score.

        Strategy
        --------
        - ``tabular_prob`` and ``waveform_prob`` are each in [0, 1].
        - If only one branch is available, it drives the full score.
        - If both are available, the score is a weighted ensemble
          (60 % waveform, 40 % tabular) favouring the more immediate
          physiological signal.
        - The fused probability is scaled to 0-100 and mapped to the
          frontend UX specification.
        """
        tp = tabular_result.get("prob", 0.5)
        wp = waveform_result.get("prob", 0.5)

        both = tabular_used and waveform_used
        if both:
            fused = 0.4 * tp + 0.6 * wp
        elif tabular_used:
            fused = tp
        elif waveform_used:
            fused = wp
        else:
            fused = 0.5

        score = float(np.clip(fused * 100.0, 0.0, 100.0))
        risk = _classify_risk(score)

        return {
            "risk_score": round(score, 1),
            "risk_level": risk["level"],
            "risk_label": risk["label"],
            "ui_indicator": risk["ui"],
            "tabular_probability": round(tp, 4),
            "waveform_probability": round(wp, 4),
            "ensemble_method": "both" if both else ("tabular_only" if tabular_used else "waveform_only"),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, packet: TelemetryPacket | dict[str, Any]) -> dict[str, Any]:
        """Run the full assessment pipeline on a single telemetry packet.

        Parameters
        ----------
        packet : TelemetryPacket | dict
            Incoming telemetry from a chest patch / hospital gateway.

        Returns
        -------
        dict
            ``risk_score``, ``risk_level``, ``risk_label``, ``ui_indicator``,
            ``tabular_probability``, ``waveform_probability``,
            ``realtime_features``, ``patient_id``, ``timestamp``,
            ``assessment_ms`` (latency).
        """
        t0 = time.perf_counter()

        if isinstance(packet, dict):
            packet = TelemetryPacket(
                patient_id=packet.get("patient_id", "unknown"),
                session_token=packet.get("session_token", ""),
                timestamp=packet.get("timestamp", time.time()),
                tabular=packet.get("tabular", {}),
                ecg_window=packet.get("ecg_window"),
                fs=packet.get("fs", FS),
                meta=packet.get("meta", {}),
            )

        # ---- Tabular (XGBoost) ----
        tabular_used = self._prediction_model is not None and bool(packet.tabular)
        tabular_result: dict[str, float] = {"prob": 0.5, "class": -1}
        if tabular_used:
            if self._feature_cols:
                vec = np.array([packet.tabular.get(c, 0.0) for c in self._feature_cols], dtype=np.float64)
            else:
                vec = np.array(list(packet.tabular.values()), dtype=np.float64)
            tabular_result = self._predict_tabular(vec)

        # ---- Waveform (CNNBiLSTM) ----
        waveform_used = self._detection_model is not None and packet.ecg_window is not None
        waveform_result: dict[str, float] = {"prob": 0.5, "class": -1}
        if waveform_used:
            waveform_result = self._predict_waveform(packet.ecg_window)

        # ---- Real-time feature extraction ----
        rt_features: dict[str, Any] = {}
        if packet.ecg_window is not None:
            rt_features = extract_realtime_features(packet.ecg_window, packet.fs)

        # ---- Fuse → risk score ----
        risk = self._compute_risk_score(tabular_result, waveform_result, tabular_used, waveform_used)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            **risk,
            "realtime_features": rt_features,
            "patient_id": packet.patient_id,
            "session_token": packet.session_token,
            "timestamp": packet.timestamp,
            "assessment_ms": latency_ms,
        }

    def assess_stream(
        self,
        packet_stream: list[dict[str, Any]] | Generator[dict[str, Any], None, None],
    ) -> Generator[dict[str, Any], None, None]:
        """Process a stream of telemetry packets, yielding assessments.

        Yields one assessment dict per packet.
        """
        for pkt in packet_stream:
            yield self.assess(pkt)


# ---------------------------------------------------------------------------
# Legacy alias  (backward-compatible wrapper)
# ---------------------------------------------------------------------------

class RealtimeInferenceEngine:
    """Legacy wrapper.  Use :class:`CardiacAssessmentEngine` instead."""

    def __init__(self, prediction_model=None, detection_model=None, scaler=None,
                 feature_cols=None, device="auto", fs=FS, class_names=None):
        self._engine = CardiacAssessmentEngine()
        if prediction_model is not None:
            self._engine._prediction_model = prediction_model
        if detection_model is not None:
            self._engine._detection_model = detection_model
        if scaler is not None:
            self._engine._scaler = scaler
        if feature_cols is not None:
            self._engine._feature_cols = feature_cols

    @property
    def is_ready(self) -> bool:
        return self._engine.is_ready

    def predict_static(self, features: np.ndarray) -> dict[str, Any]:
        res = self._engine._predict_tabular(features)
        return {"class_id": res.get("class", -1), "probabilities": [1 - res["prob"], res["prob"]]}

    def predict_waveform(self, ecg_window: np.ndarray) -> dict[str, Any]:
        res = self._engine._predict_waveform(ecg_window)
        return {"class_id": res.get("class", -1), "probabilities": [1 - res["prob"], res["prob"]]}

    def predict_live(self, ecg_window: np.ndarray, static_features: np.ndarray | None = None, r_peaks=None) -> dict[str, Any]:
        pkt = {"ecg_window": ecg_window}
        if static_features is not None:
            from ml.feature_engineering.predict_features import INTERVAL_COLS
            pkt["tabular"] = dict(zip(INTERVAL_COLS, static_features.tolist())) if hasattr(static_features, 'tolist') else {}
        return self._engine.assess(pkt)
