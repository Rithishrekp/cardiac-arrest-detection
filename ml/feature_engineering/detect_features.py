from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal as sp_signal

_MODULE_DIR = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.abspath(
    os.path.join(_MODULE_DIR, "..", "datasets", "processed", "ecg_filtered.npy")
)
_DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(_MODULE_DIR, "..", "datasets", "processed")
)

FS = 1000.0

# ---------------------------------------------------------------------------
# Pan-Tompkins QRS detector (full implementation)
# ---------------------------------------------------------------------------


@dataclass
class PanTompkinsResult:
    r_peaks: np.ndarray
    qrs_onsets: np.ndarray
    qrs_offsets: np.ndarray
    signal_used: np.ndarray
    integrated: np.ndarray


def _pan_tompkins_preprocess(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Derivative → squaring → moving-window integration.

    Returns the integrated energy signal used for R-peak detection.
    """
    # 5-point derivative  y[n] = (1/8)[x[n] + 2x[n-1] - 2x[n-3] - x[n-4]]
    b = np.array([1, 2, 0, -2, -1], dtype=np.float64) / 8.0
    deriv = sp_signal.lfilter(b, [1.0], ecg)
    squared = deriv ** 2
    int_win = int(0.150 * fs)  # 150 ms window
    b_int = np.ones(int_win, dtype=np.float64) / int_win
    return sp_signal.lfilter(b_int, [1.0], squared)


def _refine_peaks(
    ecg: np.ndarray,
    candidates: np.ndarray,
    search_radius: int = 60,
    integration_delay: int = 75,
) -> np.ndarray:
    """For each integration-signal candidate, compensate for the moving-average
    group delay and find the maximum of ``|ecg|`` within the search window.

    The group delay of an ``N``-tap moving average is ``(N-1)/2``
    (≈75 samples for 150 ms at 1 kHz).  We shift back by that amount
    before searching for the R-wave extremum.
    """
    n = len(ecg)
    refined: list[int] = []
    for c in candidates:
        center = max(0, c - integration_delay)
        lo = max(0, center - search_radius)
        hi = min(n, center + search_radius)
        seg = np.abs(ecg[lo:hi])
        refined.append(int(np.argmax(seg) + lo))
    return np.array(refined, dtype=int)


def _remove_close_peaks(
    peaks: np.ndarray, min_distance: int = 200
) -> np.ndarray:
    """Greedy suppression: keep the first peak, discard any within
    *min_distance* following it."""
    if len(peaks) <= 1:
        return peaks
    kept = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] >= min_distance:
            kept.append(p)
    return np.array(kept, dtype=int)


def _qrs_boundaries(
    ecg: np.ndarray, r_peak: int, fs: float
) -> tuple[int, int]:
    """Estimate QRS onset and offset from the R-peak.

    Uses a two-pass strategy:
    1. Find the baseline noise level from the signal.
    2. Scan backward/forward from the R-peak to the first point where
       the absolute signal drops within the noise envelope.
    """
    n = len(ecg)
    abs_ecg = np.abs(ecg)

    # estimate noise floor using a low percentile (bypass QRS peaks)
    noise_floor = float(np.percentile(abs_ecg, 15))

    ref = float(abs_ecg[r_peak])
    threshold = max(noise_floor * 2.0, 0.03 * ref)

    def scan(start: int, step: int) -> int:
        pos = start
        while 0 <= pos < n and abs_ecg[pos] > threshold:
            pos += step
        return pos

    onset = scan(r_peak, -1)
    offset = scan(r_peak, 1)
    return onset, offset


def pan_tompkins_detector(
    ecg: np.ndarray,
    fs: float = FS,
    refractory_ms: float = 200.0,
) -> PanTompkinsResult:
    """Pan-Tompkins–inspired QRS detector with adaptive thresholding.

    The algorithm applies Pan-Tompkins internal processing (derivative,
    squaring, integration) to build an energy signal, but uses a modern
    adaptive peak-detection backend for robustness across ECG morphologies.

    Steps
    -----
    1. 5-point derivative approximation.
    2. Squaring (non-linear amplification).
    3. Moving-window integration (150 ms).
    4. Adaptive threshold peak detection in the integrated signal.
    5. Refine each peak position from the absolute-value envelope.

    Parameters
    ----------
    ecg : np.ndarray
        1-D ECG, shape ``(n_samples,)``.
    fs : float
        Sampling frequency (Hz).
    refractory_ms : float
        Minimum interval between successive R-peaks (ms).

    Returns
    -------
    PanTompkinsResult
    """
    refractory = int(refractory_ms * fs / 1000.0)

    # ---- steps 1–3: derivative → squaring → integration ----
    integrated = _pan_tompkins_preprocess(ecg, fs)

    # ---- step 4: adaptive peak detection on the integrated signal ----
    candidates, _ = sp_signal.find_peaks(
        integrated,
        distance=refractory,
    )

    if len(candidates) == 0:
        candidates = np.array([np.argmax(integrated)])

    # ---- step 5: refine using absolute-value envelope ----
    # retain candidates well above the noise floor
    if len(candidates) > 0:
        peak_vals = integrated[candidates]
        noise_floor = np.percentile(integrated, 30)
        candidates = candidates[peak_vals >= 2.0 * noise_floor]

    if len(candidates) == 0:
        candidates = np.array([np.argmax(integrated)])

    int_delay = int(0.150 * fs) // 2
    r_peaks = _refine_peaks(ecg, candidates, integration_delay=int_delay)
    r_peaks = _remove_close_peaks(r_peaks, min_distance=refractory)

    # QRS boundaries
    onsets: list[int] = []
    offsets: list[int] = []
    for rp in r_peaks:
        o, f = _qrs_boundaries(ecg, rp, fs)
        onsets.append(o)
        offsets.append(f)

    return PanTompkinsResult(
        r_peaks=r_peaks,
        qrs_onsets=np.array(onsets, dtype=int),
        qrs_offsets=np.array(offsets, dtype=int),
        signal_used=ecg,
        integrated=integrated,
    )


# ---------------------------------------------------------------------------
# Morphology indicators
# ---------------------------------------------------------------------------


@dataclass
class MorphologyResult:
    st_elevation_depression: np.ndarray
    qrs_durations_ms: np.ndarray
    t_wave_inversions: int
    beat_metadata: list[dict[str, Any]] = field(default_factory=list)


def extract_morphology_indicators(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    fs: float = FS,
    st_measure_offset_ms: float = 80.0,
    t_search_start_ms: float = 200.0,
    t_search_end_ms: float = 400.0,
    isoelectric_window_ms: float = 40.0,
) -> MorphologyResult:
    """Compute beat-by-beat morphology indicators from the ECG and R-peaks.

    For each detected beat the following are extracted:

    * **QRS duration** — time from onset to offset (ms).
    * **ST deviation** — ST level measured *st_measure_offset_ms* after the
      R-peak, relative to the isoelectric baseline (PR segment).
    * **T-wave inversion** — a T-wave is counted as inverted when its
      dominant peak lies below the isoelectric baseline.

    Parameters
    ----------
    ecg : np.ndarray
        1-D bandpassed ECG.
    r_peaks : np.ndarray
        Sample indices of detected R-peaks.
    fs : float
        Sampling frequency (Hz).
    st_measure_offset_ms : float
        Offset from R-peak to sample the ST segment (ms).
    t_search_start_ms, t_search_end_ms : float
        Search window for T-wave peak, relative to R-peak (ms).
    isoelectric_window_ms : float
        Window length before QRS onset for baseline reference (ms).

    Returns
    -------
    MorphologyResult
    """
    if len(r_peaks) == 0:
        return MorphologyResult(
            st_elevation_depression=np.array([], dtype=np.float64),
            qrs_durations_ms=np.array([], dtype=np.float64),
            t_wave_inversions=0,
        )

    n = len(ecg)
    st_offs = int(st_measure_offset_ms * fs / 1000.0)
    t_start = int(t_search_start_ms * fs / 1000.0)
    t_end = int(t_search_end_ms * fs / 1000.0)
    iso_win = int(isoelectric_window_ms * fs / 1000.0)

    st_vals: list[float] = []
    qrs_durs: list[float] = []
    t_inv_count = 0
    beat_meta: list[dict[str, Any]] = []

    for i, peak in enumerate(r_peaks):
        # --- QRS boundaries ---
        onset, offset = _qrs_boundaries(ecg, peak, fs)
        qrs_durs.append((offset - onset) / fs * 1000.0)

        # --- isoelectric baseline (PR segment) ---
        iso_start = max(0, onset - iso_win)
        iso_seg = ecg[iso_start:onset] if onset > iso_start else np.array([0.0])
        iso_level = float(np.mean(iso_seg))

        # --- R-peak amplitude & polarity ---
        r_amp = float(ecg[peak])

        # --- ST segment ---
        st_idx = min(peak + st_offs, n - 1)
        st_start = max(0, st_idx - 5)
        st_end = min(n, st_idx + 5)
        st_level = float(np.mean(ecg[st_start:st_end]))
        st_dev = st_level - iso_level
        st_vals.append(st_dev)

        # --- T-wave ---
        tw_start = min(peak + t_start, n - 1)
        tw_end = min(peak + t_end, n)
        tw_inverted = False
        if tw_end > tw_start:
            t_wave_segment = ecg[tw_start:tw_end]
            t_peak_idx = min(max(0, int(np.argmax(np.abs(t_wave_segment)))), len(t_wave_segment) - 1)
            t_peak_amp = float(t_wave_segment[t_peak_idx])
            tw_inverted = (t_peak_amp - iso_level) < 0
            if tw_inverted:
                t_inv_count += 1

        beat_meta.append({
            "beat_index": i,
            "r_peak_sample": int(peak),
            "r_amplitude": r_amp,
            "qrs_onset": int(onset),
            "qrs_offset": int(offset),
            "qrs_duration_ms": float(qrs_durs[-1]),
            "st_deviation": st_dev,
            "t_wave_inverted": tw_inverted,
        })

    return MorphologyResult(
        st_elevation_depression=np.array(st_vals, dtype=np.float64),
        qrs_durations_ms=np.array(qrs_durs, dtype=np.float64),
        t_wave_inversions=t_inv_count,
        beat_metadata=beat_meta,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def process_ecg_features(
    input_path: str | None = None,
    output_dir: str | None = None,
    fs: float = FS,
    save: bool = True,
) -> dict[str, Any]:
    """End-to-end ECG feature extraction for both channels.

    Pipeline
    --------
    1. Load ``ecg_filtered.npy`` (shape ``(2, 9000)``).
    2. For each channel, run :func:`pan_tompkins_detector`.
    3. For each channel, run :func:`extract_morphology_indicators`.
    4. Aggregate per-channel and cross-channel morphology features.

    Parameters
    ----------
    input_path : str | None
        Path to ``ecg_filtered.npy``.  Defaults to
        ``ml/datasets/processed/ecg_filtered.npy``.
    output_dir : str | None
        Output directory.  Defaults to ``ml/datasets/processed/``.
    fs : float
        Sampling frequency (Hz).
    save : bool
        Persist results (default ``True``).

    Returns
    -------
    dict[str, Any]
        ``{"per_channel": …, "aggregated": …, "metadata": …}``
    """
    input_path = input_path or _DEFAULT_INPUT
    output_dir = output_dir or _DEFAULT_OUTPUT

    ecg = np.load(input_path)
    if ecg.ndim != 2 or ecg.shape[0] != 2:
        raise ValueError(f"Expected (2, n_samples) array, got {ecg.shape}")

    n_samples = ecg.shape[1]
    channels_data: list[dict[str, Any]] = []
    all_beats: list[dict[str, Any]] = []

    for ch_idx in range(2):
        sig = ecg[ch_idx]

        pt_result = pan_tompkins_detector(sig, fs=fs)
        morph_result = extract_morphology_indicators(sig, pt_result.r_peaks, fs=fs)

        channels_data.append(
            {
                "channel": ch_idx,
                "num_beats": len(pt_result.r_peaks),
                "r_peaks_samples": pt_result.r_peaks.tolist(),
                "qrs_onsets_samples": pt_result.qrs_onsets.tolist(),
                "qrs_offsets_samples": pt_result.qrs_offsets.tolist(),
                "st_elevation_depression": morph_result.st_elevation_depression.tolist(),
                "qrs_durations_ms": morph_result.qrs_durations_ms.tolist(),
                "t_wave_inversions": morph_result.t_wave_inversions,
                "mean_qrs_duration_ms": float(np.mean(morph_result.qrs_durations_ms))
                if len(morph_result.qrs_durations_ms) > 0
                else 0.0,
                "mean_st_deviation": float(np.mean(morph_result.st_elevation_depression))
                if len(morph_result.st_elevation_depression) > 0
                else 0.0,
                "total_t_inversions": morph_result.t_wave_inversions,
                "beats": morph_result.beat_metadata,
            }
        )
        all_beats.extend(morph_result.beat_metadata)

    ch0_peaks = channels_data[0]["r_peaks_samples"]
    heart_rate = len(ch0_peaks) / (n_samples / fs) * 60.0 if n_samples > 0 else 0.0

    aggregated = {
        "total_beats_both_channels": sum(
            c["num_beats"] for c in channels_data
        ),
        "mean_qrs_duration_ms": float(
            np.mean(
                [c["mean_qrs_duration_ms"] for c in channels_data if c["num_beats"] > 0]
            )
            or 0.0
        ),
        "mean_st_deviation": float(
            np.mean(
                [c["mean_st_deviation"] for c in channels_data if c["num_beats"] > 0]
            )
            or 0.0
        ),
        "total_t_wave_inversions_both": sum(
            c["total_t_inversions"] for c in channels_data
        ),
        "heart_rate_bpm_est": round(heart_rate, 1),
    }

    result: dict[str, Any] = {
        "per_channel": channels_data,
        "all_beats": all_beats,
        "aggregated": aggregated,
        "metadata": {
            "fs_hz": fs,
            "signal_length_samples": n_samples,
            "signal_duration_s": n_samples / fs,
        },
    }

    if save:
        os.makedirs(output_dir, exist_ok=True)
        npy_path = os.path.join(output_dir, "detect_features.npy")
        np.save(
            npy_path,
            result,
            allow_pickle=True,
        )

    return result
