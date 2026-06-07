from __future__ import annotations

import os
import zipfile
from typing import Optional

import numpy as np
from scipy import signal
from scipy.io import loadmat

_MODULE_DIR = os.path.dirname(__file__)
_DEFAULT_MAT = os.path.abspath(
    os.path.join(
        _MODULE_DIR, "..", "..", "SportDB2", "AMF", "S1", "CRD1", "Data.mat"
    )
)
_DEFAULT_PROCESSED = os.path.abspath(
    os.path.join(_MODULE_DIR, "..", "datasets", "processed")
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_sportdb2_zip(
    zip_path: str = "SportDB2.zip", extract_dir: str = "SportDB2"
) -> str:
    """Extract *SportDB2.zip* if the target directory does not exist."""
    if os.path.exists(extract_dir):
        print(f"'{extract_dir}' already exists, skipping extraction.")
        return extract_dir
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("Extraction complete.")
    else:
        print(f"'{zip_path}' not found.")
    return extract_dir


def load_mat_signals(mat_path: str) -> tuple[np.void, dict]:
    """Load the full structured ``.mat`` and return the raw void + metadata."""
    data = loadmat(mat_path)
    signals: np.void = data["Data"][0][0]
    info = {
        "keys": list(data.keys()),
        "num_components": len(signals),
        "shapes": [sig.shape for sig in signals],
        "field_names": signals.dtype.names,
    }
    return signals, info


def isolate_kardia_ecg(
    signals: np.void,
    leads: tuple[int, int] = (0, 1),
    max_samples: int = 9000,
) -> np.ndarray:
    """Extract a ``(2, max_samples)`` ECG matrix from the Kardia sub-struct.

    The 5th field (index ``4``) of the top-level void is a nested struct
    containing fields ``ECG_1`` … ``ECG_4``, each of shape ``(18000, 1)``.

    This function:

    1. Selects two leads via *leads* (default ``(0, 1)`` → ECG_1, ECG_2).
    2. Flattens each to 1-D.
    3. Slices to *max_samples* (default ``9000``).
    4. Stacks into a ``(2, 9000)`` float64 matrix.

    Parameters
    ----------
    signals : np.void
        Raw struct from :func:`load_mat_signals`.
    leads : tuple[int, int]
        Indices into ``('ECG_1', 'ECG_2', 'ECG_3', 'ECG_4')``.
    max_samples : int
        Number of samples to retain per lead (default ``9000``).

    Returns
    -------
    np.ndarray
        Shape ``(2, max_samples)``, dtype ``float64``.
    """
    kardia = signals[4][0, 0]
    lead_names: list[str] = list(kardia.dtype.names)

    channels: list[np.ndarray] = []
    for idx in leads:
        raw = np.asarray(kardia[lead_names[idx]]).ravel().astype(np.float64)
        channels.append(raw[:max_samples])

    return np.stack(channels, axis=0)


def bandpass_filter(
    data: np.ndarray,
    fs: float = 1000.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (SOS format).

    Parameters
    ----------
    data : np.ndarray
        Shape ``(channels, samples)`` or ``(samples,)``.
    fs : float
        Sampling frequency in Hz (default ``1000.0``).
    lowcut : float
        Low cut-off — removes baseline wander / breathing (default ``0.5``).
    highcut : float
        High cut-off — removes muscle / motion artifacts (default ``45.0``).
    order : int
        Filter order (default ``4``).

    Returns
    -------
    np.ndarray
        Filtered array, same shape as *data*.
    """
    sos = signal.butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")

    if data.ndim == 1:
        return signal.sosfiltfilt(sos, data)

    filtered = np.empty_like(data, dtype=np.float64)
    for ch in range(data.shape[0]):
        filtered[ch] = signal.sosfiltfilt(sos, data[ch].astype(np.float64))
    return filtered


def process_ecg_waveforms(
    mat_path: str | None = None,
    output_dir: str | None = None,
    fs: float = 1000.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    filter_order: int = 4,
    leads: tuple[int, int] = (0, 1),
    max_samples: int = 9000,
    save: bool = True,
) -> dict[str, np.ndarray]:
    """End-to-end pipeline: load → isolate → filter → save.

    Steps
    -----
    1. Load the ``.mat`` file via :func:`load_mat_signals`.
    2. Extract ``(2, max_samples)`` ECG matrix via :func:`isolate_kardia_ecg`.
    3. Apply zero-phase Butterworth bandpass via :func:`bandpass_filter`.
    4. (Optionally) save the cleaned matrix as ``.npy`` and ``.csv``.

    Parameters
    ----------
    mat_path : str | None
        Path to ``Data.mat``.  Defaults to
        ``SportDB2/AMF/S1/CRD1/Data.mat``.
    output_dir : str | None
        Directory for output files.  Defaults to ``ml/datasets/processed/``.
    fs, lowcut, highcut, filter_order
        See :func:`bandpass_filter`.
    leads : tuple[int, int]
        See :func:`isolate_kardia_ecg`.
    max_samples : int
        Samples per channel (default ``9000``).
    save : bool
        Persist results as ``.npy`` and ``.csv`` (default ``True``).

    Returns
    -------
    dict[str, np.ndarray]
        ``{"raw": …, "filtered": …}``, each shape ``(2, max_samples)``.
    """
    mat_path = mat_path or _DEFAULT_MAT
    output_dir = output_dir or _DEFAULT_PROCESSED

    signals, _ = load_mat_signals(mat_path)

    raw_ecg = isolate_kardia_ecg(signals, leads=leads, max_samples=max_samples)
    filtered_ecg = bandpass_filter(
        raw_ecg, fs=fs, lowcut=lowcut, highcut=highcut, order=filter_order
    )

    if save:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "ecg_raw.npy"), raw_ecg)
        np.save(os.path.join(output_dir, "ecg_filtered.npy"), filtered_ecg)
        np.savetxt(
            os.path.join(output_dir, "ecg_filtered.csv"),
            filtered_ecg,
            delimiter=",",
            header="ECG_1,ECG_2",
        )

    return {"raw": raw_ecg, "filtered": filtered_ecg}
