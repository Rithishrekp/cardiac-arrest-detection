from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

_MODULE_DIR = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.abspath(
    os.path.join(_MODULE_DIR, "..", "datasets", "processed", "static_features_clean.csv")
)
_DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(_MODULE_DIR, "..", "datasets", "processed")
)

INTERVAL_COLS: list[str] = [
    "RRInterval",
    "PPInterval",
    "QTInterval",
]


def _sliding_features(col: pd.Series, window: int) -> pd.DataFrame:
    """Compute local statistical snapshots over a sliding window.

    For each position *i*, the window ``col[max(0, i-window+1):i+1]`` is
    summarised as mean, variance, min, max, and delta (first difference).
    """
    roll = col.rolling(window=window, min_periods=1)
    out = pd.DataFrame(index=col.index)
    out[f"{col.name}_mean"] = roll.mean()
    out[f"{col.name}_var"] = roll.var(ddof=0)
    out[f"{col.name}_min"] = roll.min()
    out[f"{col.name}_max"] = roll.max()
    out[f"{col.name}_delta"] = col.diff().fillna(0.0)
    return out


def extract_interval_features(
    input_path: str | None = None,
    output_dir: str | None = None,
    window: int = 5,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a model-ready feature tensor from temporal interval columns.

    Pipeline
    --------
    1. Load ``static_features_clean.csv``.
    2. Extract the three high-yield intervals: ``RRInterval``,
       ``PPInterval``, ``QTInterval``.
    3. For each interval, compute sliding-window statistics:
       *mean*, *variance*, *min*, *max*, and *delta* (rate of change).
    4. Concatenate the raw + derived columns into the output tensor.

    Parameters
    ----------
    input_path : str | None
        Path to the cleaned static CSV.  Defaults to
        ``ml/datasets/processed/static_features_clean.csv``.
    output_dir : str | None
        Directory to save ``predict_features.parquet`` and ``predict_features.csv``.
        Defaults to ``ml/datasets/processed/``.
    window : int
        Rolling window size (number of patient rows).  Default ``5``.
    save : bool
        Persist the feature tensor (default ``True``).

    Returns
    -------
    df : pd.DataFrame
        Feature tensor of shape ``(n_patients, n_features)``.
    info : dict
        Metadata describing which columns are raw vs. derived.
    """
    input_path = input_path or _DEFAULT_INPUT
    output_dir = output_dir or _DEFAULT_OUTPUT

    raw = pd.read_csv(input_path)

    present = [c for c in INTERVAL_COLS if c in raw.columns]
    missing = [c for c in INTERVAL_COLS if c not in raw.columns]
    if not present:
        raise ValueError(f"None of the required columns {INTERVAL_COLS} found in {input_path}")

    interval_df = raw[present].copy()

    derived_frames: list[pd.DataFrame] = [interval_df]
    for col in present:
        derived_frames.append(_sliding_features(raw[col], window=window))

    feature_tensor = pd.concat(derived_frames, axis=1)

    info: dict[str, Any] = {
        "raw_columns": present,
        "missing_columns": missing,
        "window_size": window,
        "shape": list(feature_tensor.shape),
        "derived_columns": [
            c for c in feature_tensor.columns if c not in present
        ],
    }

    if save:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "predict_features.csv")
        feature_tensor.to_csv(csv_path, index=False)
        try:
            parquet_path = os.path.join(output_dir, "predict_features.parquet")
            feature_tensor.to_parquet(parquet_path, index=False)
        except ImportError:
            pass

    return feature_tensor, info
