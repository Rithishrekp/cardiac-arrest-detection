from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_MODULE_DIR = os.path.dirname(__file__)
_DEFAULT_RAW = os.path.abspath(
    os.path.join(_MODULE_DIR, "..", "datasets", "raw", "dataset-1.xlsx")
)
_DEFAULT_PROCESSED = os.path.abspath(
    os.path.join(_MODULE_DIR, "..", "datasets", "processed")
)

RENAME_MAP: dict[str, str] = {
    "Age (years)": "Age",
    "Weight (Kg)": "Weight",
    "Height (cm)": "Height",
    "SysBP (mmHg)": "SysBP",
    "DiaBP (mmHg)": "DiaBP",
    "VentricularRate (bpm)": "VentricularRate",
    "PQInterval (ms)": "PQInterval",
    "QRSDuration (ms)": "QRSDuration",
    "QTInterval (ms)": "QTInterval",
    "QTCInterval (ms)": "QTCInterval",
    "RRInterval (ms)": "RRInterval",
    "PPInterval (ms)": "PPInterval",
    "Paxis (\u00ba)": "Paxis",
    "RAxis (\u00ba)": "RAxis",
    "TAxis (\u00ba)": "TAxis",
}

NUMERIC_COLS: list[str] = [
    "Age",
    "Weight",
    "Height",
    "SysBP",
    "DiaBP",
    "VentricularRate",
    "PQInterval",
    "QRSDuration",
    "QTInterval",
    "QTCInterval",
    "RRInterval",
    "PPInterval",
]

CATEGORICAL_DEMOGRAPHICS: list[str] = ["Race", "Sex"]


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply ``pd.to_numeric(…, errors='coerce')`` to every column in *cols*
    that exists in *df*, then convert to ``float32`` for uniformity."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df


def _encode_demographic(
    df: pd.DataFrame,
    column: str,
    encoder_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Label-encode a single demographic column and store its mapping.

    The fitted ``LabelEncoder`` (and its ``classes_``) are saved
    into *encoder_map* under *column* so decoding is possible later.
    """
    if column not in df.columns or pd.api.types.is_numeric_dtype(df[column]):
        return df

    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))

    encoder_map[column] = {
        "classes": encoder.classes_.tolist(),
        "encoded": [int(c) for c in encoder.transform(encoder.classes_)],
    }
    return df


def clean_static_data(
    filepath: str | None = None,
    header_row: int = 1,
    fill_value: float = 0.0,
    output_dir: str | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Load, clean, and standardise the static clinical xlsx dataset.

    The pipeline:

    1. Read Excel (``header_row=1`` by default).
    2. Drop unnamed / spurious columns.
    3. Strip whitespace from column headers.
    4. Rename verbose headers via :attr:`RENAME_MAP`.
    5. Drop duplicate column names (keep first).
    6. **Strict numeric coercion** — each column in :attr:`NUMERIC_COLS`
       is forced to ``float32`` via ``pd.to_numeric(…, errors='coerce')``.
    7. **Demographic encoding** — ``Race`` and ``Sex`` are label-encoded
       with a persistent mapping dict returned alongside the DataFrame.
    8. **MAP computation** — ``MAP = (SysBP + 2 * DiaBP) / 3``.
    9. Fill remaining ``NaN`` and drop duplicate rows.

    Parameters
    ----------
    filepath : str | None
        Path to the ``.xlsx`` file.  Defaults to
        ``ml/datasets/raw/dataset-1.xlsx``.
    header_row : int
        Row index for column names (default ``1``).
    fill_value : float
        Value for ``fillna`` (default ``0.0``).
    output_dir : str | None
        Directory to write the cleaned CSV.  Defaults to
        ``ml/datasets/processed/``.
    save : bool
        Whether to persist the cleaned DataFrame as CSV (default ``True``).

    Returns
    -------
    df : pd.DataFrame
        Cleaned tabular matrix, all numeric columns ``float32``,
        demographics label-encoded as ``int``.
    encoder_map : dict[str, dict[str, Any]]
        Per-column encoder state, e.g.
        ``{"Race": {"classes": [...], "encoded": [...]}}``.
    """
    filepath = filepath or _DEFAULT_RAW
    output_dir = output_dir or _DEFAULT_PROCESSED

    df = pd.read_excel(filepath, header=header_row)

    unnamed_mask = df.columns.str.contains("^Unnamed", na=False)
    df = df.loc[:, ~unnamed_mask]

    df.columns = [str(c).strip() for c in df.columns]

    existing_renames = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df.rename(columns=existing_renames, inplace=True)

    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    df = _coerce_numeric(df, NUMERIC_COLS)

    encoder_map: dict[str, dict[str, Any]] = {}
    for demo_col in CATEGORICAL_DEMOGRAPHICS:
        df = _encode_demographic(df, demo_col, encoder_map)

    if "SysBP" in df.columns and "DiaBP" in df.columns:
        df["MAP"] = (df["SysBP"] + 2.0 * df["DiaBP"]) / 3.0
        df["MAP"] = df["MAP"].astype(np.float32)

    if "Weight" in df.columns and "Height" in df.columns:
        df["BMI"] = df["Weight"] / ((df["Height"] / 100.0) ** 2)
        df["BMI"] = df["BMI"].astype(np.float32)

    df = df.fillna(fill_value)
    df = df.drop_duplicates()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "static_features_clean.csv")
        df.to_csv(csv_path, index=False)
        map_path = os.path.join(
            output_dir, "demographic_encoder_map.json"
        )
        with open(map_path, "w") as f:
            json.dump(encoder_map, f, indent=2)

    return df, encoder_map
