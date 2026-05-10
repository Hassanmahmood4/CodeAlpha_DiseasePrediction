"""Preprocessing utilities for Pima diabetes features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ZERO_AS_MISSING


def mask_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel zeros with NaN for clinical fields where 0 is invalid."""
    out = df.copy()
    for col in ZERO_AS_MISSING:
        if col in out.columns:
            out.loc[out[col] == 0, col] = np.nan
    return out


def training_medians(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Median values for optional fields used when the UI collects a subset of inputs."""
    medians: dict[str, float] = {}
    for c in cols:
        if c in df.columns:
            medians[c] = float(df[c].median())
    return medians
