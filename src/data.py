"""Load Pima Indians Diabetes dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import FEATURE_COLUMNS, PIMA_URL, TARGET_COLUMN


def load_pima_dataframe(cache_path: Path | None = None) -> pd.DataFrame:
    """
    Fetch Pima Indians Diabetes CSV and return a typed DataFrame.

    Uses a local cache file when ``cache_path`` is provided to avoid repeat downloads.
    """
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    if cache_path is not None and cache_path.is_file():
        df = pd.read_csv(cache_path)
        df.columns = columns
        return df

    resp = requests.get(PIMA_URL, timeout=60)
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text), header=None)
    df.columns = columns
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df
