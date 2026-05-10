#!/usr/bin/env python3
"""Train (if needed) and run one deterministic inference smoke check."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS  # noqa: E402
from src.training import predict_row, train_and_evaluate  # noqa: E402


def main() -> None:
    lr = ARTIFACTS_DIR / "logistic_regression_bundle.joblib"
    if not lr.is_file():
        train_and_evaluate()

    row = pd.DataFrame(
        [[2, 120.0, 70.0, 20.0, 80.0, 32.0, 0.47, 35]],
        columns=FEATURE_COLUMNS,
    )
    pred, proba = predict_row(lr, row)
    assert pred in (0, 1)
    assert abs(proba.sum() - 1.0) < 1e-3
    print(f"smoke_ok pred={pred} proba={proba.round(4).tolist()}")


if __name__ == "__main__":
    main()
