#!/usr/bin/env python3
"""
CLI entry: train Random Forest and Logistic Regression on Pima Indians Diabetes data.

Usage (from project root):

    python scripts/train_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import train_and_evaluate  # noqa: E402


def main() -> None:
    metrics = train_and_evaluate()
    print("Training complete. Metrics:")
    for model_name, m in metrics.items():
        print(
            f"  {model_name}: accuracy={m['accuracy']:.4f} "
            f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
