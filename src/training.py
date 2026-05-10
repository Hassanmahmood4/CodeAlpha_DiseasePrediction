"""Train classifiers, evaluate, and persist artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACTS_DIR, FEATURE_COLUMNS, TARGET_COLUMN
from .data import load_pima_dataframe
from .preprocess import mask_invalid_zeros, training_medians


def build_preprocess_pipeline() -> Pipeline:
    """Median imputation + scaling for numeric clinical inputs."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def train_and_evaluate(
    random_state: int = 42,
    test_size: float = 0.25,
    cache_path: Path | None = None,
) -> dict:
    """Train Random Forest and Logistic Regression; save models and metrics JSON."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = ARTIFACTS_DIR / "pima_raw_cache.csv"
    df = load_pima_dataframe(cache_path=cache_path or raw_path)
    df = mask_invalid_zeros(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    medians = training_medians(X_train, ["SkinThickness", "DiabetesPedigreeFunction"])
    with open(ARTIFACTS_DIR / "ui_defaults.json", "w", encoding="utf-8") as f:
        json.dump({"training_medians": medians}, f, indent=2)

    preprocess = build_preprocess_pipeline()
    X_train_p = preprocess.fit_transform(X_train)
    X_test_p = preprocess.transform(X_test)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    metrics_out: dict[str, dict] = {}

    for name, clf in models.items():
        clf.fit(X_train_p, y_train)
        y_pred = clf.predict(X_test_p)
        proba = clf.predict_proba(X_test_p)[:, 1]

        cm = confusion_matrix(y_test, y_pred).tolist()
        metrics_out[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "confusion_matrix": cm,
            "support_negative": int((y_test == 0).sum()),
            "support_positive": int((y_test == 1).sum()),
        }

        bundle = {"preprocess": preprocess, "model": clf, "feature_columns": FEATURE_COLUMNS}
        joblib.dump(bundle, ARTIFACTS_DIR / f"{name}_bundle.joblib")

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    return metrics_out


def predict_row(bundle_path: Path, row: pd.DataFrame) -> tuple[int, np.ndarray]:
    """Run inference using a saved preprocess + model bundle."""
    bundle = joblib.load(bundle_path)
    preprocess: Pipeline = bundle["preprocess"]
    model = bundle["model"]
    cols: list[str] = bundle["feature_columns"]
    X = row.reindex(columns=cols)
    X = mask_invalid_zeros(X)
    X_t = preprocess.transform(X)
    pred = model.predict(X_t)[0]
    proba = model.predict_proba(X_t)[0]
    return int(pred), proba
