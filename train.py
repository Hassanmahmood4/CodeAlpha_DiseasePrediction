#!/usr/bin/env python3
"""
Train disease-prediction classifiers on UCI-style tabular medical datasets.
Supports: breast_cancer (sklearn), diabetes (sklearn), heart_disease (UCI CSV via OpenML ID 1590).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # optional: needs libomp on macOS (`brew install libomp`)


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DATA_HOME = Path(__file__).resolve().parent / ".sklearn_data"


def load_dataset(name: str):
    name = name.lower().strip()
    if name == "breast_cancer":
        raw = load_breast_cancer()
        X = pd.DataFrame(raw.data, columns=raw.feature_names)
        y = raw.target
        task = "binary"
        pos_label = 1
    elif name == "diabetes":
        raw = load_diabetes()
        X = pd.DataFrame(raw.data, columns=raw.feature_names)
        # sklearn diabetes target is regression; binarize for classification demo
        y = (raw.target > np.median(raw.target)).astype(int)
        task = "binary"
        pos_label = 1
    elif name == "heart_disease":
        from sklearn.datasets import fetch_openml

        DATA_HOME.mkdir(parents=True, exist_ok=True)
        bunch = fetch_openml(
            name="heart-disease",
            version=1,
            as_frame=True,
            parser="auto",
            data_home=str(DATA_HOME),
        )
        frame = bunch.frame
        target_col = "target" if "target" in frame.columns else frame.columns[-1]
        y = pd.factorize(frame[target_col])[0]
        X = frame.drop(columns=[target_col])
        X = pd.get_dummies(X, drop_first=True)
        task = "binary"
        pos_label = 1
    else:
        raise ValueError(f"Unknown dataset: {name}. Use breast_cancer, diabetes, or heart_disease.")
    return X, y, task, pos_label


def build_models(random_state: int = 42):
    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=random_state),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=random_state,
            eval_metric="logloss",
        )
    return models


def main():
    parser = argparse.ArgumentParser(description="Train disease prediction models.")
    parser.add_argument(
        "--dataset",
        default="breast_cancer",
        choices=("breast_cancer", "diabetes", "heart_disease"),
        help="Medical dataset to use.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    X, y, task, pos_label = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = build_models(random_state=args.seed)
    best_name, best_auc = None, -1.0

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        proba = getattr(model, "predict_proba", None)
        y_score = proba(X_test_s)[:, 1] if proba else model.decision_function(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = float("nan")

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_name = name

        joblib.dump(model, ARTIFACTS / f"{args.dataset}_{name}.joblib")

    joblib.dump(scaler, ARTIFACTS / f"{args.dataset}_scaler.joblib")
    joblib.dump(list(X.columns), ARTIFACTS / f"{args.dataset}_feature_columns.joblib")

    print(f"\nBest model by ROC-AUC: {best_name} ({best_auc:.4f})")
    print(f"Artifacts saved under {ARTIFACTS}")


if __name__ == "__main__":
    main()
