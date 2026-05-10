#!/usr/bin/env python3
"""Streamlit UI for disease-risk prediction using trained joblib models."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train import load_dataset

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FORM_COL_LIMIT = 40


def list_trained_datasets() -> list[str]:
    if not ARTIFACTS.is_dir():
        return []
    names = []
    for p in ARTIFACTS.glob("*_scaler.joblib"):
        names.append(p.name.removesuffix("_scaler.joblib"))
    return sorted(set(names))


def list_models(dataset: str) -> list[str]:
    prefix = f"{dataset}_"
    out = []
    for path in ARTIFACTS.glob(f"{dataset}_*.joblib"):
        suffix = path.stem[len(prefix) :]
        if suffix in ("scaler", "feature_columns"):
            continue
        out.append(suffix)
    return sorted(out)


def load_bundle(dataset: str):
    scaler = joblib.load(ARTIFACTS / f"{dataset}_scaler.joblib")
    cols = joblib.load(ARTIFACTS / f"{dataset}_feature_columns.joblib")
    return scaler, cols


def predict_row(dataset: str, model_name: str, row: pd.DataFrame):
    scaler, cols = load_bundle(dataset)
    model = joblib.load(ARTIFACTS / f"{dataset}_{model_name}.joblib")
    X = row.reindex(columns=list(cols), fill_value=0)
    X = X.astype(float)
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)[0]
    return int(pred), proba


def default_reference_frame(dataset: str) -> pd.Series:
    X, _, _, _ = load_dataset(dataset)
    return X.median(numeric_only=True)


def target_labels(dataset: str) -> list[str] | None:
    from sklearn.datasets import load_breast_cancer

    if dataset == "breast_cancer":
        return list(load_breast_cancer().target_names)
    if dataset == "diabetes":
        return ["below_median_progression", "above_median_progression"]
    return None


def main():
    st.set_page_config(page_title="Disease prediction", layout="wide")
    st.title("Disease prediction demo")
    st.caption("Uses models saved by `python train.py --dataset …`. Train first if `artifacts/` is empty.")

    datasets = list_trained_datasets()
    if not datasets:
        st.warning(f"No trained artifacts found in `{ARTIFACTS}`. Run training, then refresh.")
        st.code("python train.py --dataset breast_cancer", language="bash")
        return

    dataset = st.selectbox("Dataset", datasets)
    models = list_models(dataset)
    if not models:
        st.error("Scaler present but no model files found.")
        return
    model_name = st.selectbox("Model", models)

    _, cols = load_bundle(dataset)
    ref = default_reference_frame(dataset)
    labels = target_labels(dataset)

    row = None
    if len(cols) > FORM_COL_LIMIT:
        st.info(
            "This dataset has many engineered columns. Upload a CSV with **one data row** "
            "whose headers align with training features (missing columns are filled with 0)."
        )
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            if raw.empty:
                st.error("CSV has no rows.")
            else:
                row = raw.iloc[[0]].copy()
    else:
        st.subheader("Feature values")
        values: dict[str, float] = {}
        n = len(cols)
        n_cols = 3
        grid = [cols[i : i + n_cols] for i in range(0, n, n_cols)]
        for chunk in grid:
            cst = st.columns(len(chunk))
            for j, col in enumerate(chunk):
                default = float(ref[col]) if col in ref.index and pd.notna(ref[col]) else 0.0
                values[col] = float(
                    cst[j].number_input(col, value=default, format="%.6g", key=f"f_{dataset}_{col}")
                )
        row = pd.DataFrame([values])[list(cols)]

    if st.button("Predict", type="primary") and row is not None:
        pred, proba = predict_row(dataset, model_name, row)
        if labels and 0 <= pred < len(labels):
            st.success(f"Predicted class: **{pred}** — *{labels[pred]}*")
        else:
            st.success(f"Predicted class: **{pred}**")
        if proba is not None:
            names = labels if labels else [f"class_{i}" for i in range(len(proba))]
            chart = pd.DataFrame({"probability": proba}, index=names[: len(proba)])
            st.bar_chart(chart)


if __name__ == "__main__":
    main()
