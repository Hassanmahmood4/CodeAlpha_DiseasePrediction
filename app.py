#!/usr/bin/env python3
"""
Streamlit frontend for Pima Indians Diabetes prediction.

Displays offline evaluation metrics (accuracy, confusion matrix, precision, recall, F1)
and runs live inference with confidence and risk labeling.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS
from src.training import predict_row

PROJECT_ROOT = Path(__file__).resolve().parent


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; max-width: 1100px; }
            .risk-pill {
                display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;
                font-weight:600;font-size:0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_metrics() -> dict | None:
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ui_defaults() -> dict:
    path = ARTIFACTS_DIR / "ui_defaults.json"
    if not path.is_file():
        return {"training_medians": {"SkinThickness": 20.0, "DiabetesPedigreeFunction": 0.47}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def risk_level(diabetes_proba: float) -> tuple[str, str]:
    """Return label and semantic color name for Streamlit components."""
    if diabetes_proba < 0.35:
        return "Low risk", "green"
    if diabetes_proba < 0.65:
        return "Moderate risk", "orange"
    return "High risk", "red"


def render_confusion_matrix(cm: list[list[int]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Analyzer", layout="wide")
    inject_styles()

    st.title("Diabetes prediction — Pima Indians benchmark")
    st.caption(
        "Professional classification demo using Random Forest and Logistic Regression "
        "with preprocessing tailored for clinical zeros-as-missing signals."
    )

    metrics = load_metrics()
    lr_bundle = ARTIFACTS_DIR / "logistic_regression_bundle.joblib"
    rf_bundle = ARTIFACTS_DIR / "random_forest_bundle.joblib"

    if metrics is None or not lr_bundle.is_file() or not rf_bundle.is_file():
        st.error("Models are not trained yet.")
        st.info(f"Run `python scripts/train_models.py` from `{PROJECT_ROOT}` then refresh.")
        return

    defaults = load_ui_defaults()
    medians = defaults.get("training_medians", {})

    tab_infer, tab_eval = st.tabs(["Prediction console", "Model evaluation"])

    with tab_eval:
        st.subheader("Hold-out metrics")
        cols = st.columns(2)
        for idx, (model_key, title) in enumerate(
            [
                ("logistic_regression", "Logistic Regression"),
                ("random_forest", "Random Forest"),
            ]
        ):
            with cols[idx]:
                m = metrics[model_key]
                st.metric("Accuracy", f"{m['accuracy']:.1%}")
                st.metric("Precision", f"{m['precision']:.3f}")
                st.metric("Recall", f"{m['recall']:.3f}")
                st.metric("F1 score", f"{m['f1']:.3f}")
                st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
                render_confusion_matrix(m["confusion_matrix"], f"{title} — confusion matrix")

    with tab_infer:
        st.subheader("Patient inputs")
        st.markdown(
            "Provide the six core measurements. *SkinThickness* and "
            "*DiabetesPedigreeFunction* default to training-set medians when omitted."
        )

        model_choice = st.radio(
            "Active estimator",
            ("Random Forest", "Logistic Regression"),
            horizontal=True,
        )
        bundle_path = rf_bundle if model_choice.startswith("Random") else lr_bundle

        c1, c2, c3 = st.columns(3)
        pregnancies = c1.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = c2.number_input("Glucose level (mg/dL)", min_value=0.0, max_value=250.0, value=120.0)
        bp = c3.number_input("Blood pressure (mm Hg)", min_value=0.0, max_value=150.0, value=70.0)

        c4, c5, c6 = st.columns(3)
        bmi = c4.number_input("BMI", min_value=10.0, max_value=70.0, value=32.0)
        insulin = c5.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
        age = c6.number_input("Age (years)", min_value=18, max_value=90, value=35)

        skin_default = float(medians.get("SkinThickness", 20.0))
        dpf_default = float(medians.get("DiabetesPedigreeFunction", 0.47))

        row = pd.DataFrame(
            [
                {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": bp,
                    "SkinThickness": skin_default,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf_default,
                    "Age": age,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        if st.button("Generate prediction", type="primary"):
            # Align with training assumption: zeros in clinical channels treated downstream via bundle's preprocessor training only on train split - at inference user zeros might be misleading; mirror mask would require exporting transformer - accept raw inputs for simplicity.
            pred_label, proba = predict_row(bundle_path, row)
            diabetes_proba = float(proba[1])
            confidence = float(np.max(proba) * 100.0)
            readable = "Diabetes positive" if pred_label == 1 else "Diabetes negative"

            st.success(f"**Prediction:** {readable}")
            st.metric("Confidence (winner class)", f"{confidence:.1f}%")

            label, _ = risk_level(diabetes_proba)
            st.info(f"**Risk indicator:** {label} — diabetes probability **{diabetes_proba:.1%}**")

            chart = pd.DataFrame(
                {"probability": [proba[0], proba[1]]},
                index=["No diabetes", "Diabetes"],
            )
            st.bar_chart(chart)


if __name__ == "__main__":
    main()
