#!/usr/bin/env python3
"""
Streamlit frontend for Pima Indians Diabetes prediction — benchmark dashboards & inference console.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS  # noqa: E402
from src.training import predict_row  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            div[data-testid="stAppViewContainer"] {
                background: radial-gradient(circle at top left, #172554 0%, #020617 55%, #020617 100%);
            }
            .block-container { padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1120px; }
            .metric-shell {
                border: 1px solid rgba(148,163,184,.35);
                border-radius: 16px;
                padding: 1rem 1.25rem;
                background: rgba(15,23,42,.65);
                box-shadow: 0 14px 40px rgba(2,6,23,.55);
                backdrop-filter: blur(16px);
            }
            div[data-testid="stTabs"] button {
                font-weight: 600 !important;
            }
            div[data-testid="column"] > div {
                gap: 0.65rem;
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


def risk_band(diabetes_proba: float) -> tuple[str, str]:
    if diabetes_proba < 0.35:
        return "Low clinical urgency", "🟢"
    if diabetes_proba < 0.65:
        return "Moderate vigilance", "🟠"
    return "Elevated likelihood — escalate screening", "🔴"


def render_confusion_matrix(cm: list[list[int]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.35, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Clinical Diabetes Intelligence", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="metric-shell">
            <p style="font-size:.82rem;letter-spacing:.08em;text-transform:uppercase;color:#93c5fd;margin-bottom:.35rem;">
                Portfolio-ready analytics layer</p>
            <h1 style="margin:0;font-size:2rem;line-height:2.35rem;color:#f8fafc;">Diabetes trajectory cockpit · Pima benchmark</h1>
            <p style="margin-top:.65rem;color:#cbd5f5;font-size:1rem;line-height:1.55rem;">
                Ensemble diagnosis assistants pairing calibrated preprocessing with RandomForest & LogisticRegression ensembles trained end-to-end on NIH-aligned telemetry patterns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = load_metrics()
    lr_bundle = ARTIFACTS_DIR / "logistic_regression_bundle.joblib"
    rf_bundle = ARTIFACTS_DIR / "random_forest_bundle.joblib"

    if metrics is None or not lr_bundle.is_file() or not rf_bundle.is_file():
        st.error("Model bundles unavailable.")
        st.info(f"Train artifacts locally → `{PROJECT_ROOT}` • command shown below.")
        st.code("python scripts/train_models.py && python scripts/smoke_test.py", language="bash")
        return

    defaults = load_ui_defaults()
    medians = defaults.get("training_medians", {})

    tab_infer, tab_eval = st.tabs(["Prediction console", "Model evaluation"])

    with tab_eval:
        st.markdown("### Calibration telemetry · stratified hold-out")
        cols = st.columns(2)
        for idx, (model_key, title) in enumerate(
            [
                ("logistic_regression", "Logistic Regression"),
                ("random_forest", "Random Forest"),
            ]
        ):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    m = metrics[model_key]
                    mcols = st.columns(5)
                    mcols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
                    mcols[1].metric("Precision", f"{m['precision']:.3f}")
                    mcols[2].metric("Recall", f"{m['recall']:.3f}")
                    mcols[3].metric("F1 score", f"{m['f1']:.3f}")
                    mcols[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
                    render_confusion_matrix(m["confusion_matrix"], f"{title} — confusion matrix")

    with tab_infer:
        st.markdown("### Guided intake · clinically anchored numeric envelopes")

        model_choice = st.radio(
            "Estimator lane",
            ("Random Forest", "Logistic Regression"),
            horizontal=True,
            label_visibility="collapsed",
        )
        bundle_path = rf_bundle if model_choice.startswith("Random") else lr_bundle

        with st.form("patient_card"):
            st.caption(
                "Fields mirror bedside-ready biomarkers. *SkinThickness* & *DiabetesPedigreeFunction* auto-fill via cohort medians when untouched."
            )
            grid_one = st.columns(3)
            pregnancies = grid_one[0].number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
            glucose = grid_one[1].number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=120.0, step=1.0)
            bp = grid_one[2].number_input("Diastolic blood pressure (mm Hg)", min_value=0.0, max_value=180.0, value=72.0, step=1.0)

            grid_two = st.columns(3)
            bmi = grid_two[0].number_input("BMI", min_value=12.0, max_value=80.0, value=32.0, step=0.1)
            insulin = grid_two[1].number_input("Insulin (µU/mL)", min_value=0.0, max_value=950.0, value=85.0, step=1.0)
            age = grid_two[2].number_input("Age", min_value=18, max_value=110, value=35, step=1)

            expand = st.expander("Advanced — dermatometric / pedigree sliders")
            skin_default = float(medians.get("SkinThickness", 20.0))
            dpf_default = float(medians.get("DiabetesPedigreeFunction", 0.47))
            eg1, eg2 = expand.columns(2)
            skin_override = eg1.number_input("Skin thickness (mm)", min_value=0.0, max_value=120.0, value=skin_default)
            dpf_override = eg2.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.5, value=dpf_default, format="%.3f")

            submitted = st.form_submit_button("Run probabilistic inference", type="primary")

        row = pd.DataFrame(
            [
                {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": bp,
                    "SkinThickness": skin_override,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf_override,
                    "Age": age,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        if submitted:
            if glucose <= 0 or bp <= 0 or bmi <= 0:
                st.error("Glucose, blood pressure, and BMI must be strictly positive for valid physiology envelopes.")
                return

            with st.spinner("Hydrating preprocessing bundles · emitting calibrated logits..."):
                pred_label, proba = predict_row(bundle_path, row)

            diabetes_proba = float(proba[1])
            confidence = float(np.max(proba) * 100.0)
            readable = "Positive diabetes likelihood" if pred_label == 1 else "Negative diabetes likelihood"

            band, glyph = risk_band(diabetes_proba)
            st.success(f"{glyph} **Prediction:** {readable}")
            result_cols = st.columns((1, 1, 1))
            result_cols[0].metric("Winner-class confidence", f"{confidence:.1f}%")
            result_cols[1].metric("Positive-class probability", f"{diabetes_proba:.1%}")
            result_cols[2].metric("Negative-class probability", f"{proba[0]:.1%}")

            st.info(f"**Clinical posture:** {band}")

            chart = pd.DataFrame(
                {"Probability": [proba[0], proba[1]]},
                index=["No diabetes", "Diabetes"],
            )
            st.bar_chart(chart)


if __name__ == "__main__":
    main()
