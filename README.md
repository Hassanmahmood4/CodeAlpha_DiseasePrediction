# CodeAlpha — Disease Prediction System

Professional **Pima Indians Diabetes** classification stack with Streamlit, scikit-learn preprocessing, dual estimators (Random Forest + Logistic Regression), and packaged evaluation artifacts.

## Features

- Median imputation + scaling after masking invalid clinical zeros (glucose/BP/BMI/insulin/skin thickness).
- Streamlit UI with responsive layout, prediction console, confidence, probability chart, and categorical risk indicator.
- Evaluation tab summarizing accuracy, precision, recall, F1, ROC-AUC, and confusion matrices for both models.
- Modular layout (`src/` package, `scripts/` trainers, `artifacts/` outputs).

## Requirements

- Python 3.10+
- Network access on first training run (dataset download)

## Setup

```bash
cd CodeAlpha_DiseasePrediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train models & metrics

```bash
python scripts/train_models.py
```

This writes:

- `artifacts/logistic_regression_bundle.joblib`
- `artifacts/random_forest_bundle.joblib`
- `artifacts/metrics.json`
- `artifacts/ui_defaults.json`
- `artifacts/pima_raw_cache.csv` (cached dataset)

## Launch Streamlit

```bash
streamlit run app.py
```

Use the **Prediction console** tab for live inference (six primary inputs; optional fields default to training medians). Review **Model evaluation** for offline metrics.

## QA automation

```bash
pip install -r requirements.txt
python scripts/train_models.py
python scripts/smoke_test.py
```

Matplotlib caches render inside `.mplconfig/` (gitignored) so hosted demos avoid permission issues on read-only home directories.

## GitHub

Repository name recommended by CodeAlpha: `CodeAlpha_DiseasePrediction`.

```bash
git remote add origin https://github.com/<your-user>/CodeAlpha_DiseasePrediction.git
git branch -M main
git push -u origin main
```

## Deployment notes

- Pin Python in your hosting provider; ensure `artifacts/` from training is available to the app container/image or train during image build.
- Streamlit reads `.streamlit/config.toml` for theme defaults.
