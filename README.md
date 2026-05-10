# CodeAlpha — Disease Prediction from Medical Data

Predict disease risk from tabular patient-style features using classical ML (logistic regression, SVM, random forest, XGBoost).

## Requirements

- Python 3.10+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --dataset breast_cancer
python train.py --dataset diabetes
python train.py --dataset heart_disease
```

The heart dataset is fetched from OpenML (requires network the first time).

Outputs: trained models and scaler under `artifacts/`. Metrics printed include accuracy, precision, recall, F1, and ROC-AUC.

**XGBoost:** On macOS, if XGBoost fails to load, install OpenMP (`brew install libomp`) or rely on the other models (training continues without XGBoost).

## GitHub

Create a new repository named `CodeAlpha_DiseasePrediction`, then:

```bash
git remote add origin https://github.com/<your-user>/CodeAlpha_DiseasePrediction.git
git branch -M main
git push -u origin main
```
