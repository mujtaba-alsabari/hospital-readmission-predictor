# Hospital Readmission Risk Predictor

A machine learning system that predicts the probability of a patient being readmitted to the hospital within 30 days of discharge. Built to support clinical decision-making by identifying high-risk patients who would benefit from enhanced follow-up care.

## Project Status
End-to-end pipeline complete: data exploration, cleaning, modeling, interpretability, API deployment, and dashboard. Final cloud deployment in progress.

## Problem

Approximately 15% of hospital patients are readmitted within 30 days of discharge, costing the U.S. healthcare system $26 billion annually. Medicare penalizes hospitals with high readmission rates under the Hospital Readmissions Reduction Program. Identifying high-risk patients before discharge enables targeted interventions like enhanced follow-up care, medication reconciliation, and home visits.

## Approach

- **Dataset:** Diabetes 130-US Hospitals (UCI), ~100,000 hospital encounters across 130 US hospitals
- **Cleaning:** Standardized missing values, deduplicated to first encounter per patient, grouped 700+ ICD-9 codes into 9 clinical categories
- **Models compared:** Logistic Regression, Random Forest, XGBoost (default), XGBoost (tuned via 30-iteration RandomizedSearchCV)
- **Interpretability:** SHAP LinearExplainer for global feature importance and per-prediction explanations
- **Deployment:** FastAPI REST endpoint + Streamlit dashboard, both running locally

## Results

| Model | AUC-ROC | F1 | Recall | Precision |
|---|---|---|---|---|
| **Logistic Regression** | **0.6704** | 0.2081 | 0.5570 | 0.1279 |
| Random Forest | 0.6699 | 0.2159 | 0.3632 | 0.1536 |
| XGBoost (default) | 0.6603 | 0.2060 | 0.4722 | 0.1318 |
| XGBoost (tuned) | 0.6676 | 0.2017 | 0.5716 | 0.1224 |

All four models clustered within 0.01 AUC — consistent with published academic literature on this dataset. Logistic Regression was selected as the deployment model based on best AUC, simpler architecture, faster inference, and direct interpretability of coefficients.

### Top features driving predictions (SHAP)

1. `number_inpatient` — prior hospitalizations
2. `discharge_disposition_id_1` — discharged to home (protective)
3. `number_diagnoses` — total comorbidities
4. `has_prior_inpatient` — engineered binary flag (top 4 finish for an engineered feature)
5. `diabetesMed` — on diabetes medication

All top features align with established clinical readmission research.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, Uvicorn, Pydantic, Matplotlib, Seaborn, Jupyter, Git

## Project Structure

    .
    ├── data/             # Raw and processed datasets (gitignored)
    ├── notebooks/        # Jupyter notebooks for EDA, cleaning, modeling, interpretability
    ├── src/              # Reusable Python modules
    ├── models/           # Saved trained models and SHAP explainer
    ├── api/              # FastAPI application — see api/README.m