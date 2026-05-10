# Hospital Readmission Risk Predictor

A machine learning system that predicts the probability of a patient being readmitted to the hospital within 30 days of discharge. Built to support clinical decision-making by identifying high-risk patients who would benefit from enhanced follow-up care.

## Live Demo

>   **Important — Free-tier hosting:** This project runs on Render's free tier which spins down after 15 minutes of inactivity. To test the live demo:
> 1. **Open the API link FIRST** and wait ~30-50 seconds for it to wake up
> 2. **Then open the Dashboard link**
> 3. All subsequent requests are sub-second

- **Live API (open this first):** [hospital-readmission-predictor-5k8p.onrender.com/docs](https://hospital-readmission-predictor-5k8p.onrender.com/docs)
- **Live Dashboard:** [readmission-dashboard.onrender.com](https://readmission-dashboard.onrender.com)

## Problem

Approximately 15% of hospital patients are readmitted within 30 days of discharge, costing the U.S. healthcare system $26 billion annually. Medicare penalizes hospitals with high readmission rates under the Hospital Readmissions Reduction Program. Identifying high-risk patients before discharge enables targeted interventions like enhanced follow-up care, medication reconciliation, and home visits.

## Approach

- **Dataset:** Diabetes 130-US Hospitals (UCI), ~100,000 encounters across 130 US hospitals, deduplicated to ~68,000 unique patients
- **Cleaning:** Standardized missing values, grouped 700+ ICD-9 codes into 9 clinical categories, engineered 4 domain-specific features
- **Models compared:** Logistic Regression, Random Forest, XGBoost (default), XGBoost (tuned via 30-iteration RandomizedSearchCV)
- **Interpretability:** SHAP LinearExplainer for global feature importance and per-prediction explanations
- **Deployment:** FastAPI REST API + Streamlit dashboard, both deployed to Render with HTTPS

## Results

| Model | AUC-ROC | F1 | Recall | Precision |
|---|---|---|---|---|
| **Logistic Regression** | **0.6704** | 0.2081 | 0.5570 | 0.1279 |
| Random Forest | 0.6699 | 0.2159 | 0.3632 | 0.1536 |
| XGBoost (default) | 0.6603 | 0.2060 | 0.4722 | 0.1318 |
| XGBoost (tuned) | 0.6676 | 0.2017 | 0.5716 | 0.1224 |

All four models clustered within 0.01 AUC — consistent with published academic literature on this dataset (Strack et al., 2014). Logistic Regression was selected for deployment based on best AUC, simpler architecture, faster inference, and direct interpretability.

### Top features driving predictions (SHAP)

1. `number_inpatient` — prior hospitalizations (strongest signal)
2. `discharge_disposition_id_1` — discharged to home (protective)
3. `number_diagnoses` — total comorbidities
4. `has_prior_inpatient` — engineered binary flag
5. `diabetesMed` — on diabetes medication

All top features align with established clinical readmission research.

## Architecture

    User (browser)
        |
        v
    Streamlit Dashboard (Render)
        |  HTTP requests
        v
    FastAPI Backend (Render)
        |  model + SHAP
        v
    Logistic Regression + LinearExplainer
        |
        v
    JSON response with prediction + explanation

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, Uvicorn, Pydantic, Matplotlib, Seaborn, Jupyter, Git

## Project Structure

    .
    ├── data/               # Raw and processed datasets (gitignored)
    ├── notebooks/          # Jupyter notebooks for EDA, cleaning, modeling, interpretability
    ├── models/             # Trained model artifacts (LR, scaler, SHAP explainer)
    ├── api/                # FastAPI application — see api/README.md
    ├── streamlit_app/      # Streamlit dashboard — see streamlit_app/README.md
    ├── src/                # Reusable Python modules
    └── tests/              # Unit tests

## Notebooks

- `01_data_exploration.ipynb` — EDA, target distribution, missing-value investigation
- `02_data_cleaning.ipynb` — full cleaning pipeline, ICD-9 grouping, feature engineering
- `03_modeling.ipynb` — train, tune, and compare 4 model configurations
- `04_interpretability.ipynb` — SHAP global and local explanations

## Running Locally

### Prerequisites

    conda create -n readmission python=3.11
    conda activate readmission
    conda install pandas numpy scikit-learn matplotlib seaborn jupyter notebook joblib -c conda-forge
    pip install xgboost shap fastapi uvicorn streamlit requests

### Start the API (terminal 1)

    uvicorn api.main:app --host 127.0.0.1 --port 8000

### Start the dashboard (terminal 2)

    streamlit run streamlit_app/app.py

## Lessons Learned

- Class imbalance (7.5% positive) makes accuracy misleading — AUC and recall are better metrics for this problem
- Prior hospitalizations (`number_inpatient`) dominates predictions, consistent with clinical literature
- Engineered features (`has_prior_inpatient`, `total_prior_visits`) ranked in SHAP top 10, validating domain-driven feature engineering
- Model selection isn't always about the "fanciest" algorithm — Logistic Regression matched XGBoost on AUC while being faster and more interpretable
- Deployment revealed real issues (sklearn version mismatch, Python version pinning) that notebooks never surface

## Author

Mujtaba Alsabari, M.S. Data Science | [LinkedIn](https://linkedin.com/in/mujtaba-alsabari1/) | [GitHub](https://github.com/mujtaba-alsabari)