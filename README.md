# Hospital Readmission Risk Predictor

A machine learning system that predicts the probability of a patient being readmitted to the hospital within 30 days of discharge. Built to support clinical decision-making by identifying high-risk patients who would benefit from enhanced follow-up care.

## Project Status
In active development

## Problem
Approximately 15% of hospital patients are readmitted within 30 days of discharge, costing the U.S. healthcare system $26 billion annually. Medicare penalizes hospitals with high readmission rates under the Hospital Readmissions Reduction Program. Identifying high-risk patients before discharge enables targeted interventions.

## Approach
- **Dataset:** Diabetes 130-US Hospitals dataset (100,000+ patient encounters)
- **Models:** Logistic Regression, Random Forest, XGBoost (with comparison)
- **Interpretability:** SHAP values for individual prediction explanations
- **Deployment:** FastAPI REST endpoint + Streamlit dashboard

## Tech Stack
Python, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit

## Project Structure

    .
    ├── data/          # Raw and processed datasets
    ├── notebooks/     # Exploratory analysis and model development
    ├── src/           # Production-ready code modules
    ├── models/        # Saved trained models
    ├── api/           # FastAPI application
    └── tests/         # Unit tests

## Setup

This project uses a conda environment with Python 3.11.

    conda create -n readmission python=3.11
    conda activate readmission
    conda install pandas numpy scikit-learn matplotlib seaborn jupyter notebook joblib -c conda-forge
    pip install xgboost shap fastapi uvicorn streamlit

## Author
Mujtaba Alsabari, M.S. Data Science
