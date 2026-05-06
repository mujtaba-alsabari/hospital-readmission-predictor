# Hospital Readmission Risk Predictor — Streamlit Dashboard

Interactive web dashboard for the readmission prediction API. Lets users select a patient, adjust key clinical features, and see the model's prediction with SHAP-based explanations.

## Features

- Patient feature input (selector + sliders for key clinical fields)
- Live prediction calls to the FastAPI backend
- Visual display of:
  - Predicted readmission probability
  - Categorical risk level (low / medium / high)
  - Top 5 contributing features (SHAP-based bar chart)
  - Raw SHAP values in expandable table
- Sidebar with API health check and model metadata

## Architecture

    Streamlit (port 8501)
            |
            | HTTP requests
            v
    FastAPI (port 8000)
            |
            | predict + SHAP
            v
    Logistic Regression model + SHAP LinearExplainer

## Running locally

### 1. Activate the conda environment

    conda activate readmission

### 2. Start the FastAPI backend (window 1)

From the project root:

    uvicorn api.main:app --host 127.0.0.1 --port 8000

### 3. Start the Streamlit dashboard (window 2)

In a second terminal:

    streamlit run streamlit_app/app.py

The dashboard auto-opens at http://localhost:8501.

## File structure

    streamlit_app/
    ├── app.py             # main Streamlit application
    ├── components.py      # reusable UI components (form, results display)
    ├── api_client.py      # functions for calling the FastAPI backend
    ├── README.md          # this file
    └── __init__.py        # marks streamlit_app/ as a package

## Design notes

- **Separation of concerns**: UI code is split into a thin `app.py` orchestrator, reusable components, and a dedicated API client. This keeps each file under 100 lines and makes future changes safer.
- **Client-side feature calculation**: when the user adjusts inputs, derived features (`has_prior_inpatient`, `total_prior_visits`, `meds_per_day`, `total_procedures`) are recomputed locally before being sent to the API. This keeps the contract simple — the API expects a complete feature dict.
- **Health check on load**: the sidebar verifies the FastAPI server is reachable before allowing prediction requests, providing clearer error messages.

## Future improvements

- Allow uploading a CSV of patients for batch predictions
- Add a comparison view (predict for multiple patients side-by-side)
- Replace the current matplotlib chart with an interactive Plotly visualization
- Add session history of recent predictions
