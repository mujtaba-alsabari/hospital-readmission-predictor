# Hospital Readmission Risk Predictor — API

FastAPI service that exposes the trained Logistic Regression model for predicting 30-day hospital readmission risk, with SHAP-based per-prediction explanations.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/predict` | Predict readmission risk for one patient |
| GET | `/docs` | Auto-generated interactive API documentation |
| GET | `/redoc` | Alternative API documentation view |

## Running locally

### 1. Activate the conda environment

    conda activate readmission

### 2. Make sure model artifacts exist
The API expects these files in `../models/`:
- `final_model.joblib` — trained Logistic Regression
- `final_scaler.joblib` — StandardScaler fit on training data
- `feature_columns.joblib` — expected feature order
- `shap_explainer.joblib` — SHAP LinearExplainer

These are produced by Notebook 03 (modeling) and Notebook 04 (interpretability).

### 3. Launch the server
From the project root directory:

    uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

The server loads model artifacts at startup (1-2 seconds), then serves predictions.

### 4. Test the API
Open your browser to:

    http://127.0.0.1:8000/docs

Or run the included test script (in a separate terminal):

    python api/test_request.py

The test script picks one patient from the cleaned test set and sends them to `/predict`. Edit `PATIENT_INDEX` in the script to test different patients.

## Example response

POST `/predict` returns:

    {
      "readmission_probability": 0.581,
      "risk_level": "high",
      "top_contributing_features": [
        {"feature": "number_inpatient", "shap_value": 0.347, "feature_value": 2.0},
        {"feature": "has_prior_inpatient", "shap_value": 0.218, "feature_value": 1.0},
        {"feature": "diag_1_grouped_Respiratory", "shap_value": -0.143, "feature_value": 1.0},
        {"feature": "discharge_disposition_id_1", "shap_value": -0.113, "feature_value": 1.0},
        {"feature": "diag_2_grouped_Diabetes", "shap_value": 0.065, "feature_value": 1.0}
      ],
      "model_version": "logistic_regression_v1"
    }

## File structure

    api/
    ├── main.py           # FastAPI application and routes
    ├── schemas.py        # Pydantic request/response models
    ├── inference.py      # Model loading and prediction logic
    ├── test_request.py   # Local script to test /predict against a real patient
    ├── README.md         # This file
    └── __init__.py       # Marks api/ as a Python package

## Architecture notes

- **Artifact loading happens once** at module import time (in `inference.py`), not per request — adds 1 second to startup but reduces inference latency to ~50 ms per request.
- **Feature ordering is enforced** via the saved `feature_columns.joblib` — incoming features must be present, and the model receives them in the exact training order.
- **Scaling consistency**: the saved StandardScaler is applied to inputs without re-fitting. This preserves the data leakage rule from training.
- **Pydantic validation** rejects malformed requests before they reach the model with a 422 error.

## Future improvements

- Explicitly type all 132 features in the Pydantic schema (currently uses a flexible `dict` for brevity)
- Add request logging for monitoring
- Add rate limiting
- Containerize with Docker for cloud deployment
- Add authentication for production use