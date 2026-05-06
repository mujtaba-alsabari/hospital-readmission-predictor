\# Hospital Readmission Risk Predictor — API



FastAPI service that exposes the trained Logistic Regression model for predicting 30-day hospital readmission risk, with SHAP-based per-prediction explanations.



\## Endpoints



| Method | Path | Description |

|---|---|---|

| GET | `/health` | Service health check |

| POST | `/predict` | Predict readmission risk for one patient |

| GET | `/docs` | Auto-generated interactive API documentation |

| GET | `/redoc` | Alternative API documentation view |



\## Running locally



\### 1. Activate the conda environment



&#x20;   conda activate readmission



\### 2. Make sure model artifacts exist

The API expects these files in `../models/`:

\- `final\_model.joblib` — trained Logistic Regression

\- `final\_scaler.joblib` — StandardScaler fit on training data

\- `feature\_columns.joblib` — expected feature order

\- `shap\_explainer.joblib` — SHAP LinearExplainer



These are produced by Notebook 03 (modeling) and Notebook 04 (interpretability).



\### 3. Launch the server

From the project root directory:



&#x20;   uvicorn api.main:app --reload --host 127.0.0.1 --port 8000



The server loads model artifacts at startup (1-2 seconds), then serves predictions.



\### 4. Test the API

Open your browser to:



&#x20;   http://127.0.0.1:8000/docs



Or run the included test script (in a separate terminal):



&#x20;   python api/test\_request.py



The test script picks one patient from the cleaned test set and sends them to `/predict`. Edit `PATIENT\_INDEX` in the script to test different patients.



\## Example response



POST `/predict` returns:



&#x20;   {

&#x20;     "readmission\_probability": 0.581,

&#x20;     "risk\_level": "high",

&#x20;     "top\_contributing\_features": \[

&#x20;       {"feature": "number\_inpatient", "shap\_value": 0.347, "feature\_value": 2.0},

&#x20;       {"feature": "has\_prior\_inpatient", "shap\_value": 0.218, "feature\_value": 1.0},

&#x20;       {"feature": "diag\_1\_grouped\_Respiratory", "shap\_value": -0.143, "feature\_value": 1.0},

&#x20;       {"feature": "discharge\_disposition\_id\_1", "shap\_value": -0.113, "feature\_value": 1.0},

&#x20;       {"feature": "diag\_2\_grouped\_Diabetes", "shap\_value": 0.065, "feature\_value": 1.0}

&#x20;     ],

&#x20;     "model\_version": "logistic\_regression\_v1"

&#x20;   }



\## File structure



&#x20;   api/

&#x20;   ├── main.py           # FastAPI application and routes

&#x20;   ├── schemas.py        # Pydantic request/response models

&#x20;   ├── inference.py      # Model loading and prediction logic

&#x20;   ├── test\_request.py   # Local script to test /predict against a real patient

&#x20;   ├── README.md         # This file

&#x20;   └── \_\_init\_\_.py       # Marks api/ as a Python package



\## Architecture notes



\- \*\*Artifact loading happens once\*\* at module import time (in `inference.py`), not per request — adds 1 second to startup but reduces inference latency to \~50 ms per request.

\- \*\*Feature ordering is enforced\*\* via the saved `feature\_columns.joblib` — incoming features must be present, and the model receives them in the exact training order.

\- \*\*Scaling consistency\*\*: the saved StandardScaler is applied to inputs without re-fitting. This preserves the data leakage rule from training.

\- \*\*Pydantic validation\*\* rejects malformed requests before they reach the model with a 422 error.



\## Future improvements



\- Explicitly type all 132 features in the Pydantic schema (currently uses a flexible `dict` for brevity)

\- Add request logging for monitoring

\- Add rate limiting

\- Containerize with Docker for cloud deployment

\- Add authentication for production use

