"""
FastAPI application for hospital readmission prediction.
"""
from fastapi import FastAPI, HTTPException

from api.schemas import (
    PatientFeatures,
    PredictionResponse,
    HealthResponse,
    FeatureContribution,
)
from api.inference import predict_with_explanation


app = FastAPI(
    title="Hospital Readmission Risk Predictor",
    description="Predicts 30-day readmission risk for diabetic patients with SHAP explanations.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures):
    try:
        result = predict_with_explanation(payload.features, top_k=5)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return PredictionResponse(
        readmission_probability=result["readmission_probability"],
        risk_level=result["risk_level"],
        top_contributing_features=[
            FeatureContribution(**f) for f in result["top_contributing_features"]
        ],
        model_version=result["model_version"],
    )