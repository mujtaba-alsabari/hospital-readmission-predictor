"""
Request and response schemas for the readmission prediction API.
"""
from pydantic import BaseModel, Field
from typing import List


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="Signed SHAP contribution")
    feature_value: float = Field(..., description="The patient's value for this feature")


class PredictionResponse(BaseModel):
    readmission_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="low / medium / high")
    top_contributing_features: List[FeatureContribution]
    model_version: str = Field(default="logistic_regression_v1")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PatientFeatures(BaseModel):
    features: dict = Field(..., description="Flat dict of feature_name -> value")