"""
ML inference logic for hospital readmission prediction.
"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "final_scaler.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "feature_columns.joblib"
EXPLAINER_PATH = PROJECT_ROOT / "models" / "shap_explainer.joblib"

print("Loading model artifacts...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)
explainer = joblib.load(EXPLAINER_PATH)
print(f"Model loaded ({type(model).__name__}), {len(feature_columns)} features")


def _classify_risk(probability: float) -> str:
    if probability < 0.20:
        return "low"
    elif probability < 0.50:
        return "medium"
    else:
        return "high"


def predict_with_explanation(features: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    missing = set(feature_columns) - set(features.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} required features")

    feature_values = np.array([[features[col] for col in feature_columns]])
    feature_values_scaled = scaler.transform(feature_values)

    proba = float(model.predict_proba(feature_values_scaled)[0, 1])
    risk_level = _classify_risk(proba)

    shap_values = explainer.shap_values(feature_values_scaled)[0]

    contributions = sorted(
        [
            {
                "feature": col,
                "shap_value": float(shap_values[i]),
                "feature_value": float(feature_values[0, i]),
            }
            for i, col in enumerate(feature_columns)
        ],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    return {
        "readmission_probability": proba,
        "risk_level": risk_level,
        "top_contributing_features": contributions[:top_k],
        "model_version": "logistic_regression_v1",
    }