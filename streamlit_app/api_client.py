"""
Client functions for calling the readmission prediction API.
"""
import os
import requests
from typing import Dict, Any


API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def check_api_health() -> bool:
    """Return True if the API is reachable and the model is loaded."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except (requests.ConnectionError, requests.Timeout):
        return False


def predict_readmission(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the /predict endpoint with patient features.
    """
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"features": features},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()