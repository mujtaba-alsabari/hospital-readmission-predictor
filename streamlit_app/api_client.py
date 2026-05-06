"""
Client functions for calling the readmission prediction API.

Isolates all HTTP/network concerns so the UI code (app.py, components.py)
can stay focused on rendering.
"""
import requests
from typing import Dict, Any


API_BASE_URL = "http://127.0.0.1:8000"


def check_api_health() -> bool:
    """Return True if the API is reachable and the model is loaded."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except (requests.ConnectionError, requests.Timeout):
        return False


def predict_readmission(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the /predict endpoint with patient features.

    Parameters
    ----------
    features : dict
        Flat dict mapping feature_name -> value (132 features).

    Returns
    -------
    dict
        API response with readmission_probability, risk_level,
        top_contributing_features, and model_version.

    Raises
    ------
    requests.HTTPError if the API returns an error
    requests.ConnectionError if the API is unreachable
    """
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"features": features},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()