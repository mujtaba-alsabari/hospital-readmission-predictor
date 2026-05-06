"""
Send a real patient from the test set to the /predict endpoint and print the response.
Usage:  python api/test_request.py
"""
import json
import requests
import pandas as pd
from pathlib import Path

# Load cleaned data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cleaned.csv")

# Pick a patient — change this index to test different patients
PATIENT_INDEX = 100

# Build the feature dict (drop the target column)
patient_row = df.drop(columns=['readmitted_30d']).iloc[PATIENT_INDEX].to_dict()
actual_outcome = int(df.iloc[PATIENT_INDEX]['readmitted_30d'])

# Build payload
payload = {"features": patient_row}

# Send request
print(f"Sending patient #{PATIENT_INDEX} to /predict ...")
print(f"Actual outcome (ground truth): {'Readmitted' if actual_outcome == 1 else 'Not readmitted'}")
print()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload,
    timeout=10,
)

# Pretty-print response
if response.status_code == 200:
    result = response.json()
    print("=" * 60)
    print("PREDICTION RESPONSE")
    print("=" * 60)
    print(f"Predicted probability: {result['readmission_probability']:.3f}")
    print(f"Risk level:            {result['risk_level']}")
    print(f"Model version:         {result['model_version']}")
    print()
    print("Top contributing features:")
    for i, feat in enumerate(result['top_contributing_features'], start=1):
        direction = "raises risk" if feat['shap_value'] > 0 else "lowers risk"
        print(f"  {i}. {feat['feature']:<35} value={feat['feature_value']:>7.2f}  "
              f"shap={feat['shap_value']:+.3f}  {direction}")
else:
    print(f"Request failed with status {response.status_code}")
    print(f"Response: {response.text}")