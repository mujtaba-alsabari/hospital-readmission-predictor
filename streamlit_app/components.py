"""
Reusable Streamlit UI components for the readmission predictor dashboard.

Components:
    render_input_form()  - patient feature input form
    render_results()     - prediction display with SHAP chart
"""
from typing import Dict, Any
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Path to cleaned data — used to load a sample patient as a "starter" template
SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "sample_patients.csv"


@st.cache_data
def load_sample_patients() -> pd.DataFrame:
    """Load cleaned data once and cache it. Returns DataFrame without target."""
    df = pd.read_csv(SAMPLE_DATA_PATH)
    return df.drop(columns=['readmitted_30d'])


def render_input_form() -> Dict[str, Any]:
    """
    Show form for entering patient features.

    For simplicity, the user picks a sample patient from the test set
    rather than typing all 132 features. They can then "tweak" key features.

    Returns
    -------
    dict of feature_name -> value, ready to send to /predict.
    """
    st.subheader("Patient Information")

    # Load sample patients
    df = load_sample_patients()

    # Patient selector
    patient_idx = st.number_input(
        "Sample patient index (0-{})".format(len(df) - 1),
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
        help="Pick a patient from the cleaned dataset as a starting template."
    )

    # Get the chosen patient's features
    features = df.iloc[int(patient_idx)].to_dict()

    # Show a few key features the user can tweak
    st.markdown("##### Adjust key features (optional)")

    col1, col2 = st.columns(2)

    with col1:
        features['age'] = st.slider(
            "Age decile (0=[0-10), 9=[90-100))",
            min_value=0, max_value=9,
            value=int(features['age']),
        )
        features['number_inpatient'] = st.number_input(
            "Number of prior inpatient visits",
            min_value=0, max_value=20,
            value=int(features['number_inpatient']),
        )
        features['time_in_hospital'] = st.number_input(
            "Days in hospital this stay",
            min_value=1, max_value=14,
            value=int(features['time_in_hospital']),
        )

    with col2:
        features['number_diagnoses'] = st.number_input(
            "Number of diagnoses",
            min_value=1, max_value=20,
            value=int(features['number_diagnoses']),
        )
        features['num_medications'] = st.number_input(
            "Number of medications",
            min_value=1, max_value=80,
            value=int(features['num_medications']),
        )
        features['num_lab_procedures'] = st.number_input(
            "Number of lab procedures",
            min_value=1, max_value=132,
            value=int(features['num_lab_procedures']),
        )

    # Recompute derived features in case the user changed inputs
    features['has_prior_inpatient'] = int(features['number_inpatient'] > 0)
    features['total_prior_visits'] = (
        features['number_outpatient']
        + features['number_emergency']
        + features['number_inpatient']
    )
    features['meds_per_day'] = features['num_medications'] / (features['time_in_hospital'] + 1)
    features['total_procedures'] = features['num_procedures'] + features['num_lab_procedures']

    return features


def render_results(prediction: Dict[str, Any]) -> None:
    """Display the prediction result with risk level, probability, and SHAP chart."""
    proba = prediction['readmission_probability']
    risk_level = prediction['risk_level']

    # Top metrics row
    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Readmission Probability",
            value=f"{proba:.1%}",
        )

    with col2:
        # Color-coded risk badge
        risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "⚪")
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {risk_level.upper()}",
        )

    with col3:
        st.metric(
            label="Model",
            value=prediction['model_version'],
        )

    # Progress bar for probability
    st.progress(proba)

    # SHAP top features chart
    st.subheader("Top Contributing Features")
    st.caption("Features driving this prediction. Red = pushes risk UP, Blue = pushes risk DOWN.")

    contributions = prediction['top_contributing_features']
    feature_names = [f['feature'] for f in contributions]
    shap_values = [f['shap_value'] for f in contributions]
    colors = ['#D62728' if v > 0 else '#1F77B4' for v in shap_values]

    # Reverse so most-impactful is at top
    feature_names = feature_names[::-1]
    shap_values = shap_values[::-1]
    colors = colors[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, shap_values, color=colors, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP value (impact on log-odds of readmission)')
    ax.set_title('Top features for this prediction')
    plt.tight_layout()

    st.pyplot(fig)

    # Detail table
    with st.expander("Show raw SHAP values"):
        df_contributions = pd.DataFrame(prediction['top_contributing_features'])
        df_contributions = df_contributions[['feature', 'feature_value', 'shap_value']]
        df_contributions.columns = ['Feature', 'Value', 'SHAP contribution']
        st.dataframe(df_contributions, use_container_width=True)