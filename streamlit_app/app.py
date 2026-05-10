"""
Hospital Readmission Risk Predictor - Streamlit Dashboard.

Run with:  streamlit run streamlit_app/app.py
"""
import streamlit as st
import requests

from api_client import check_api_health, predict_readmission
from components import render_input_form, render_results


# Page config
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
)


# Header
st.title("Hospital Readmission Risk Predictor")
st.markdown(
    "Predicts the probability of 30-day hospital readmission for diabetic patients, "
    "with SHAP-based explanations of which features drove each prediction."
)


# Health check sidebar
with st.sidebar:
    st.header("System Status")
    if check_api_health():
        st.success("API connected")
    else:
        st.error("API unreachable")
        st.info(
            "Start the FastAPI server first:\n\n"
            "```\nuvicorn api.main:app --host 127.0.0.1 --port 8000\n```"
        )
        st.stop()

    st.divider()
    st.markdown("##### About this model")
    st.markdown(
        "- **Algorithm**: Logistic Regression\n"
        "- **Dataset**: ~68,000 hospital encounters\n"
        "- **AUC-ROC**: 0.67\n"
        "- **Features**: 132\n"
        "- **Class balance**: 7.5% positive (handled with class weighting)"
    )


# Main content
features = render_input_form()


# Predict button
if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
    with st.spinner("Calling model..."):
        try:
            prediction = predict_readmission(features)
            render_results(prediction)
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.status_code}")
            st.code(e.response.text)
        except requests.ConnectionError:
            st.error("Could not reach the API. Is it running?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")