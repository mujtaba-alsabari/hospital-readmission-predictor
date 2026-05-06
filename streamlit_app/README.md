\# Hospital Readmission Risk Predictor — Streamlit Dashboard



Interactive web dashboard for the readmission prediction API. Lets users select a patient, adjust key clinical features, and see the model's prediction with SHAP-based explanations.



\## Features



\- Patient feature input (selector + sliders for key clinical fields)

\- Live prediction calls to the FastAPI backend

\- Visual display of:

&#x20; - Predicted readmission probability

&#x20; - Categorical risk level (low / medium / high)

&#x20; - Top 5 contributing features (SHAP-based bar chart)

&#x20; - Raw SHAP values in expandable table

\- Sidebar with API health check and model metadata



\## Architecture



&#x20;   Streamlit (port 8501)

&#x20;           |

&#x20;           | HTTP requests

&#x20;           v

&#x20;   FastAPI (port 8000)

&#x20;           |

&#x20;           | predict + SHAP

&#x20;           v

&#x20;   Logistic Regression model + SHAP LinearExplainer



\## Running locally



\### 1. Activate the conda environment



&#x20;   conda activate readmission



\### 2. Start the FastAPI backend (window 1)



From the project root:



&#x20;   uvicorn api.main:app --host 127.0.0.1 --port 8000



\### 3. Start the Streamlit dashboard (window 2)



In a second terminal:



&#x20;   streamlit run streamlit\_app/app.py



The dashboard auto-opens at http://localhost:8501.



\## File structure



&#x20;   streamlit\_app/

&#x20;   ├── app.py             # main Streamlit application

&#x20;   ├── components.py      # reusable UI components (form, results display)

&#x20;   ├── api\_client.py      # functions for calling the FastAPI backend

&#x20;   ├── README.md          # this file

&#x20;   └── \_\_init\_\_.py        # marks streamlit\_app/ as a package



\## Design notes



\- \*\*Separation of concerns\*\*: UI code is split into a thin `app.py` orchestrator, reusable components, and a dedicated API client. This keeps each file under 100 lines and makes future changes safer.

\- \*\*Client-side feature calculation\*\*: when the user adjusts inputs, derived features (`has\_prior\_inpatient`, `total\_prior\_visits`, `meds\_per\_day`, `total\_procedures`) are recomputed locally before being sent to the API. This keeps the contract simple — the API expects a complete feature dict.

\- \*\*Health check on load\*\*: the sidebar verifies the FastAPI server is reachable before allowing prediction requests, providing clearer error messages.



\## Future improvements



\- Allow uploading a CSV of patients for batch predictions

\- Add a comparison view (predict for multiple patients side-by-side)

\- Replace the current matplotlib chart with an interactive Plotly visualization

\- Add session history of recent predictions

