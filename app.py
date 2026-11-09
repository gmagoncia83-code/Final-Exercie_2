"""
Streamlit application for Wine Quality Prediction System.

Run:
    python train_model.py
    streamlit run app.py
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

DATA_FILE = "wine_quality.csv"
MODEL_FILE = "wine_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE)

st.title("🍷 Wine Quality Prediction System")
st.caption("Classroom decision-support prototype — preliminary screening only.")

if not Path(MODEL_FILE).exists() or not Path(DATA_FILE).exists():
    st.warning("Model/data files not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = load_dataset()

c1, c2, c3 = st.columns(3)
c1.metric("Dataset records", len(data))
c2.metric("Mean quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Prediction")
st.write("Upload a CSV file containing the 11 predictor columns. Do not include the original `quality` column.")

required_columns = list(model.feature_names_in_)
sample = data.drop(columns=["quality"]).head(5)
st.download_button(
    "Download sample input CSV",
    sample.to_csv(index=False),
    file_name="sample_wine_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Upload wine sample CSV", type=["csv"])
threshold = st.slider(
    "GOOD QUALITY probability threshold",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="Higher threshold usually increases precision but may reduce recall for GOOD QUALITY."
)

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [col for col in required_columns if col not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            X_new = new_data[required_columns]
            probabilities = model.predict_proba(X_new)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            results = new_data.copy()
            results["Good quality probability"] = probabilities.round(4)
            results["Prediction"] = [
                "GOOD QUALITY" if pred == 1 else "NEEDS IMPROVEMENT"
                for pred in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted GOOD QUALITY samples", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                file_name="wine_quality_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process the uploaded file: {exc}")

st.subheader("Responsible use")
st.info(
    "This is a classroom prototype trained on public red-wine data. "
    "It is not a substitute for laboratory testing, sensory evaluation, food-safety "
    "procedures, or professional quality-control decisions. Confirm predictions "
    "with qualified staff before taking action."
)
