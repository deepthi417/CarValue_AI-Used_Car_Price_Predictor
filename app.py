import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="🚗 CarValue AI",
    page_icon="🚗",
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_pipeline.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)
# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🚗 CarValue AI")
st.subheader("Used Car Price Prediction")

st.markdown(
    """
    Enter the vehicle details below and get an AI-powered
    estimate of its market value.
    """
)

st.divider()

# ---------------------------------------------------
# INPUT FORM
# ---------------------------------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox(
            "Brand",
            [
                "Toyota",
                "Honda",
                "BMW",
                "Ford",
                "Hyundai",
                "Nissan",
                "Chevrolet",
                "Kia",
                "Volkswagen",
                "Tesla"
            ]
        )

        make_year = st.number_input(
            "Make Year",
            min_value=1995,
            max_value=2025,
            value=2018
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "Electric"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

        color = st.selectbox(
            "Color",
            [
                "White",
                "Black",
                "Blue",
                "Red",
                "Gray",
                "Silver"
            ]
        )

    with col2:
        mileage_kmpl = st.number_input(
            "Mileage (km/l)",
            min_value=1.0,
            max_value=50.0,
            value=18.0
        )

        engine_cc = st.number_input(
            "Engine Capacity (cc)",
            min_value=500,
            max_value=5000,
            value=1500
        )

        owner_count = st.number_input(
            "Previous Owners",
            min_value=0,
            max_value=10,
            value=1
        )

        accidents_reported = st.number_input(
            "Accidents Reported",
            min_value=0,
            max_value=20,
            value=0
        )

        service_history = st.selectbox(
            "Service History",
            ["Full", "Partial", "Unknown"]
        )

        insurance_valid = st.selectbox(
            "Insurance Valid",
            ["Yes", "No"]
        )

    submitted = st.form_submit_button(
        "🔍 Estimate Market Price"
    )

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if submitted:

    input_data = pd.DataFrame({
        "make_year": [make_year],
        "mileage_kmpl": [mileage_kmpl],
        "engine_cc": [engine_cc],
        "fuel_type": [fuel_type],
        "owner_count": [owner_count],
        "brand": [brand],
        "transmission": [transmission],
        "color": [color],
        "service_history": [service_history],
        "accidents_reported": [accidents_reported],
        "insurance_valid": [insurance_valid]
    })

    try:
        prediction = model.predict(input_data)[0]

        prediction = max(0, prediction)

        st.success("✅ Prediction Successful")

        st.metric(
            label="Estimated Market Value",
            value=f"${prediction:,.0f}"
        )

        with st.expander("View Input Data"):
            st.dataframe(input_data)

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()

st.caption(
    "CarValue AI • Machine Learning Powered Used Car Valuation"
)
