import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="CarValue AI — Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ---------------------------------------------------
# YOUR CSS BLOCK GOES HERE
# ---------------------------------------------------
# st.markdown("""...""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_pipeline.joblib"
    )

    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Failed to load model")
    st.exception(e)
    st.stop()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI-Powered Valuation</div>
    <div class="hero-title">CarValue <span>AI</span></div>
    <div class="hero-subtitle">
        Used Car Price Predictor — Enter car details to get an instant market estimate
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------
brand = st.selectbox(
    "Brand",
    [
        "Toyota","Honda","BMW","Ford","Hyundai",
        "Nissan","Chevrolet","Kia","Volkswagen","Tesla"
    ]
)

make_year = st.number_input(
    "Make Year",
    min_value=1995,
    max_value=2023,
    value=2015
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
    ["White","Black","Blue","Red","Gray","Silver"]
)

mileage_kmpl = st.number_input(
    "Mileage (kmpl)",
    value=18.0
)

engine_cc = st.number_input(
    "Engine Capacity (cc)",
    value=1500
)

owner_count = st.number_input(
    "Number of Previous Owners",
    value=1
)

accidents_reported = st.number_input(
    "Accidents Reported",
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

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
if st.button("🔍 Estimate Market Price"):

    input_data = pd.DataFrame({
        "make_year":[make_year],
        "mileage_kmpl":[mileage_kmpl],
        "engine_cc":[engine_cc],
        "fuel_type":[fuel_type],
        "owner_count":[owner_count],
        "brand":[brand],
        "transmission":[transmission],
        "color":[color],
        "service_history":[service_history],
        "accidents_reported":[accidents_reported],
        "insurance_valid":[insurance_valid]
    })

    try:
        predicted_price = model.predict(input_data)[0]

        st.success(
            f"Estimated Market Value: ${predicted_price:,.0f}"
        )

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)
