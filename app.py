import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Trained Pipeline
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Car Value AI",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Value AI")
st.write("Predict the estimated market value of a used car.")

st.divider()

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    make_year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2035,
        value=2020
    )

    mileage_kmpl = st.number_input(
        "Mileage (km/l)",
        min_value=1.0,
        max_value=50.0,
        value=18.0
    )

    engine_cc = st.number_input(
        "Engine Capacity (CC)",
        min_value=500,
        max_value=6000,
        value=1200
    )

    owner_count = st.number_input(
        "Number of Previous Owners",
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

    fuel_type = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "CNG",
            "Electric",
            "Hybrid"
        ]
    )

    brand = st.text_input(
        "Brand",
        value="Hyundai"
    )

    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )

    color = st.text_input(
        "Color",
        value="White"
    )

    service_history = st.selectbox(
        "Service History",
        [
            "Full",
            "Partial",
            "Unknown"
        ]
    )

    insurance_valid = st.selectbox(
        "Insurance Valid",
        [
            "Yes",
            "No"
        ]
    )

    predict = st.form_submit_button("Predict Price")

# -----------------------------
# Prediction
# -----------------------------
if predict:

    input_df = pd.DataFrame({
        "make_year": [make_year],
        "mileage_kmpl": [mileage_kmpl],
        "engine_cc": [engine_cc],
        "owner_count": [owner_count],
        "accidents_reported": [accidents_reported],
        "fuel_type": [fuel_type],
        "brand": [brand],
        "transmission": [transmission],
        "color": [color],
        "service_history": [service_history],
        "insurance_valid": [insurance_valid]
    })

    prediction = model.predict(input_df)[0]

    st.success("Prediction Completed!")

    st.metric(
        label="Estimated Market Value",
        value=f"${prediction:,.2f}"
    )

    st.caption(
        "Generated using the trained Car Value AI Machine Learning Pipeline."
    )
