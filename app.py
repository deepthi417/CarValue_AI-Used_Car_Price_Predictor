import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CarValue AI — Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    .hero-title span {
        color: #00d4ff;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #8892a4;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }

    /* Section labels */
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #00d4ff;
        margin-bottom: 0.8rem;
        margin-top: 1.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(0, 212, 255, 0.15);
    }

    /* Input styling */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stNumberInput"] > div > div > input {
        background-color: #1e2235 !important;
        border: 1px solid #2e3548 !important;
        border-radius: 8px !important;
        color: #e0e6f0 !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Predict button */
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff, #0099cc);
        color: #0f0f1a;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        margin-top: 1.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(90deg, #33ddff, #00b8f5);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.35);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1a2540, #1e2d50);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #8892a4;
        margin-bottom: 0.5rem;
    }
    .result-price {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .result-note {
        font-size: 0.8rem;
        color: #5a6478;
        margin-top: 0.8rem;
    }

    /* Divider */
    hr {
        border-color: #1e2535 !important;
        margin: 1.5rem 0 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_car_price_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI-Powered Valuation</div>
    <div class="hero-title">CarValue <span>AI</span></div>
    <div class="hero-subtitle">Used Car Price Predictor — Enter car details to get an instant market estimate</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────

# — Car Identity —
st.markdown('<div class="section-label">Car Identity</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", [
        "Toyota", "Honda", "BMW", "Ford", "Hyundai",
        "Nissan", "Chevrolet", "Kia", "Volkswagen", "Tesla"
    ])
with col2:
    make_year = st.number_input("Make Year", min_value=1995, max_value=2023, value=2015, step=1)

col3, col4 = st.columns(2)
with col3:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
with col4:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

color = st.selectbox("Color", ["White", "Black", "Blue", "Red", "Gray", "Silver"])

# — Performance & Condition —
st.markdown('<div class="section-label">Performance & Condition</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=35.0, value=18.0, step=0.1)
with col6:
    engine_cc = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=1500, step=50)

col7, col8 = st.columns(2)
with col7:
    owner_count = st.number_input("Number of Previous Owners", min_value=1, max_value=10, value=1, step=1)
with col8:
    accidents_reported = st.number_input("Accidents Reported", min_value=0, max_value=10, value=0, step=1)

# — History & Documentation —
st.markdown('<div class="section-label">History & Documentation</div>', unsafe_allow_html=True)

col9, col10 = st.columns(2)
with col9:
    service_history = st.selectbox("Service History", ["Full", "Partial", "Unknown"])
with col10:
    insurance_valid = st.selectbox("Insurance Valid", ["Yes", "No"])

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
if st.button("🔍  Estimate Market Price"):
    input_data = pd.DataFrame({
        "make_year":           [make_year],
        "mileage_kmpl":        [mileage_kmpl],
        "engine_cc":           [engine_cc],
        "fuel_type":           [fuel_type],
        "owner_count":         [owner_count],
        "brand":               [brand],
        "transmission":        [transmission],
        "color":               [color],
        "service_history":     [service_history],
        "accidents_reported":  [accidents_reported],
        "insurance_valid":     [insurance_valid],
    })

    predicted_price = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Market Value</div>
        <div class="result-price">$ {predicted_price:,.0f}</div>
        <div class="result-note">Prediction based on current market data · Results are indicative estimates</div>
    </div>
    """, unsafe_allow_html=True)
