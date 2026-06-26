import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="CarValue AI — Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

.stApp{
    background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

.hero-header{
    text-align:center;
    padding-top:35px;
    padding-bottom:10px;
}

.hero-badge{
    display:inline-block;
    background:#0ea5e920;
    border:1px solid #0ea5e9;
    color:#38bdf8;
    padding:6px 14px;
    border-radius:30px;
    font-size:12px;
    font-weight:600;
    letter-spacing:1px;
}

.hero-title{
    font-family:'Space Grotesk',sans-serif;
    color:white;
    font-size:48px;
    font-weight:700;
    margin-top:10px;
}

.hero-title span{
    color:#38bdf8;
}

.hero-subtitle{
    color:#cbd5e1;
    font-size:16px;
    margin-top:-10px;
}

.section-label{
    color:#38bdf8;
    font-size:13px;
    font-weight:700;
    margin-top:25px;
    margin-bottom:10px;
    letter-spacing:1px;
}

.stButton>button{
    width:100%;
    background:#0ea5e9;
    color:white;
    border:none;
    border-radius:10px;
    padding:14px;
    font-size:18px;
    font-weight:700;
}

.stButton>button:hover{
    background:#0284c7;
}

.result-card{
    margin-top:25px;
    background:#111827;
    border:1px solid #38bdf8;
    border-radius:15px;
    padding:30px;
    text-align:center;
}

.result-label{
    color:#94a3b8;
    font-size:14px;
}

.result-price{
    color:#38bdf8;
    font-size:46px;
    font-weight:bold;
    margin-top:10px;
}

.result-note{
    color:#94a3b8;
    margin-top:10px;
    font-size:13px;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

#MainMenu{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

@st.cache_resource
def load_model():

    model_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_pipeline.joblib"
    )

    if not os.path.exists(model_path):
        st.error(f"""
❌ Model not found

Expected path:

{model_path}

Upload your trained model into the models folder.
        """)
        st.stop()

    return joblib.load(model_path)

try:
    model=load_model()

except Exception as e:
    st.error("Unable to load trained model.")
    st.exception(e)
    st.stop()

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("""
<div class="hero-header">

<div class="hero-badge">
AI Powered Valuation
</div>

<div class="hero-title">
CarValue <span>AI</span>
</div>

<div class="hero-subtitle">
Predict the Market Price of Your Used Car in Seconds
</div>

</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# INPUT FORM
# -------------------------------------------------------

st.markdown('<div class="section-label">CAR DETAILS</div>', unsafe_allow_html=True)

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

with col2:
    make_year = st.number_input(
        "Manufacturing Year",
        min_value=1995,
        max_value=2023,
        value=2018,
        step=1
    )

col3, col4 = st.columns(2)

with col3:
    fuel_type = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "Electric"
        ]
    )

with col4:
    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )

color = st.selectbox(
    "Color",
    [
        "White",
        "Black",
        "Silver",
        "Gray",
        "Blue",
        "Red"
    ]
)

st.markdown('<div class="section-label">PERFORMANCE</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    mileage_kmpl = st.number_input(
        "Mileage (kmpl)",
        min_value=5.0,
        max_value=40.0,
        value=18.0,
        step=0.1
    )

with col6:
    engine_cc = st.number_input(
        "Engine Capacity (cc)",
        min_value=600,
        max_value=5000,
        value=1500,
        step=100
    )

col7, col8 = st.columns(2)

with col7:
    owner_count = st.number_input(
        "Previous Owners",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

with col8:
    accidents_reported = st.number_input(
        "Accidents Reported",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

st.markdown('<div class="section-label">SERVICE HISTORY</div>', unsafe_allow_html=True)

col9, col10 = st.columns(2)

with col9:
    service_history = st.selectbox(
        "Service History",
        [
            "Full",
            "Partial",
            "Unknown"
        ]
    )

with col10:
    insurance_valid = st.selectbox(
        "Insurance Valid",
        [
            "Yes",
            "No"
        ]
    )

st.write("")

predict_btn = st.button(
    "🚗 Estimate Market Price",
    use_container_width=True
)

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if predict_btn:

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

    with st.spinner("🤖 AI is analyzing your vehicle..."):

        try:

            prediction = model.predict(input_data)[0]

            if prediction < 0:
                prediction = 0

            st.markdown(f"""
            <div class="result-card">

                <div class="result-label">
                    Estimated Market Value
                </div>

                <div class="result-price">
                    ${prediction:,.0f}
                </div>

                <div class="result-note">
                    Generated using the trained CarValue AI Machine Learning Pipeline
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.success("Prediction completed successfully!")

            st.divider()

            st.subheader("Vehicle Summary")

            summary = pd.DataFrame({

                "Feature":[
                    "Brand",
                    "Year",
                    "Fuel Type",
                    "Transmission",
                    "Mileage",
                    "Engine",
                    "Owners",
                    "Accidents",
                    "Service History",
                    "Insurance"
                ],

                "Value":[
                    brand,
                    make_year,
                    fuel_type,
                    transmission,
                    f"{mileage_kmpl} kmpl",
                    f"{engine_cc} cc",
                    owner_count,
                    accidents_reported,
                    service_history,
                    insurance_valid
                ]

            })

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True
            )

        except Exception as e:

            st.error("Prediction failed.")

            st.exception(e)

            st.info(
                """
Possible reasons:

• Wrong model file

• Feature names don't match training data

• Corrupted model

• Different sklearn version
                """
            )


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    st.title("🚗 CarValue AI")

    st.markdown("---")

    st.markdown("### About")

    st.write("""
CarValue AI is an end-to-end Machine Learning application that estimates the market price of a used car using a trained regression model.

The prediction is based on:

- Brand
- Manufacturing Year
- Fuel Type
- Transmission
- Mileage
- Engine Capacity
- Owner Count
- Accident History
- Service History
- Insurance Status
""")

    st.markdown("---")

    st.markdown("### Model")

    st.success("Best Pipeline")

    st.write("Algorithm selected automatically after Hyperparameter Tuning.")

    st.markdown("---")

    st.markdown("### Tech Stack")

    st.write("""
- Python
- Streamlit
- Scikit-Learn
- XGBoost
- Joblib
- Pandas
- NumPy
""")

    st.markdown("---")

    st.info(
        "This application predicts the approximate resale value of a used car. "
        "The prediction is based on historical training data and should be treated as an estimate rather than an exact market price."
    )


# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <hr style="border:1px solid #334155">

    <div style='text-align:center;color:#94a3b8;font-size:14px;'>

    🚗 <b>CarValue AI</b><br>

    End-to-End Machine Learning Project<br><br>

    Developed using
    <b>Python • Streamlit • Scikit-Learn • XGBoost • MLflow • Optuna</b>

    </div>
    """,
    unsafe_allow_html=True
)
