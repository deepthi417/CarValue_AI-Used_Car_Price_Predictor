# 🚗 CarValue AI – Used Car Price Prediction

CarValue AI is a Machine Learning-powered web application that predicts the estimated market value of a used car based on its specifications. The project leverages a complete Scikit-learn pipeline with preprocessing and regression models, and is deployed using Streamlit.

---

## 📌 Features

- 🚗 Predicts used car market prices instantly
- 🤖 End-to-end Machine Learning Pipeline
- 📊 Data preprocessing using Scikit-learn Pipelines
- ⚙️ Feature scaling, encoding, and outlier handling
- 🧠 Multiple regression algorithms evaluated
- 🔍 Hyperparameter tuning using Optuna
- 📈 Experiment tracking with MLflow
- 🌐 Interactive Streamlit web application
- 💰 Displays estimated market value in Indian Rupees (₹)

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- MLflow
- Joblib
- Streamlit

---

## 📂 Project Structure

```text
CarValue_AI/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│
├── data/
│   └── Used_Car_Price_Prediction.csv
│
└── images/
```

---

## 📊 Dataset Features

### Numerical Features

- Manufacturing Year
- Mileage (km/l)
- Engine Capacity (CC)
- Previous Owners
- Accidents Reported

### Categorical Features

- Brand
- Fuel Type
- Transmission
- Color
- Service History
- Insurance Status

### Target

- Price (USD)

The deployed application converts the predicted value into Indian Rupees (₹) for user convenience.

---

## ⚙️ Machine Learning Workflow

1. Data Cleaning
2. Duplicate Removal
3. Missing Value Handling
4. Outlier Clipping using IQR
5. Feature Scaling
6. One-Hot Encoding
7. Model Training
8. Hyperparameter Optimization
9. Cross Validation
10. Model Evaluation
11. Model Deployment

---

## 🤖 Models Evaluated

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors
- XGBoost Regressor

### Best Performing Model

**Linear Regression**

| Metric | Score |
|---------|-------|
| Cross Validation R² | 0.8705 |
| Train R² | 0.8715 |
| Test R² | 0.8736 |
| MAE | 794.15 |
| RMSE | 998.44 |

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/CarValue_AI.git
```

Navigate to the project folder

```bash
cd CarValue_AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💻 Application Preview

The application allows users to:

- Enter vehicle specifications
- Predict the estimated market value
- View results instantly
- Download prediction report

---

## 📈 Future Improvements

- Vehicle Image Upload
- Car Recommendation System
- Price Trend Visualization
- Model Explainability using SHAP
- Cloud Deployment
- REST API Integration

---

## 📚 Learning Outcomes

This project demonstrates:

- Machine Learning Pipelines
- Feature Engineering
- Hyperparameter Optimization
- Experiment Tracking
- Model Deployment
- Web Application Development
- Regression Modeling
- Production-ready ML Workflow

---

## 👩‍💻 Author

**Deepthi**

Machine Learning & Artificial Intelligence Enthusiast

GitHub: https://github.com/deepthi417

LinkedIn: https://www.linkedin.com/in/deepthi-podila/]

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
