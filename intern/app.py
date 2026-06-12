import streamlit as st
import numpy as np
import pickle
import os

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "heart_disease_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

st.title("Heart Disease Prediction System")
st.write("Fill patient details carefully")

# -----------------------------
# INPUT SECTION WITH EXPLANATIONS
# -----------------------------

age = st.number_input("Age", 1, 120, 50)

sex = st.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x: "0 = Female" if x == 0 else "1 = Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 = Typical Angina (serious chest pain)",
        1: "1 = Atypical Angina",
        2: "2 = Non-anginal Pain",
        3: "3 = Asymptomatic (no pain)"
    }[x]
)

trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

chol = st.number_input("Cholesterol", 100, 600, 200)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1],
    format_func=lambda x: "0 = ≤120 mg/dl (normal)" if x == 0 else "1 = >120 mg/dl (high)"
)

restecg = st.selectbox(
    "Rest ECG",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 = Normal",
        1: "1 = ST-T Wave Abnormality",
        2: "2 = Left Ventricular Hypertrophy"
    }[x]
)

thalach = st.number_input("Max Heart Rate", 60, 220, 150)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1],
    format_func=lambda x: "0 = No" if x == 0 else "1 = Yes"
)

oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.2, 1.0)

slope = st.selectbox(
    "Slope of ST Segment",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 = Downsloping (high risk)",
        1: "1 = Flat (medium risk)",
        2: "2 = Upsloping (normal)"
    }[x]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4],
    format_func=lambda x: f"{x} vessel(s) colored by fluoroscopy"
)

thal = st.selectbox(
    "Thalassemia",
    [0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 = Unknown",
        1: "1 = Fixed Defect",
        2: "2 = Normal Blood Flow",
        3: "3 = Reversible Defect"
    }[x]
)

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict"):

    # basic validation
    if oldpeak > 6.2:
        st.error("Invalid Oldpeak value (must be 0–6.2)")
        st.stop()

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -----------------------------
    # OUTPUT
    # -----------------------------

    st.subheader("Result")

    st.write(f"Heart Disease Probability: {probability*100:.2f}%")

    if probability < 0.4:
        st.success("LOW RISK")
    elif probability < 0.7:
        st.warning("MEDIUM RISK")
    else:
        st.error("HIGH RISK")

    if prediction == 1:
        st.error("Prediction: Heart Disease Detected")
    else:
        st.success("Prediction: No Heart Disease Detected")