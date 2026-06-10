import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

Base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
model = joblib.load(os.path.join(Base_dir, "../models/loan_model.pkl"))
scaler = joblib.load(os.path.join(Base_dir, "../models/scaler.pkl"))

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")
st.markdown("Enter applicant details and predict loan approval status.")

st.sidebar.header("Applicant Information")

# Categorical Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
credit_history = st.sidebar.selectbox("Credit History", [0, 1])
property_area = st.sidebar.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

# Numerical Inputs
applicant_income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    value=5000.0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=0.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=150.0
)

loan_term = st.number_input(
    "Loan Amount Term (Months)",
    min_value=0.0,
    value=360.0
)

# Encoding
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

property_area_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

property_area = property_area_map[property_area]

# Create DataFrame
input_data = pd.DataFrame(
    [[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]],
    columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ]
)

st.subheader("Input Summary")
st.dataframe(input_data)

if st.button("Predict Loan Approval"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.progress(int(probability[1] * 100))
        st.write(
            f"Approval Probability: {probability[1]*100:.2f}%"
        )
    else:
        st.error("❌ Loan Rejected")
        st.progress(int(probability[0] * 100))
        st.write(
            f"Rejection Probability: {probability[0]*100:.2f}%"
        )

    st.subheader("Model Confidence")

    result_df = pd.DataFrame({
        "Class": ["Rejected", "Approved"],
        "Probability": probability
    })

    st.dataframe(result_df)
