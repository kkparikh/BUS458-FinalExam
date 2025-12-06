import streamlit as st
import pandas as pd
import pickle

# Load the saved model
model = pickle.load(open("my_model.pkl", "rb"))

st.title("Loan Approval Predictor")

st.write("Enter the applicant's information below to estimate their loan approval probability.")

# User Inputs
fico = st.number_input("FICO Score", min_value=300, max_value=850, value=700)
income = st.number_input("Monthly Gross Income ($)", min_value=0, value=5000)
loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=10000)
housing_payment = st.number_input("Monthly Housing Payment ($)", min_value=0, value=1200)

employment_status = st.selectbox("Employment Status", ["full_time", "part_time", "self_employed", "unemployed"])
employment_sector = st.selectbox("Employment Sector", ["finance", "healthcare", "retail", "IT", "Unknown"])
reason = st.selectbox("Loan Reason", ["debt_consolidation", "credit_card_refinancing", "home_improvement", "major_purchase"])

bankrupt = st.selectbox("Ever Bankrupt or Foreclosed?", [0, 1])
lender = st.selectbox("Lender", ["A", "B", "C"])

# Prepare input as DataFrame
input_data = pd.DataFrame({
    "FICO_score": [fico],
    "Monthly_Gross_Income": [income],
    "Requested_Loan_Amount": [loan_amount],
    "Monthly_Housing_Payment": [housing_payment],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Reason": [reason],
    "Ever_Bankrupt_or_Foreclose": [bankrupt],
    "Lender": [lender]
})

# One-hot encode like training
input_encoded = pd.get_dummies(input_data, drop_first=True)
input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
if st.button("Predict"):
    prob = model.predict_proba(input_encoded)[0][1]
    st.write(f"### Approval Probability: {prob:.2f}")

    if prob >= 0.5:
        st.success("This applicant is likely to be APPROVED.")
    else:
        st.error("This applicant is likely to be DENIED.")
