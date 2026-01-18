import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_risk_model.pkl")

st.title("Credit Risk Prediction")

loan_amount = st.number_input("Loan Amount")
income = st.number_input("Income")
credit_score = st.number_input("Credit Score")
employment_length = st.number_input("Employment Length")

if st.button("Predict"):
    debt_to_income = loan_amount / income if income > 0 else 0
    data = pd.DataFrame([[loan_amount, income, credit_score, employment_length, debt_to_income]],
                        columns=['loan_amount','income','credit_score','employment_length','debt_to_income'])
    prediction = model.predict(data)[0]
    st.write("Prediction:", "Default" if prediction==1 else "Safe")
