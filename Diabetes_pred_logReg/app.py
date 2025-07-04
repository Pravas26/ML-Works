import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model, scaler = joblib.load('models/logistic_model.pkl')

st.title("Diabetes Prediction App")
st.write("Predict whether a patient is likely to have diabetes based on health parameters.")

# Input fields
preg = st.number_input("Pregnancies", min_value=0)
glu = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    user_input = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]

    st.write("### Result:")
    st.success("Diabetic" if prediction == 1 else "Not Diabetic")
    st.write(f"Probability of being diabetic: **{probability:.2f}**")
