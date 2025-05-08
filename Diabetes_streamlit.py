import streamlit as st
import pandas as pd
import numpy as np
import joblib  

# Load the trained model and scaler
scaler = joblib.load("scalar.joblib")  # Load the standard scaler
model = joblib.load("Diabetes.joblib")  # Load the trained model

st.title("Diabetes Prediction App")

# Sidebar inputs
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=15, step=1)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
glucose = st.sidebar.number_input("Glucose Level", min_value=50, max_value=300, step=1)
bp = st.sidebar.number_input("Blood Pressure", min_value=40, max_value=180, step=1)
hba1c = st.sidebar.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, step=0.1)
ldl = st.sidebar.number_input("LDL Cholesterol", min_value=50, max_value=250, step=1)
hdl = st.sidebar.number_input("HDL Cholesterol", min_value=20, max_value=100, step=1)
triglycerides = st.sidebar.number_input("Triglycerides", min_value=50, max_value=500, step=5)
waist = st.sidebar.number_input("Waist Circumference (cm)", min_value=50, max_value=200, step=1)
hip = st.sidebar.number_input("Hip Circumference (cm)", min_value=50, max_value=200, step=1)
family_history = st.sidebar.selectbox("Family History of Diabetes?", ['No', 'Yes'])
diet_type = st.sidebar.selectbox("Diet Type", ["Healthy", "Unhealthy"])
hypertension = st.sidebar.checkbox("Has Hypertension?")
medication_use = st.sidebar.checkbox("Uses Medication?")

# Compute WHR
whr = round(waist / hip, 2) if hip != 0 else 0

# Convert categorical inputs to numerical
family_history = 1 if family_history == "Yes" else 0
diet_type = 1 if diet_type == "Unhealthy" else 0
hypertension = int(hypertension)
medication_use = int(medication_use)

# Create DataFrame for prediction
input_data = np.array([[age, pregnancies, bmi, glucose, bp, hba1c, ldl, hdl, 
                        triglycerides, waist, hip, whr, family_history, diet_type, 
                        hypertension, medication_use]])

btn = st.sidebar.button("Predict")

if btn:
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # Convert to percentage
    
    # Show result
    if prediction == 1:
        st.error(f"High Risk of Diabetes ({probability:.2f}%)")
    else:
        st.success(f"Low Risk of Diabetes ({probability:.2f}%)")