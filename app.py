import streamlit as st
import numpy as np
import pandas as pd
from model import load_model

# Load the trained model and scaler
model, scaler, imputer = load_model()

# Define the column names for consistency
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

st.title("Diabetes Prediction")

st.write("Enter the details of the patient or choose a sample input:")

# Sample inputs
samples = {
    "Non-Diabetic Case": [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    "Diabetic Case": [8, 183, 64, 0, 0, 23.3, 0.672, 32],
    "Borderline Case": [3, 144, 80, 15, 0, 32.5, 0.465, 40],
    "High Risk Case": [10, 168, 74, 0, 0, 38.0, 0.537, 34],
    "Low Risk Case": [0, 89, 66, 23, 94, 28.1, 0.167, 21],
    "Moderate Risk Case": [5, 116, 74, 0, 0, 25.6, 0.201, 30],
    "Elderly Case": [2, 120, 68, 20, 80, 30.0, 0.550, 65]
}

# Dropdown to select sample input
sample_input = st.selectbox("Select a sample input", options=["Manual Input"] + list(samples.keys()))

if sample_input == "Manual Input":
    # Input fields for user to enter data
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
else:
    # Use selected sample input
    sample_data = samples[sample_input]
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age = sample_data

    # Display the selected sample input values
    st.write("Selected sample input values:")
    st.write(f"Pregnancies: {pregnancies}")
    st.write(f"Glucose: {glucose}")
    st.write(f"BloodPressure: {blood_pressure}")
    st.write(f"SkinThickness: {skin_thickness}")
    st.write(f"Insulin: {insulin}")
    st.write(f"BMI: {bmi}")
    st.write(f"DiabetesPedigreeFunction: {dpf}")
    st.write(f"Age: {age}")

# Button to make the prediction
if st.button("Predict"):
    try:
        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        user_data_df = pd.DataFrame(user_data, columns=column_names)
        user_data_imputed = imputer.transform(user_data_df)
        user_data_scaled = scaler.transform(user_data_imputed)
        prediction = model.predict(user_data_scaled)
        prediction_proba = model.predict_proba(user_data_scaled)

        st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
        st.write(f"Prediction Probability: {prediction_proba[0][prediction[0]]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
