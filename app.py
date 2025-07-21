import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

# 🎯 Load the Gradient Boosting model from MLflow Registry
model_uri = "models:/insurance_model_gradient_boosting@champion"
model = mlflow.sklearn.load_model(model_uri)

# 🎯 Load the scaler from the correct folder
scaler = joblib.load("models/scalers/minmax_scaler.pkl")

# 🌟 Streamlit UI
st.set_page_config(page_title="Insurance Charges Predictor", page_icon="💰")
st.title("💸 Insurance Charges Prediction App")

# Input fields
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ['southwest', 'northwest', 'northeast', 'southeast'])

# 🔁 Encode categorical variables
sex_male = 1 if sex == "male" else 0
smoker_flag = 1 if smoker == "yes" else 0
region_map = {'southwest': 0, 'northwest': 1, 'northeast': 2, 'southeast': 3}
region_encoded = region_map[region]

# 🔢 Construct input DataFrame
input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "smoker": smoker_flag,
    "region": region_encoded,
    "sex_male": sex_male
}])

# 📏 Scale numerical features
input_df[["age", "bmi"]] = scaler.transform(input_df[["age", "bmi"]])

# ✅ Ensure correct column order
input_df = input_df[['age', 'bmi', 'children', 'smoker', 'region', 'sex_male']]

# 🧠 Predict
if st.button("Predict Charges"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Insurance Charges: **${prediction:,.2f}**")
