import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SmartSante - Estimation Charges", layout="centered")

st.title("Estimation des charges d’assurance santé")

xgb_reg = joblib.load("../src/SmartSante_IA")       
scaler = joblib.load("../src/SmartSante_Scaler")          

st.header("Informations personnelles")
age = st.number_input("Âge", min_value=0)
children = st.number_input("Nombre d'enfants", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
sex = st.selectbox("Sexe", ["male", "female"])
smoker = st.selectbox("Fumeur ?", ["yes", "no"])
region = st.selectbox("Région", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Estimer les Charges"):
    if age <= 10 or bmi <=15 :
        st.warning("⚠️ Veuillez entrer des valeurs valides : âge > 10 ans et BMI > 15.")
    else:
        new_person = pd.DataFrame({
            "age": [age],
            "sex": [0 if sex=="male" else 1],          
            "bmi": [bmi],
            "children": [children],
            "smoker": [1 if smoker=="yes" else 0],         
            "region_northeast": [1 if region=="northeast" else 0],
            "region_northwest": [1 if region=="northwest" else 0],
            "region_southeast": [1 if region=="southeast" else 0],
            "region_southwest": [1 if region=="southwest" else 0]
        })

        new_person["bmi"] = np.log1p(new_person["bmi"])

        feature_scaled = ["age", "children"]
        new_person[feature_scaled] = scaler.transform(new_person[feature_scaled])

        predicted_log = xgb_reg.predict(new_person)
        predicted_charge = np.expm1(predicted_log)

        st.success(f"💶 Charges estimées : {predicted_charge[0]:.2f} €")
