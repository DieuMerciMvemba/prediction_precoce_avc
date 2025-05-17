import streamlit as st
import joblib, numpy as np

@st.cache_resource
def load_pipeline():
    data = joblib.load('stroke_pipeline.pkl')
    return data['model'], data['scaler'], data['encoder']

model, scaler, encoder = load_pipeline()

st.header("Prédiction Précoce d’AVC")
st.write("Remplissez les données cliniques ci-dessous pour obtenir un risque estimé.")

# Sidebar inputs
with st.sidebar:
    st.subheader("Paramètres de prédiction")
    gender = st.selectbox("Genre", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Marié(e)", ["No", "Yes"])
    work_type = st.selectbox("Type de travail", [
        "Private", "Self-employed", "Govt_job", "Children", "Never_worked"
    ])
    Residence_type = st.selectbox("Type de résidence", ["Urban", "Rural"])
    smoking_status = st.selectbox("Statut tabagique", [
        "formerly smoked", "never smoked", "smokes", "Unknown"
    ])
    age = st.number_input("Âge", value=50.0, min_value=0.0, max_value=120.0, step=1.0)
    avg_glucose_level = st.number_input("Glucose moyen", value=100.0, min_value=0.0, max_value=300.0, step=0.1)
    bmi = st.number_input("IMC", value=25.0, min_value=0.0, max_value=70.0, step=0.1)
    if st.button("Prédire"):
        X_num = scaler.transform([[age, avg_glucose_level, bmi]])
        X_cat = encoder.transform([[gender, ever_married, work_type, Residence_type, smoking_status]])
        X = np.hstack([X_num, X_cat])
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0][1]
        risk_percent = round(proba * 100, 2)
        st.subheader("Résultats")
        st.write(f"**Risque d’AVC détecté :** {'Oui' if pred==1 else 'Non'}")
        st.write(f"**Probabilité estimée :** {risk_percent} %")