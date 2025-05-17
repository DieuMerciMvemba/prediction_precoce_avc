import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_pipeline():
    data = joblib.load('stroke_pipeline.pkl')
    return data['model'], data['scaler'], data['encoder']

model, scaler, encoder = load_pipeline()

# Dictionnaires de correspondance : valeur interne -> libellé affiché
gender_labels = {
    "Male": "Homme",
    "Female": "Femme",
    "Other": "Autre"
}

married_labels = {
    "Yes": "Oui",
    "No": "Non"
}

work_type_labels = {
    "Private": "Privé",
    "Self-employed": "Indépendant",
    "Govt_job": "Fonctionnaire",
    "Children": "Enfant",
    "Never_worked": "Jamais travaillé"
}

residence_labels = {
    "Urban": "Urbain",
    "Rural": "Rural"
}

smoking_labels = {
    "formerly smoked": "Ancien fumeur",
    "never smoked": "Jamais fumé",
    "smokes": "Fumeur",
    "Unknown": "Inconnu"
}

# Zone principale
st.header("Prédiction Précoce d’AVC")

# Affichage des résultats si disponibles
if "prediction_done" in st.session_state and st.session_state.prediction_done:
    st.subheader("Résultats")
    st.write(f"**Risque d’AVC détecté :** {'Oui' if st.session_state.pred == 1 else 'Non'}")
    st.write(f"**Probabilité estimée :** {st.session_state.risk_percent} %")
    st.markdown("---")

st.write("Remplissez les données cliniques ci-dessous pour obtenir un risque estimé.")

# Formulaire de saisie
with st.form(key="prediction_form"):
    st.subheader("Paramètres de prédiction")
    gender = st.selectbox(
        "Genre",
        options=list(gender_labels.keys()),
        format_func=lambda x: gender_labels[x]
    )
    ever_married = st.selectbox(
        "Marié(e)",
        options=list(married_labels.keys()),
        format_func=lambda x: married_labels[x]
    )
    work_type = st.selectbox(
        "Type de travail",
        options=list(work_type_labels.keys()),
        format_func=lambda x: work_type_labels[x]
    )
    Residence_type = st.selectbox(
        "Type de résidence",
        options=list(residence_labels.keys()),
        format_func=lambda x: residence_labels[x]
    )
    smoking_status = st.selectbox(
        "Statut tabagique",
        options=list(smoking_labels.keys()),
        format_func=lambda x: smoking_labels[x]
    )
    age = st.number_input("Âge", value=50.0, min_value=0.0, max_value=120.0, step=1.0)
    avg_glucose_level = st.number_input("Glucose moyen", value=100.0, min_value=0.0, max_value=300.0, step=0.1)
    bmi = st.number_input("IMC", value=25.0, min_value=0.0, max_value=70.0, step=0.1)
    submit_button = st.form_submit_button(label="Prédire")

    if submit_button:
        # Préparation des données
        X_num = scaler.transform([[age, avg_glucose_level, bmi]])
        X_cat = encoder.transform([[gender, ever_married, work_type, Residence_type, smoking_status]])
        X = np.hstack([X_num, X_cat])

        # Prédiction
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0][1]
        risk_percent = round(proba * 100, 2)

        # Stockage dans session_state pour affichage principal
        st.session_state.pred = pred
        st.session_state.risk_percent = risk_percent
        st.session_state.prediction_done = True

        # Rafraîchir la page pour afficher les résultats
        st.rerun()
