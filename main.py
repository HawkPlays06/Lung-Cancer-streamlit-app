import streamlit as st
import pandas as pd
import joblib


pipeline = joblib.load("pipeline.pickle")
model = joblib.load("model_dt.pickle")


st.title("Patient Response Prediction")
st.markdown("**Please provide patient information**:")

"""
gender
age group
race
histology
smoker
background
stage
marker present
marker
result
treatment

number of family members with lung
number of comorbidities

"""

sex = st.selectbox("Sex", ["female", "male"])
age = st.selectbox("Age group", ["30 - 49", "49 - 64", "64 - 90"])
race = st.selectbox("Race", ["Caucasian", "Latin American", "African", "Asian"])
smoker = st.selectbox("Smoking habits", ["Active smoker", "Not a smoker (<100 cigarettes/life time)", "Former smoker (> 1 year)", "Unknown"])

histology = st.selectbox("Histology results", ["Small cell carcinoma  (microcytic)", "Large cell carcinoma","Adenocarcinoma", "Squamous", "NOS/Undifferentiated", "Other","Adenosquamous", "Large cell neuroendocrine carcinoma","Sarcomatoid"])

background = st.selectbox("Has the patient's family suffered from cancer", ["Unknown", "Yes", "No"])
numfamlung = int(st.number_input("Number of family members suffered from lung cancer",min_value=0, max_value=10, value=0))
numfamother = int(st.number_input("Number of family members suffered from cancer",min_value=0, max_value=10, value=0))

stage = st.selectbox("Cancer stage", ["IIB", "IIIB", "IV", "IIIA", "IIIC", "IVB", "Limited", "IVA", "Extended", "IA", "IIA", "IB", "Other"])
marker_present = "Yes" if st.checkbox("Marker present") else "No"
marker = st.selectbox("Marker", ["nan", "EGFR", "ALK", "PDL1", "Other", "BRAF", "HER2"])
result = st.selectbox("Result", ["nan", "Negative", "Exon 19", "Negative IHQ", "non-traslocated FISH", "Exon 21", "Positive", "Traslocated FISH", "Not detected", "Negative T790M", "unamplified FISH", "Positive T790M", "Other"])
treatment = st.selectbox("treatment", ["Intravenous and oral chemotherapy", "Sequential chemotherapy-radiotherapy", "Adjuvant chemotherapy", "Intravenous chemotherapy", "Targeted oral therapy", "Concomitant chemotherapy-radiotherapy", "Immunotherapy", "Neoadjuvant chemotherapy", "Intravenous chemotherapy + immunotherapy", "Neoadjuvant  chemotherapy-radiotherapy", "Adjuvant chemotherapy-radiotherapy"])

comorbidity_count = int(st.number_input("Number of comorbidities",min_value=0, value=0))

prediction_state = st.markdown("calculating...")

patient = pd.DataFrame([{
    "Gender": sex,
    "AgeGroup": age,
    "race": race,
    "smokinghabit": smoker,
    "backgroundind": background,
    "numfamlung": numfamlung,
    "inistage": stage,
    "histology": histology,
    "molmarind": marker_present,
    "molmar1": marker,
    "result1": result,
    "numcir": comorbidity_count,
    "phartreat1": treatment,
    "numfamother" : numfamother
}])

patient_transform = pipeline.transform(patient)

y_pred = model.predict(patient_transform)

prediction_state.markdown(f"Prediction: **{y_pred[0]}**")