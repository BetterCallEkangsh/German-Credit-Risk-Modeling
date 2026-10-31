import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="German Credit Risk Predictor", page_icon="💳", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return None


st.title("💳 German Credit Risk Predictor")
st.caption("Estimates whether a loan applicant is a good or bad credit risk.")

if not os.path.exists("model.joblib"):
    st.error("model.joblib not found. Run `python train.py german_credit_data.csv` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Risk threshold", 0.10, 0.90, 0.50, 0.05,
        help="Flag an applicant as high risk when P(bad) is at or above this value. "
             "Lower it to catch more risky applicants (higher recall, more false alarms).",
    )
    if metrics:
        st.header("Model performance")
        st.caption("Held-out test set (200 applicants), AdaBoost + SMOTE")
        st.metric("Test ROC-AUC", metrics["test_auc"])
        st.metric("Recall (bad class)", metrics["test_recall_bad"])
        st.metric("F1 (bad class)", metrics["test_f1_bad"])
        st.metric("Accuracy", metrics["test_accuracy"])

st.subheader("Applicant details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 80, 35)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.selectbox(
        "Job level",
        [0, 1, 2, 3],
        index=2,
        format_func=lambda x: {0: "0 - Unskilled, non-resident", 1: "1 - Unskilled, resident",
                               2: "2 - Skilled", 3: "3 - Highly skilled"}[x],
    )
    housing = st.selectbox("Housing", ["own", "rent", "free"])
with c2:
    credit = st.number_input("Credit amount", 250, 20000, 3000, step=100)
    duration = st.slider("Duration (months)", 4, 72, 20)
    purpose = st.selectbox(
        "Purpose",
        ["car", "radio/TV", "furniture/equipment", "business", "education",
         "domestic appliances", "repairs", "vacation/others"],
    )

c3, c4 = st.columns(2)
with c3:
    saving = st.selectbox("Saving accounts", ["no info", "little", "moderate", "quite rich", "rich"])
with c4:
    checking = st.selectbox("Checking account", ["no info", "little", "moderate", "rich"])

if st.button("Predict risk", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "Sex": sex, "Housing": housing, "Saving accounts": saving,
        "Checking account": checking, "Purpose": purpose,
        "Age": age, "Job": job, "Credit amount": credit, "Duration": duration,
    }])
    p_bad = float(model.predict_proba(row)[0, 1])

    st.divider()
    st.subheader("Result")
    st.progress(p_bad, text=f"Probability of bad credit risk: {p_bad:.1%}")
    if p_bad >= threshold:
        st.error(f"High risk: P(bad) = {p_bad:.1%} is at or above the {threshold:.0%} threshold.")
    else:
        st.success(f"Lower risk: P(bad) = {p_bad:.1%} is below the {threshold:.0%} threshold.")
    st.caption(
        "Trained on only 1,000 records (test ROC-AUC around 0.67), so treat this as a "
        "demonstration, not a lending decision tool."
    )
