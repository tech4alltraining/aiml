```{python}
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "rf_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Feature order
# Must match training feature order
# -----------------------------
FEATURE_COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file"
]

# -----------------------------
# LabelEncoder mappings
# Based on sklearn LabelEncoder alphabetical encoding
# -----------------------------
label_mappings = {
    "person_gender": {
        "female": 0,
        "male": 1
    },
    "person_education": {
        "Associate": 0,
        "Bachelor": 1,
        "Doctorate": 2,
        "High School": 3,
        "Master": 4
    },
    "person_home_ownership": {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3
    },
    "loan_intent": {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3,
        "PERSONAL": 4,
        "VENTURE": 5
    },
    "previous_loan_defaults_on_file": {
        "No": 0,
        "Yes": 1
    }
}

# -----------------------------
# App title
# -----------------------------
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant and loan details to predict loan approval status.")

st.divider()

# -----------------------------
# User input form
# -----------------------------
with st.form("loan_prediction_form"):

    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=29,
            step=1
        )

        person_gender = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        person_education = st.selectbox(
            "Education",
            ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
        )

        person_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=39704.0,
            step=1000.0
        )

        person_emp_exp = st.number_input(
            "Employment Experience",
            min_value=0,
            max_value=80,
            value=8,
            step=1
        )

    with col2:
        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["MORTGAGE", "OTHER", "OWN", "RENT"],
            index=3
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=607,
            step=1
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        )

    st.subheader("Loan Information")

    col3, col4 = st.columns(2)

    with col3:
        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=10800.0,
            step=500.0
        )

        loan_intent = st.selectbox(
            "Loan Intent",
            [
                "DEBTCONSOLIDATION",
                "EDUCATION",
                "HOMEIMPROVEMENT",
                "MEDICAL",
                "PERSONAL",
                "VENTURE"
            ],
            index=3
        )

    with col4:
        loan_int_rate = st.number_input(
            "Loan Interest Rate",
            min_value=0.0,
            max_value=100.0,
            value=13.57,
            step=0.01
        )

        loan_percent_income = st.number_input(
            "Loan Percent Income",
            min_value=0.0,
            max_value=1.0,
            value=0.27,
            step=0.01
        )

        previous_loan_defaults_on_file = st.selectbox(
            "Previous Loan Defaults on File",
            ["No", "Yes"]
        )

    submitted = st.form_submit_button("Predict")


# -----------------------------
# Prediction logic
# -----------------------------
if submitted:

    input_data = {
        "person_age": person_age,
        "person_gender": label_mappings["person_gender"][person_gender],
        "person_education": label_mappings["person_education"][person_education],
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": label_mappings["person_home_ownership"][person_home_ownership],
        "loan_amnt": loan_amnt,
        "loan_intent": label_mappings["loan_intent"][loan_intent],
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": label_mappings["previous_loan_defaults_on_file"][previous_loan_defaults_on_file]
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_COLUMNS]

    st.subheader("Model Input After Label Encoding")
    st.dataframe(input_df)

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    # Assumption: 1 = Approved, 0 = Rejected
    # Change this if your target variable was encoded differently.
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        st.write("### Prediction Probability")

        if len(probabilities) == 2:
            st.write(f"Rejected Probability: **{probabilities[0] * 100:.2f}%**")
            st.write(f"Approved Probability: **{probabilities[1] * 100:.2f}%**")
        else:
            st.write(probabilities)
```