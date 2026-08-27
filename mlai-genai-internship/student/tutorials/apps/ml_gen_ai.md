# Integrating Machine Learning with Generative AI

**Module 5 · Session 5.1** — the strongest single idea of the week, and the best thing you can put in a capstone project.

**Machine Learning model predicts loan approval. Generative AI explains the prediction in simple language. Streamlit provides the user interface.**

> 🧠 **The analogy to hold onto:** the **doctor** reads the test results and makes the diagnosis; the **receptionist** explains what happens next in words you understand. The ML model is the doctor. The LLM is the receptionist. A hospital where the receptionist diagnoses patients is a disaster — and so is an app where the LLM decides the loan.

## Before you start

Complete [`loan-app.md`](loan-app.md) first — this tutorial extends that app and reuses its `rf_model.joblib`.

```bash
conda activate genai
pip install streamlit pandas scikit-learn joblib google-genai
```

You also need a Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Setup instructions: [Student Handbook](../../setup-guide.md#storing-your-gemini-api-key-safely).

---


So the demo flow is:

```text
User enters loan details
        ↓
Streamlit sends input to trained ML model
        ↓
ML model predicts Approved / Rejected
        ↓
Generative AI explains why the model may have predicted that result
        ↓
User sees prediction + explanation + suggestions
```

Streamlit is suitable for this because it is designed for quickly building interactive ML/data apps, and Gemini can be called from Python using Google’s GenAI SDK and `generate_content`. ([Streamlit Docs][1]) ([Google AI for Developers][2])

---

## Folder structure

Create a folder like this:

```text
loan_genai_demo/
│
├── app.py
├── rf_model.joblib
├── requirements.txt
└── .streamlit/
    └── secrets.toml
```

Your trained model file should be placed in the same folder as `app.py`.

---

## requirements.txt

```txt
streamlit
pandas
numpy
scikit-learn
joblib
google-genai
```

Scikit-learn supports persisting trained models so that they can later be reused for prediction instead of retraining. ([Scikit-Learn][3])

---

## .streamlit/secrets.toml

Do not hard-code the Gemini API key inside the Python file. Streamlit supports storing secrets such as API keys in `secrets.toml`, and `st.secrets` provides dictionary-like access to those values. ([Streamlit Docs][4])

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

## app.py

```python
import streamlit as st
import pandas as pd
import joblib
from google import genai

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="ML + GenAI Loan Approval Demo",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Integrating Machine Learning with Generative AI")
st.subheader("Loan Approval Prediction + GenAI Explanation")

st.write(
    """
    This demo uses a trained Machine Learning model to predict loan approval.
    Then, Generative AI explains the prediction in simple language.
    """
)

# -----------------------------
# Load ML model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

model = load_model()

# -----------------------------
# Load Gemini client
# -----------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "gemini-3.5-flash"

# -----------------------------
# IMPORTANT:
# Use the exact same encoding used during training.
# These mappings are examples. Replace them with your actual LabelEncoder mappings.
# -----------------------------
gender_map = {
    "female": 0,
    "male": 1
}

education_map = {
    "Associate": 0,
    "Bachelor": 1,
    "Doctorate": 2,
    "High School": 3,
    "Master": 4
}

home_ownership_map = {
    "MORTGAGE": 0,
    "OTHER": 1,
    "OWN": 2,
    "RENT": 3
}

loan_intent_map = {
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5
}

previous_default_map = {
    "No": 0,
    "Yes": 1
}

feature_order = [
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
# User input form
# -----------------------------
st.header("Enter Applicant Details")

with st.form("loan_form"):
    person_age = st.number_input("Age", min_value=18, max_value=100, value=29)
    person_gender = st.selectbox("Gender", ["female", "male"])
    person_education = st.selectbox(
        "Education",
        ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
    )
    person_income = st.number_input("Annual Income", min_value=0.0, value=39704.0)
    person_emp_exp = st.number_input("Employment Experience in Years", min_value=0, max_value=60, value=8)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10800.0)
    loan_intent = st.selectbox(
        "Loan Intent",
        ["MEDICAL", "EDUCATION", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=50.0, value=13.57)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.27)
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, max_value=50, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=607)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", ["No", "Yes"])

    submitted = st.form_submit_button("Predict and Explain")

# -----------------------------
# Prediction + GenAI explanation
# -----------------------------
if submitted:
    raw_input = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    encoded_input = {
        "person_age": person_age,
        "person_gender": gender_map[person_gender],
        "person_education": education_map[person_education],
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": home_ownership_map[person_home_ownership],
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent_map[loan_intent],
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_default_map[previous_loan_defaults_on_file]
    }

    input_df = pd.DataFrame([encoded_input], columns=feature_order)

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0]
        approval_probability = probability[1]
    else:
        approval_probability = None

    if prediction == 1:
        result = "Approved"
        st.success("Loan Status: Approved")
    else:
        result = "Rejected"
        st.error("Loan Status: Rejected")

    if approval_probability is not None:
        st.metric("Approval Probability", f"{approval_probability:.2%}")

    # Do not allow GenAI to invent the decision.
    # The ML model has already made the prediction.
    prompt = f"""
    You are an AI assistant explaining a machine learning loan approval prediction.

    The ML model prediction is: {result}

    Applicant details:
    {raw_input}

    Approval probability:
    {approval_probability}

    Explain the prediction in simple language for students.

    Important rules:
    1. Do not say the decision is final.
    2. Do not claim that you know the internal exact reasoning of the model.
    3. Explain possible contributing factors such as income, loan amount, credit score,
       loan-to-income ratio, interest rate, employment experience, and previous default.
    4. Do not use gender as a reason for approval or rejection.
    5. Give 3 practical suggestions to improve approval chances if the prediction is rejected.
    6. Keep the explanation concise and easy to read.
    """

    with st.spinner("Generating explanation using Generative AI..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

    st.header("GenAI Explanation")
    st.write(response.text)

    st.header("Input Sent to ML Model")
    st.dataframe(input_df)
```

---

## Run it

From the terminal:

```bash
conda activate genai
cd loan_genai_demo
pip install -r requirements.txt
streamlit run app.py
```

Stop the server with `Ctrl+C`.

---

## How the two halves fit together

**Step 1 — the ML model predicts.** The trained Random Forest receives structured input: age, income, credit score, loan amount, interest rate, loan intent, previous defaults. It outputs a class label — approved or rejected.

**Step 2 — the GenAI model explains.** It does **not** make the decision. It receives the prediction *and* the applicant details, and turns them into a readable explanation.

**Step 3 — Streamlit joins them.** The user enters values, clicks a button, sees the prediction, then sees the explanation.

> **Machine Learning is used for prediction. Generative AI is used for explanation and interaction.** That distinction is the whole point of this tutorial.

---

## Try it

1. Look at the trained model file `rf_model.joblib` — it was trained on historical loan data before you ever ran the app
2. Run the app and enter one applicant's details
3. Click **Predict and Explain**
4. Read the ML result, then read the GenAI explanation
5. Change one field — credit score, income, loan amount, previous defaults
6. Run again and compare the two explanations

**Then answer:** why should the GenAI layer *explain* the result rather than *replace* the ML model?

---

## What you have built

```text
Machine Learning = prediction engine
Generative AI    = explanation engine
Streamlit        = user interface
```

Traditional ML and GenAI combined into an application that is more interactive and more understandable than either alone.

[1]: https://docs.streamlit.io/ "Streamlit documentation"
[2]: https://ai.google.dev/gemini-api/docs/text-generation "Text generation - generateContent API"
[3]: https://scikit-learn.org/stable/model_persistence.html "11. Model persistence"
[4]: https://docs.streamlit.io/develop/api-reference/connections/st.secrets "st.secrets - Streamlit Docs"
