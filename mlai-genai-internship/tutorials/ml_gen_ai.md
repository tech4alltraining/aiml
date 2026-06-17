**Machine Learning model predicts loan approval. Generative AI explains the prediction in simple language. Streamlit provides the user interface.**

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

## Demo title

**“Integrating Machine Learning with Generative AI for Loan Approval Explanation”**

This is a good classroom demo because it clearly shows the difference between:

| Component        | Role                                               |
| ---------------- | -------------------------------------------------- |
| Machine Learning | Makes the actual prediction                        |
| Generative AI    | Explains the prediction in human-readable language |
| Streamlit        | Provides the web interface                         |

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
    6. Keep the explanation concise and classroom-friendly.
    """

    with st.spinner("Generating explanation using Generative AI..."):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

    st.header("GenAI Explanation")
    st.write(response.text)

    st.header("Input Sent to ML Model")
    st.dataframe(input_df)
```

---

## How to run the demo

From the terminal:

```bash
cd loan_genai_demo
pip install -r requirements.txt
streamlit run app.py
```

---

## How to explain this demo to students

You can explain it like this:

**Step 1: Traditional ML prediction**

The trained Random Forest model receives structured input such as age, income, credit score, loan amount, interest rate, loan intent, and previous default status. It outputs a class label such as approved or rejected.

**Step 2: GenAI explanation**

The Generative AI model does not make the loan decision. It receives the ML prediction and the applicant details, then generates a human-readable explanation.

**Step 3: Streamlit integration**

Streamlit connects both parts into a single web app. The user enters values, clicks a button, sees the ML prediction, and then sees a GenAI-generated explanation.

---

## Important point for your demo

Say this clearly during the session:

> “Machine Learning is used for prediction. Generative AI is used for explanation and interaction.”

That distinction is the main learning outcome.

---

## Recommended demo script for classroom

You can present it in this order:

1. Show the trained model file: `rf_model.joblib`.
2. Explain that the model was already trained using historical loan data.
3. Run the Streamlit app.
4. Enter one applicant’s details.
5. Click **Predict and Explain**.
6. Show the ML result: Approved/Rejected.
7. Show the GenAI explanation.
8. Change values such as credit score, income, loan amount, or previous default.
9. Run again and compare the explanation.
10. Discuss why GenAI should explain the result, not replace the ML model.

---

## Best teaching conclusion

This demo shows a practical AI-powered application:

```text
Machine Learning = prediction engine
Generative AI = explanation engine
Streamlit = user interface
```

This is a strong example of how traditional ML and GenAI can be combined to build more interactive, understandable, and user-friendly AI applications.

[1]: https://docs.streamlit.io/?utm_source=chatgpt.com "Streamlit documentation"
[2]: https://ai.google.dev/gemini-api/docs/text-generation?utm_source=chatgpt.com "Text generation - generateContent API"
[3]: https://scikit-learn.org/stable/model_persistence.html?utm_source=chatgpt.com "11. Model persistence"
[4]: https://docs.streamlit.io/develop/api-reference/connections/st.secrets?utm_source=chatgpt.com "st.secrets - Streamlit Docs"
