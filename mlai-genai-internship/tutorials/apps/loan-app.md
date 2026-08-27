# Deploying an ML Model with Streamlit: Loan Approval Prediction

**Module 5 · Session 5.2** — your first application that serves a *trained Machine Learning model* rather than calling an API.

Until now your models have lived in a notebook and printed to a terminal. Nobody but you could use them. This tutorial takes a trained Random Forest and puts it behind a web form that anyone can fill in.

```text
Trained model file (rf_model.joblib)
          ↓
Streamlit form collects applicant details
          ↓
The SAME encoding used during training is applied
          ↓
model.predict()
          ↓
Approved / Rejected + probability shown in the browser
```

---

## Before you start

You need the `genai` environment from the [Student Handbook](../../setup-guide.md):

```bash
conda activate genai
pip install streamlit pandas scikit-learn joblib
```

Create a folder for this app:

```bash
mkdir loan_app
cd loan_app
```

You will create two files in it:

```text
loan_app/
├── train_model.py     # run ONCE - produces the model file
├── rf_model.joblib    # created by train_model.py
└── app.py             # the Streamlit app
```

---

## Step 1: Train and save the model

**This step is the one students most often skip, and then wonder why the app crashes with `FileNotFoundError: rf_model.joblib`.** The app does not train anything. It loads a model that must already exist.

Create **`train_model.py`**:

```python
"""
train_model.py
Trains a Random Forest on the loan dataset and saves it as rf_model.joblib.
Run this ONCE before starting the Streamlit app.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_URL = (
    "https://raw.githubusercontent.com/tech4alltraining/aiml/"
    "refs/heads/main/datasets/loan_data_10k.csv"
)

TARGET = "loan_status"

# The ORDER of this list matters. The app must send columns in exactly
# this order or the model will misread the values.
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
    "previous_loan_defaults_on_file",
]

CATEGORICAL_COLUMNS = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

# 1. Load the data
df = pd.read_csv(DATA_URL)
print("Rows and columns:", df.shape)

# 2. Remove rows with missing values (this dataset has only three)
before = len(df)
df = df.dropna().reset_index(drop=True)
print("Removed", before - len(df), "rows with missing values")

# 3. Encode text columns as numbers.
#    LabelEncoder assigns 0, 1, 2 ... in ALPHABETICAL order of the values.
#    Print the mapping - you must copy it into app.py exactly.
encoders = {}
for column in CATEGORICAL_COLUMNS:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print(column, "->", encoders[column])

# 4. Split features and target
X = df[FEATURE_COLUMNS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate - never ship a model you have not measured
y_pred = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

# 7. Save the trained model next to app.py
joblib.dump(model, "rf_model.joblib")
print("Saved rf_model.joblib")
```

Run it:

```bash
conda activate genai
python train_model.py
```

**Expected output** (your numbers will be very close to these):

```text
Rows and columns: (10000, 14)
Removed 3 rows with missing values
person_gender -> {'female': 0, 'male': 1}
person_education -> {'Associate': 0, 'Bachelor': 1, 'Doctorate': 2, 'High School': 3, 'Master': 4}
person_home_ownership -> {'MORTGAGE': 0, 'OTHER': 1, 'OWN': 2, 'RENT': 3}
loan_intent -> {'DEBTCONSOLIDATION': 0, 'EDUCATION': 1, 'HOMEIMPROVEMENT': 2, 'MEDICAL': 3, 'PERSONAL': 4, 'VENTURE': 5}
previous_loan_defaults_on_file -> {'No': 0, 'Yes': 1}
Test accuracy: 0.891
...
Saved rf_model.joblib
```

> **Read those mapping lines carefully.** They are printed for a reason. The `label_mappings` dictionary in `app.py` must match them **exactly**. If you train on a different dataset, or the categories differ, copy the new mappings across — otherwise the app will silently send "Master" where the model expects "Doctorate", and every prediction will be quietly wrong.

**✅ Check yourself:** run `ls` (or `dir` on Windows). Do you see `rf_model.joblib`? If not, do not continue — Step 2 cannot work.

---

## Step 2: The Streamlit app

Create **`app.py`** in the same folder:

```python
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

---

## Step 3: Run it

```bash
conda activate genai
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Stop the server with `Ctrl+C`.

---

## How this app works

### The three things that must line up

This is the whole lesson of the tutorial. A deployed model is fragile in exactly three ways:

| Must match | Why | What breaks if it does not |
|---|---|---|
| **The feature list** | The model learned 13 columns | `ValueError: X has 12 features, but RandomForestClassifier is expecting 13` |
| **The column order** | The model identifies columns by position, not name | No error at all — just silently wrong predictions |
| **The label encoding** | The model learned that `RENT` means `3` | No error — the app sends the wrong number and the prediction is wrong |

Only the first one gives you an error message. **The other two fail silently**, which makes them far more dangerous. That is why `app.py` ends with `input_df = input_df[FEATURE_COLUMNS]` — it forces the order every single time.

### Why `@st.cache_resource`

```python
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
```

Streamlit re-runs your entire script on **every** interaction. Without this decorator, the model would be read from disk every time a user moves a slider. With it, the model loads once and stays in memory.

### Why `Path(__file__).parent`

```python
MODEL_PATH = Path(__file__).parent / "rf_model.joblib"
```

This finds the model next to `app.py`, regardless of which folder you ran `streamlit run` from. Writing `joblib.load("rf_model.joblib")` works only when your terminal happens to be in the right directory — a very common source of `FileNotFoundError`.

---

## Try these

### 1. Find the tipping point

Enter an applicant who is clearly approved. Now change **one field at a time** until the prediction flips.

Which single field flipped it most easily? Compare with a classmate.

### 2. Check it against feature importance

Add this to `train_model.py` and re-run it:

```python
importances = (pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
                 .sort_values(ascending=False))
print("\nFeature importance:")
print(importances.round(4))
```

**Does the field that flipped predictions most easily appear near the top?** If a field flips predictions easily but has low importance, that is worth investigating — it may mean your test applicant was sitting right on the decision boundary.

### 3. Look at the probability, not just the label

The app prints both. Find an applicant where the model says "Approved" with only 55% confidence.

**Would you show that applicant a flat "✅ Loan Approved"?** A prediction at 55% and one at 99% are very different claims, and a good interface says so.

---

## Responsible AI checkpoints

Before you would ever deploy something like this:

- **The target encoding is an assumption.** The app assumes `1 = Approved`. Verify that against your training data — getting it backwards inverts every decision in the app.
- **`person_gender` is a feature in this model.** Should a loan decision use gender at all? Check the accuracy separately for each gender group. If they differ, you have a fairness problem, not a technical one.
- **A rejected applicant deserves a reason.** A red ❌ with no explanation is not acceptable for a decision affecting someone's money. [`ml_gen_ai.md`](ml_gen_ai.md) adds that explanation layer.
- **A human must sign off.** This model is roughly 89% accurate — meaning it is wrong about one applicant in nine.

---

## Next

[`ml_gen_ai.md`](ml_gen_ai.md) takes this exact app and adds a Generative AI layer, so that the ML model makes the prediction and an LLM explains it in plain language.
