# Session 5C — Model Deployment & Streamlit

**Saving models · Loading and inference · Pipelines in production · Building a web app with Streamlit**

| | |
|---|---|
| **Notebook** | [session-05c-deployment.ipynb](../notebooks/session-05c-deployment.ipynb) |
| **Previous** | [Session 5B — Classification](session-05b-classification.md) |
| **Next** | [Session 6 — Augmentation & Feature Engineering](session-06-augmentation-feature-engg-red.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **You have trained models. Nobody can use them yet.**
>
> A model that only exists inside a notebook helps no one. **This session turns one into something a person can open in a browser and actually use** — which is also what your capstone will need.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Save a trained model to a file, and load it in a completely separate program
2. **Explain why you save the pipeline rather than the bare model**
3. Make predictions on new data, with the columns in the right shape
4. Store metadata alongside a model so it is still usable in six months
5. Build a Streamlit app with input widgets and a live prediction
6. Cache a model so the app stays fast
7. **Deploy something with an honest statement of its limitations**

---

## The three topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Saving, loading and inference](#1-saving-loading-and-inference) | Save the **pipeline**, not the bare model |
| 2 | [Streamlit app development](#2-streamlit-app-development) | Cache the model, or it reloads on every click |
| 3 | [Putting it together](#3-putting-it-together) | Ship something, with its limitations stated |

---

# 1. Saving, loading and inference

A model that only exists inside a notebook is useless. **Training happens once; predicting happens thousands of times.**

🧠 **Analogy: a cooked meal versus the recipe.** Training is cooking the meal — slow, and you only do it once. Saving the model is putting it in the fridge. Inference is reheating a portion whenever someone is hungry. **You do not re-cook the whole meal for every guest.**

```python
# illustrative: a syntax reference, not runnable as written.
import joblib

joblib.dump(model, "model.joblib")     # save
loaded = joblib.load("model.joblib")   # load, in a totally different program
loaded.predict(new_data)               # inference
```

> ⚠️ **The number one deployment bug:** saving the model but not the scaler. Your app then feeds raw values into a model that expects scaled ones, and the predictions are quietly wrong. **The fix is to save a `Pipeline`.**

## The Pipeline: one object, everything inside

```python
# illustrative: a syntax reference, not runnable as written.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=2000)),
])
pipe.fit(X_train, y_train)     # scaler fits on TRAIN only - leak-proof by design
joblib.dump(pipe, "pipeline.joblib")
```

Now `pipeline.joblib` contains the scaler **and** the model. Load it and pass raw values: the scaling happens inside. **A pipeline makes the leakage bug from Session 3 structurally impossible.**

## 📘 Examples

**Example 1 — save, load, predict**

```python
# illustrative: a syntax reference, not runnable as written.
import joblib, os

pipe.fit(Xtr, ytr)
joblib.dump(pipe, "diabetes_pipeline.joblib")
print("saved:", round(os.path.getsize("diabetes_pipeline.joblib") / 1024, 1), "KB")

fresh = joblib.load("diabetes_pipeline.joblib")     # imagine a different file
print("score after reload:", round(fresh.score(Xte, yte), 4))
```

The reloaded score is **identical** to the original. That is the whole point.

**Example 2 — inference on one new person**

```python
# illustrative: a syntax reference, not runnable as written.
import pandas as pd

new_patient = pd.DataFrame([{
    "gender": 1, "age": 54.0, "hypertension": 1, "heart_disease": 0,
    "smoking_history": 4, "bmi": 31.2, "HbA1c_level": 6.8,
    "blood_glucose_level": 160,
}])

print("prediction:", fresh.predict(new_patient)[0])
print("probability:", round(fresh.predict_proba(new_patient)[0][1], 4))
```

> ⚠️ **Column order and column names must match training exactly.** Build the row as a `DataFrame` with the same columns, not a bare list.

**Example 3 — save the metadata too**

```python
# illustrative: a syntax reference, not runnable as written.
bundle = {
    "pipeline": pipe,
    "features": list(X.columns),      # so the app can build the form
    "target": "diabetes",
    "test_score": float(pipe.score(Xte, yte)),
    "sklearn_version": __import__("sklearn").__version__,
}
joblib.dump(bundle, "diabetes_bundle.joblib")

b = joblib.load("diabetes_bundle.joblib")
print(b["features"])
print("trained with sklearn", b["sklearn_version"], "score", round(b["test_score"], 4))
```

**Six months from now you will not remember the column order.** The bundle does.

> ⚠️ **`.joblib` files are executable code.** Only ever load one you produced or fully trust — loading a file from an untrusted source can run arbitrary code on your machine.

# 2. Streamlit App Development

Streamlit turns a Python script into a web app. **No HTML, no JavaScript.**

🧠 **Analogy: Streamlit reruns your whole script top to bottom on every interaction.** Think of a whiteboard that is wiped and rewritten each time someone moves a slider. That explains almost everything about how it behaves — including why you must cache the model.

## The minimum app

```python
# app.py
import streamlit as st
import pandas as pd
import joblib

st.title("Diabetes Risk Checker")

@st.cache_resource            # <- load ONCE, not on every click
def load_model():
    return joblib.load("diabetes_pipeline.joblib")

model = load_model()

age = st.slider("Age", 1, 100, 45)
bmi = st.slider("BMI", 10.0, 60.0, 27.0)
hba1c = st.slider("HbA1c level", 3.0, 9.0, 5.7)
glucose = st.slider("Blood glucose", 70, 300, 140)
hypertension = st.checkbox("Hypertension")

if st.button("Predict"):
    row = pd.DataFrame([{
        "gender": 1, "age": age, "hypertension": int(hypertension),
        "heart_disease": 0, "smoking_history": 4, "bmi": bmi,
        "HbA1c_level": hba1c, "blood_glucose_level": glucose,
    }])
    prob = model.predict_proba(row)[0][1]
    st.metric("Risk", f"{prob:.1%}")
    st.warning("Elevated risk") if prob > 0.5 else st.success("Low risk")
    st.caption("Educational demo only. Not medical advice.")
```

Run it:

```bash
streamlit run app.py
```

## The widgets you actually need

| Widget | Use for |
|---|---|
| `st.slider(label, min, max, default)` | A number in a known range |
| `st.number_input(label)` | An exact number |
| `st.selectbox(label, options)` | One of several categories |
| `st.checkbox(label)` | Yes/no |
| `st.file_uploader(label)` | Letting the user upload a CSV |
| `st.button(label)` | Triggering the prediction |
| `st.metric(label, value)` | Showing the headline result |
| `st.dataframe(df)` / `st.line_chart(df)` | Tables and charts |

## The three rules that fix most Streamlit bugs

1. **`@st.cache_resource` on anything you load** — models, database connections. Without it your 22MB forest reloads on every keypress.
2. **`@st.cache_data` on anything you compute** — loaded CSVs, aggregations.
3. **`st.session_state` for anything that must survive a rerun** — a running history, a counter, a chat log.

```python
# streamlit-only: run with `streamlit run app.py`, not as a plain script
if "history" not in st.session_state:
    st.session_state.history = []          # runs once, not on every rerun

st.session_state.history.append(prob)
st.line_chart(st.session_state.history)
```

## 📘 Examples

**Example 1 — a CSV explorer in twelve lines**

```python
import streamlit as st
import pandas as pd

st.title("CSV Explorer")
f = st.file_uploader("Upload a CSV", type="csv")

if f:
    df = pd.read_csv(f)
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    col = st.selectbox("Column to chart", df.select_dtypes("number").columns)
    st.bar_chart(df[col].value_counts().head(20))
```

**Example 2 — batch predictions from an upload**

```python
uploaded = st.file_uploader("Upload rows to score", type="csv")
if uploaded:
    new = pd.read_csv(uploaded)
    new["risk"] = model.predict_proba(new)[:, 1]
    st.dataframe(new.sort_values("risk", ascending=False))
    st.download_button("Download results", new.to_csv(index=False), "scored.csv")
```

**Example 3 — layout that does not look like a student project**

```python
st.set_page_config(page_title="Loan Checker", page_icon="🏦", layout="wide")

left, right = st.columns(2)
with left:
    income = st.number_input("Annual income", 0, 1_000_000, 50_000, step=1_000)
with right:
    amount = st.number_input("Loan amount", 0, 500_000, 10_000, step=1_000)

with st.sidebar:
    st.header("About")
    st.write("Random Forest, ROC-AUC 0.96 on held-out data.")

with st.expander("How this works"):
    st.write("Trained on 10,000 historical applications.")
```

> ⚠️ **Never hard-code an API key in a Streamlit app.** Put it in `.streamlit/secrets.toml` and read `st.secrets["KEY"]`. Add `.streamlit/secrets.toml` to `.gitignore`. See [setup-guide.md](../setup-guide.md).

---

# 3. Putting it together

**A complete, working deployment: train, save, and serve.**

## Part 1 — `train.py`

**This script runs once. It produces a file.**

```python
# train.py
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"

NUMERICAL = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
             'loan_int_rate', 'loan_percent_income',
             'cb_person_cred_hist_length', 'credit_score']
CATEGORICAL = ['person_gender', 'person_education', 'person_home_ownership',
               'loan_intent', 'previous_loan_defaults_on_file']

df = pd.read_csv(dataset_url).dropna()
encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le                       # keep them - the app needs them

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

bundle = {
    'pipeline': pipe,
    'encoders': encoders,
    'features': list(X.columns),
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
}
joblib.dump(bundle, 'loan_model.joblib')

print(f"saved. accuracy {bundle['accuracy']:.4f}, recall {bundle['recall']:.4f}")
```

> **The bundle is the important idea.** It carries the pipeline **and** the encoders **and** the column order **and** the measured scores. **Six months from now you will not remember any of those**, and the app needs all of them.

## Part 2 — `app.py`

**This script runs every time someone opens the page.**

```python
# app.py
# streamlit-only: run with `streamlit run app.py`, not as a plain script
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Assessment", page_icon="🏦")
st.title("Loan Application Assessment")

@st.cache_resource                 # load ONCE, not on every interaction
def load_bundle():
    return joblib.load('loan_model.joblib')

bundle = load_bundle()
pipe = bundle['pipeline']

# --- inputs
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 80, 30)
    income = st.number_input("Annual income", 0, 1_000_000, 60_000, step=1_000)
    emp_exp = st.slider("Years of employment", 0, 40, 5)
    loan_amnt = st.number_input("Loan amount", 0, 500_000, 10_000, step=1_000)
with col2:
    int_rate = st.slider("Interest rate (%)", 5.0, 25.0, 11.0)
    hist_length = st.slider("Credit history (years)", 0, 30, 5)
    credit_score = st.slider("Credit score", 300, 850, 640)
    defaults = st.selectbox("Previous defaults", ["No", "Yes"])

if st.button("Assess application", type="primary"):
    row = pd.DataFrame([{
        'person_age': age,
        'person_gender': 1,
        'person_education': 2,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': 3,
        'loan_amnt': loan_amnt,
        'loan_intent': 4,
        'loan_int_rate': int_rate,
        'loan_percent_income': loan_amnt / max(income, 1),
        'cb_person_cred_hist_length': hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': 1 if defaults == "Yes" else 0,
    }])[bundle['features']]        # <- the SAME column order as training

    probability = pipe.predict_proba(row)[0][1]
    decision = "Approved" if probability > 0.5 else "Declined"

    st.metric("Assessment", decision, f"{probability:.0%} confidence")
    st.progress(float(probability))

    if probability > 0.5:
        st.success("This application resembles those that were approved.")
    else:
        st.warning("This application resembles those that were declined.")

# --- honesty, on screen
with st.sidebar:
    st.header("About this model")
    st.metric("Test accuracy", f"{bundle['accuracy']:.1%}")
    st.metric("Test recall", f"{bundle['recall']:.1%}")
    st.caption(
        "Trained on 10,000 historical applications. "
        "**This is a decision-support tool, not an automatic approver.** "
        "A loan officer makes the decision. It has not been audited for "
        "fairness across applicant groups, and should not be used for "
        "automated decisions affecting individuals."
    )
```

```bash
streamlit run app.py
```

## What makes this deployment rather than a demo

| | |
|---|---|
| **`[bundle['features']]`** | Forces the column order to match training. **Without it, a silent wrong answer** |
| **`@st.cache_resource`** | The 200-tree forest loads once, not on every click |
| **`predict_proba`** | Shows a **probability**, so a human can judge borderline cases |
| **The sidebar** | States the measured scores **and the limitations**, on screen |

> **The sidebar is the part beginners leave out and professionals never do.** **A deployed model without a stated limitation is a claim you have not made honestly.**

## ⚠️ Before you deploy anything

- [ ] The **pipeline** is saved, not the bare model
- [ ] Column names and order match training exactly
- [ ] The model file is in `.gitignore` — it is a large binary that changes on every retrain
- [ ] No API key is hard-coded anywhere
- [ ] The interface shows a **probability**, not just a label
- [ ] The limitations are visible **on the page**, not buried in a README
- [ ] There is a route to a human when the model is wrong

---

# ✅ Before you move on

- [ ] I can save a model with `joblib` and load it in a separate program
- [ ] **I save the pipeline, so the scaler travels with the model**
- [ ] I know the number one deployment bug is a model without its scaler
- [ ] I build prediction input as a DataFrame with matching column names and order
- [ ] I store metadata — features, scores, versions — alongside the model
- [ ] I know model files belong in `.gitignore`
- [ ] I use `@st.cache_resource` so the model loads once
- [ ] I show a probability rather than a bare label
- [ ] **I state the model's limitations on the page itself**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-05c-deployment.ipynb) | Everything above |
| [Streamlit simple](../tutorials/apps/streamlit-app-simple.md) · [advanced](../tutorials/apps/streamlit-app-advanced.md) | Step-by-step app guides |
| [Loan app walkthrough](../tutorials/apps/loan-app.md) | An end-to-end ML app |
| [Session 11 — AI-Powered Applications](session-11-ai-apps.md) | Adding an LLM that explains the decision |
