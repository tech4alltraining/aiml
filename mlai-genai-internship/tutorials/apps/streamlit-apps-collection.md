# Streamlit Apps Collection

**Fifteen complete, runnable apps — from a single input box to a full dashboard.**

Every app here is self-contained: copy the file, run one command, and it works. They are ordered so that each one adds **exactly one new idea** to the one before it.

---

## How to use this collection

Work down a column, not across. Finish all three simple ML apps before you touch an advanced one.

| | **Machine Learning** | **Generative AI** | **ML + GenAI** |
|---|---|---|---|
| **Simple** | [A1 Salary Predictor](#a1--salary-predictor-one-input) · [A2 Sales Predictor](#a2--sales-predictor-two-inputs) · [A3 Iris Classifier](#a3--iris-flower-classifier-four-inputs) | [B1 Topic Explainer](#b1--topic-explainer-one-input) · [B2 Text Summariser](#b2--text-summariser-two-inputs) · [B3 Persona Chatbot](#b3--persona-chatbot) | [C1 Loan Advisor](#c1--loan-advisor) · [C2 Iris Botanist](#c2--iris-botanist) |
| **Advanced** | [A4 Diabetes Risk](#a4--diabetes-risk-dashboard) · [A5 Batch Predictor](#a5--batch-csv-predictor) · [A6 Model Comparison](#a6--model-comparison-lab) | [B4 Document Q&A](#b4--document-qa-grounded) · [B5 Quiz Generator](#b5--quiz-generator) · [B6 Image Describer](#b6--image-describer) | [C3 Segment Namer](#c3--customer-segment-namer) · [C4 Feedback Analyser](#c4--bulk-feedback-analyser) |

### Difficulty

| Level | You should already understand |
|---|---|
| ⭐ Simple | `st.text_input`, `st.button`, `st.write` |
| ⭐⭐ Intermediate | `st.form`, `st.session_state`, `@st.cache_resource` |
| ⭐⭐⭐ Advanced | File uploads, caching data, plotting, structured output |

---

## Before you start

Activate your environment. **Every command below assumes your prompt shows `(genai)`.**

```bash
conda activate genai
```

Or, if you used venv:

```bash
source genai/bin/activate
```

Windows venv users:

```powershell
.\genai\Scripts\Activate.ps1
```

Install what the apps need:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib google-genai python-dotenv pillow
```

Set up your project folder:

```bash
mkdir streamlit-apps
cd streamlit-apps
mkdir .streamlit
```

Create `.streamlit/secrets.toml` for the GenAI apps (skip if you are only doing section A):

```toml
GEMINI_API_KEY = "your_actual_key_here"
```

And a `.gitignore`:

```text
.streamlit/secrets.toml
*.joblib
__pycache__/
```

### The three things every Streamlit app does

```text
1. Draw widgets    st.text_input(), st.slider(), st.selectbox()
2. Do something    when a button is pressed
3. Show a result   st.write(), st.metric(), st.dataframe()
```

**Streamlit re-runs your entire script from line 1 every time the user touches anything.** That single fact explains `st.session_state` (values that must survive a re-run) and `@st.cache_resource` (things too expensive to redo every time).

---

# Section A — Machine Learning Apps

These apps serve a **trained model**. No API key needed, no internet needed after the first run.

---

## A1 — Salary Predictor (one input)

⭐ **Simple** · one number in, one number out. The smallest useful ML app that exists.

**What it does:** you type years of experience, it predicts a salary.

### Step 1: train the model

**`a1_train.py`**

```python
"""Trains a one-feature linear regression and saves it."""

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/regression/salary_data.csv")

df = pd.read_csv(URL).dropna()

X = df[["Experience"]]     # 2-D: a table with one column
y = df["Salary"]           # 1-D: a single column

model = LinearRegression()
model.fit(X, y)

print(f"Learned: Salary = {model.coef_[0]:.0f} * Experience + {model.intercept_:.0f}")
print(f"R2 on training data: {model.score(X, y):.4f}")

joblib.dump(model, "a1_model.joblib")
print("Saved a1_model.joblib")
```

```bash
python a1_train.py
```

### Step 2: the app

**`a1_app.py`**

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Salary Predictor", page_icon="💰")

# @st.cache_resource loads the model ONCE instead of on every interaction.
@st.cache_resource
def load_model():
    return joblib.load("a1_model.joblib")

model = load_model()

st.title("💰 Salary Predictor")
st.write("Enter years of experience to estimate a salary.")

# --- The single input ---
experience = st.number_input(
    "Years of experience",
    min_value=0.0,
    max_value=40.0,
    value=5.0,
    step=0.5,
)

if st.button("Predict salary", type="primary"):
    # The model expects a TABLE, not a single number.
    input_df = pd.DataFrame({"Experience": [experience]})
    prediction = model.predict(input_df)[0]

    st.metric("Estimated salary", f"{prediction:,.0f}")

    st.caption(
        f"The model learned: salary = {model.coef_[0]:,.0f} x experience "
        f"+ {model.intercept_:,.0f}"
    )

with st.expander("How does this work?"):
    st.write(
        """
        The model is a straight line fitted through 375 real data points.
        Predicting just means reading a value off that line.

        It knows **nothing** except years of experience - not your skills,
        your city, or your industry. That is why it is an estimate, not an answer.
        """
    )
```

```bash
streamlit run a1_app.py
```

### ✏️ Exercises

1. **Easy.** Change the currency label from `{prediction:,.0f}` to `₹{prediction:,.0f}`.
2. **Easy.** Set the slider maximum to 50 years. What does the model predict at 50? Is that believable?
3. **Medium.** Add `st.slider` as an alternative input and let the user pick which to use with `st.radio`.
4. **Medium.** Show a warning when experience is above 25: the training data barely covers that range, so the model is **extrapolating**.
5. **Hard.** Plot the fitted line with `st.line_chart` and mark the user's prediction on it.

> ⚠️ **The lesson of exercise 4.** A model is only trustworthy inside the range it was trained on. Ask this one for a 200-year career and it will confidently return a number. Confidence is not correctness.

---

## A2 — Sales Predictor (two inputs)

⭐ **Simple** · two numbers in, one number out. Introduces `st.columns` for side-by-side layout.

**What it does:** you enter TV and radio advertising spend, it predicts sales.

### Step 1: train

**`a2_train.py`**

```python
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/regression/advertising.csv")

df = pd.read_csv(URL).dropna()
print(df.head())

X = df[["TV", "Radio"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("R2 on unseen test data:", round(r2_score(y_test, model.predict(X_test)), 4))
print("TV coefficient   :", round(model.coef_[0], 4))
print("Radio coefficient:", round(model.coef_[1], 4))

joblib.dump(model, "a2_model.joblib")
print("Saved a2_model.joblib")
```

```bash
python a2_train.py
```

**Read those two coefficients.** They tell you how much one extra unit of spend on each channel is worth. That is a business insight, not just a number.

### Step 2: the app

**`a2_app.py`**

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sales Predictor", page_icon="📈")

@st.cache_resource
def load_model():
    return joblib.load("a2_model.joblib")

model = load_model()

st.title("📈 Advertising Sales Predictor")
st.write("Set your advertising budget and see the predicted sales.")

# --- Two inputs, side by side ---
left, right = st.columns(2)

with left:
    tv = st.slider("TV budget (thousands)", 0.0, 300.0, 150.0, step=5.0)

with right:
    radio = st.slider("Radio budget (thousands)", 0.0, 50.0, 25.0, step=1.0)

input_df = pd.DataFrame({"TV": [tv], "Radio": [radio]})
prediction = model.predict(input_df)[0]

# No button needed - Streamlit re-runs when a slider moves, so this
# updates live as the user drags.
st.divider()

a, b, c = st.columns(3)
a.metric("Total budget", f"{tv + radio:,.0f}k")
b.metric("Predicted sales", f"{prediction:,.2f}")
c.metric("Sales per unit spent", f"{prediction / max(tv + radio, 1):.3f}")

st.divider()

st.subheader("Which channel is working harder?")
st.write(
    f"- Each extra 1k on **TV** adds about **{model.coef_[0]:.3f}** to sales\n"
    f"- Each extra 1k on **Radio** adds about **{model.coef_[1]:.3f}** to sales"
)

if model.coef_[1] > model.coef_[0]:
    st.success("Radio gives more sales per unit spent in this model.")
else:
    st.info("TV gives more sales per unit spent in this model.")
```

```bash
streamlit run a2_app.py
```

### ✏️ Exercises

1. **Easy.** Add a third slider for `Newspaper` and retrain with all three columns.
2. **Easy.** Replace the sliders with `st.number_input` so users can type exact figures.
3. **Medium.** Add a total-budget cap: warn if TV + Radio exceeds 250.
4. **Medium.** Add `st.session_state` to keep a history of every combination tried, shown in a table below.
5. **Hard.** Given a fixed total budget, find the TV/radio split that maximises predicted sales. Show it as a recommendation.

> **Notice there is no button in this app.** Streamlit re-runs on every slider move, so the result updates live. Buttons matter when the work is slow or has a side effect — not for instant calculations.

---

## A3 — Iris Flower Classifier (four inputs)

⭐ **Simple** · your first **classification** app. Introduces `predict_proba` and `st.bar_chart`.

**What it does:** you enter four flower measurements, it predicts the species **and how confident it is**.

### Step 1: train

**`a3_train.py`**

```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/classification/iris.csv")

df = pd.read_csv(URL).dropna()
print(df["species"].value_counts())

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X = df[FEATURES]
y = df["species"]        # already text - RandomForest handles string labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
print("Classes in order:", list(model.classes_))

joblib.dump(model, "a3_model.joblib")
print("Saved a3_model.joblib")
```

```bash
python a3_train.py
```

### Step 2: the app

**`a3_app.py`**

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

@st.cache_resource
def load_model():
    return joblib.load("a3_model.joblib")

model = load_model()

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

st.title("🌸 Iris Flower Classifier")
st.write("Enter four measurements in centimetres.")

# --- Four inputs in a 2x2 grid ---
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 3.7, 0.1)

with col2:
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=FEATURES,
)

prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
confidence = probabilities.max()

st.divider()
st.subheader(f"Prediction: **{prediction}**")

# Confidence is often more useful than the label on its own.
if confidence > 0.90:
    st.success(f"Confident: {confidence:.1%}")
elif confidence > 0.60:
    st.warning(f"Fairly confident: {confidence:.1%}")
else:
    st.error(f"Not confident: {confidence:.1%} - this flower sits near a boundary")

# Show ALL the probabilities, not just the winner.
prob_df = pd.DataFrame(
    {"probability": probabilities},
    index=model.classes_,
)
st.bar_chart(prob_df)

with st.expander("Your input"):
    st.dataframe(input_df, hide_index=True)
```

```bash
streamlit run a3_app.py
```

### ✏️ Exercises

1. **Easy.** Change the title emoji and the page icon.
2. **Easy.** Add `st.image` showing a photo of the predicted species (find one, save it locally).
3. **Medium.** Add a **"Try a random flower"** button that fills the sliders with a real row from the dataset.
4. **Medium.** Set petal length to 1.5 and petal width to 0.3. Which species? Now move petal length slowly to 5.0. **At what value does the prediction flip?** That is the decision boundary.
5. **Hard.** Add a scatter plot of the training data with the user's flower marked on it.
6. **Hard.** Show feature importances and explain which measurement matters most.

> **Why show all three probabilities?** A prediction of "versicolor at 51%" and one at "versicolor at 99%" are completely different claims. An interface that hides that difference is lying by omission.

---
## A4 — Diabetes Risk Dashboard

⭐⭐⭐ **Advanced** · a full pipeline: mixed input types, a `Pipeline` object, sidebar layout, risk gauge, feature importance.

**What it does:** collects eight patient details, predicts diabetes risk, and shows which factors drove the result.

> ⚠️ **This is a teaching exercise, not a medical tool.** Say that in the app. We do, below.

### Step 1: train

**`a4_train.py`**

```python
"""
Trains a diabetes risk model using a scikit-learn Pipeline.

A Pipeline bundles preprocessing AND the model into one object. This
matters enormously for deployment: the app loads ONE file and does not
have to remember how to encode anything. It also makes data leakage
much harder, because the scaler is fitted inside cross-validation.
"""

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/classification/diabetes_prediction_dataset.csv")

df = pd.read_csv(URL).dropna()

# 100,000 rows trains slowly in a classroom. Sample it, keeping the
# class balance with stratify.
df = df.sample(n=20000, random_state=42).reset_index(drop=True)

print("Shape:", df.shape)
print(df["diabetes"].value_counts(normalize=True).round(3))

NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL = ["gender", "smoking_history"]
BINARY = ["hypertension", "heart_disease"]

FEATURES = NUMERIC + CATEGORICAL + BINARY

X = df[FEATURES]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# The preprocessor: scale numbers, one-hot the categories, leave
# binary 0/1 columns alone.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ],
    remainder="passthrough",
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",   # this dataset is imbalanced
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))

# Save the pipeline AND the column order the app must send.
joblib.dump({"pipeline": pipeline, "features": FEATURES}, "a4_model.joblib")
print("Saved a4_model.joblib")
```

```bash
python a4_train.py
```

> **Why `class_weight="balanced"`?** Only about 8% of these patients have diabetes. Without it, the model learns that guessing "no" almost always is a good strategy — high accuracy, terrible recall. This is exactly the failure you built deliberately in Activity 2.6.

### Step 2: the app

**`a4_app.py`**

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk", page_icon="🩺", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("a4_model.joblib")

bundle = load_bundle()
pipeline = bundle["pipeline"]
FEATURES = bundle["features"]

st.title("🩺 Diabetes Risk Screening")

st.warning(
    "**Educational demonstration only.** This is a student project trained on "
    "a public dataset. It is not a medical device and must never be used to "
    "make a health decision. Consult a qualified doctor."
)

# --- Inputs live in the sidebar, results in the main area ---
with st.sidebar:
    st.header("Patient details")

    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    bmi = st.slider("BMI", 10.0, 60.0, 27.0, 0.1)

    st.subheader("Medical history")
    hypertension = st.checkbox("Has hypertension")
    heart_disease = st.checkbox("Has heart disease")
    smoking_history = st.selectbox(
        "Smoking history",
        ["never", "former", "current", "not current", "ever", "No Info"],
    )

    st.subheader("Lab results")
    hba1c = st.slider("HbA1c level", 3.0, 9.0, 5.5, 0.1)
    glucose = st.slider("Blood glucose level", 80, 300, 120)

    submitted = st.button("Assess risk", type="primary", use_container_width=True)

if not submitted:
    st.info("← Fill in the patient details in the sidebar, then press **Assess risk**.")
    st.stop()      # stops the script here; nothing below runs

# --- Build the input row. Column names must match training exactly. ---
input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose,
    "gender": gender,
    "smoking_history": smoking_history,
    "hypertension": int(hypertension),
    "heart_disease": int(heart_disease),
}])[FEATURES]

risk = pipeline.predict_proba(input_df)[0][1]

# --- Results ---
left, right = st.columns([1, 1])

with left:
    st.subheader("Estimated risk")
    st.progress(float(risk))
    st.metric("Probability of diabetes", f"{risk:.1%}")

    if risk >= 0.60:
        st.error("**High** estimated risk. Clinical review recommended.")
    elif risk >= 0.30:
        st.warning("**Moderate** estimated risk. Monitor and re-test.")
    else:
        st.success("**Low** estimated risk on these inputs.")

with right:
    st.subheader("What drives this model overall")
    # Feature names come out of the fitted ColumnTransformer.
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(8)
        .set_index("feature")
    )
    st.bar_chart(importance_df)

st.divider()

with st.expander("Values sent to the model"):
    st.dataframe(input_df, hide_index=True, use_container_width=True)

with st.expander("How to read this, and what it cannot tell you"):
    st.markdown(
        """
        **The number is a probability, not a diagnosis.** A 40% risk does not
        mean you have 40% of a disease. It means that among people in the
        training data with similar values, roughly 40% had diabetes.

        **Feature importance is global, not personal.** The chart shows what
        matters across all 20,000 patients, not what drove *this* result.

        **The model has never seen you.** It knows eight numbers. It knows
        nothing about family history, diet, medication, or ethnicity - all of
        which genuinely matter.
        """
    )
```

```bash
streamlit run a4_app.py
```

### ✏️ Exercises

1. **Easy.** Change the risk thresholds from 30%/60% to 20%/50%. How many more patients get flagged?
2. **Medium.** Add a "Load example patient" button with three presets: low, moderate and high risk.
3. **Medium.** Store every assessment in `st.session_state` and show a history table.
4. **Medium.** Add a "Download this assessment" button using `st.download_button` and a CSV string.
5. **Hard.** Replace global feature importance with a **per-patient** explanation: re-run the prediction with one feature set to its dataset average, and report how much the risk moved. That is a simple form of what SHAP does properly.
6. **Hard, and the important one.** Compute the model's recall separately for each `gender` value. Are they equal? If not, write two sentences on what that would mean if this were deployed.

---

## A5 — Batch CSV Predictor

⭐⭐⭐ **Advanced** · file upload, column validation, bulk prediction, download. This is the pattern most real business ML apps actually use.

**What it does:** you upload a CSV of many rows, it predicts every row and gives you a file back.

Uses the model from A3. Run `python a3_train.py` first.

**`a5_app.py`**

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Batch Predictor", page_icon="📦", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("a3_model.joblib")

model = load_model()
REQUIRED = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

st.title("📦 Batch Iris Predictor")
st.write("Upload a CSV and get a prediction for every row.")

# --- Give the user a template. Never make them guess your format. ---
template = pd.DataFrame({
    "sepal_length": [5.1, 6.7],
    "sepal_width": [3.5, 3.0],
    "petal_length": [1.4, 5.2],
    "petal_width": [0.2, 2.3],
})

st.download_button(
    "⬇️ Download a template CSV",
    data=template.to_csv(index=False),
    file_name="template.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin. Use the template above if you need the format.")
    st.stop()

# --- Read it, and fail helpfully if it is malformed ---
try:
    df = pd.read_csv(uploaded)
except Exception as error:
    st.error(f"Could not read that file as CSV: {error}")
    st.stop()

st.subheader("What you uploaded")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"{len(df)} rows, {len(df.columns)} columns")

# --- Validate BEFORE predicting. This is what separates a demo from a tool. ---
missing = [column for column in REQUIRED if column not in df.columns]
if missing:
    st.error(f"Your file is missing these required columns: {', '.join(missing)}")
    st.write("Required columns:", REQUIRED)
    st.stop()

work = df.copy()
blank_rows = work[REQUIRED].isnull().any(axis=1).sum()
if blank_rows:
    st.warning(f"{blank_rows} row(s) have missing values and will be skipped.")
    work = work.dropna(subset=REQUIRED)

if work.empty:
    st.error("No complete rows left to predict.")
    st.stop()

# --- Predict every row at once. Never loop row by row - it is far slower. ---
predictions = model.predict(work[REQUIRED])
probabilities = model.predict_proba(work[REQUIRED])

work["prediction"] = predictions
work["confidence"] = probabilities.max(axis=1).round(4)

st.divider()
st.subheader("Results")

a, b, c = st.columns(3)
a.metric("Rows predicted", len(work))
b.metric("Mean confidence", f"{work['confidence'].mean():.1%}")
c.metric("Low confidence (<70%)", int((work["confidence"] < 0.70).sum()))

st.dataframe(work, use_container_width=True)

st.subheader("How many of each class?")
st.bar_chart(work["prediction"].value_counts())

# --- Flag the rows a human should check ---
uncertain = work[work["confidence"] < 0.70]
if not uncertain.empty:
    st.warning(
        f"{len(uncertain)} row(s) came out below 70% confidence. "
        "In a real deployment these would be routed to a person."
    )
    st.dataframe(uncertain, use_container_width=True)

st.download_button(
    "⬇️ Download results",
    data=work.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv",
    type="primary",
)
```

```bash
streamlit run a5_app.py
```

Test it by downloading the template, then re-uploading it.

### ✏️ Exercises

1. **Easy.** Change the low-confidence threshold to 0.85. How many more rows get flagged?
2. **Medium.** Make the threshold a `st.slider` the user controls.
3. **Medium.** Add a column showing the runner-up class and its probability.
4. **Medium.** Show a friendly error if the user uploads a CSV with the right columns but text where numbers should be.
5. **Hard.** Add a progress bar for files over 1,000 rows using `st.progress` and chunked prediction.
6. **Hard, scenario.** *A colleague uploads a file where `petal_length` is in millimetres, not centimetres. Your app happily predicts nonsense.* Add a sanity check that warns when values fall far outside the training range.

> **Exercise 6 is the real lesson.** Your model cannot tell you the units are wrong; it will produce confident predictions from meaningless input. **Validating input is your job, not the model's.**

---

## A6 — Model Comparison Lab

⭐⭐⭐ **Advanced** · trains several models live, compares metrics, and draws a confusion matrix. Turns Day 2 and Day 3 into something you can *see*.

**`a6_app.py`**

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

st.set_page_config(page_title="Model Comparison Lab", page_icon="🔬", layout="wide")

DATASETS = {
    "Loan approval (10k rows)": (
        "https://raw.githubusercontent.com/tech4alltraining/aiml/"
        "refs/heads/main/datasets/loan_data_10k.csv", "loan_status"),
    "Iris (150 rows)": (
        "https://raw.githubusercontent.com/tech4alltraining/aiml/"
        "refs/heads/main/datasets/classification/iris.csv", "species"),
    "Titanic (891 rows)": (
        "https://raw.githubusercontent.com/tech4alltraining/aiml/"
        "refs/heads/main/datasets/classification/archive/titanic.csv", "survived"),
}

MODELS = {
    "Baseline (always majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(),
}

# @st.cache_data caches the RESULT of a function that returns data.
# (@st.cache_resource is for objects like models and connections.)
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

st.title("🔬 Model Comparison Lab")
st.write("Train several models on the same data and compare them honestly.")

with st.sidebar:
    st.header("Setup")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    chosen = st.multiselect(
        "Models to compare",
        list(MODELS.keys()),
        default=["Baseline (always majority)", "Logistic Regression", "Random Forest"],
    )
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    use_cv = st.checkbox("Also run 5-fold cross-validation", value=True)
    run = st.button("Train and compare", type="primary", use_container_width=True)

url, target = DATASETS[dataset_name]
df = load_data(url).dropna().reset_index(drop=True)

st.subheader(f"Dataset: {dataset_name}")
a, b, c = st.columns(3)
a.metric("Rows", len(df))
b.metric("Columns", len(df.columns))
c.metric("Target", target)

with st.expander("Preview the data"):
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Target balance:**")
    st.write(df[target].value_counts())

if not run:
    st.info("← Choose your setup in the sidebar, then press **Train and compare**.")
    st.stop()

if not chosen:
    st.error("Select at least one model.")
    st.stop()

# --- Prepare the data ---
work = df.copy()
for column in work.select_dtypes(include="object").columns:
    work[column] = LabelEncoder().fit_transform(work[column].astype(str))

X = work.drop(columns=[target])
y = work[target]
is_binary = y.nunique() == 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# Scale for the distance- and gradient-based models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on TRAIN only
X_test_scaled = scaler.transform(X_test)         # transform test

results = []
confusions = {}
average = "binary" if is_binary else "macro"

progress = st.progress(0.0, text="Training...")

for index, name in enumerate(chosen):
    model = MODELS[name]
    needs_scaling = name in {"Logistic Regression", "K-Nearest Neighbours"}

    fit_X, eval_X = (X_train_scaled, X_test_scaled) if needs_scaling else (X_train, X_test)

    model.fit(fit_X, y_train)
    y_pred = model.predict(eval_X)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    if use_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            MODELS[name],
            scaler.fit_transform(X) if needs_scaling else X,
            y, cv=cv, scoring="accuracy",
        )
        row["CV mean"] = scores.mean()
        row["CV std"] = scores.std()

    results.append(row)
    confusions[name] = confusion_matrix(y_test, y_pred)
    progress.progress((index + 1) / len(chosen), text=f"Trained {name}")

progress.empty()

results_df = pd.DataFrame(results).set_index("Model")

st.divider()
st.subheader("Results")
st.dataframe(
    results_df.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d32"),
    use_container_width=True,
)

st.bar_chart(results_df[["Accuracy", "Precision", "Recall", "F1"]])

# --- Did anything actually beat the baseline? ---
if "Baseline (always majority)" in results_df.index:
    baseline = results_df.loc["Baseline (always majority)", "Accuracy"]
    best_name = results_df["Accuracy"].idxmax()
    best = results_df["Accuracy"].max()
    if best_name == "Baseline (always majority)":
        st.error(
            f"**No model beat the baseline ({baseline:.1%}).** "
            "On this data, with these features, none of them learned anything useful."
        )
    else:
        st.success(
            f"**{best_name}** leads at {best:.1%}, against a baseline of {baseline:.1%} "
            f"— an improvement of {best - baseline:+.1%}."
        )

if use_cv and "CV std" in results_df.columns:
    shaky = results_df[results_df["CV std"] > 0.05]
    if not shaky.empty:
        st.warning(
            "These models scored very differently across folds, so their single "
            f"test score is not reliable: {', '.join(shaky.index)}"
        )

st.divider()
st.subheader("Confusion matrices")

columns = st.columns(min(len(chosen), 3))
for index, name in enumerate(chosen):
    with columns[index % len(columns)]:
        st.caption(name)
        matrix = confusions[name]
        figure, axis = plt.subplots(figsize=(3, 2.6))
        axis.imshow(matrix, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, matrix[i, j], ha="center", va="center", fontsize=9)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks(range(matrix.shape[0]))
        axis.set_yticks(range(matrix.shape[0]))
        st.pyplot(figure)
        plt.close(figure)
```

```bash
streamlit run a6_app.py
```

### ✏️ Exercises

1. **Easy.** Add `SVC` from `sklearn.svm` to the `MODELS` dictionary.
2. **Easy.** Run the loan dataset with only the Baseline and Decision Tree. Does the tree beat the baseline?
3. **Medium.** Add a slider for `max_depth` on the Decision Tree and watch overfitting appear in the CV std.
4. **Medium.** Add a timing column: how long did each model take to fit? Is the slowest also the best?
5. **Hard.** Add your own dataset to the `DATASETS` dictionary — a CSV from the `datasets/` folder that is not already listed.
6. **Hard, scenario.** *On the Titanic dataset, Random Forest gets 82% and Logistic Regression gets 79%. Your project partner wants to ship Random Forest.* Use the CV std column to argue whether that 3-point difference is real or noise.

---

# Section B — Generative AI Apps

These call the Gemini API. **You need a key in `.streamlit/secrets.toml` before any of them will run.**

Common pattern for all of Section B:

```python
from google import genai

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3.5-flash"

response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
st.write(response.text)
```

---
## B1 — Topic Explainer (one input)

⭐ **Simple** · one text box, one API call. The smallest possible GenAI app.

**`b1_app.py`**

```python
import streamlit as st
from google import genai

st.set_page_config(page_title="Topic Explainer", page_icon="📚")

MODEL_NAME = "gemini-3.5-flash"

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

st.title("📚 Topic Explainer")
st.write("Type any topic and get a simple explanation.")

topic = st.text_input("Topic", placeholder="Example: overfitting")

if st.button("Explain", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    # A five-part prompt: role, task, context, constraints, format.
    prompt = f"""
    You are a patient teacher explaining to a first-year engineering student.

    Explain this topic: {topic}

    The student knows basic Python but no advanced maths.

    Use one everyday analogy. Keep it under 150 words. Avoid jargon
    unless you define it in the same sentence.

    Return two short paragraphs. No headings, no bullet points.
    """

    with st.spinner("Thinking..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

    st.divider()
    st.write(response.text)

    st.caption("⚠️ AI-generated. Verify anything you plan to rely on.")
```

```bash
streamlit run b1_app.py
```

### ✏️ Exercises

1. **Easy.** Change "first-year engineering student" to "a 10-year-old". Compare outputs for the same topic.
2. **Easy.** Change the word limit to 50, then 400. Does it obey?
3. **Medium.** Add a `st.selectbox` for language (English, Hindi, Malayalam, Tamil) and put it in the prompt.
4. **Medium.** Add a "Simpler" button that re-explains the last answer at a lower level, using `st.session_state`.
5. **Hard.** Add `st.write_stream` so the answer appears word by word instead of all at once. (Hint: `client.models.generate_content_stream`.)

> **The whole app is the prompt.** Everything else is a text box and a button. When students complain that "the AI gives bad answers", 90% of the time the fix is in those twelve lines of prompt.

---

## B2 — Text Summariser (two inputs)

⭐ **Simple** · text area plus a control. Introduces `st.tabs` and `st.download_button`.

**`b2_app.py`**

```python
import streamlit as st
from google import genai

st.set_page_config(page_title="Text Summariser", page_icon="✂️", layout="wide")

MODEL_NAME = "gemini-3.5-flash"

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

st.title("✂️ Text Summariser")

left, right = st.columns([2, 1])

with left:
    text = st.text_area("Paste your text here", height=280,
                        placeholder="Paste an article, lecture notes, or a report...")

with right:
    style = st.radio("Summary style",
                     ["Short paragraph", "Bullet points", "One sentence", "Study notes"])
    length = st.slider("Approximate words", 30, 400, 120, 10)
    audience = st.selectbox("Written for",
                            ["A student", "A busy manager", "A subject expert"])

# Live word count - useful feedback while the user is pasting.
word_count = len(text.split())
st.caption(f"Input: {word_count} words")

if st.button("Summarise", type="primary", disabled=word_count < 20):
    prompt = f"""
    Summarise the text below for this reader: {audience}.

    Format: {style}
    Target length: about {length} words.

    Use ONLY information present in the text. If something is not stated,
    do not add it. Do not add your own opinions or outside facts.

    TEXT:
    \"\"\"
    {text}
    \"\"\"
    """

    with st.spinner("Summarising..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

    summary = response.text

    st.divider()
    tab1, tab2 = st.tabs(["Summary", "Compression"])

    with tab1:
        st.write(summary)
        st.download_button("⬇️ Download summary", summary, "summary.txt")

    with tab2:
        summary_words = len(summary.split())
        a, b, c = st.columns(3)
        a.metric("Original", f"{word_count} words")
        b.metric("Summary", f"{summary_words} words")
        c.metric("Reduced by", f"{100 * (1 - summary_words / max(word_count, 1)):.0f}%")

if word_count < 20:
    st.info("Paste at least 20 words to enable the button.")
```

```bash
streamlit run b2_app.py
```

### ✏️ Exercises

1. **Easy.** Add "Explain Like I'm Five" to the style options.
2. **Medium.** Add a file uploader accepting `.txt` files, so users can upload instead of paste.
3. **Medium.** Add a "key terms" tab that asks the model to extract the five most important terms with definitions.
4. **Hard.** Add a fact-check tab: for each sentence in the summary, ask the model whether the original text supports it. This is a simple **grounding check**.
5. **Hard, scenario.** *Someone pastes 50,000 words.* The request will exceed the context window or cost a fortune. Add a word-count guard and a message explaining the limit.

> **Look at the prompt's third paragraph.** "Use ONLY information present in the text" is what makes this a summariser rather than an essay generator. Remove that line and run it on a short paragraph — the model will happily invent supporting detail.

---

## B3 — Persona Chatbot

⭐⭐ **Intermediate** · multi-turn conversation. The app that teaches `st.session_state` properly.

**`b3_app.py`**

```python
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Persona Chatbot", page_icon="💬")

MODEL_NAME = "gemini-3.5-flash"

PERSONAS = {
    "Python Tutor": "You are a patient Python tutor. Always show a short code "
                    "example. Never give the full solution to a homework problem - "
                    "give a hint and ask a guiding question.",
    "Interview Coach": "You are a technical interview coach for data roles. Ask one "
                       "question at a time, wait for the answer, then give specific "
                       "feedback before moving on.",
    "Career Guide": "You are a career counsellor for engineering students in India. "
                    "Be practical and concrete. Never promise job outcomes.",
    "Study Buddy": "You are a study partner. Explain simply, then quiz the student "
                   "with one question to check they understood.",
}

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

st.title("💬 Persona Chatbot")

with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Persona", list(PERSONAS.keys()))
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1,
                            help="0 = predictable, 1.5 = creative")
    st.caption(PERSONAS[persona])

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_persona = persona
        st.rerun()

# --- st.session_state survives the re-run that happens on every interaction ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_persona" not in st.session_state:
    st.session_state.active_persona = persona

# Switching persona mid-conversation is confusing. Start fresh.
if persona != st.session_state.active_persona:
    st.session_state.messages = []
    st.session_state.active_persona = persona
    st.info(f"Switched to **{persona}**. Conversation cleared.")

# --- Replay the whole history. Streamlit redraws everything each run. ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # The API is STATELESS. To have a conversation we must resend the history.
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=PERSONAS[persona],
                    temperature=temperature,
                ),
            )
        reply = response.text
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.sidebar.caption(f"Messages in history: {len(st.session_state.messages)}")
```

```bash
streamlit run b3_app.py
```

### ✏️ Exercises

1. **Easy.** Add a "Maths Tutor" persona.
2. **Easy.** Ask "What is Python?" then "Who created it?" — it answers correctly because history is resent. Now comment out the history loop and send only `user_input`. Watch it break.
3. **Medium.** Show the message count as a token estimate (roughly `words × 1.3`).
4. **Medium.** Add a download button that exports the conversation as a Markdown file.
5. **Hard.** Cap the history at the last 10 messages. Longer conversations otherwise grow the prompt — and the cost — on every single turn.
6. **Hard.** Summarise older messages instead of dropping them, so the bot keeps long-term context cheaply. This is what production chatbots do.

> **Exercise 2 is the point of the whole app.** The model has no memory. `st.session_state` plus resending the history *is* the memory. Every chatbot you have used works exactly this way.

---

## B4 — Document Q&A (grounded)

⭐⭐⭐ **Advanced** · file upload plus **grounding**. The core idea behind RAG, without the vector database.

**`b4_app.py`**

```python
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")

MODEL_NAME = "gemini-3.5-flash"
REFUSAL = "Not stated in the provided document."

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

st.title("📄 Document Q&A")
st.write(
    "Upload a text file and ask questions about it. The app answers **only** "
    "from the document - it will refuse rather than guess."
)

uploaded = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])

sample = """
The internship programme runs for four weeks. Week 1 is five days of
classroom training covering Python, machine learning, and generative AI.
Weeks 2 to 4 are the capstone project phase, with online reviews at the
end of each week. Students must submit a repository and a report by the
end of week 4, and give an 8 to 10 minute final presentation.
Attendance in week 1 is compulsory. The pass mark is 50 percent.
"""

if uploaded is not None:
    document = uploaded.read().decode("utf-8", errors="ignore")
    st.success(f"Loaded **{uploaded.name}** — {len(document.split())} words")
else:
    document = sample
    st.info("No file uploaded — using a short built-in sample document.")

with st.expander("View the document"):
    st.text(document[:3000] + ("..." if len(document) > 3000 else ""))

# A rough guard. Long documents cost more and may exceed the context window.
word_count = len(document.split())
if word_count > 8000:
    st.warning(
        f"This document is {word_count:,} words. Every question resends all of "
        "it, so each answer will be slow and relatively expensive."
    )

question = st.text_input("Ask a question about the document")

if st.button("Answer", type="primary") and question.strip():
    # THE GROUNDING PROMPT. Every line here is doing work.
    prompt = f"""
    Answer the question using ONLY the document below.

    If the answer is not in the document, reply with exactly this sentence
    and nothing else: "{REFUSAL}"

    Do not use any outside knowledge. Do not guess. Do not infer beyond
    what is written.

    After your answer, on a new line, quote the exact sentence from the
    document that supports it, prefixed with "SOURCE: ".

    DOCUMENT:
    \"\"\"
    {document}
    \"\"\"

    QUESTION: {question}
    """

    with st.spinner("Reading the document..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )

    answer = response.text.strip()

    st.divider()
    if REFUSAL.lower() in answer.lower():
        st.warning(f"❓ {answer}")
        st.caption(
            "**This is the app working correctly.** A refusal is far more "
            "useful than a confident invention."
        )
    else:
        st.success(answer)

    with st.expander("Token usage"):
        usage = response.usage_metadata
        a, b, c = st.columns(3)
        a.metric("Prompt tokens", f"{usage.prompt_token_count:,}")
        a.caption("The whole document, every time")
        b.metric("Output tokens", f"{usage.candidates_token_count:,}")
        c.metric("Total", f"{usage.total_token_count:,}")

st.divider()
st.subheader("Try these to test the grounding")
st.markdown(
    """
    - **In the document:** "How long is the programme?" → should answer
    - **In the document:** "What is the pass mark?" → should answer
    - **NOT in the document:** "Who is the instructor?" → should **refuse**
    - **NOT in the document:** "What is the fee?" → should **refuse**

    A model that answers the last two has hallucinated. Grounding is what stops it.
    """
)
```

```bash
streamlit run b4_app.py
```

### ✏️ Exercises

1. **Easy.** Ask a question you know is not in the document. Does it refuse?
2. **Easy.** Delete the "Do not use any outside knowledge" line and ask again. What changes?
3. **Medium.** Support PDF upload with `pypdf` (`pip install pypdf`).
4. **Medium.** Add a chat history so users can ask follow-up questions.
5. **Hard.** Split long documents into 500-word chunks, send only the chunks most relevant to the question (keyword overlap is fine), and report how many tokens you saved. **You have now built RAG.**
6. **Hard, scenario.** *A user uploads a document containing "The refund policy is 30 days" and asks "Can I get a refund after 45 days?"* The answer requires reasoning, not lookup. Does your app handle it well? Should it?

> **Exercise 2 is the whole lesson.** Grounding is not a setting you enable — it is a sentence you write in the prompt, and it works because you told the model what it is not allowed to do.

---

## B5 — Quiz Generator

⭐⭐⭐ **Advanced** · **structured JSON output**, interactive scoring, session state. The app that turns an LLM into a component you can build on.

**`b5_app.py`**

```python
import json
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Quiz Generator", page_icon="🎯")

MODEL_NAME = "gemini-3.5-flash"

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

st.title("🎯 Quiz Generator")

for key, default in [("quiz", None), ("answers", {}), ("submitted", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Create a quiz")
    topic = st.text_input("Topic", "Overfitting in machine learning")
    count = st.slider("Number of questions", 3, 10, 5)
    difficulty = st.select_slider("Difficulty", ["Easy", "Medium", "Hard"], "Medium")
    generate = st.button("Generate quiz", type="primary", use_container_width=True)

if generate:
    prompt = f"""
    Create a {difficulty.lower()} multiple-choice quiz on: {topic}

    Produce exactly {count} questions. Each question must have exactly 4
    options, exactly one of which is correct.

    Return ONLY valid JSON in this exact shape:
    {{
      "questions": [
        {{
          "question": "the question text",
          "options": ["option A", "option B", "option C", "option D"],
          "correct_index": 0,
          "explanation": "why that answer is correct, in one sentence"
        }}
      ]
    }}

    correct_index is a 0-based integer into the options array.
    Vary which position the correct answer sits in.
    """

    with st.spinner("Writing questions..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",   # forces valid JSON
                temperature=0.7,
            ),
        )

    try:
        st.session_state.quiz = json.loads(response.text)["questions"]
        st.session_state.answers = {}
        st.session_state.submitted = False
    except (json.JSONDecodeError, KeyError) as error:
        st.error(f"The model did not return usable JSON: {error}")
        st.code(response.text[:600])
        st.stop()

quiz = st.session_state.quiz

if not quiz:
    st.info("← Choose a topic in the sidebar and press **Generate quiz**.")
    st.stop()

st.subheader(f"{len(quiz)} questions on: {topic}")

for index, item in enumerate(quiz):
    st.markdown(f"**Q{index + 1}. {item['question']}**")
    st.session_state.answers[index] = st.radio(
        "Choose one",
        options=range(len(item["options"])),
        format_func=lambda i, opts=item["options"]: opts[i],
        key=f"q{index}",
        index=None,
        label_visibility="collapsed",
    )
    st.write("")

if st.button("Submit answers", type="primary"):
    st.session_state.submitted = True

if st.session_state.submitted:
    st.divider()
    score = 0

    for index, item in enumerate(quiz):
        chosen = st.session_state.answers.get(index)
        correct = item["correct_index"]

        if chosen == correct:
            score += 1
            st.success(f"**Q{index + 1}** ✅ Correct — {item['options'][correct]}")
        elif chosen is None:
            st.warning(f"**Q{index + 1}** ⚠️ Not answered. "
                       f"Correct: {item['options'][correct]}")
        else:
            st.error(f"**Q{index + 1}** ❌ You chose *{item['options'][chosen]}*. "
                     f"Correct: **{item['options'][correct]}**")
        st.caption(item["explanation"])

    percentage = score / len(quiz)
    st.divider()
    st.metric("Your score", f"{score} / {len(quiz)}", f"{percentage:.0%}")
    st.progress(percentage)

    if percentage == 1.0:
        st.balloons()

    st.caption(
        "⚠️ Questions and answers are AI-generated and may contain errors. "
        "Check anything that surprises you against your notes."
    )
```

```bash
streamlit run b5_app.py
```

### ✏️ Exercises

1. **Easy.** Add a "Very Hard" difficulty level.
2. **Easy.** Generate a quiz on a topic you know well. **Find a wrong answer.** There will usually be one in ten.
3. **Medium.** Add true/false as a question type.
4. **Medium.** Add a timer using `st.empty()` and a countdown.
5. **Hard.** Save results to a CSV so a student can track scores over time.
6. **Hard.** Generate the quiz from an uploaded document instead of a topic — combine this app with B4.

> **`response_mime_type="application/json"` is what makes this an application.** Without it you get a paragraph that mostly looks like JSON, and `json.loads` fails perhaps one time in five. With it, you get a Python dictionary you can build a UI on.

---

## B6 — Image Describer

⭐⭐⭐ **Advanced** · **multimodal** input. Sends a picture, not just text.

**`b6_app.py`**

```python
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="Image Describer", page_icon="🖼️", layout="wide")

MODEL_NAME = "gemini-3.5-flash"

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

st.title("🖼️ Image Describer")
st.write("Upload a picture and ask a question about it.")

TASKS = {
    "Describe it": "Describe this image in detail. Mention the main subject, "
                   "the setting, and anything notable.",
    "Extract the text": "Transcribe all text visible in this image, exactly as "
                        "written. If there is no text, say 'No text found.'",
    "Alt text for accessibility": "Write a single-sentence alt text for this image, "
                                  "suitable for a screen reader. Be factual and "
                                  "concise. Do not start with 'An image of'.",
    "Count the objects": "Count the distinct objects in this image and list them "
                         "with their counts. If you are unsure, say so.",
    "Ask my own question": None,
}

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    task = st.selectbox("What should the model do?", list(TASKS.keys()))

    custom = ""
    if TASKS[task] is None:
        custom = st.text_input("Your question about the image")

    go = st.button("Analyse", type="primary", disabled=uploaded is None)

with right:
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        st.caption(f"{image.width} x {image.height} pixels, {image.format}")

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

if go:
    instruction = TASKS[task] or custom
    if not instruction.strip():
        st.warning("Type a question first.")
        st.stop()

    image = Image.open(uploaded)

    # Multimodal input: a list mixing text and an image.
    with st.spinner("Looking at the image..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[instruction, image],
            config=types.GenerateContentConfig(temperature=0.2),
        )

    st.divider()
    st.subheader("Result")
    st.write(response.text)

    st.caption(
        "⚠️ Vision models misread handwriting, small text, and unusual angles. "
        "Never rely on extracted text without checking it."
    )

with st.expander("What is this useful for?"):
    st.markdown(
        """
        | Real use | Why it matters |
        |---|---|
        | Alt text generation | Accessibility, at scale |
        | Receipt and form reading | Data entry without typing |
        | Product cataloguing | Descriptions from photographs |
        | Content moderation | First-pass filtering, human review after |

        **What it is not good for:** anything where being wrong is expensive and
        nobody checks. Medical images, legal documents, safety inspections —
        these all need a qualified human making the decision.
        """
    )
```

```bash
streamlit run b6_app.py
```

### ✏️ Exercises

1. **Easy.** Add a "Write a social media caption" task.
2. **Easy.** Photograph handwritten notes and try "Extract the text". How accurate is it?
3. **Medium.** Support multiple images at once and ask the model to compare them.
4. **Medium.** Add a temperature slider and compare descriptions at 0.0 and 1.2.
5. **Hard.** Combine with B5: generate quiz questions *from a photograph of a textbook page*.
6. **Hard, scenario.** *You build this into an app that reads meter readings from photographs.* It gets 95% right. Design the workflow for the other 5% — how does a human find and fix them?

---

# Section C — ML + GenAI Apps

**The ML model decides. The GenAI model explains.** Never the other way round.

> 🧠 **The doctor and the receptionist.** The doctor reads the results and makes the diagnosis — trained, measurable, accountable. The receptionist explains what happens next in words you understand. A hospital where the receptionist diagnoses patients is a disaster, and so is an app where the LLM decides the loan.

---
## C1 — Loan Advisor

⭐⭐ **Intermediate** · your first app where both halves work together.

Needs `rf_model.joblib` from [`loan-app.md`](loan-app.md). Run that tutorial's `train_model.py` first, or copy the file into this folder.

**`c1_app.py`**

```python
import streamlit as st
import pandas as pd
import joblib
from google import genai
from google.genai import types

st.set_page_config(page_title="Loan Advisor", page_icon="🏦", layout="wide")

MODEL_NAME = "gemini-3.5-flash"

FEATURE_COLUMNS = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
]

MAPPINGS = {
    "person_gender": {"female": 0, "male": 1},
    "person_education": {"Associate": 0, "Bachelor": 1, "Doctorate": 2,
                         "High School": 3, "Master": 4},
    "person_home_ownership": {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3},
    "loan_intent": {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
                    "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5},
    "previous_loan_defaults_on_file": {"No": 0, "Yes": 1},
}

@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

model = load_model()
client = get_client()

st.title("🏦 Loan Advisor")
st.caption("Machine Learning makes the decision. Generative AI explains it.")

with st.sidebar:
    st.header("Applicant")
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", list(MAPPINGS["person_gender"]))
    education = st.selectbox("Education", list(MAPPINGS["person_education"]))
    income = st.number_input("Annual income", 0, 5_000_000, 60_000, step=1_000)
    experience = st.number_input("Years of employment", 0, 60, 5)
    home = st.selectbox("Home ownership", list(MAPPINGS["person_home_ownership"]))

    st.header("Loan")
    amount = st.number_input("Loan amount", 500, 500_000, 12_000, step=500)
    intent = st.selectbox("Purpose", list(MAPPINGS["loan_intent"]))
    rate = st.slider("Interest rate (%)", 5.0, 25.0, 11.0, 0.1)
    credit_score = st.slider("Credit score", 300, 850, 650)
    history = st.number_input("Credit history (years)", 0, 40, 5)
    defaults = st.selectbox("Previous defaults", ["No", "Yes"])

    go = st.button("Assess", type="primary", use_container_width=True)

if not go:
    st.info("← Enter the details in the sidebar and press **Assess**.")
    st.stop()

percent_income = round(amount / max(income, 1), 4)

# ---------- PART 1: the ML model DECIDES ----------
input_df = pd.DataFrame([{
    "person_age": age,
    "person_gender": MAPPINGS["person_gender"][gender],
    "person_education": MAPPINGS["person_education"][education],
    "person_income": float(income),
    "person_emp_exp": experience,
    "person_home_ownership": MAPPINGS["person_home_ownership"][home],
    "loan_amnt": float(amount),
    "loan_intent": MAPPINGS["loan_intent"][intent],
    "loan_int_rate": rate,
    "loan_percent_income": percent_income,
    "cb_person_cred_hist_length": history,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": MAPPINGS["previous_loan_defaults_on_file"][defaults],
}])[FEATURE_COLUMNS]

prediction = int(model.predict(input_df)[0])
probability = model.predict_proba(input_df)[0][1]
decision = "APPROVED" if prediction == 1 else "REJECTED"

left, right = st.columns([1, 2])

with left:
    st.subheader("ML decision")
    if prediction == 1:
        st.success(f"✅ {decision}")
    else:
        st.error(f"❌ {decision}")
    st.metric("Model confidence", f"{probability:.1%}")
    st.progress(float(probability))
    st.caption(f"Loan is {percent_income:.1%} of annual income")

# ---------- PART 2: the LLM EXPLAINS (it does not decide) ----------
prompt = f"""
You are a helpful loan officer assistant.

A machine learning model has already reviewed this application and
decided: {decision}

Applicant details:
- Age: {age}
- Education: {education}
- Annual income: {income:,}
- Years employed: {experience}
- Home ownership: {home}
- Loan amount: {amount:,} ({percent_income:.1%} of annual income)
- Purpose: {intent}
- Interest rate: {rate}%
- Credit score: {credit_score} (range 300-850)
- Credit history: {history} years
- Previous defaults: {defaults}

Write a short, respectful message to the applicant:
1. State the decision in one sentence.
2. Give the two or three factors that most likely influenced it.
3. If rejected, give two specific, actionable steps to improve.

Do not invent any information that is not listed above.
Do not state or imply a specific approval probability.
Do not present this as financial advice.
Keep it under 160 words.
"""

with right:
    st.subheader("Explanation")
    with st.spinner("Writing the explanation..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )
    st.write(response.text)
    st.caption("⚠️ AI-generated explanation of an automated decision. "
               "A human officer must review any real lending decision.")

with st.expander("What each component did"):
    st.markdown(
        """
        | Component | Its job | What it did NOT do |
        |---|---|---|
        | **Random Forest** | Made the approve/reject decision from 13 numbers | Write any words |
        | **Gemini** | Turned that decision into a readable message | Decide anything |

        Swap these around and you have a system that cannot be tested,
        cannot be audited, and cannot explain itself the same way twice.
        """
    )

with st.expander("Values sent to the ML model"):
    st.dataframe(input_df, hide_index=True, use_container_width=True)
```

```bash
streamlit run c1_app.py
```

### ✏️ Exercises

1. **Easy.** Change the explanation to a friendlier tone. Which prompt line did you edit?
2. **Easy.** Remove `"Do not invent any information..."`, run a rejection, and look for invented facts. Put it back.
3. **Medium.** Add a language selector so the explanation can be produced in a regional language.
4. **Medium.** Add "What would change this decision?" — try a few credit scores, find the flip point, and mention it in the prompt.
5. **Hard.** Pass the model's top three feature importances into the prompt so the explanation reflects what the model *actually* weighted, not what the LLM guesses it weighted.
6. **Hard, scenario.** *An applicant asks "why was I rejected when my friend with a lower income was approved?"* Can your app answer that? What would it need in order to?

> **Exercise 5 is the difference between a plausible explanation and a true one.** Right now the LLM is guessing which factors mattered, based on what usually matters in loans. Feed it the real feature importances and it stops guessing.

---

## C2 — Iris Botanist

⭐⭐ **Intermediate** · the smallest possible ML + GenAI app. Good for understanding the pattern before C1's complexity.

Needs `a3_model.joblib` from [A3](#a3--iris-flower-classifier-four-inputs).

**`c2_app.py`**

```python
import streamlit as st
import pandas as pd
import joblib
from google import genai
from google.genai import types

st.set_page_config(page_title="Iris Botanist", page_icon="🌷")

MODEL_NAME = "gemini-3.5-flash"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

@st.cache_resource
def load_model():
    return joblib.load("a3_model.joblib")

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

model = load_model()
client = get_client()

st.title("🌷 Iris Botanist")
st.caption("The model identifies the species. Gemini tells you about it.")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 3.7, 0.1)
with col2:
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

if st.button("Identify", type="primary"):
    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,
    )

    # ML decides
    species = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0].max()

    st.divider()
    st.subheader(f"Identified: *{species}*")
    st.metric("Confidence", f"{confidence:.1%}")

    if confidence < 0.70:
        st.warning(
            "Low confidence. These measurements sit close to the boundary "
            "between two species."
        )

    # GenAI explains
    prompt = f"""
    A classification model identified an iris flower as: {species}
    Model confidence: {confidence:.0%}

    Measurements given:
    - Sepal: {sepal_length} cm long, {sepal_width} cm wide
    - Petal: {petal_length} cm long, {petal_width} cm wide

    Write a short note for a botany student:
    1. Two sentences about this species.
    2. One sentence on which of these four measurements most distinguishes
       it from the other two iris species.
    3. If confidence is below 70%, add a sentence explaining that this
       specimen is unusual for its species.

    Under 120 words. Do not invent measurements that were not given.
    """

    with st.spinner("Consulting the botanist..."):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5),
        )

    st.write(response.text)
    st.caption("⚠️ AI-generated botanical notes. Verify before citing.")
```

```bash
streamlit run c2_app.py
```

### ✏️ Exercises

1. **Easy.** Set petal length to 4.8 and petal width to 1.7 — a genuinely ambiguous flower. Does the low-confidence branch fire?
2. **Medium.** Add a "Compare with the other two species" button.
3. **Medium.** Cache explanations so identifying the same species twice does not cost a second API call.
4. **Hard.** Pass the *probabilities of all three species* into the prompt and ask the model to explain the uncertainty, not just the winner.

---

## C3 — Customer Segment Namer

⭐⭐⭐ **Advanced** · **unsupervised** ML plus GenAI. Clustering finds groups; the LLM gives them names a human can act on.

**`c3_app.py`**

```python
import streamlit as st
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from google import genai
from google.genai import types

st.set_page_config(page_title="Segment Namer", page_icon="🎯", layout="wide")

MODEL_NAME = "gemini-3.5-flash"
URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/clustering/Mall_Customers.csv")

@st.cache_data
def load_data():
    return pd.read_csv(URL)

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

df = load_data()
client = get_client()

st.title("🎯 Customer Segment Namer")
st.caption("K-Means finds the groups. Gemini gives them names a manager can use.")

with st.sidebar:
    st.header("Clustering")
    k = st.slider("Number of segments (k)", 2, 8, 5)
    features = st.multiselect(
        "Features to cluster on",
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        default=["Annual Income (k$)", "Spending Score (1-100)"],
    )
    run = st.button("Find and name segments", type="primary", use_container_width=True)

st.subheader("The data")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"{len(df)} customers")

if not run:
    st.info("← Choose your settings and press **Find and name segments**.")
    st.stop()

if len(features) < 2:
    st.error("Select at least two features.")
    st.stop()

# ---------- PART 1: ML finds the groups ----------
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, df["segment"])

profiles = df.groupby("segment")[features].mean().round(1)
profiles["customers"] = df["segment"].value_counts().sort_index()

left, right = st.columns([1, 1])

with left:
    st.subheader("Segment profiles")
    st.dataframe(profiles, use_container_width=True)
    st.metric("Silhouette score", f"{silhouette:.3f}",
              help="Higher is better. Above 0.5 is usually a good separation.")

with right:
    st.subheader("The segments")
    if len(features) >= 2:
        chart_df = df[[features[0], features[1], "segment"]]
        st.scatter_chart(chart_df, x=features[0], y=features[1], color="segment")

# ---------- PART 2: GenAI names them ----------
profile_text = profiles.to_string()

prompt = f"""
A K-Means clustering model grouped {len(df)} mall customers into {k} segments.
Here are the average values for each segment:

{profile_text}

For EACH segment, produce:
- a short memorable name (2-4 words) a retail manager would understand
- a one-sentence description
- one concrete marketing action suited to that group

Base everything ONLY on the numbers above. Do not invent behaviours,
demographics, or preferences that the numbers do not support.

Return ONLY valid JSON:
{{
  "segments": [
    {{"id": 0, "name": "...", "description": "...", "action": "..."}}
  ]
}}
"""

st.divider()
st.subheader("Named segments")

with st.spinner("Naming the segments..."):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.6,
        ),
    )

try:
    named = json.loads(response.text)["segments"]
except (json.JSONDecodeError, KeyError) as error:
    st.error(f"Could not parse the model's JSON: {error}")
    st.code(response.text[:600])
    st.stop()

columns = st.columns(min(len(named), 3))
for index, segment in enumerate(named):
    with columns[index % len(columns)]:
        size = int(profiles.loc[segment["id"], "customers"])
        st.markdown(f"### {segment['name']}")
        st.caption(f"Segment {segment['id']} · {size} customers")
        st.write(segment["description"])
        st.info(f"**Action:** {segment['action']}")

st.caption(
    "⚠️ Segment names are AI-generated interpretations of numeric averages. "
    "They are a starting point for discussion, not a finding."
)

with st.expander("Why this pairing works"):
    st.markdown(
        """
        **Clustering has no answer key.** K-Means returns segments 0, 1, 2, 3, 4 —
        numbers with no meaning. Someone must look at the averages and work out
        what each group *is*.

        That interpretation step is exactly what an LLM is good at, and exactly
        what K-Means cannot do. But notice the guardrail in the prompt: the model
        names what the numbers show, and is explicitly told not to invent
        behaviours the data does not support.
        """
    )
```

```bash
streamlit run c3_app.py
```

### ✏️ Exercises

1. **Easy.** Try k = 2, then k = 8. At which k do the names stop being distinct?
2. **Easy.** Add `Age` to the features. Do the segments change meaningfully?
3. **Medium.** Show the elbow chart so the user can choose k with evidence.
4. **Medium.** Add a download button producing a CSV of customers with their segment name attached.
5. **Hard.** Run the naming twice at temperature 0.9 and compare. Are the names stable? What does that tell you about treating them as findings?
6. **Hard, scenario.** *Marketing wants to target the "high income, low spending" segment with discounts.* Is that supported by the data, or is it a story the LLM told? Write down what evidence you would need.

---

## C4 — Bulk Feedback Analyser

⭐⭐⭐ **Advanced** · the full pipeline: upload → ML classifies every row → GenAI summarises the whole. This is the strongest single pattern for a capstone project.

**`c4_app.py`**

```python
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from google import genai
from google.genai import types

st.set_page_config(page_title="Feedback Analyser", page_icon="📊", layout="wide")

MODEL_NAME = "gemini-3.5-flash"
URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/nlp/bbc-text.csv")

@st.cache_resource
def train_classifier():
    """Trains a text classifier once, then keeps it in memory."""
    df = pd.read_csv(URL).dropna()
    df = df.sample(n=1500, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2, random_state=42,
        stratify=df["category"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectors, y_train)

    report = classification_report(y_test, model.predict(X_test_vectors),
                                   output_dict=True)
    return vectorizer, model, report

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.title("📊 Bulk Text Analyser")
st.caption("ML classifies every document. GenAI summarises the whole collection.")

with st.spinner("Training the classifier (first run only)..."):
    vectorizer, classifier, report = train_classifier()

st.success(f"Classifier ready — {report['accuracy']:.1%} accuracy on held-out data")

st.divider()

SAMPLES = """The team secured a dramatic win in the final minutes of the match.
Shares fell sharply after the company reported lower quarterly profits.
The new smartphone features a faster processor and improved battery life.
The minister announced a new policy on education funding this week.
The film received strong reviews at its premiere last night."""

text_block = st.text_area(
    "One document per line",
    value=SAMPLES,
    height=180,
)

uploaded = st.file_uploader("Or upload a CSV with a 'text' column", type=["csv"])

if uploaded is not None:
    uploaded_df = pd.read_csv(uploaded)
    if "text" not in uploaded_df.columns:
        st.error("Your CSV must have a column named 'text'.")
        st.stop()
    documents = uploaded_df["text"].dropna().astype(str).tolist()
else:
    documents = [line.strip() for line in text_block.split("\n") if line.strip()]

if not documents:
    st.info("Enter some text or upload a CSV.")
    st.stop()

if st.button(f"Analyse {len(documents)} documents", type="primary"):

    # ---------- PART 1: ML classifies EVERY document ----------
    vectors = vectorizer.transform(documents)
    predictions = classifier.predict(vectors)
    confidences = classifier.predict_proba(vectors).max(axis=1)

    results = pd.DataFrame({
        "text": documents,
        "category": predictions,
        "confidence": confidences.round(3),
    })

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Every document classified")
        st.dataframe(results, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Category counts")
        counts = results["category"].value_counts()
        st.bar_chart(counts)
        st.metric("Mean confidence", f"{results['confidence'].mean():.1%}")
        low = int((results["confidence"] < 0.5).sum())
        if low:
            st.warning(f"{low} document(s) below 50% confidence")

    # ---------- PART 2: GenAI summarises the WHOLE SET ----------
    # Send the aggregate counts and a few examples - NOT every document.
    # That keeps the prompt small and the cost predictable.
    examples = "\n".join(
        f"- [{row.category}] {row.text[:160]}"
        for row in results.head(12).itertuples()
    )

    prompt = f"""
    A text classification model sorted {len(documents)} documents into categories.

    Counts per category:
    {counts.to_string()}

    Mean classifier confidence: {results['confidence'].mean():.1%}

    Sample of the classified documents:
    {examples}

    Write a brief report for a manager:
    1. One sentence on the overall composition of this collection.
    2. Two or three notable observations.
    3. One sentence on how much to trust these results, given the
       confidence figures above.

    Base everything ONLY on the counts and samples given. Do not invent
    documents, trends, or figures. Under 180 words.
    """

    st.divider()
    st.subheader("Summary report")

    with st.spinner("Writing the report..."):
        response = get_client().models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )

    st.write(response.text)
    st.caption("⚠️ AI-generated summary of automated classifications. "
               "Both layers can be wrong.")

    st.download_button(
        "⬇️ Download classified results",
        results.to_csv(index=False),
        "classified.csv",
        "text/csv",
    )

with st.expander("The architecture, and why it is built this way"):
    st.code(
        """
1000 documents
      |
TF-IDF + Logistic Regression      <- cheap, fast, measurable
      |                              (one model run, no API cost)
1000 labels + confidences
      |
aggregate counts + 12 examples    <- a SMALL prompt
      |
Gemini                            <- ONE API call
      |
a readable report
        """,
        language="text",
    )
    st.markdown(
        """
        **Why not just send all 1000 documents to the LLM?** It would work, and it
        would cost roughly a thousand times more, take far longer, and give you a
        different answer every run. The ML layer does the repetitive work
        cheaply and reproducibly; the LLM is called **once**, on the summary.

        That cost asymmetry is the main reason ML + GenAI architectures exist.
        """
    )
```

```bash
streamlit run c4_app.py
```

### ✏️ Exercises

1. **Easy.** Add your own sentences to the text area. Are they classified sensibly?
2. **Easy.** Type something in none of the five categories (a recipe, say). Look at the confidence.
3. **Medium.** Increase the training sample from 1,500 to 3,000. Does accuracy improve enough to justify the extra training time?
4. **Medium.** Send only documents below 60% confidence to the LLM and ask it to re-classify them. That is a **cascade**: cheap model first, expensive model only where needed.
5. **Hard.** Swap the dataset for `cyberbullying_tweets.csv` and adapt the report prompt. Then write two sentences on the responsible-AI risks of deploying that.
6. **Hard, scenario.** *Your classifier is 96% accurate on 10,000 support tickets. The GenAI summary says "most complaints concern billing".* How would you verify that claim before repeating it to management?

---

# Where to go next

| You finished | Go to |
|---|---|
| All of Section A | [Section B](#section-b--generative-ai-apps) — add an API |
| All of Section B | [Section C](#section-c--ml--genai-apps) — combine them |
| All of Section C | [Capstone guide](../../student-handbook.md#capstone-project-guide) |
| Want more exercises | [exercises-assignments.md](../../exercises-assignments.md) |

## Deploying your app so others can use it

Streamlit Community Cloud hosts these for free:

1. Push your project to a **public** GitHub repository — **without** `secrets.toml`.
2. Include a `requirements.txt` listing your packages.
3. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repository.
4. Add your `GEMINI_API_KEY` in the app's **Settings → Secrets** panel, not in the code.

> ⚠️ **Before you push:** run `git status` and confirm `secrets.toml` and `.env` do **not** appear. A key in a public repository is found by automated scanners within minutes.

## Common problems

| Symptom | Cause and fix |
|---|---|
| `streamlit: command not found` | Environment not activated, or Streamlit not installed |
| `FileNotFoundError: *.joblib` | You skipped the training script, or you are in the wrong folder |
| `StreamlitSecretNotFoundError` | `.streamlit/secrets.toml` missing, or you ran from the parent folder |
| `KeyError: 'GEMINI_API_KEY'` | Key name misspelled in `secrets.toml` |
| App reloads and loses state | Normal — put anything that must survive into `st.session_state` |
| Very slow on every click | Missing `@st.cache_resource` on model loading |
| `Port 8501 is already in use` | Another app is running: `streamlit run app.py --server.port 8502` |
| `429 RESOURCE_EXHAUSTED` | Free-tier rate limit. Wait a minute; do not call the API in a loop |
| `X has 12 features, but ... expecting 13` | Your input columns do not match training — check names **and order** |
