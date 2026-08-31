# Session 5C — Model Deployment & Streamlit

**Turning a trained model into something a person can actually use**

| | |
|---|---|
| **Notebook** | [session-05c-deployment.ipynb](../notebooks/session-05c-deployment.ipynb) |
| **Previous** | [Session 5B — Classification](session-05b-classification.md) |
| **Next** | [Session 6 — Augmentation, Feature Engineering & Reduction](session-06-augmentation-feature-engg-red.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **A model in a notebook helps nobody.** **A loan officer will not open Jupyter, and neither will a doctor.**
>
> **This session builds four complete applications** — iris, loan approval, diabetes screening and salary prediction. **Each one is train, save, serve, in full.** By the end you can hand somebody a link.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Set up a working environment for a Streamlit project
2. **Save a trained pipeline and load it back**, correctly
3. Build a Streamlit app that collects inputs and shows a prediction
4. **Name the common Streamlit widgets** and choose the right one for each input
5. Handle **numeric and categorical inputs** in the same form
6. Show a **probability, not just a label** — and let the user move the threshold
7. Build a **regression** app as well as classification apps
8. **Deploy an app publicly** and say what to check before you do

---

## How this session is organised

| Part | What you get |
|---|---|
| **A — [Before you build](#part-a--before-you-build)** | Environment, folder layout, and what "deployment" means |
| **B — [App 1: Iris](#part-b--app-1--iris-species)** | **The complete pattern, in full** — train, save, serve, run |
| **C — [Streamlit reference](#part-c--streamlit-reference)** | Widgets, layout, caching, and where to look things up |
| **D — [Three more apps](#part-d--three-more-apps)** | Loan approval, diabetes screening, salary prediction |
| **E — [Going live](#part-e--going-live)** | Publishing, and what to check first |

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [Demo or deployment?](#1-demo-or-deployment) | | 8 | [Common widgets](#8-common-widgets) |
| 2 | [Environment setup](#2-environment-setup) | | 9 | [Layout and display](#9-layout-and-display) |
| 3 | [Project layout](#3-project-layout) | | 10 | [The three rules](#10-the-three-rules-that-fix-most-streamlit-bugs) |
| 4 | [App 1 — `train.py`](#4-app-1--trainpy) | | 11 | [Where to look things up](#11-where-to-look-things-up) |
| 5 | [App 1 — `app.py`](#5-app-1--apppy) | | 12 | [App 2 — Loan approval](#12-app-2--loan-approval) |
| 6 | [Running it](#6-running-it) | | 13 | [App 3 — Diabetes screening](#13-app-3--diabetes-screening) |
| 7 | [The pattern, extracted](#7-the-pattern-extracted) | | 14 | [App 4 — Salary prediction](#14-app-4--salary-prediction) |

**The [20 MCQs](#-session-5c--20-mcqs) and [tasks](#-session-5c--tasks) are at the end.**

---

# Part A — Before you build

# 1. Demo or deployment?

🧠 **Analogy: a chef's tasting spoon versus a restaurant.** **The spoon proves the sauce works. The restaurant has to serve it to a stranger, at speed, without the chef standing there explaining it.**

| | **A demo** | **Deployment** |
|---|---|---|
| Who runs it | You | **Anyone** |
| Where the model lives | In memory, in a notebook | **On disk, loaded on demand** |
| Retrained on each use | Often | **Never** — trained once |
| Bad input | Crashes | **Handled** |
| Uncertainty | Ignored | **Shown** |
| Preprocessing | Loose cells | **Inside the saved pipeline** |

> **The single most important idea in this session:** **save the whole pipeline, not just the model.**
>
> **If your scaler lives in a notebook cell and only the classifier is saved, your app will scale nothing** — and it will not error. It will just be quietly wrong forever.

---

# 2. Environment setup

**Everything here runs in the course's `genai` environment.**

```bash
conda activate genai
```

**Check that the four things you need are present:**

```bash
python -c "import sklearn, joblib, streamlit, pandas; print('sklearn', sklearn.__version__); print('streamlit', streamlit.__version__)"
```

**Expected output (versions may differ slightly):**

```text
sklearn 1.9.0
streamlit 1.59.2
```

## If Streamlit is missing

```bash
pip install streamlit joblib
```

**Both are already in the course [`requirements.txt`](../requirements.txt)** — if you installed from that file, you have them.

## Verify Streamlit itself

```bash
streamlit hello
```

> **This opens a demo app in your browser at `http://localhost:8501`.** **If that works, your setup is fine.** **Press `Ctrl+C` in the terminal to stop it.**

⚠️ **A Streamlit app is not a notebook and not a plain script.** **`python app.py` will run the file and produce nothing useful.** **The only correct way to start one is:**

```bash
streamlit run app.py
```

---

# 3. Project layout

**Every app in this session uses the same three-file shape.**

```text
iris-app/
├── train.py              run ONCE - trains and saves the pipeline
├── app.py                the Streamlit app - loads and serves
├── models/
│   └── iris_model.joblib the saved pipeline
└── requirements.txt      what the deployment server must install
```

| File | Runs | Job |
|---|---|---|
| **`train.py`** | **Once, by you** | Load data, preprocess, train, evaluate, **save** |
| **`app.py`** | **Every time a user opens the page** | **Load** the saved file, collect inputs, predict |
| **`models/*.joblib`** | — | The trained pipeline, frozen |
| **`requirements.txt`** | On the deployment server | Reproduce the environment |

> **The separation is the point.** **Training is slow and happens once. Serving is fast and happens constantly.** **An app that retrains on every page load is the most common beginner mistake, and it makes the app unusably slow.**

---

# Part B — App 1 — Iris species

**The simplest useful case: four numeric inputs, three classes.** **Build this one completely and the other three are variations.**

---

# 4. App 1 — `train.py`

```python
# train.py  -  run this ONCE:  python train.py
import pathlib
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. TRAIN - scaler and model in ONE pipeline object
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, random_state=42))
pipeline.fit(X_train, y_train)

# 4. EVALUATE - you must know what you are shipping
y_pred = pipeline.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. SAVE - the whole pipeline, plus everything the app needs
pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump({
    "pipeline": pipeline,
    "features": list(X.columns),
    "classes": list(iris.target_names),
    "accuracy": accuracy_score(y_test, y_pred),
}, "models/iris_model.joblib")

print("saved -> models/iris_model.joblib")
```

**Output:**

```text
Test accuracy: 0.9

              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       0.82      0.90      0.86        10
   virginica       0.89      0.80      0.84        10
    accuracy                           0.90        30

saved -> models/iris_model.joblib
```

## Three decisions worth explaining

**1. `make_pipeline(StandardScaler(), RandomForestClassifier(...))`**

> **The scaler is *inside* the saved object.** When the app later calls `pipeline.predict(...)`, the scaling happens automatically, **using the exact minimum and maximum learned at training time.**
>
> **Save the model alone and you must remember to scale in the app — with the same numbers.** **You will not remember. Nobody does.**

**2. Saving a dictionary, not just the pipeline**

> **The app needs more than the model:** the feature names in the right order, the class names, and the accuracy to display honestly. **Bundle them, and the app has no hard-coded assumptions to drift out of date.**

**3. Printing the classification report before saving**

> **Never ship a model whose score you have not looked at.** **This app is about to tell people what species a flower is, and 0.90 is the number you owe them.**

---

# 5. App 1 — `app.py`

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# --- load the saved pipeline ONCE, not on every interaction
@st.cache_resource
def load_model():
    return joblib.load("models/iris_model.joblib")

bundle = load_model()
pipeline = bundle["pipeline"]

st.title("🌸 Iris Species Classifier")
st.write("Enter four measurements in centimetres and the model will name the species.")

# --- collect the inputs
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, 3.0, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, 1.3, 0.1)

# --- build a one-row DataFrame with the SAME column names as training
sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=bundle["features"])

# --- predict
if st.button("Predict species", type="primary"):
    prediction = pipeline.predict(sample)[0]
    probabilities = pipeline.predict_proba(sample)[0]

    st.success(f"Predicted species: **{bundle['classes'][prediction]}**")

    st.subheader("How confident is it?")
    st.bar_chart(pd.Series(probabilities, index=bundle["classes"]))

    confidence = probabilities.max()
    if confidence < 0.60:
        st.warning(f"Low confidence ({confidence:.0%}). Treat this as a guess.")

# --- honesty, on screen
st.caption(f"Model test accuracy: {bundle['accuracy']:.1%}. "
           "Trained on 150 flowers — a teaching dataset, not a botanical tool.")
```

## Line by line

| Line | What it does |
|---|---|
| `st.set_page_config(...)` | **Must be the first Streamlit call in the file.** Sets the tab title and icon |
| `@st.cache_resource` | **Loads the model once and keeps it.** Without this the file is read from disk on every click |
| `st.columns(2)` | Two side-by-side columns, so the form is not one long strip |
| `st.slider(label, min, max, default, step)` | **A bounded numeric input — the user cannot type nonsense** |
| `pd.DataFrame([...], columns=bundle["features"])` | ⚠️ **The column names must match training exactly** |
| `st.button(..., type="primary")` | Nothing below it runs until the button is clicked |
| `st.bar_chart(...)` | **Shows all three probabilities, not just the winner** |
| `st.caption(...)` | The small print — **and it should be true** |

---

# 6. Running it

```bash
cd iris-app
python train.py          # once
streamlit run app.py     # every time you want the app
```

**Streamlit prints:**

```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.5:8501
```

> **The Network URL works from any device on the same Wi-Fi** — a phone, a colleague's laptop. **That alone is often enough to demonstrate a model to a stakeholder.**
>
> **Edit `app.py` and save: the browser offers to rerun.** **You do not restart anything.**

---

# 7. The pattern, extracted

**Every app in Part D is this same shape.**

```text
train.py                              app.py
─────────────────────────────         ─────────────────────────────
1. load data                          1. load the saved bundle  (cached)
2. preprocess  ┐                      2. draw input widgets
3. split       ├ all inside           3. build a ONE-ROW DataFrame
4. train       ┘ the Pipeline            with the SAME column names
5. evaluate and PRINT                 4. pipeline.predict(...)
6. joblib.dump(bundle)                5. show the answer AND the confidence
                                      6. state the model's real accuracy
```

## The four mistakes this pattern prevents

| Mistake | What the pattern does |
|---|---|
| **Scaler left behind in the notebook** | It is inside the pipeline |
| **Column names in a different order** | They come from the saved bundle |
| **Retraining on every page load** | `train.py` is separate, and `@st.cache_resource` caches the load |
| **Showing a label with false certainty** | `predict_proba` is displayed too |

---

# Part C — Streamlit reference

# 8. Common widgets

**Streamlit's whole design: every widget is a function that *returns the value the user chose*.** **There are no callbacks, no event handlers, no forms to wire up.**

```python
# illustrative: a syntax reference, not runnable as written.
age = st.slider("Age", 18, 100, 30)     # `age` IS the number the user picked
```

## Input widgets

| Widget | Returns | Use it for |
|---|---|---|
| **`st.slider(label, min, max, default, step)`** | number | **A bounded number** — the user cannot type nonsense |
| `st.number_input(label, min_value, max_value, value)` | number | **A number with a wide range** — income, loan amount |
| **`st.selectbox(label, options)`** | the chosen option | **One choice from a list** — education, loan intent |
| `st.multiselect(label, options)` | a list | Several choices from a list |
| **`st.radio(label, options)`** | the chosen option | **2–4 choices, all visible at once** |
| `st.checkbox(label)` | `True` / `False` | A single yes/no |
| **`st.toggle(label)`** | `True` / `False` | A yes/no that looks like a switch |
| `st.text_input(label)` | string | Names, free text |
| `st.text_area(label)` | string | Longer text |
| `st.date_input(label)` | a date | Dates |
| `st.file_uploader(label, type=["csv"])` | a file object | **Letting the user upload their own data** |
| **`st.button(label, type="primary")`** | `True` on the click | **Gate the prediction behind it** |
| `st.form(...)` + `st.form_submit_button()` | — | **Collect many inputs and submit them together** |

## Output widgets

| Widget | Shows |
|---|---|
| `st.title` / `st.header` / `st.subheader` | Headings |
| `st.write(...)` | **Almost anything** — text, DataFrames, charts, figures |
| `st.markdown(...)` | Formatted text |
| **`st.metric(label, value, delta)`** | **A big number with an optional change indicator** |
| `st.dataframe(df)` | An interactive table |
| `st.table(df)` | A static table |
| **`st.bar_chart` / `st.line_chart` / `st.area_chart`** | **A quick chart, straight from a DataFrame or Series** |
| `st.pyplot(fig)` | A matplotlib figure |
| **`st.success` / `st.info` / `st.warning` / `st.error`** | **A coloured message box** |
| `st.progress(value)` | A progress bar |
| `st.caption(...)` | Small print |

## Layout and structure

| Function | What it does |
|---|---|
| **`st.columns(n)`** | **Side-by-side columns** |
| `st.tabs(["A", "B"])` | Tabbed sections |
| **`st.sidebar`** | **The left panel** — put inputs here, results in the main area |
| `st.expander("More")` | A collapsible section |
| `st.container()` | A group you can write into later |
| `st.divider()` | A horizontal rule |

---

# 9. Layout and display

**Two layouts cover almost everything.**

## Layout A — inputs in the sidebar

```python
# illustrative: a syntax reference, not runnable as written.
with st.sidebar:
    st.header("Applicant details")
    age = st.slider("Age", 20, 80, 30)
    income = st.number_input("Annual income", 0, 500_000, 60_000, step=5_000)

st.title("Loan Approval")
st.write("Fill in the form on the left.")
```

> **Best when there are many inputs.** **The form stays put while the results area updates.**

## Layout B — columns in the main area

```python
# illustrative: a syntax reference, not runnable as written.
col1, col2, col3 = st.columns(3)
col1.metric("Prediction", "Approved")
col2.metric("Confidence", "87%")
col3.metric("Model accuracy", "88.8%")
```

> **Best for showing results.** **`st.metric` is the right widget for a headline number.**

## Grouping inputs so the app does not rerun on every keystroke

```python
# illustrative: a syntax reference, not runnable as written.
with st.form("application"):
    age = st.slider("Age", 20, 80, 30)
    income = st.number_input("Income", 0, 500_000, 60_000)
    submitted = st.form_submit_button("Check eligibility")

if submitted:
    ...
```

> **Without a form, Streamlit reruns the whole script every time any widget changes.** **Inside a form, nothing happens until the submit button is pressed** — which matters when there are ten inputs.

---

# 10. The three rules that fix most Streamlit bugs

## Rule 1 — the script reruns top to bottom on every interaction

> **Move a slider and Streamlit runs your entire `app.py` again from line 1.** **This is the single fact that explains most confusing behaviour.**
>
> **It is why loading a model at the top of the file, unguarded, means reading it from disk on every click.**

## Rule 2 — cache anything expensive

| Decorator | Use for |
|---|---|
| **`@st.cache_resource`** | **Models, database connections** — things you load once and share |
| **`@st.cache_data`** | **DataFrames, computation results** — things that are data |

```python
# illustrative: a syntax reference, not runnable as written.
@st.cache_resource          # the model: loaded once, shared by all users
def load_model():
    return joblib.load("models/iris_model.joblib")

@st.cache_data              # a dataframe: cached per set of arguments
def load_reference_data():
    return pd.read_csv("data/reference.csv")
```

> **Forgetting `@st.cache_resource` on the model loader is the most common performance bug in Streamlit apps.**

## Rule 3 — anything you want to survive a rerun goes in `st.session_state`

```python
# illustrative: a syntax reference, not runnable as written.
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append(prediction)
st.write(f"You have made {len(st.session_state.history)} predictions.")
```

> **An ordinary Python variable is destroyed and recreated on every rerun.** **`st.session_state` is the only thing that persists** — use it for prediction history, multi-step forms and login state.

---

# 11. Where to look things up

| Resource | What it is best for |
|---|---|
| **[API reference](https://docs.streamlit.io/develop/api-reference)** | **Every widget, with its arguments.** The page to bookmark |
| [Input widgets](https://docs.streamlit.io/develop/api-reference/widgets) | Sliders, selectboxes, uploaders |
| [Chart elements](https://docs.streamlit.io/develop/api-reference/charts) | Built-in charts, matplotlib, plotly |
| [Layouts and containers](https://docs.streamlit.io/develop/api-reference/layout) | Columns, tabs, sidebar, expanders |
| [Caching](https://docs.streamlit.io/develop/concepts/architecture/caching) | `cache_data` against `cache_resource` |
| [Session state](https://docs.streamlit.io/develop/concepts/architecture/session-state) | Keeping values across reruns |
| **[Cheat sheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet)** | **One page with everything on it** |
| [App gallery](https://streamlit.io/gallery) | Working examples to borrow from |
| **[Community Cloud docs](https://docs.streamlit.io/deploy/streamlit-community-cloud)** | **Deploying free, from GitHub** |

> **In the terminal, `streamlit docs` opens the documentation, and `streamlit hello` opens the demo app.**

---

# Part D — Three more apps

**Same pattern, three different shapes of problem.**

| App | Problem | New thing it teaches |
|---|---|---|
| **[Loan approval](#12-app-2--loan-approval)** | Binary classification | **Categorical inputs, and a `ColumnTransformer`** |
| **[Diabetes screening](#13-app-3--diabetes-screening)** | Binary, **imbalanced** | **A user-adjustable threshold** |
| **[Salary prediction](#14-app-4--salary-prediction)** | Regression | **A number, with an honest error range** |

---

# 12. App 2 — Loan approval

**10,000 applications, 13 features — eight numeric and five categorical — and a yes/no decision.**

## Step 1 — the data needs cleaning first

**[Session 3](session-03-eda-preprocessing.md#the-sequence)'s sequence, and it finds things.**

```python
import pandas as pd

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"
loan = pd.read_csv(dataset_url)

print("shape:", loan.shape, " duplicates:", loan.duplicated().sum())
print("missing:", loan.isnull().sum()[loan.isnull().sum() > 0].to_dict())
print("age > 100      :", (loan["person_age"] > 100).sum(), loan.loc[loan["person_age"] > 100, "person_age"].tolist())
print("emp_exp > 60   :", (loan["person_emp_exp"] > 60).sum(), loan.loc[loan["person_emp_exp"] > 60, "person_emp_exp"].tolist())
print("int_rate > 40  :", (loan["loan_int_rate"] > 40).sum(), loan.loc[loan["loan_int_rate"] > 40, "loan_int_rate"].tolist())
```

**Output:**

```text
shape: (10000, 14)  duplicates: 0
missing: {'person_gender': 1, 'loan_int_rate': 1, 'loan_percent_income': 1}
age > 100      : 1 [144]
emp_exp > 60   : 1 [121]
int_rate > 40  : 1 [101.0]
```

> **A 144-year-old applicant with 121 years of work experience, borrowing at 101% interest.** **Three impossible values in 10,000 rows — and if you build the app first, your age slider will run to 144 and look ridiculous.**
>
> **The widget ranges come from the data. Clean the data before you choose them.**

## Step 2 — `train.py`

```python
# train.py  -  python train.py
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"
loan = pd.read_csv(dataset_url)

# 1. CLEAN - impossible values out (Session 3's step 4)
loan = loan[(loan["person_age"] <= 100)
            & (loan["person_emp_exp"] <= 60)
            & (loan["loan_int_rate"] <= 40)].reset_index(drop=True)

NUMERIC = ["person_age", "person_income", "person_emp_exp", "loan_amnt",
           "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
           "credit_score"]
CATEGORICAL = ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]

X = loan[NUMERIC + CATEGORICAL]
y = loan["loan_status"]

# 2. PREPROCESSOR - different treatment per column type, all inside the pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, max_depth=12,
                                     random_state=42, n_jobs=-1)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=["rejected", "approved"]))

# 3. SAVE - including the widget ranges, taken from the CLEANED data
pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump({
    "pipeline": pipeline,
    "numeric": NUMERIC,
    "categorical": CATEGORICAL,
    "ranges": {c: (float(loan[c].min()), float(loan[c].max()),
                   float(loan[c].median())) for c in NUMERIC},
    "options": {c: sorted(loan[c].dropna().unique().tolist()) for c in CATEGORICAL},
    "accuracy": accuracy_score(y_test, y_pred),
}, "models/loan_model.joblib")
print("saved -> models/loan_model.joblib")
```

**Output:**

```text
Test accuracy: 0.8865

              precision    recall  f1-score   support
    rejected       0.92      0.85      0.88      1000
    approved       0.86      0.93      0.89      1000
    accuracy                           0.89      2000

saved -> models/loan_model.joblib
```

## ⚠️ Two things this `train.py` does that App 1 did not

**1. `ColumnTransformer` — different preprocessing per column type**

> **Numeric columns get median imputation and scaling. Categorical columns get most-frequent imputation and one-hot encoding.** **`ColumnTransformer` applies each to the right columns and stitches the results back together — inside the pipeline, so the app never sees any of it.**

**2. `handle_unknown="ignore"` on the encoder**

> ⚠️ **Without this, a category the model never saw in training crashes the app.** **With it, the unknown value becomes all-zeros and the model carries on.** **This is not optional in a deployed app** — real users will always find a value you did not anticipate.

## Step 3 — `app.py`

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/loan_model.joblib")

bundle = load_model()
pipeline, ranges, options = bundle["pipeline"], bundle["ranges"], bundle["options"]

st.title("🏦 Loan Approval Predictor")

with st.sidebar:
    st.header("Applicant")
    age = st.slider("Age", 20, 80, 26)
    gender = st.selectbox("Gender", options["person_gender"])
    education = st.selectbox("Education", options["person_education"])
    income = st.number_input("Annual income", 8_000, 500_000, 60_950, step=1_000)
    emp_exp = st.slider("Years of employment", 0, 58, 4)
    home = st.selectbox("Home ownership", options["person_home_ownership"])

    st.header("Loan")
    amount = st.number_input("Loan amount", 1_000, 100_000, 8_500, step=500)
    intent = st.selectbox("Purpose", options["loan_intent"])
    rate = st.slider("Interest rate (%)", 5.0, 20.0, 11.4, 0.1)
    hist = st.slider("Credit history (years)", 2, 30, 4)
    score = st.slider("Credit score", 418, 768, 639)
    defaults = st.radio("Previous defaults on file",
                        options["previous_loan_defaults_on_file"], horizontal=True)

# one row, columns named exactly as in training
applicant = pd.DataFrame([{
    "person_age": age, "person_income": income, "person_emp_exp": emp_exp,
    "loan_amnt": amount, "loan_int_rate": rate,
    "loan_percent_income": round(amount / income, 4),
    "cb_person_cred_hist_length": hist, "credit_score": score,
    "person_gender": gender, "person_education": education,
    "person_home_ownership": home, "loan_intent": intent,
    "previous_loan_defaults_on_file": defaults,
}])

st.subheader("Application summary")
st.dataframe(applicant, use_container_width=True)

if st.button("Assess application", type="primary"):
    approved = pipeline.predict(applicant)[0]
    probability = pipeline.predict_proba(applicant)[0][1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", "Approved" if approved else "Rejected")
    c2.metric("Approval probability", f"{probability:.1%}")
    c3.metric("Loan-to-income", f"{amount / income:.2f}")

    if approved:
        st.success("The model predicts this application would be approved.")
    else:
        st.error("The model predicts this application would be rejected.")

    if 0.40 < probability < 0.60:
        st.warning("This is a borderline case. It should go to a human reviewer.")

st.caption(f"Model test accuracy: {bundle['accuracy']:.1%}. "
           "A teaching demonstration — not a lending decision.")
```

## ⚠️ What this app must never be used for

> **A real lending decision.** **The model was trained on historic approvals, so it has learned whatever bias was in those approvals** — and it takes `person_gender` as an input.
>
> **In most jurisdictions an automated credit decision on that basis is illegal**, and a "borderline → human reviewer" rule is the *minimum* safeguard, not a solution. **[Session 12](session-12-opensource-ethics.md) covers this properly.**

---

# 13. App 3 — Diabetes screening

**100,000 patients, and only 8.5% of them have diabetes.** **That imbalance changes the app.**

## Step 1 — what the data looks like

```python
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/diabetes_prediction_dataset.csv"
diabetes = pd.read_csv(dataset_url)

print("shape:", diabetes.shape)
print("duplicates:", diabetes.duplicated().sum())
print("missing:", diabetes.isnull().sum().sum())
print("target:", diabetes["diabetes"].value_counts().to_dict())
print("smoking_history:", diabetes["smoking_history"].unique().tolist())
```

**Output:**

```text
shape: (100000, 9)
duplicates: 3854
missing: 0
target: {0: 91500, 1: 8500}
smoking_history: ['never', 'No Info', 'current', 'former', 'ever', 'not current']
```

> **3,854 duplicate rows.** With only nine columns, identical rows are plausible coincidences — **but 3.9% is too many for coincidence, and they inflate any score that counts them twice.** **Remove them here.**
>
> **And `'No Info'` is a category, not a missing value.** **Leave it as its own level** — "we do not know whether this patient smokes" is itself informative.

## Step 2 — `train.py`

```python
# train.py
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/diabetes_prediction_dataset.csv"
diabetes = pd.read_csv(dataset_url).drop_duplicates().reset_index(drop=True)

NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
           "hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_history"]

X = diabetes[NUMERIC + CATEGORICAL]
y = diabetes["diabetes"]

pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
    # class_weight="balanced" - the 91.5% / 8.5% split would otherwise
    # produce a model that predicts "no diabetes" for everybody
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["no diabetes", "diabetes"]))

pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump({
    "pipeline": pipeline,
    "options": {c: sorted(diabetes[c].unique().tolist()) for c in CATEGORICAL},
    "accuracy": accuracy_score(y_test, y_pred),
}, "models/diabetes_model.joblib")
print("saved -> models/diabetes_model.joblib")
```

**Output:**

```text
Test accuracy: 0.8845

[[15516  2018]
 [  204  1492]]

              precision    recall  f1-score   support
 no diabetes       0.99      0.88      0.93     17534
    diabetes       0.43      0.88      0.57      1696
    accuracy                           0.88     19230
```

## ⚠️ Read that report before you build the app

| Metric | Value | What it means |
|---|---|---|
| **Recall on diabetes** | **0.88** | **It finds 1,492 of the 1,696 patients who have diabetes — and misses 204** |
| **Precision on diabetes** | **0.43** | **It flags 3,510 people, of whom 2,018 do not have diabetes** |

> **That is a deliberate trade, and `class_weight="balanced"` bought it.** **For a screening tool it is the right one:** a false alarm costs a blood test; a miss costs a missed diagnosis.
>
> **But the app must say so.** **An app that prints "DIABETES DETECTED" at 43% precision would be actively harmful.**

## Step 3 — `app.py`, with a threshold the user controls

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺")

@st.cache_resource
def load_model():
    return joblib.load("models/diabetes_model.joblib")

bundle = load_model()
pipeline, options = bundle["pipeline"], bundle["options"]

st.title("🩺 Diabetes Risk Screening")
st.info("This is a **screening** tool. It flags people who should be tested. "
        "It does not diagnose anybody.")

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 1, 100, 45)
        gender = st.selectbox("Gender", options["gender"])
        bmi = st.slider("BMI", 10.0, 60.0, 27.0, 0.1)
        smoking = st.selectbox("Smoking history", options["smoking_history"])
    with c2:
        hba1c = st.slider("HbA1c level", 3.5, 9.0, 5.5, 0.1)
        glucose = st.slider("Blood glucose level", 80, 300, 140, 5)
        hypertension = st.toggle("Diagnosed with hypertension")
        heart_disease = st.toggle("Diagnosed with heart disease")
    submitted = st.form_submit_button("Assess risk", type="primary")

threshold = st.slider(
    "Flagging threshold", 0.10, 0.90, 0.50, 0.05,
    help="Lower it to catch more cases at the cost of more false alarms.")

if submitted:
    patient = pd.DataFrame([{
        "age": age, "bmi": bmi, "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": int(hypertension), "heart_disease": int(heart_disease),
        "gender": gender, "smoking_history": smoking,
    }])

    risk = pipeline.predict_proba(patient)[0][1]

    st.metric("Estimated risk", f"{risk:.1%}")
    st.progress(min(risk, 1.0))

    if risk >= threshold:
        st.warning(f"**Above the {threshold:.0%} threshold — recommend a diabetes test.**")
    else:
        st.success(f"Below the {threshold:.0%} threshold. No test recommended by this tool.")

st.caption(
    f"Model accuracy {bundle['accuracy']:.1%}, but read this instead: it finds 88% of "
    "true cases (recall) and fewer than half of the people it flags actually have "
    "diabetes (precision 0.43). It is tuned to over-flag on purpose. "
    "Not a medical device. Not a diagnosis."
)
```

> **The threshold slider is the point of this app.** **[Session 5B](session-05b-classification.md) argued that 0.5 is only a default; here the user can move it and watch the trade-off happen.**
>
> **And notice the caption leads with recall and precision, not accuracy.** **"88.5% accurate" would be true and misleading.**

---

# 14. App 4 — Salary prediction

**A regression app: the output is a number, not a class.**

## Step 1 — the data needs Session 3's judgement

```python
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/salary_data.csv"
salary = pd.read_csv(dataset_url)

print("shape:", salary.shape)
print("missing:", salary.isnull().sum().to_dict())
print("duplicates:", salary.duplicated().sum())
print("salaries below 10,000:", sorted(salary.loc[salary["Salary"] < 10000, "Salary"].dropna().tolist()))
```

**Output:**

```text
shape: (375, 2)
missing: {'Experience': 2, 'Salary': 2}
duplicates: 220
salaries below 10,000: [350.0]
```

## ⚠️ 220 duplicates out of 375 — and you must NOT remove them

**[Session 3](session-03-eda-preprocessing.md#when-not-to-remove-duplicates) warned about exactly this case. Here is the measurement.**

| Treatment | Rows | **5-fold CV R²** |
|---|---|---|
| Drop missing only | 373 | **0.8622** |
| Drop missing + the ₹350 salary | **372** | **0.8568** |
| **…and remove duplicates** | **153** | **0.7360** |

> **Removing the duplicates costs 0.12 of R² and throws away 60% of the data.**
>
> **Why?** **The table has two columns. Two different people with 5 years of experience earning ₹60,000 produce identical rows** — and both are real observations. **`drop_duplicates()` would delete one of them.**
>
> **A duplicate is only an error when the same *record* was entered twice.** With two columns and 375 people, repeats are arithmetic, not mistakes. **Keep them.**
>
> **The ₹350 salary is a different matter — that is an impossible annual salary, and it goes.**

## Step 2 — `train.py`

```python
# train.py
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/salary_data.csv"
salary = pd.read_csv(dataset_url)

# drop the 4 missing values and the one impossible salary - but KEEP duplicates
salary = salary.dropna()
salary = salary[salary["Salary"] >= 10_000].reset_index(drop=True)
print("rows kept:", len(salary))

X = salary[["Experience"]]
y = salary["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
cv = cross_val_score(LinearRegression(), X, y,
                     cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean()

print(f"holdout R2 {r2:.4f}   5-fold CV R2 {cv:.4f}   MAE {mae:,.0f}")
print(f"salary = {model.coef_[0]:,.0f} x experience + {model.intercept_:,.0f}")

pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump({"model": model, "r2": r2, "cv_r2": cv, "mae": mae,
             "max_experience": float(X["Experience"].max())},
            "models/salary_model.joblib")
print("saved -> models/salary_model.joblib")
```

**Output:**

```text
rows kept: 372
holdout R2 0.9090   5-fold CV R2 0.8568   MAE 11,613
salary = 6,780 x experience + 32,324
saved -> models/salary_model.joblib
```

> **The model is one line: `salary = 6,780 × experience + 32,324`.** **Every extra year of experience is worth about ₹6,780, and someone with no experience starts at about ₹32,300.**

## Step 3 — `app.py`

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Salary Estimator", page_icon="💰")

@st.cache_resource
def load_model():
    return joblib.load("models/salary_model.joblib")

bundle = load_model()
model = bundle["model"]

st.title("💰 Salary Estimator")
st.write("Estimate an annual salary from years of experience.")

experience = st.slider("Years of experience", 0.0, 30.0, 5.0, 0.5)

prediction = model.predict(pd.DataFrame({"Experience": [experience]}))[0]
mae = bundle["mae"]

c1, c2 = st.columns(2)
c1.metric("Estimated salary", f"₹{prediction:,.0f}")
c2.metric("Typical error (±)", f"₹{mae:,.0f}")

st.info(f"**A realistic range is ₹{prediction - mae:,.0f} to ₹{prediction + mae:,.0f}.** "
        "The single number above is the middle of that range, not a promise.")

# --- the whole model, drawn
years = np.arange(0, 31)
curve = model.predict(pd.DataFrame({"Experience": years}))
st.line_chart(pd.DataFrame({"predicted salary": curve}, index=years))

if experience > bundle["max_experience"]:
    st.warning(
        f"The model never saw anyone with more than {bundle['max_experience']:.0f} years. "
        "This prediction is an extrapolation and should not be trusted.")

st.caption(f"Cross-validated R² {bundle['cv_r2']:.2f} — the model explains about "
           f"{bundle['cv_r2']:.0%} of the variation in salary from experience alone. "
           "Real salaries also depend on role, location and industry, none of which "
           "this model has.")
```

## ⚠️ Three things a regression app must do that a classifier need not

| | Why |
|---|---|
| **Show an error range, not just a number** | **"₹98,000" reads as a promise. "₹98,000 ± ₹11,600" reads as an estimate** |
| **Warn on extrapolation** | **The slider goes to 30; the data stops at 25.** A linear model will happily predict for 30 years and has no basis for it |
| **State what is missing** | **R² 0.86 means 14% is unexplained** — role, location, industry, none of which the model has |

---

# Part E — Going live

# 15. Deploying to Streamlit Community Cloud

**Free, and it takes about five minutes.**

## What the repository needs

```text
your-app/
├── app.py
├── train.py
├── models/
│   └── model.joblib          <- MUST be committed
└── requirements.txt          <- MUST list every import
```

**`requirements.txt`:**

```text
streamlit
scikit-learn==1.9.0
pandas
numpy
joblib
```

> ⚠️ **Pin the scikit-learn version.** **A pipeline saved by 1.9.0 and loaded by a different version may warn, misbehave, or refuse to load.** **This is the single most common deployment failure.**

## The steps

| # | Step |
|---|---|
| 1 | **Push the repo to GitHub** — including `models/*.joblib` |
| 2 | Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub |
| 3 | **New app** → pick the repository, branch and `app.py` |
| 4 | **Deploy.** It installs `requirements.txt` and starts the app |
| 5 | You get a public URL like `https://your-app.streamlit.app` |

## ⚠️ Model file size

**The loan model is the awkward one. Measured:**

| Random Forest settings | Test accuracy | **File size** |
|---|---|---|
| `n_estimators=200`, no depth limit | 0.8850 | **23.01 MB** |
| **`n_estimators=100, max_depth=12`** | **0.8865** | **5.24 MB** |
| `n_estimators=60, max_depth=10` | 0.8810 | **1.82 MB** |

> **The smaller model is also the more accurate one** — the depth limit is [Session 8](session-08-evaluation-tuning.md#6-fixing-each-problem)'s regularisation, doing its job.
>
> **Git is unhappy above about 50 MB and refuses above 100 MB.** **If your saved model is enormous, that is usually a signal to constrain the model, not to reach for Git LFS.**

## Other options

| Where | Good for |
|---|---|
| **Streamlit Community Cloud** | **Demos and coursework. Free** |
| **Hugging Face Spaces** | Free, and the natural home for ML demos — **[Session 12](session-12-opensource-ethics.md)** |
| Render, Railway | Small production apps |
| Docker + a cloud VM | Full control, real production |

---

# 16. Before you deploy anything

**A short checklist, and none of it is optional.**

| Check | Why |
|---|---|
| **Does the app state the model's real accuracy?** | A prediction with no error rate attached is a claim |
| **Does it show a probability, not just a label?** | **The user needs to know when it is unsure** |
| **Does it warn on extrapolation and low confidence?** | The model does not know what it does not know |
| **Are the input ranges taken from the cleaned data?** | Otherwise your age slider goes to 144 |
| **Is `handle_unknown="ignore"` set on the encoder?** | Real users will find a category you did not anticipate |
| **Is the whole pipeline saved, not just the model?** | Otherwise preprocessing silently vanishes |
| **Is the model file committed and under 50 MB?** | The app cannot start without it |
| **Is the scikit-learn version pinned?** | Version mismatches break `joblib.load` |
| **Does it say what the app must NOT be used for?** | **A loan model with a gender input is not a lending tool** |

> **The last row is the one people skip.** **Every app in this session ends with a caption saying what it is and is not.** **Write yours before you deploy, not after somebody misuses it.**

---

# ❓ Session 5C — 20 MCQs

**Answer from memory first, then check.**

### Saving and loading

**Q1.** The most important thing to save is…
- (a) The model object  (b) **The whole pipeline, including every preprocessing step**  (c) The training data  (d) The accuracy

**Q2.** If you save only the classifier and scale in the notebook, the app will…
- (a) Crash  (b) **Run happily and be quietly wrong forever**  (c) Retrain  (d) Warn you

**Q3.** `train.py` and `app.py` are separate files because…
- (a) Style  (b) **Training is slow and happens once; serving is fast and happens on every page load**  (c) Streamlit requires it  (d) To reduce file size

**Q4.** Bundling `features`, `classes` and `accuracy` alongside the pipeline means…
- (a) A bigger file  (b) **The app has no hard-coded assumptions that can drift out of date**  (c) Faster loading  (d) Nothing useful

**Q5.** The correct way to start a Streamlit app is…
- (a) `python app.py`  (b) **`streamlit run app.py`**  (c) `jupyter app.py`  (d) `streamlit app.py`

### Streamlit behaviour

**Q6.** When a user moves a slider, Streamlit…
- (a) Updates that one widget  (b) **Reruns the entire script from line 1**  (c) Calls a callback  (d) Does nothing until you click

**Q7.** `@st.cache_resource` on the model loader…
- (a) Makes predictions faster  (b) **Loads the file from disk once instead of on every interaction**  (c) Caches user inputs  (d) Is optional styling

**Q8.** The difference between `@st.cache_data` and `@st.cache_resource` is…
- (a) None  (b) **`cache_data` is for data like DataFrames; `cache_resource` is for shared objects like models and connections**  (c) One is faster  (d) One is deprecated

**Q9.** A variable you want to survive a rerun must be stored in…
- (a) A global  (b) **`st.session_state`**  (c) A file  (d) A cache

**Q10.** `st.form` is useful because…
- (a) It looks better  (b) **Nothing runs until the submit button is pressed, instead of rerunning on every keystroke**  (c) It validates inputs  (d) It caches

**Q11.** For a bounded numeric input where you do not want nonsense typed in, use…
- (a) `st.text_input`  (b) **`st.slider`**  (c) `st.write`  (d) `st.selectbox`

**Q12.** `st.set_page_config()` must be…
- (a) At the end  (b) **The first Streamlit call in the file**  (c) Anywhere  (d) Inside a function

### The apps

**Q13.** `ColumnTransformer` is needed in the loan app because…
- (a) There is a lot of data  (b) **Numeric and categorical columns need different preprocessing, applied to the right columns**  (c) It is faster  (d) The target is binary

**Q14.** `OneHotEncoder(handle_unknown="ignore")` matters in a deployed app because…
- (a) It is faster  (b) **A category the model never saw would otherwise crash the app, and real users always find one**  (c) It improves accuracy  (d) It handles missing values

**Q15.** The loan data contained an applicant aged 144. If you build the app before cleaning…
- (a) Nothing happens  (b) **Your age slider's range is taken from the dirty data and runs to 144**  (c) The model fails  (d) Streamlit rejects it

**Q16.** The diabetes model has recall 0.88 and precision 0.43. The app should…
- (a) Report "88.5% accurate"  (b) **Say it finds 88% of true cases but that fewer than half of the people it flags actually have diabetes**  (c) Hide the numbers  (d) Report precision only

**Q17.** `class_weight="balanced"` was used on the diabetes model because…
- (a) It is faster  (b) **The 91.5% / 8.5% split would otherwise produce a model that predicts "no diabetes" for everybody**  (c) It scales the data  (d) It is required by LogisticRegression

**Q18.** The salary data had 220 duplicates in 375 rows, and the guide keeps them because…
- (a) Laziness  (b) **With two columns, two different people can genuinely have the same experience and salary — removing them cost 0.12 of R² and 60% of the data**  (c) Duplicates never matter  (d) Pandas cannot remove them

**Q19.** A regression app should show an error range because…
- (a) It looks professional  (b) **"₹98,000" reads as a promise; "₹98,000 ± ₹11,600" reads as an estimate**  (c) It is required  (d) To hide errors

**Q20.** Pinning `scikit-learn==1.9.0` in `requirements.txt`…
- (a) Makes it faster  (b) **Prevents a version mismatch from breaking `joblib.load` on the deployment server**  (c) Reduces the model size  (d) Is optional

<details><summary>Answers</summary>

**A1 — (b) The whole pipeline.** **This is the single most important idea in the session.** Preprocessing is part of the model.

**A2 — (b) Quietly wrong forever.** **It will not error.** The model will receive unscaled numbers and produce plausible-looking nonsense.

**A3 — (b) Different speeds, different frequencies.** **An app that retrains on every page load is the most common beginner mistake**, and it makes the app unusable.

**A4 — (b) No hard-coded assumptions.** Feature order, class names and the accuracy all come from the file that was saved with the model.

**A5 — (b) `streamlit run app.py`.** **`python app.py` runs the file and produces nothing useful.**

**A6 — (b) Reruns the entire script.** **This one fact explains most confusing Streamlit behaviour.**

**A7 — (b) Loads once.** **Forgetting it is the most common performance bug in Streamlit apps.**

**A8 — (b) Data versus shared objects.** `cache_data` caches per set of arguments; `cache_resource` holds one shared instance.

**A9 — (b) `st.session_state`.** **An ordinary variable is destroyed and recreated on every rerun.**

**A10 — (b) Nothing runs until submit.** Which matters when there are ten inputs and each keystroke would otherwise trigger a full rerun.

**A11 — (b) `st.slider`.** **A bounded widget is input validation you get for free.**

**A12 — (b) First.** Streamlit raises an error if anything else has already drawn to the page.

**A13 — (b) Different preprocessing per column type.** Numeric columns get imputation and scaling; categorical columns get imputation and one-hot encoding.

**A14 — (b) Unknown categories would crash it.** **This is not optional in a deployed app.**

**A15 — (b) Your slider runs to 144.** **The widget ranges come from the data, so clean the data first.**

**A16 — (b) Report recall and precision.** **"88.5% accurate" would be true and misleading.** The model is tuned to over-flag on purpose, and the caption must say so.

**A17 — (b) The imbalance.** Without it, always predicting "no diabetes" scores 91.5% and finds nobody.

**A18 — (b) They are legitimate repeated observations.** **A duplicate is only an error when the same *record* was entered twice.** Measured: CV R² fell from 0.8568 to 0.7360.

**A19 — (b) A single number reads as a promise.** **Also warn when the input is outside the training range** — the model has no basis for those predictions.

**A20 — (b) Version mismatch.** **A pipeline saved by one version and loaded by another may warn, misbehave, or refuse to load.** This is the most common deployment failure.
</details>

---

# 🎯 Session 5C — Tasks

## Setup and the pattern

**Task 1 — Verify your environment.** Activate `genai`, print the versions of scikit-learn, streamlit, pandas and joblib, and run `streamlit hello`. **Report what you saw.**

**Task 2 — Build App 1 end to end.** Create the folder, write `train.py` and `app.py`, run both. **Take a screenshot of the working app.**

**Task 3 — Break it deliberately.** Save *only* the `RandomForestClassifier` instead of the pipeline, and load that in the app. **Report what happens to the predictions**, and explain why nothing errored.

**Task 4 — Prove the cache matters.** Remove `@st.cache_resource`, add a `print()` inside `load_model()`, and move a slider ten times. **Count the printed lines.** Put the decorator back and repeat.

**Task 5 — Share it.** Start the app and open the **Network URL** on your phone. **Report whether it worked and what you had to do.**

## Widgets

**Task 6 — A widget tour.** Build a single app that uses ten different widgets and prints each one's returned value. **Which surprised you?**

**Task 7 — Two layouts.** Rebuild App 1 twice — once with inputs in the sidebar, once with `st.columns` in the main area. **Which would you give a user, and why?**

**Task 8 — Add a form.** Wrap App 2's inputs in `st.form`. **Describe the difference in how the app feels.**

**Task 9 — Session state.** Add a prediction history to App 1: every prediction is appended to a list and shown as a table. **This cannot be done without `st.session_state` — explain why.**

**Task 10 — Read the docs.** Find three widgets in the [API reference](https://docs.streamlit.io/develop/api-reference) that are not in §8's tables. **Describe what each does and when you would use it.**

## The apps

**Task 11 — Loan approval, end to end.** Build App 2. **Then find an input combination the model gets obviously wrong** and describe it.

**Task 12 — Skip the cleaning.** Train the loan model *without* removing the impossible values, and set the slider ranges from that data. **Report what the age slider looks like.**

**Task 13 — Unknown category.** Remove `handle_unknown="ignore"` from App 2, then feed the app a category the model never saw. **Report the exact error.**

**Task 14 — Diabetes, end to end.** Build App 3. **Move the threshold slider from 0.1 to 0.9 for one patient and record where the recommendation flips.**

**Task 15 — The imbalance, measured.** Retrain the diabetes model *without* `class_weight="balanced"`. **Report accuracy, precision and recall for both versions in one table, and say which you would deploy.**

**Task 16 — Salary, end to end.** Build App 4. **Set the slider to 30 years and report the prediction and the warning.**

**Task 17 — The duplicates experiment.** Train the salary model with and without `drop_duplicates()`. **Report CV R² for both and write two sentences on when a duplicate is an error and when it is data.**

**Task 18 — A regression app of your own.** Build one for the car price data from Session 5. **Include an error range and an extrapolation warning.**

## Deployment

**Task 19 — Deploy one.** Push any of the four apps to GitHub and deploy it on Streamlit Community Cloud. **Report the public URL and anything that went wrong.**

**Task 20 — The honesty audit.** Take any deployed app — yours or someone else's — and go through §16's checklist row by row. **Report which rows it fails and what you would change.**

---

## ✅ Session 5C checklist

- [ ] I save the **whole pipeline**, never just the model
- [ ] I keep `train.py` and `app.py` separate
- [ ] I bundle feature names, class names and the accuracy with the model
- [ ] I run apps with **`streamlit run app.py`**
- [ ] I know the script **reruns top to bottom** on every interaction
- [ ] I put `@st.cache_resource` on the model loader
- [ ] I use `st.session_state` for anything that must survive a rerun
- [ ] I set **widget ranges from the cleaned data**
- [ ] I set `handle_unknown="ignore"` on any encoder in a deployed app
- [ ] I show a **probability**, not just a label
- [ ] I show an **error range** on regression, and warn on extrapolation
- [ ] I **state the model's real accuracy on screen** — and the right metric for the problem
- [ ] I write down what the app **must not be used for**
- [ ] I pin the scikit-learn version in `requirements.txt`

---

| | |
|---|---|
| **Previous** | [Session 5B — Classification](session-05b-classification.md) |
| **Next** | [Session 6 — Augmentation, Feature Engineering & Reduction](session-06-augmentation-feature-engg-red.md) |
| **Notebook** | [session-05c-deployment.ipynb](../notebooks/session-05c-deployment.ipynb) |
| **More practice** | [Exercises & assignments](../exercises-assignments.md) |
