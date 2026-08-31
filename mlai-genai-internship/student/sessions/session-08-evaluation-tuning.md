# Session 8 — Model Evaluation & Improvement

**Overfitting & Underfitting · Model Validation · Holdout · Cross-Validation · K-Fold · Leave-One-Out · Bootstrapping · Grid Search · Random Search · Bayesian Optimization**

| | |
|---|---|
| **Notebook** | [session-08-evaluation-tuning.ipynb](../notebooks/session-08-evaluation-tuning.ipynb) |
| **Previous** | [Session 7 — Unsupervised Learning: Clustering](session-07-unsupervised.md) |
| **Next** | [Session 9 — Deep Learning](session-09-deep-learning.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **In Sessions 5 and 5B you trained models and reported a number. This session asks three questions in order:**
>
> **Is my model the right size? — Is my number real? — Can I make it better?**

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Preprocess a dataset with **Session 3's sequence** before evaluating anything on it
2. Recognise **underfitting and overfitting** from the train–test gap
3. **Fix** each one deliberately
4. Say **what validation is and why it is needed**
5. **Explain the difference between validation and testing** — and why it matters
6. Name the five validation strategies and say which fits which situation
7. Run **k-fold, stratified k-fold and leave-one-out** cross-validation
8. Select between models with cross-validation, then evaluate the winner **once**
9. Read a **validation curve** and a **learning curve**
10. Spot the two leaks — scaling before the split, resampling before the split — and **fix both with a pipeline**
11. Distinguish parameters from hyperparameters
12. Tune with **manual, grid, random and Bayesian search** — and say what each costs
13. **Never let the test set choose a hyperparameter**

---

## How this session is organised

| Part | Question it answers | Trainer notebook |
|---|---|---|
| **A — [Underfitting & Overfitting](#part-a--underfitting--overfitting)** | *Is my model too simple, too complex, or right?* | `overfitting_underfitting.ipynb` |
| **B — [Model Validation Techniques](#part-b--model-validation-techniques)** | *How do I measure it honestly?* | `cross_validation.ipynb`, `k_fold_cross_validation.ipynb` |
| **C — [Hyperparameter Tuning](#part-c--hyperparameter-tuning)** | *How do I make it better, without cheating?* | `hyperparameter_session.ipynb` |

**The three parts are a chain.** **Part A finds a problem it cannot settle. Part B gives you the tools to settle it. Part C uses those tools to fix it.**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [The three fits](#1-the-three-fits) | | 13 | [Model selection with CV](#13-model-selection-with-cross-validation) |
| 2 | [Use case 1 — heart failure](#2-use-case-1--heart-failure) | | 14 | [Back to Part A's question](#14-back-to-part-as-question) |
| 3 | [Reading the gap](#3-reading-the-gap) | | 15 | [Validation curves](#15-validation-curves) |
| 4 | [Why the smallest model won](#4-why-the-smallest-model-won) | | 16 | [Learning curves](#16-learning-curves) |
| 5 | [Use case 2 — car prices](#5-use-case-2--car-prices) | | 17 | [The two leaks](#17-the-two-leaks-that-make-every-number-a-lie) |
| 6 | [Fixing each problem](#6-fixing-each-problem) | | 18 | [The tuning setup](#18-the-tuning-setup) |
| 7 | [What validation is](#7-what-validation-is-and-why-it-is-needed) | | 19 | [Parameters vs hyperparameters](#19-parameters-vs-hyperparameters) |
| 8 | [Validation vs testing](#8-validation-and-testing--the-difference) | | 20 | [Manual search](#20-manual-search--and-the-trap-in-it) |
| 9 | [Validation strategies](#9-validation-strategies) | | 21 | [Grid search](#21-grid-search) |
| 10 | [K-Fold cross-validation](#10-k-fold-cross-validation) | | 22 | [Random search](#22-random-search) |
| 11 | [Stratified K-Fold](#11-stratified-k-fold) | | 23 | [Tuning an SVM](#23-tuning-an-svm) |
| 12 | [Leave-One-Out and all four](#12-leave-one-out-and-all-four-side-by-side) | | 24 | [Tuning inside a pipeline](#24-tuning-inside-a-pipeline) |
| | | | 25 | [Bayesian optimization](#25-bayesian-optimization) |

**Practices sit between the parts.** The [20 MCQs](#-session-8--20-mcqs) and [tasks](#-session-8--tasks) are at the end.

---
# The datasets — preprocessed the Session 3 way

**Two datasets carry this session. Neither can be used until [Session 3](session-03-eda-preprocessing.md#the-sequence)'s sequence has been run on it.**

```text
1. LOAD    2. EXPLORE    3. DUPLICATES    4. IMPOSSIBLE VALUES
5. MISSING VALUES    6. OUTLIERS    7. ENCODING    8. SPLIT    9. SCALING
```

**Steps 8 and 9 belong to the modelling code, not here** — and [§17](#17-the-two-leaks-that-make-every-number-a-lie) measures what happens when you get their order wrong.

---

## Dataset 1 — heart failure clinical records

**299 patients, 13 measurements, and whether the patient died.** **Parts A and C use it throughout.**

### Explore first

```python
import numpy as np
import pandas as pd

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/heart_failure_raw.csv"
heart = pd.read_csv(dataset_url)

print("shape:", heart.shape)
print("duplicate rows:", heart.duplicated().sum())
print("\nmissing:")
print(heart.isnull().sum()[heart.isnull().sum() > 0])
print("\n", heart[["age", "ejection_fraction", "serum_creatinine"]].describe().round(2))
```

**Output:**

```text
shape: (299, 14)
duplicate rows: 0

missing:
age                  15
ejection_fraction    15
serum_creatinine     15

          age  ejection_fraction  serum_creatinine
count  284.00             284.00            284.00
mean    61.50              38.28              1.55
min     40.00              14.00              0.50
max    160.00              80.00             27.00
```

**Three findings, and only the first two show up in `info()`:**

| Finding | Step that handles it |
|---|---|
| **0 duplicates** | Step 3 — nothing to do, but you checked |
| **45 missing values** | Step 5 |
| **Maximum age of 160** | **Step 4 — nobody is 160** |

### Impossible values, then missing values

**An impossible value is an error, not an extreme case.** **Convert it to `NaN` and let the imputation step deal with it** — the other thirteen measurements for those patients are perfectly good.

```python
impossible = heart["age"] > 120
print("impossible ages:", heart.loc[impossible, "age"].tolist())
heart.loc[impossible, "age"] = np.nan

for col in ["age", "ejection_fraction", "serum_creatinine"]:
    heart[col] = heart[col].fillna(heart[col].median())

print("missing now:", heart.isnull().sum().sum(), " age max now:", heart["age"].max())
```

**Output:**

```text
impossible ages: [150.0, 160.0]
missing now: 0  age max now: 95.0
```

> **Impossible values come *before* missing values in the sequence.** Do it the other way round and the errors survive imputation untouched.

### Outliers — checked, and kept

**Session 3's IQR rule flags 77 rows here, a quarter of the dataset.** **Their death rate is 46.8% against an overall 32.1%** — the "outliers" are disproportionately the patients who died. **They stay.** *(Worked through in [Session 6](session-06-augmentation-feature-engg-red.md#11-example-2--heart-failure).)*

### Encoding — `LabelEncoder`, every text column

```python
from sklearn.preprocessing import LabelEncoder

TEXT = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
        "treatment_type", "DEATH_EVENT"]

le = LabelEncoder()
for col in TEXT:
    heart[col] = le.fit_transform(heart[col])
    print(f"{col:<22} {le.classes_.tolist()} -> {list(range(len(le.classes_)))}")

X = heart.drop(columns=["DEATH_EVENT"])
y = heart["DEATH_EVENT"]

print("\nX:", X.shape, " missing:", X.isnull().sum().sum())
print("class balance:", y.value_counts().to_dict())
```

**Output:**

```text
anaemia                ['No', 'Yes'] -> [0, 1]
diabetes               ['No', 'Yes'] -> [0, 1]
high_blood_pressure    ['No', 'Yes'] -> [0, 1]
sex                    ['No', 'Yes'] -> [0, 1]
smoking                ['No', 'Yes'] -> [0, 1]
treatment_type         ['Lifestyle', 'Medication', 'Other', 'Surgery'] -> [0, 1, 2, 3]
DEATH_EVENT            ['No', 'Yes'] -> [0, 1]

X: (299, 13)  missing: 0
class balance: {0: 203, 1: 96}
```

> **Print `le.classes_` every time.** **It is the only thing that tells you which way round the codes went.** Alphabetical order happens to put `'No'` first here, which is what you want — **but check, do not assume.**

⚠️ **A note on `treatment_type`.** It has four unordered categories, and Label Encoding gives them 0, 1, 2, 3 — **which implies `Surgery` is three times `Lifestyle`.** **That is not true.** Session 3's table says who is hurt by this:

| Model | Does the false order hurt? |
|---|---|
| **Decision Tree, Random Forest** | **Barely** — they can split anywhere |
| kNN, SVM | **Yes** — they measure distance |

> **Most of this session uses trees and forests, so the choice is defensible — but it is a choice, and you should be able to say why you made it.** **Dummy variables are the alternative when a distance-based model is the final answer.**

**299 rows is small, and 203 against 96 is imbalanced. Both facts matter enormously in the next few pages.**

---

## Dataset 2 — car prices

**15,244 cars, predicting `selling_price` with a decision tree.** **Part B uses it: a regression problem shows over- and underfitting far more starkly than a 299-row classification one.**

**The file is already encoded — every column is numeric. That does not mean it is clean.**

```python
cars_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/cardekho_preprocessed.csv"
cars = pd.read_csv(cars_url)

print("shape:", cars.shape)
print("duplicates:", cars.duplicated().sum(), " missing:", cars.isnull().sum().sum())
print(cars[["vehicle_age", "km_driven", "seats", "selling_price"]].describe().round(0))
```

**Output:**

```text
shape: (15244, 13)
duplicates: 0  missing: 0

       vehicle_age   km_driven    seats  selling_price
count      15244.0     15244.0  15244.0        15244.0
min            0.0       100.0      0.0        40000.0
50%            6.0     50000.0      5.0       559000.0
max           29.0   3800000.0      9.0     39500000.0
```

> ⚠️ **`seats` has a minimum of 0, and `km_driven` a maximum of 3,800,000.** **A car with no seats does not exist, and 3.8 million kilometres is about five return trips to the Moon.**

```python
print("cars with seats == 0        :", (cars["seats"] == 0).sum())
print("cars driven over 1,000,000km:", (cars["km_driven"] > 1_000_000).sum())

cars = cars[(cars["seats"] > 0) & (cars["km_driven"] <= 1_000_000)].reset_index(drop=True)
print("rows kept:", len(cars))
```

**Output:**

```text
cars with seats == 0        : 2
cars driven over 1,000,000km: 2
rows kept: 15240
```

**Four rows out of 15,244 — 0.03% of the data. [§14](#14-back-to-part-as-question) measures what they were costing.**

---

# Part A — Underfitting & Overfitting

**Follows `overfitting_underfitting.ipynb`.**

**Before you can ask whether a number is real, you have to ask whether the model is the right size.** **That question comes first, and it has a simple diagnostic.**

---

# 1. The three fits

🧠 **Analogy: a student preparing for an exam.**
>
> - **The student who skims one chapter** fails the practice questions *and* the exam. **Underfitting.**
> - **The student who memorises last year's paper word for word** scores 100% on last year's paper and fails the new one. **Overfitting.**
> - **The student who understands the subject** does well on both. **A good fit.**

| | What it looks like | Cause |
|---|---|---|
| **Underfitting** | **Bad on training data AND on test data** | The model is **too simple** for the pattern |
| **Good fit** | **Good on both, with a small difference** | The model matches the pattern |
| **Overfitting** | **Excellent on training data, poor on test data** | The model is **too complex** — it memorised |

> **You cannot tell these apart from the test score alone.** **A poor test score is produced by both underfitting and overfitting, and the fixes are opposites.**
>
> **What separates them is the *gap* between the training score and the test score.** **That is the one number to always print.**

---

# 2. Use case 1 — heart failure

**239 training patients, and a decision tree at three different sizes.**

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

configs = [
    ("Underfitting (depth=1)", dict(max_depth=1)),
    ("Good fit (depth=3)",     dict(max_depth=3, min_samples_leaf=10)),
    ("Overfitting (no limit)", dict(max_depth=None, min_samples_leaf=1)),
]

results = []
for name, settings in configs:
    model = DecisionTreeClassifier(random_state=42, **settings).fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results.append({"Model": name,
                    "Train Accuracy": round(train_acc, 4),
                    "Test Accuracy": round(test_acc, 4),
                    "Gap": round(train_acc - test_acc, 4),
                    "F1 (deaths)": round(f1_score(y_test, model.predict(X_test)), 4)})

results_df = pd.DataFrame(results)
print("\nFinal Comparison:")
print(results_df.to_string(index=False))
```

**Output:**

```text
Final Comparison:
                 Model  Train Accuracy  Test Accuracy     Gap  F1 (deaths)
Underfitting (depth=1)          0.8494         0.8333  0.0160       0.6667
    Good fit (depth=3)          0.8828         0.8000  0.0828       0.5714
Overfitting (no limit)          1.0000         0.7500  0.2500       0.5946
```

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(results_df["Model"], results_df["Train Accuracy"], marker="o", label="Training Accuracy")
plt.plot(results_df["Model"], results_df["Test Accuracy"], marker="o", label="Testing Accuracy")
plt.xlabel("Model Type"); plt.ylabel("Accuracy")
plt.title("Underfitting vs Good Fit vs Overfitting")
plt.xticks(rotation=20); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
```

![Heart failure: train and test accuracy against tree depth](images/s8-heart-tree-depth.png)

**Read the columns in order.**

| | Train | Test | **Gap** |
|---|---|---|---|
| depth 1 | 0.8494 | 0.8333 | **+0.016** |
| depth 3 | 0.8828 | 0.8000 | **+0.083** |
| no limit | **1.0000** | **0.7500** | **+0.250** |

> **Training accuracy rises with every step up in capacity — 0.85, 0.88, 1.00.** **It always will.** **More capacity always fits the training data better, which is why the training score on its own tells you nothing.**
>
> **Test accuracy falls — 0.83, 0.80, 0.75.** **And the gap widens from +0.016 to +0.250.**
>
> **The unrestricted tree scored 1.0000 on training data. It has memorised all 239 patients.**

---

# 3. Reading the gap

**This is the interpretation the trainer notebook prints, and it is worth committing to memory.**

```text
Underfitting:
    Low Train score
    Low Test score
    Model too simple

Good Fit:
    High Train score
    High Test score
    Small difference between them

Overfitting:
    Very High Train score
    Lower Test score
    Large gap between Train and Test performance
```

## The diagnosis table

| What you see | Diagnosis | What to do |
|---|---|---|
| Train **low**, test **low** | **Underfitting** | **Add** capacity, features, or training time |
| Train **high**, test **high**, small gap | **Good fit** | Ship it |
| Train **high**, test **lower**, wide gap | **Overfitting** | **Remove** capacity; add data; regularise |
| Train **low**, test **high** | Usually a bug — or a lucky split | Check your split |

> **The single most useful habit in this session: print the train score alongside the test score, every time.**

---

# 4. Why the smallest model won

**Look again at that table. The model labelled "underfitting" has the best test accuracy (0.8333) *and* the best F1 on the patients who died (0.6667).**

> **The labels are the trainer's hypotheses, not verdicts.** **A model with a 1.6-point gap and the best test score is not underfitting — it is the right size for 239 rows.**
>
> **Do not assume a label. Read the numbers.**

## But *why* does one split get 83%?

**A tree with `max_depth=1` uses exactly one column. Which one?**

```python
from sklearn.tree import export_text

stump = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_train, y_train)
print(export_text(stump, feature_names=list(X.columns)))
```

**Output:**

```text
|--- time <= 73.50
|   |--- class: 1
|--- time >  73.50
|   |--- class: 0
```

**`time` is the follow-up period in days. Look at it.**

```python
print("correlation with DEATH_EVENT:", round(heart[["time", "DEATH_EVENT"]].corr().iloc[0, 1], 4))
print("mean follow-up, died     :", round(heart.loc[heart.DEATH_EVENT == 1, "time"].mean(), 1))
print("mean follow-up, survived :", round(heart.loc[heart.DEATH_EVENT == 0, "time"].mean(), 1))
```

**Output:**

```text
correlation with DEATH_EVENT: -0.527
mean follow-up, died     : 70.9
mean follow-up, survived : 158.3
```

## ⚠️ `time` is an outcome artefact

> **Follow-up stopped early *because the patient died*.** **The column is not a predictor of death — it is partly a record of it.**
>
> **Remove it and every model in this session drops from about 0.84 to 0.69** — close to the rate you get by always predicting "survived". *(Measured in [§14](#14-back-to-part-as-question).)*
>
> **No validation method catches this.** **Everything in Part B is scrupulously correct on this data and every method is measuring a column that would not exist when a real prediction is needed.** **The only defence is knowing what your columns mean.**

**We keep `time` for the rest of the session, because the subject here is the machinery rather than the medicine — but now you know what the numbers rest on.**

---

# 5. Use case 2 — car prices

**15,240 cars, predicting `selling_price`. Same experiment, a regression problem, and sixty times as much data.**

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

y_cars = cars["selling_price"]

def fit_and_score(features, **tree_settings):
    X_sub = cars[features]
    a, b, c, d = train_test_split(X_sub, y_cars, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42, **tree_settings).fit(a, c)
    return r2_score(c, model.predict(a)), r2_score(d, model.predict(b))

print("UNDERFIT  ", fit_and_score(["vehicle_age"], max_depth=1))
print("GOOD FIT  ", fit_and_score(["vehicle_age", "km_driven", "engine", "max_power"],
                                  max_depth=5, min_samples_leaf=10))
print("OVERFIT   ", fit_and_score(["vehicle_age", "km_driven", "mileage",
                                   "engine", "max_power", "seats"],
                                  max_depth=None, min_samples_leaf=1))
```

**Output:**

```text
UNDERFIT   (0.0442, 0.0305)
GOOD FIT   (0.8527, 0.6123)
OVERFIT    (0.9991, 0.7049)
```

![Car prices: one 80/20 split](images/s8-fit-spectrum.png)

| Model | Train R² | Test R² | **Gap** |
|---|---|---|---|
| underfit | 0.0442 | 0.0305 | **+0.014** |
| good fit | 0.8527 | 0.6123 | **+0.240** |
| overfit | **0.9991** | **0.7049** | **+0.294** |

## ⚠️ Something is wrong with this table

> **The model labelled "overfitting" has the *best* test score.** **0.7049 against the good-fit model's 0.6123.**
>
> **The gap column still works** — it correctly ranks underfit (+0.014) below good fit (+0.240) below overfit (+0.294). **But the test scores do not rank the models at all.**

## The problem is the split, not the models

**Change nothing but `random_state` and rerun the overfitting model.**

```text
seed 0   test R²  0.6784        seed 5   test R²  0.8596
seed 1   test R²  0.8504        seed 6   test R²  0.2439
seed 2   test R²  0.8599        seed 7   test R²  0.8703
seed 3   test R²  0.6277        seed 8   test R²  0.8682
seed 4   test R²  0.8799        seed 9   test R²  0.8372
```

> **0.2439 to 0.8799 — a swing of 0.64, from nothing but which 3,048 cars happened to land in the test set.**
>
> **The 0.7049 in the table above is one draw from that lottery.** **It is not wrong; it is just not an answer.**

## What Part A can and cannot tell you

| Question | Can one split answer it? |
|---|---|
| **Is this model memorising?** | **Yes — read the gap** |
| **Which of these three models is best?** | **No** |
| **How good is this model, really?** | **No** |

> **This is where Part A hands over.** **You have a diagnostic that works and a measurement that does not.** **[Part B](#part-b--model-validation-techniques) fixes the measurement, and [§14](#14-back-to-part-as-question) comes back and settles this table.**

---

# 6. Fixing each problem

**You do not need cross-validation to know *what* the fixes are.**

## Fixing underfitting

| Fix | Example |
|---|---|
| **More capacity** | `max_depth` from 1 to 10; a forest instead of one tree |
| **More features** | The good-fit car model used 4 features; the underfit one used 1 |
| **Better features** | **[Session 6](session-06-augmentation-feature-engg-red.md#8-what-is-feature-engineering)'s feature engineering** |
| **Less regularisation** | Lower `alpha` in Ridge/Lasso; higher `C` in an SVM |
| **A better target** | **[§15](#15-validation-curves) shows a log transform buying 0.06 of R² without touching the model** |

## Fixing overfitting

| Fix | Example |
|---|---|
| **Less capacity** | `max_depth=10` instead of `None`; `min_samples_leaf=10` instead of 1 |
| **More data** | **If, and only if, the learning curve says it would help — [§16](#16-learning-curves)** |
| **Regularisation** | Ridge/Lasso; `C` in an SVM; `alpha` in a network |
| **Fewer features** | **[Session 6](session-06-augmentation-feature-engg-red.md#13-types-of-feature-reduction)'s feature selection** |
| **Ensembling** | **A Random Forest averages many overfitted trees into one that is not** |
| **Early stopping** | Stop training when validation stops improving |

## The fix on the heart data, measured

```python
for name, settings in [("no limit ", dict(max_depth=None, min_samples_leaf=1)),
                       ("depth 3  ", dict(max_depth=3, min_samples_leaf=10)),
                       ("depth 1  ", dict(max_depth=1))]:
    m = DecisionTreeClassifier(random_state=42, **settings).fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    print(f"{name}  train {tr:.4f}   test {te:.4f}   gap {tr-te:+.4f}")
```

**Output:**

```text
no limit   train 1.0000   test 0.7500   gap +0.2500
depth 3    train 0.8828   test 0.8000   gap +0.0828
depth 1    train 0.8494   test 0.8333   gap +0.0160
```

> **Every step down in capacity raised the test score and shrank the gap.** **On 239 rows, less really is more.**

## ✏️ Practice — diagnosing the fit

1. Build the three heart-failure trees and print the comparison table and the plot. **Which column diagnoses the problem, and which model actually wins?**
2. Print the depth-1 tree with `export_text`. **Which column does it use? Investigate that column and say whether the model should be trusted.**
3. Build the three car models. **Why does the "overfitting" one have the best test score?**
4. Rerun the overfitting car model with `random_state` 0 to 9. **Report the minimum, maximum and range.**
5. **Write two sentences on what a single train/test split can and cannot tell you.**

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import accuracy_score, f1_score, r2_score

heart_url = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
             "refs/heads/main/datasets/classification/heart_failure_raw.csv")
heart = pd.read_csv(heart_url)
heart.loc[heart["age"] > 120, "age"] = np.nan
for c in ["age", "ejection_fraction", "serum_creatinine"]:
    heart[c] = heart[c].fillna(heart[c].median())
le = LabelEncoder()
for c in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
          "treatment_type", "DEATH_EVENT"]:
    heart[c] = le.fit_transform(heart[c])
X = heart.drop(columns=["DEATH_EVENT"]); y = heart["DEATH_EVENT"]
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

rows = []                                                              # 1
for nm, kw in [("Underfitting (depth=1)", dict(max_depth=1)),
               ("Good fit (depth=3)", dict(max_depth=3, min_samples_leaf=10)),
               ("Overfitting (no limit)", dict(max_depth=None, min_samples_leaf=1))]:
    m = DecisionTreeClassifier(random_state=42, **kw).fit(a, c)
    t1 = accuracy_score(c, m.predict(a)); t2 = accuracy_score(d, m.predict(b))
    rows.append({"Model": nm, "Train": round(t1, 4), "Test": round(t2, 4),
                 "Gap": round(t1 - t2, 4), "F1": round(f1_score(d, m.predict(b)), 4)})
print(pd.DataFrame(rows).to_string(index=False))
# The GAP column diagnoses: +0.016, +0.083, +0.250. But the "UNDERFITTING"
# model WINS on test accuracy (0.8333) and F1 (0.6667). The labels are
# hypotheses; a model with a 1.6-point gap and the best score is the
# right size for 239 rows.

stump = DecisionTreeClassifier(max_depth=1, random_state=42).fit(a, c)  # 2
print(export_text(stump, feature_names=list(X.columns)))
print("corr:", round(heart[["time", "DEATH_EVENT"]].corr().iloc[0, 1], 4))
print("mean time | died:", round(heart.loc[heart.DEATH_EVENT == 1, "time"].mean(), 1),
      "| survived:", round(heart.loc[heart.DEATH_EVENT == 0, "time"].mean(), 1))
# It splits on `time`, the follow-up period - which is short BECAUSE the
# patient died. An outcome artefact, not a predictor. The model should
# NOT be trusted for real prediction, and no validation method catches
# it. Only reading the column does.

cars_url = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"       # 3
            "refs/heads/main/datasets/regression/cardekho_preprocessed.csv")
raw = pd.read_csv(cars_url)
cars = raw[(raw["seats"] > 0) & (raw["km_driven"] <= 1_000_000)].reset_index(drop=True)
FE = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
def sc(feats, **kw):
    p, q, r, s = train_test_split(cars[feats], cars["selling_price"],
                                  test_size=.2, random_state=42)
    m = DecisionTreeRegressor(random_state=42, **kw).fit(p, r)
    return round(r2_score(r, m.predict(p)), 4), round(r2_score(s, m.predict(q)), 4)
print("underfit", sc(["vehicle_age"], max_depth=1))
print("good    ", sc(["vehicle_age", "km_driven", "engine", "max_power"],
                     max_depth=5, min_samples_leaf=10))
print("overfit ", sc(FE, max_depth=None, min_samples_leaf=1))
# Because ONE SPLIT cannot rank models on this data. `selling_price` is
# heavily skewed, so R2 is decided by which few luxury cars land in the
# test set. 0.7049 is one draw from a lottery.

scores = []                                                            # 4
for s in range(10):
    p, q, r, t = train_test_split(cars[FE], cars["selling_price"],
                                  test_size=.2, random_state=s)
    m = DecisionTreeRegressor(random_state=42).fit(p, r)
    scores.append(r2_score(t, m.predict(q)))
print([round(v, 4) for v in scores])
print(f"min {min(scores):.4f}  max {max(scores):.4f}  range {max(scores)-min(scores):.4f}")
# 0.2439 to 0.8799 - a range of 0.64 from the seed alone.

# 5 - A single split CAN tell you whether a model is memorising, because
#     the train-test gap is a comparison of two numbers from the SAME
#     split and the split's luck largely cancels. It CANNOT tell you how
#     good a model is, or which of several models is best, because both
#     of those read one test score, and one test score is one draw from
#     a distribution.
```
</details>

---
# Part B — Model Validation Techniques

**Follows `cross_validation.ipynb` and `k_fold_cross_validation.ipynb`.**

**Part A ended with a table it could not read. This part is why, and what to do instead.**

---

# 7. What validation is, and why it is needed

> **Validation is the practice of estimating how a model will perform on data it has not seen — using data it has not seen.**

**That sounds obvious. It is not what beginners do.**

🧠 **Analogy: a driving test.**
>
> **Practising in the same car park every day is *training*.** You get very good at that car park.
>
> **Being tested on that same car park would tell the examiner nothing** — you have memorised where every cone is.
>
> **Validation is a different car park.** **Same skills required, different layout.** Only that answers the real question: *can this person drive?*

## Why it is needed — three reasons

| | The problem | What validation does |
|---|---|---|
| **1. A model can memorise** | A tree with no depth limit scored **1.0000** on its training data in [§2](#2-use-case-1--heart-failure). It had learned nothing general | **Holds out data the model never saw** |
| **2. One score is not an estimate** | The same car model scored **0.2439 and 0.8799** on the same data, depending only on the seed | **Averages over several splits, and reports the spread** |
| **3. You have to choose between models** | Three car models, three test scores, and the scores did not rank them | **Gives a comparison you can defend** |

## The core problem, stated plainly

> **Performance on data the model has seen is not evidence of anything.**
>
> **The purpose of a model is to work on rows that do not exist yet.** **Every technique in this part is a way of simulating that, using the rows you happen to have.**

## What validation cannot do

> ⚠️ **It cannot tell you your data is wrong.**
>
> **[§4](#4-why-the-smallest-model-won)'s `time` column passes every validation method in this part with excellent marks.** **Validation checks that your *measurement procedure* is honest. It cannot check that your *columns* are.**

---

# 8. Validation and testing — the difference

**These two words are used interchangeably in conversation and mean different things in practice. Getting them confused is how good work quietly becomes worthless.**

## The three sets

```text
Full data
├── TRAINING SET     the model learns from this
│                    -> it SEES the answers
│
├── VALIDATION SET   YOU learn from this
│                    -> used to CHOOSE: which model, which hyperparameters
│                    -> looked at MANY times
│
└── TEST SET         nobody learns from this
                     -> used ONCE, at the very end, to report a number
                     -> looked at exactly ONCE
```

## Side by side

| | **Validation set** | **Test set** |
|---|---|---|
| **Purpose** | **To make decisions** — which model, which settings | **To report a final, honest number** |
| **Who uses it** | **You, the developer** | **Your reader** — a client, a regulator, a marker |
| **How often it is used** | **Many times** — once per candidate | **Exactly once** |
| **Does it influence the model?** | **Yes — that is its whole job** | **No. Never** |
| **Is its score unbiased?** | **No** — you picked the winner *because* it did well here | **Yes, if it was used once** |
| **When it is used** | **During** development | **After** development is finished |

## The relationship

> **The validation set is a rehearsal. The test set is opening night.**
>
> **You rehearse as many times as you like. You open once.**

**And the crucial consequence:**

> **A validation score is optimistically biased, always.** **You tried twenty models and kept the one that scored highest on validation — so that score includes the luck of whichever model happened to suit that particular validation data.**
>
> **The test set exists to strip that luck out.** **It can only do that if it was genuinely untouched.**

## ⚠️ How the test set gets ruined

**Nobody sets out to cheat. It happens like this:**

```text
1. Train a model. Test score 0.78.
2. "Hmm, that's low." Try a different model. Test score 0.81.
3. "Better." Tune it. Test score 0.83.
4. Report 0.83.
```

> **Step 4 is a lie, and step 2 is where it started.** **The moment you changed something because of a test score, the test set became a validation set** — and you no longer have an unbiased estimate of anything.
>
> **You cannot un-look at it.** **The only fix is a fresh test set, which usually means new data you do not have.**

## Where cross-validation fits

> **Cross-validation is a way of *building* validation sets when you cannot afford to set one aside permanently.**
>
> **With 299 patients, carving off a separate validation set would leave roughly 180 for training.** **Cross-validation lets every row be training data most of the time and validation data once — so you get a validation estimate without losing rows.**
>
> **It replaces the validation set. It does not replace the test set.**

| Data size | Typical arrangement |
|---|---|
| **Large** (100k+ rows) | **Three fixed sets: train / validation / test** |
| **Small to medium** | **Split off a test set, then cross-validate inside the training half** |
| **Very small** (tens of rows) | Cross-validate everything, and be explicit that you have no clean test set |

**The second row is what the rest of this session does.**

---

# 9. Validation strategies

**Five strategies. You should be able to name all five and say when each one fits.**

| Strategy | How it works | Models trained | Gives you |
|---|---|---|---|
| **Holdout** | **One split**: train on 80%, test on 20% | **1** | One number |
| **K-Fold CV** | **k splits**; every row is tested exactly once | **k** | **Mean ± spread** |
| **Stratified K-Fold** | K-Fold, with **the class balance preserved in every fold** | k | Mean ± spread |
| **Leave-One-Out** | **k = n.** Every row gets its own turn as the test set | **n** | A stable mean |
| **Bootstrapping** | **Resample rows with replacement**; test on what was left out | 100–1000 | **A confidence interval** |

## Holdout

🧠 **One examiner marking one script.** **Fast, and entirely dependent on which examiner you got.**

| ✅ | ❌ |
|---|---|
| **Cheapest possible** — one model | **Unreliable on small data.** [§5](#5-use-case-2--car-prices) measured a 0.64 swing |
| The right choice for the **final** test | Uses only 80% of the data for training |

**Use it for: large datasets, quick checks, and the final untouched test.**

## K-Fold cross-validation

🧠 **Five examiners marking the same script.** **Five marks, averaged, tell you far more — and the spread between them tells you how much to trust the average.**

| ✅ | ❌ |
|---|---|
| **Every row is used for both training and testing** | k times the compute |
| **Reports a spread, not just a mean** | |

**Use it for: almost everything.**

## Stratified K-Fold

**K-Fold with one addition: every fold keeps the same class proportions as the whole dataset.**

> **On a 203/96 target, a random fold can drift to 3:1 while another sits at 1.5:1 — and each fold is then measuring a slightly different problem.**

**Use it for: every classification problem. There is no reason not to.**

## Leave-One-Out (LOOCV)

🧠 **Every single student marks the script.** **Exhaustive, deterministic, and expensive.**

| ✅ | ❌ |
|---|---|
| **Maximum training data** — each model sees n−1 rows | **n models.** 299 here; 100,000 on a large dataset |
| **No randomness at all** — run it twice, get the same answer | The n models are nearly identical, so their errors are correlated |

**Use it for: genuinely tiny datasets, where holding out 20% would leave nothing to test on.**

## Bootstrapping

**Sample rows *with replacement* to build a training set of the same size, and test on whatever was never drawn.**

🧠 **Drawing names from a hat and putting each one back.** **Some names get drawn twice, some not at all.** **The ones never drawn are your test set.**

> **On average each bootstrap sample contains about 63.2% of the unique rows, leaving roughly 36.8% over.** **Those left-over rows are called *out-of-bag*, and they are free test data.**

| ✅ | ❌ |
|---|---|
| **Gives a confidence interval**, not just a point estimate | Training rows are duplicated, which biases some models |
| Works on very small datasets | Expensive — hundreds of models |

**Use it for: when you need to state uncertainty as a range.**

> **You have already used bootstrapping without knowing it.** **A Random Forest bootstraps its rows for every tree — that is where the "bagging" in bagged trees comes from.**

## Choosing

| Situation | Use |
|---|---|
| **Large data**, or the final untouched test | **Holdout** |
| **The default for everything else** | **K-Fold, k=5** |
| **Any classification problem** | **Stratified K-Fold** |
| **Very small data** (tens of rows) | **LOOCV** |
| You must state uncertainty as a range | **Bootstrap** |

---

# 10. K-Fold cross-validation

> **Instead of one split, make k of them — and let every row be in the test set exactly once.**

```text
5-fold cross-validation, 299 rows:

fold 1:  [TEST ][         TRAIN          ]   -> score 1
fold 2:  [ TRAIN ][TEST][     TRAIN      ]   -> score 2
fold 3:  [    TRAIN    ][TEST][  TRAIN   ]   -> score 3
fold 4:  [        TRAIN       ][TEST][TR ]   -> score 4
fold 5:  [           TRAIN         ][TEST]   -> score 5

Five models are trained. Every row is tested exactly once.
The answer is the MEAN of the five scores; the SPREAD is the uncertainty.
```

**`k_fold_cross_validation.ipynb` splits off a test set first, then cross-validates inside `X_train`. That is [§8](#8-validation-and-testing--the-difference)'s arrangement, in code.**

```python
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

svm_model = make_pipeline(MinMaxScaler(), SVC())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(svm_model, X_train, y_train, cv=kf)

print(kf_scores.round(4))
print("K-Fold Cross-Validation Accuracy:", round(kf_scores.mean(), 4))
```

**Output:**

```text
[0.6667 0.8958 0.6875 0.7917 0.7234]
K-Fold Cross-Validation Accuracy: 0.753
```

## Read the five fold scores before the mean

> **0.6667 in one fold and 0.8958 in another — a 23-point range, on the same model and the same data.**
>
> **The mean of 0.753 is a reasonable estimate. The spread says how much to trust it — and here it says: not much.**

## Reading the two numbers

| | What it tells you |
|---|---|
| **Mean** | Your best estimate of performance |
| **Standard deviation** | **How much to trust the mean** |

> **Always report both.** **"0.75 ± 0.08" is a statement. "0.75" is a claim you cannot support.**
>
> **And when comparing two models: if their means differ by less than the spread, you have not shown a difference.**

## ⚠️ `shuffle=True` is not optional

**Without it, `KFold` takes the rows in file order.** **If the file is sorted — by date, by class, by hospital — each fold gets a systematically different slice**, and the scores become meaningless.

```python
# illustrative: a syntax reference, not runnable as written.
KFold(n_splits=5)                                    # file order - risky
KFold(n_splits=5, shuffle=True, random_state=42)     # always prefer this
```

## How many folds?

| k | Trade-off |
|---|---|
| **3** | Fast; each model trains on only 67% of the data |
| **5** | **The usual default.** A good balance |
| **10** | Twice the cost; on 239 rows each test fold is only 24 patients, so the fold scores get *noisier* |
| **n (LOOCV)** | Maximum training data, maximum cost — [§12](#12-leave-one-out-and-all-four-side-by-side) |

---

# 11. Stratified K-Fold

**Plain `KFold` splits at random. On imbalanced data, that is the problem the 23-point range in §10 was showing you.**

```python
from sklearn.model_selection import StratifiedKFold

class_distribution = y.value_counts()
print(class_distribution)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(svm_model, X_train, y_train, cv=skf)

print(skf_scores.round(4))
print("Stratified K-Fold Cross-Validation Accuracy:", round(skf_scores.mean(), 4))
```

**Output:**

```text
0    203
1     96
Name: count, dtype: int64

[0.7292 0.8125 0.7708 0.7708 0.8723]
Stratified K-Fold Cross-Validation Accuracy: 0.7911
```

## Compare the two directly

| | Fold scores | Mean | **Range** |
|---|---|---|---|
| **`KFold`** | 0.6667 … 0.8958 | 0.7530 | **0.229** |
| **`StratifiedKFold`** | 0.7292 … 0.8723 | **0.7911** | **0.143** |

> **The target is 203 to 96 — roughly 2:1. A random fold can easily come out 3:1.**
>
> **Part of what plain `KFold` reported as variation was not about the model at all — it was about how the classes happened to fall.** **Stratifying removes it.**

> ✅ **Rule: for classification, always use `StratifiedKFold`.**
>
> **`cross_val_score` uses it automatically when you pass `cv=5` with a classifier** — but write it out so a reader can see the decision was made.

---

# 12. Leave-One-Out and all four side by side

**`cross_validation.ipynb` runs all four methods on the full dataset in one cell. This is that cell.**

```python
from sklearn.model_selection import LeaveOneOut

# Holdout Method
model = make_pipeline(MinMaxScaler(), SVC()).fit(X_train, y_train)
holdout_score = model.score(X_test, y_test)
print("Holdout Method Accuracy:", round(holdout_score, 4))

# Leave-One-Out
loo = LeaveOneOut()
loo_scores = cross_val_score(make_pipeline(MinMaxScaler(), SVC()), X, y, cv=loo, n_jobs=-1)
print("Leave-One-Out Cross-Validation Accuracy:", round(loo_scores.mean(), 4),
      f"({len(loo_scores)} models)")

# K-Fold Cross-Validation (k=5)
kf_all = cross_val_score(make_pipeline(MinMaxScaler(), SVC()), X, y, cv=kf)
print("K-Fold Cross-Validation Accuracy:", round(kf_all.mean(), 4))

# Stratified K-Fold Cross-Validation (k=5)
skf_all = cross_val_score(make_pipeline(MinMaxScaler(), SVC()), X, y, cv=skf)
print("Stratified K-Fold Cross-Validation Accuracy:", round(skf_all.mean(), 4))
```

**Output:**

```text
Holdout Method Accuracy: 0.7667
Leave-One-Out Cross-Validation Accuracy: 0.786 (299 models)
K-Fold Cross-Validation Accuracy: 0.7492
Stratified K-Fold Cross-Validation Accuracy: 0.7693
```

![Four ways to measure the same model](images/s8-cv-variants.png)

**Four numbers for one model, spanning 3.7 percentage points.**

| Method | Score | Models | Why it differs |
|---|---|---|---|
| **Holdout** | 0.7667 | **1** | One arbitrary split |
| **LOOCV** | **0.7860** | **299** | **Highest — each model trains on 298 rows instead of 239** |
| **K-Fold** | 0.7492 | 5 | Lowest — the class balance drifted between folds |
| **Stratified** | 0.7693 | 5 | **The one to trust: equal fold sizes, equal class balance** |

> **More training data raises the score, which is why LOOCV is the highest.**
>
> **None of these four is "the" accuracy.** **The honest sentence is: "roughly 0.77, and the method you choose moves it by about 4 points."**

⚠️ **LOOCV's individual scores are all 0 or 1** — one patient is either classified correctly or not. **Only the mean is meaningful.**

---

# 13. Model selection with cross-validation

**Now the payoff. `k_fold_cross_validation.ipynb` ends by using cross-validation to *choose between models* — which is what it is for.**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
}

cv_scores = {}
for model_name, model in models.items():
    scores = cross_val_score(make_pipeline(MinMaxScaler(), model), X_train, y_train, cv=skf)
    cv_scores[model_name] = scores.mean()
    print(f"{model_name:<22} {scores.mean():.4f}  +/- {scores.std():.4f}")

best_model_name = max(cv_scores, key=cv_scores.get)
print(f"\nBest Model: {best_model_name} with CV Mean Accuracy: {cv_scores[best_model_name]:.4f}")
```

**Output:**

```text
SVM                    0.7911  +/- 0.0484
KNN                    0.7158  +/- 0.0557
Random Forest          0.8413  +/- 0.0532
Gaussian Naive Bayes   0.7785  +/- 0.0619

Best Model: Random Forest with CV Mean Accuracy: 0.8413
```

> ⚠️ **Read the spreads before declaring a winner.** **Random Forest 0.8413 ± 0.0532 against SVM 0.7911 ± 0.0484: those intervals overlap.** **On 239 training rows this is a real result but not an overwhelming one.**
>
> **And note what has *not* happened: the test set has not been touched.** All four models were compared on cross-validated training data.

## Now train the winner and test it — once

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

final_model = make_pipeline(MinMaxScaler(), models[best_model_name]).fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Output:**

```text
Test accuracy: 0.8

Confusion Matrix:
[[38  3]
 [ 9 10]]

              precision    recall  f1-score   support
           0       0.81      0.93      0.86        41
           1       0.77      0.53      0.62        19
    accuracy                           0.80        60
   macro avg       0.79      0.73      0.74        60
weighted avg       0.80      0.80      0.79        60
```

> **That is the complete, honest workflow: choose on validation, report on test, once.**
>
> ⚠️ **But do not stop at "0.80".** **Recall on class 1 is 0.53 — the model found 10 of the 19 patients who died and missed 9.** **[Session 5B](session-05b-classification.md)'s lesson, arriving in a real workflow.** [Part C](#part-c--hyperparameter-tuning) attacks that.

---

# 14. Back to Part A's question

**[§5](#5-use-case-2--car-prices) left a table that could not be read. Cross-validation settles it.**

```python
from sklearn.model_selection import cross_validate

FEATURES = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_score(features, **tree_settings):
    r = cross_validate(DecisionTreeRegressor(random_state=42, **tree_settings),
                       cars[features], cars["selling_price"],
                       cv=kf5, scoring="r2", return_train_score=True, n_jobs=-1)
    return r["train_score"].mean(), r["test_score"].mean(), r["test_score"].std()

for name, feats, kw in [
        ("underfit", ["vehicle_age"], dict(max_depth=1)),
        ("good fit", ["vehicle_age", "km_driven", "engine", "max_power"],
         dict(max_depth=5, min_samples_leaf=10)),
        ("overfit ", FEATURES, dict(max_depth=None, min_samples_leaf=1)),
        ("depth 10", FEATURES, dict(max_depth=10, min_samples_leaf=10))]:
    tr, cv_, sd = cv_score(feats, **kw)
    print(f"{name}  train {tr:.4f}  CV {cv_:.4f} +/- {sd:.4f}  gap {tr-cv_:+.4f}")
```

**Output:**

```text
underfit  train 0.0400  CV 0.0414 +/- 0.0059  gap -0.0014
good fit  train 0.8057  CV 0.7652 +/- 0.0812  gap +0.0406
overfit   train 0.9993  CV 0.8276 +/- 0.0646  gap +0.1717
depth 10  train 0.8528  CV 0.8069 +/- 0.0926  gap +0.0459
```

## The table, finally readable

| Model | Train | **CV** | Gap | Verdict |
|---|---|---|---|---|
| underfit | 0.0400 | 0.0414 | **−0.001** | **Underfitting, confirmed** |
| good fit | 0.8057 | 0.7652 | +0.041 | Reasonable |
| **overfit** | **0.9993** | **0.8276** | **+0.172** | **Memorising — and still the highest CV** |
| depth 10 | 0.8528 | 0.8069 | +0.046 | **Nearly as good, with a quarter of the gap** |

> **The single split said the overfitting model scored 0.7049. Cross-validation says 0.8276.** **The 0.7049 was a bad draw, not a property of the model.**
>
> **And the result that surprises people: the unrestricted tree really does have the highest CV score.** **Its gap of +0.17 correctly says it is memorising. That is not the same as saying it generalises worst.**

## Overfitting is capacity *relative to data volume*

**Take exactly the same unrestricted tree and give it less data.**

```python
for n in [300, 1000, 3000, len(cars)]:
    subset = cars.sample(n=n, random_state=42) if n < len(cars) else cars
    r = cross_validate(DecisionTreeRegressor(random_state=42),
                       subset[FEATURES], subset["selling_price"],
                       cv=kf5, scoring="r2", return_train_score=True, n_jobs=-1)
    print(f"n={n:>6}   train {r['train_score'].mean():.4f}   CV {r['test_score'].mean():>8.4f}")
```

**Output:**

```text
n=   300   train 0.9998   CV   0.4080
n=  1000   train 1.0000   CV  -0.2062
n=  3000   train 0.9999   CV   0.6678
n= 15240   train 0.9993   CV   0.8276
```

> **At 1,000 rows the unrestricted tree scores CV R² of −0.21 — *worse than predicting the average price for every car*.** Train was 1.0000.
>
> **Same algorithm. Same features. Only the number of rows changed.**
>
> **This is the honest definition:** **overfitting is not a property of a model. It is a relationship between a model's capacity and the amount of data you have.** **A depth limit that is essential at 1,000 rows is unnecessary at 15,000.**

## And what four impossible rows were costing

**Remember the two cars with `seats = 0` and the two driven over a million kilometres.**

![What four impossible rows cost](images/s8-impossible-rows-cost.png)

| Model | **Raw** CV R² | **Cleaned** CV R² | Raw fold-to-fold spread |
|---|---|---|---|
| underfit | 0.0421 | 0.0414 | 0.009 |
| good fit | 0.7605 | 0.7652 | 0.065 |
| **overfit** | **0.6612** | **0.8276** | **0.227** |
| depth 10 | 0.8045 | 0.8069 | 0.077 |

> **Four rows out of 15,244 — 0.03% of the data — were worth 0.17 of R² to the deepest model, and were tripling its fold-to-fold spread.**
>
> **Why the deepest model most?** **A tree with no depth limit will happily build a branch for a single 3.8-million-kilometre car.** Whichever fold that car lands in gets a wild prediction, and R² punishes it heavily.
>
> **This is why Session 3's sequence comes before any of this.** **You cannot validate honestly on data you have not checked.**

## And the `time` column, measured

```python
from sklearn.ensemble import RandomForestClassifier

for label, data in [("with time   ", X), ("without time", X.drop(columns=["time"]))]:
    stump_cv = cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=42),
                               data, y, cv=skf).mean()
    forest_cv = cross_val_score(make_pipeline(MinMaxScaler(),
                                RandomForestClassifier(random_state=42)), data, y, cv=skf).mean()
    print(f"{label}   one-split tree {stump_cv:.4f}   Random Forest {forest_cv:.4f}")
```

**Output:**

```text
with time      one-split tree 0.8394   Random Forest 0.8360
without time   one-split tree 0.6889   Random Forest 0.6889
```

> **Every model drops from ~0.84 to 0.69 when `time` is removed.**
>
> **Cross-validation did not object to `time` even slightly.** **Validation makes your measurement honest. It cannot make your data honest.**

---

# 15. Validation curves

> **A validation curve plots one hyperparameter against train and validation performance. It shows you where the good fit lives.**

**This is [§9](#9-validation-strategies)'s k-fold applied at every setting of one hyperparameter.**

```python
from sklearn.model_selection import validation_curve

X_cars, y_cars = cars[FEATURES], cars["selling_price"]

depths = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30]
train_scores, cv_scores_ = validation_curve(
    DecisionTreeRegressor(random_state=42), X_cars, y_cars,
    param_name="max_depth", param_range=depths, cv=kf5, scoring="r2", n_jobs=-1)

print(cv_scores_.mean(axis=1).round(4))
```

**Output:**

```text
[0.4388 0.4486 0.5385 0.6098 0.6794 0.8451 0.6926 0.7775 0.6846 0.8144]
```

> **That is not a curve. It is a zigzag.** **0.61 at depth 5, 0.85 at depth 10, 0.69 at depth 12, 0.81 at depth 30.** **You cannot choose a depth from it.**

## ⚠️ An unreadable curve is telling you something

**The problem is not the method. It is the target.**

```python
print(cars["selling_price"].describe().round(0))
```

**Output:**

```text
count       15240.0
mean       771930.0
50%        559000.0
max      39500000.0
```

> **The median car is ₹559,000 and the most expensive is ₹39,500,000 — 70× the median.** **R² is a sum of *squared* errors, so a handful of luxury cars dominate it completely.**

**Session 6's fix for a skewed column applies to a skewed target too: take the log.**

```python
y_log = np.log(y_cars)

train_scores, cv_scores_ = validation_curve(
    DecisionTreeRegressor(random_state=42), X_cars, y_log,
    param_name="max_depth", param_range=depths, cv=kf5, scoring="r2", n_jobs=-1)

for d, tr, cv_ in zip(depths, train_scores.mean(axis=1), cv_scores_.mean(axis=1)):
    print(f"depth {d:>3}   train {tr:.4f}   CV {cv_:.4f}   gap {tr-cv_:+.4f}")
```

**Output:**

```text
depth   1   train 0.4205   CV 0.4202   gap +0.0003
depth   2   train 0.6002   CV 0.5976   gap +0.0026
depth   3   train 0.7395   CV 0.7361   gap +0.0034
depth   5   train 0.8574   CV 0.8510   gap +0.0064
depth   8   train 0.9235   CV 0.9031   gap +0.0204
depth  10   train 0.9451   CV 0.9083   gap +0.0368
depth  12   train 0.9619   CV 0.9053   gap +0.0566
depth  15   train 0.9794   CV 0.8970   gap +0.0824
depth  20   train 0.9928   CV 0.8857   gap +0.1071
depth  30   train 0.9960   CV 0.8813   gap +0.1147
```

![Raw price versus log price](images/s8-validation-curve.png)

> **The same sweep, the same folds, the same model. Only the target changed — and now it is the textbook shape.**

## How to read one

```text
        train ────────────────────────────  keeps rising, always
                  ╱‾‾‾‾‾╲
        CV      ╱         ╲                 rises, peaks, then FALLS
              ╱             ╲
        ─────┴───────┴───────┴──────
         too simple  BEST   too complex
        (underfit)         (overfit)
```

| Region | Train | CV | Gap | Name |
|---|---|---|---|---|
| **Depth 1–3** | Low | Low | **~0.00** | **Underfitting** |
| **Depth 10** | 0.9451 | **0.9083** | +0.04 | **The setting you want** |
| **Depth 20–30** | 0.99+ | **Falling** | **+0.11** | **Overfitting** |

> **The train curve never turns down.** **Only the validation curve turns, and where it turns is the answer.**
>
> **And watch the gap column: it grows monotonically from +0.0003 to +0.1147.** **The gap is the amount of memorising, and it rises with capacity whether or not the CV score has started to fall.**

---

# 16. Learning curves

> **A validation curve asks "is my model the right complexity?". A learning curve asks: *would more data help?***

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve

sizes, train_scores, cv_scores_ = learning_curve(
    RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1),
    X_cars, y_cars, train_sizes=np.linspace(0.1, 1.0, 6),
    cv=kf5, scoring="r2", n_jobs=-1, shuffle=True, random_state=42)

for n, tr, cv_ in zip(sizes, train_scores.mean(axis=1), cv_scores_.mean(axis=1)):
    print(f"rows {n:>6}   train {tr:.4f}   CV {cv_:.4f}   gap {tr-cv_:+.4f}")
```

**Output:**

```text
rows   1219   train 0.9658   CV 0.7926   gap +0.1732
rows   3413   train 0.9748   CV 0.8137   gap +0.1611
rows   5608   train 0.9800   CV 0.8600   gap +0.1199
rows   7802   train 0.9780   CV 0.8581   gap +0.1199
rows   9997   train 0.9784   CV 0.8695   gap +0.1088
rows  12192   train 0.9801   CV 0.8736   gap +0.1065
```

![Learning curve — has it flattened?](images/s8-learning-curve.png)

| Shape | Meaning | What to do |
|---|---|---|
| **CV still climbing at the right edge** | The model is starved of data | **Collect more rows** |
| **CV has flattened, gap small** | Enough data and the right model | **Ship it** |
| **CV has flattened, gap still large** | **More data will not close this** | **Regularise, or simplify** |
| **Both curves low and flat** | Underfitting | **A more capable model** |

> **Here: CV climbed from 0.793 to 0.874 and is nearly flat over the last two points — the final 2,200 cars bought 0.004.**
>
> **Collecting more cars would be close to wasted effort.** **The remaining gap of 0.11 is a model problem, and [§15](#15-validation-curves)'s curve is where you fix it.**

---

# 17. The two leaks that make every number a lie

**Every number in this part used `make_pipeline(MinMaxScaler(), SVC())` rather than scaling first. That was deliberate.**

## Leak 1 — scaling before the split

**`cross_validation.ipynb` scales like this:**

```python
# illustrative: this is what NOT to do.
scaler = MinMaxScaler()
scaler.fit(X)              # <- sees every row, including the test rows
X = scaler.transform(X)
```

```python
X_scaled_all = MinMaxScaler().fit_transform(X)
leaky = cross_val_score(SVC(), X_scaled_all, y, cv=skf)
correct = cross_val_score(make_pipeline(MinMaxScaler(), SVC()), X, y, cv=skf)

print("leaky  :", round(leaky.mean(), 4))
print("correct:", round(correct.mean(), 4))
```

**Output:**

```text
leaky  : 0.776
correct: 0.7693
```

> **The gap is small — 0.7 of a point — and that is exactly what makes it dangerous.**
>
> **`MinMaxScaler` uses each column's minimum and maximum, so a single extreme test-set patient shifts the scaling of every training row.** **A leak does not reliably inflate your score, so you cannot detect one by looking for a suspiciously high number.** The result is simply *not the number you think it is*.

## Leak 2 — resampling before the split

**`hyperparameter_session.ipynb` applies SMOTE to the whole dataset and then splits. This one is far more serious.**

```python
# needs-install: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)     # BEFORE the split
a, b, c, d = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

original_rows = set(map(tuple, X.to_numpy()))
synthetic_in_test = sum(1 for row in b.to_numpy() if tuple(row) not in original_rows)

print("rows before SMOTE:", len(X), " after:", len(X_res))
print(f"test rows: {len(b)}, of which SYNTHETIC: {synthetic_in_test} "
      f"({synthetic_in_test / len(b):.0%})")
```

**Output:**

```text
rows before SMOTE: 299  after: 406
test rows: 82, of which SYNTHETIC: 19 (23%)
```

> **23% of the test set is invented data** — each synthetic row interpolated from real patients, most of whom are now in the training set.
>
> **You are testing the model on rows built out of its own training data.** **Whatever number comes out is not an estimate of anything.**

**The fix — split first, then resample the training half only:**

```python
from imblearn.over_sampling import SMOTE

X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)   # TRAIN only

print("train rows:", len(X_train), "->", len(X_bal))
print("balance now:", y_bal.value_counts().to_dict())

model = make_pipeline(MinMaxScaler(), SVC()).fit(X_bal, y_bal)
print("honest test accuracy:", round(model.score(X_test, y_test), 4))
```

**Output:**

```text
train rows: 239 -> 324
balance now: {0: 162, 1: 162}
honest test accuracy: 0.8167
```

> **This is [Session 6](session-06-augmentation-feature-engg-red.md#3-why-use-augmentation)'s rule, restated: augment the training set, never the test set.** **[Part C](#18-the-tuning-setup) uses this corrected order throughout.**

## The one habit that prevents both

> **Put every step that *learns something from the data* inside a `Pipeline`, and let cross-validation drive the pipeline.**

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", MinMaxScaler()),      # learns min and max      -> must be inside
    ("model", SVC()),                # learns everything else  -> must be inside
])
print(round(cross_val_score(pipe, X, y, cv=skf).mean(), 4))
```

**Scaling, imputation, encoding, feature selection and PCA all learn from data.** **All of them belong inside the pipeline. Structure beats discipline.**

## ✏️ Practice — validation

1. **In your own words, explain the difference between a validation set and a test set** — and say what goes wrong if you use the test set to choose a model.
2. Cross-validate the SVM on `X_train` with plain `KFold` and with `StratifiedKFold`. **Report both fold arrays. Which has the smaller range, and what was that extra variation measuring?**
3. Run all four methods from §12. **Why is LOOCV the highest?**
4. Compare 3-, 5- and 10-fold. **Does more folds mean a better estimate?**
5. Cross-validate four models on `X_train`, pick the best, and evaluate it once on the test set. **Report the classification report and say whether you would deploy it.**
6. Cross-validate Part A's three car models. **Does the ranking change? Explain what §5's single split got wrong.**
7. Build the scaling leak, measure it, then fix it with a pipeline. **Report both numbers and say why the small gap is the dangerous part.**

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate,
                                     KFold, StratifiedKFold, LeaveOneOut)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report

heart_url = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
             "refs/heads/main/datasets/classification/heart_failure_raw.csv")
heart = pd.read_csv(heart_url)
heart.loc[heart["age"] > 120, "age"] = np.nan
for c in ["age", "ejection_fraction", "serum_creatinine"]:
    heart[c] = heart[c].fillna(heart[c].median())
le = LabelEncoder()
for c in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
          "treatment_type", "DEATH_EVENT"]:
    heart[c] = le.fit_transform(heart[c])
X = heart.drop(columns=["DEATH_EVENT"]); y = heart["DEATH_EVENT"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)
pipe = lambda m=None: make_pipeline(MinMaxScaler(), m or SVC())
kf = KFold(5, shuffle=True, random_state=42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)

# 1 - The VALIDATION set is used to CHOOSE (many times); the TEST set is
#     used to REPORT (once). A validation score is optimistically biased
#     because you kept the model that happened to do best on it. If you
#     use the test set to choose, that bias moves into the test score and
#     you no longer have an unbiased estimate of anything - and you
#     cannot un-look at it.

a = cross_val_score(pipe(), X_train, y_train, cv=kf)                    # 2
b = cross_val_score(pipe(), X_train, y_train, cv=skf)
print("KFold     :", a.round(4), f"range {a.max()-a.min():.4f}")
print("Stratified:", b.round(4), f"range {b.max()-b.min():.4f}")
# Stratified has the smaller range (0.143 vs 0.229). The extra variation
# in plain KFold was measuring the SPLIT, not the model: the 203/96 class
# balance drifted from fold to fold.

print("holdout   :", round(pipe().fit(X_train, y_train)                # 3
                            .score(X_test, y_test), 4))
loo = cross_val_score(pipe(), X, y, cv=LeaveOneOut(), n_jobs=-1)
print("LOOCV     :", round(loo.mean(), 4), f"({len(loo)} models)")
print("KFold     :", round(cross_val_score(pipe(), X, y, cv=kf).mean(), 4))
print("Stratified:", round(cross_val_score(pipe(), X, y, cv=skf).mean(), 4))
# LOOCV is highest because each of its 299 models trains on 298 rows,
# while each 5-fold model trains on 239. More training data, better score.

for k in [3, 5, 10]:                                                   # 4
    s = cross_val_score(pipe(), X, y,
                        cv=StratifiedKFold(k, shuffle=True, random_state=42))
    print(f"{k:>2}-fold  mean {s.mean():.4f}  std {s.std():.4f}")
# The mean barely moves. The std does NOT reliably shrink: at k=10 each
# test fold is only 30 patients, so the fold scores get noisier even as
# the mean stabilises. 5 is the sensible default.

models = {"SVM": SVC(random_state=42), "KNN": KNeighborsClassifier(),  # 5
          "Random Forest": RandomForestClassifier(random_state=42),
          "Gaussian Naive Bayes": GaussianNB()}
cvs = {}
for nm, m in models.items():
    s = cross_val_score(make_pipeline(MinMaxScaler(), m), X_train, y_train, cv=skf)
    cvs[nm] = s.mean(); print(f"{nm:<22} {s.mean():.4f} +/- {s.std():.4f}")
best = max(cvs, key=cvs.get)
final = make_pipeline(MinMaxScaler(), models[best]).fit(X_train, y_train)
print("\nbest:", best, "| test:", round(accuracy_score(y_test, final.predict(X_test)), 4))
print(classification_report(y_test, final.predict(X_test)))
# Random Forest wins CV and scores 0.80 on test - but recall on the
# patients who died is only 0.53. It misses 9 of 19. NO, I would not
# deploy it: for a clinical tool a miss is the expensive error.

cars_url = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"  # 6
            "refs/heads/main/datasets/regression/cardekho_preprocessed.csv")
raw = pd.read_csv(cars_url)
cars = raw[(raw["seats"] > 0) & (raw["km_driven"] <= 1_000_000)].reset_index(drop=True)
FE = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
for nm, f, kw in [("underfit", ["vehicle_age"], dict(max_depth=1)),
                  ("good fit", ["vehicle_age", "km_driven", "engine", "max_power"],
                   dict(max_depth=5, min_samples_leaf=10)),
                  ("overfit ", FE, dict(max_depth=None, min_samples_leaf=1))]:
    r = cross_validate(DecisionTreeRegressor(random_state=42, **kw), cars[f],
                       cars["selling_price"], cv=kf, scoring="r2",
                       return_train_score=True, n_jobs=-1)
    print(f"{nm}  train {r['train_score'].mean():.4f}  CV {r['test_score'].mean():.4f}"
          f"  gap {r['train_score'].mean()-r['test_score'].mean():+.4f}")
# The overfitting model goes from 0.7049 on one split to 0.8276 across
# five - the single split had simply drawn a bad test set. The RANKING
# is unchanged, but now it is supported by five measurements and a
# spread rather than one lucky or unlucky draw.

leaky = cross_val_score(SVC(), MinMaxScaler().fit_transform(X), y, cv=skf)  # 7
correct = cross_val_score(pipe(), X, y, cv=skf)
print("leaky  :", round(leaky.mean(), 4), " correct:", round(correct.mean(), 4))
# 0.7760 vs 0.7693 - the leak made the score HIGHER by less than a point.
# That is what makes it dangerous: a leak does not reliably inflate, so
# you cannot spot one by looking for a suspiciously good number. The
# number is simply not measuring what you think it is.
```
</details>

---
# Part C — Hyperparameter Tuning

**[Part A](#part-a--underfitting--overfitting) showed that `max_depth=1` beats `max_depth=None` on the heart data. Part C is about how to *find* that setting without cheating.**

**This part follows `hyperparameter_session.ipynb`, in its order: set up, kNN by hand, then grid, then random, then the same for an SVM, then a pipeline, then Bayesian optimization.**

---

# 18. The tuning setup

**The notebook's setup is SMOTE, then a train/test split, then scaling. [§17](#17-the-two-leaks-that-make-every-number-a-lie) measured why that order leaves 23% of the test set synthetic.**

**Here is the same setup in the correct order.**

```python
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# 1. SPLIT FIRST - the test set is now sealed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("train:", X_train.shape, y_train.value_counts().to_dict())

# 2. BALANCE the training half only
X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
print("\nClass Distribution After SMOTE:")
print(y_bal.value_counts())

# 3. SCALE - fitted inside a pipeline, so every fold refits it
```

**Output:**

```text
train: (239, 13) {0: 162, 1: 77}

Class Distribution After SMOTE:
0    162
1    162
Name: count, dtype: int64
```

| Step | Notebook's order | **Correct order** | Why |
|---|---|---|---|
| SMOTE | **1st** | **2nd** | Synthetic rows must not reach the test set |
| Split | 2nd | **1st** | |
| Scale | 3rd, on `X_train` | **Inside the pipeline** | The scaler must refit on every CV fold |

> **Everything below tunes on `X_bal` / `y_bal` with cross-validation, and touches `X_test` only to report a final number.**

## A note on `n_jobs`

```python
import multiprocessing
print("cores:", multiprocessing.cpu_count())
```

**Output:** `cores: 8`

> **`n_jobs=-1` uses every core.** **A grid search is embarrassingly parallel — each combination is independent — so this is nearly free speed.** **Bayesian optimization is the exception: it chooses each trial based on the last one, so it cannot parallelise the same way.**

---

# 19. Parameters vs hyperparameters

**Two words that sound identical and mean opposite things.**

| | **Parameters** | **Hyperparameters** |
|---|---|---|
| Who sets them | **The model, during `fit()`** | **You, before `fit()`** |
| Learned from data? | **Yes** | **No** |
| Examples | Regression coefficients; a tree's split points | `max_depth`, `n_neighbors`, `C`, `gamma` |
| Changed by | Training on different data | **You, deliberately** |

```python
# illustrative: a syntax reference, not runnable as written.
model = DecisionTreeClassifier(max_depth=10)   # <- HYPERparameter: your choice
model.fit(X_train, y_train)                    # <- parameters: learned here
model.tree_.threshold                          # <- the learned split points
```

> **Hyperparameter tuning means searching over the choices *you* make**, so the search has to happen *outside* training — which is why every method below wraps `fit()` in a loop.

| Model | Hyperparameter | What it controls |
|---|---|---|
| **kNN** | `n_neighbors` | **How many neighbours vote** |
| **SVM** | `C`, `kernel`, `gamma` | How hard it tries to fit; the boundary's shape |
| **Decision tree** | `max_depth`, `min_samples_leaf` | **How much it can memorise** |
| **Random Forest** | `n_estimators`, `max_depth`, `criterion` | How many trees, how deep, how splits are scored |
| **PCA** | `n_components` | How many dimensions survive |

---

# 20. Manual search — and the trap in it

**The obvious approach: try every value and pick the best.**

```python
from sklearn.neighbors import KNeighborsClassifier

scaler = MinMaxScaler().fit(X_bal)
X_bal_s, X_test_s = scaler.transform(X_bal), scaler.transform(X_test)

train_score, test_score = [], []
k_values = range(1, 20)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_bal_s, y_bal)
    train_score.append(knn.score(X_bal_s, y_bal))
    test_score.append(knn.score(X_test_s, y_test))

best_k = test_score.index(max(test_score)) + 1
print(f"Best k: {best_k}")
print(f"Train score with k = {best_k}: {train_score[best_k - 1]:.4f}")
print(f"Test score with k = {best_k}: {test_score[best_k - 1]:.4f}")
```

**Output:**

```text
Best k: 11
Train score with k = 11: 0.7623
Test score with k = 11: 0.8333
```

**Two things to notice.**

**First, `k=1` scores 1.0000 on training data.** **Of course it does — the nearest neighbour of a training point is itself.** **Its test score is 0.6833, among the worst of the nineteen.** That is overfitting in its purest form.

**Second — and this is the trap — `best_k` was chosen by looking at the test score.**

## ⚠️ The test set is not allowed to vote

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

grid_search = GridSearchCV(
    make_pipeline(MinMaxScaler(), KNeighborsClassifier()),
    {"kneighborsclassifier__n_neighbors": list(range(1, 20))},
    cv=5)
grid_search.fit(X_bal, y_bal)

print("k chosen by peeking at test :", best_k, "-> reported", round(max(test_score), 4))
print("k chosen by CV on train only:", grid_search.best_params_,
      "-> CV", round(grid_search.best_score_, 4))
print("that model's HONEST test score:", round(grid_search.score(X_test, y_test), 4))
```

**Output:**

```text
k chosen by peeking at test : 11 -> reported 0.8333
k chosen by CV on train only: {'kneighborsclassifier__n_neighbors': 17} -> CV 0.7195
that model's HONEST test score: 0.7833
```

![Choosing k without letting the test set vote](images/s8-knn-tuning.png)

> **The manual search would have reported 0.8333. The honest answer is 0.7833 — five points lower.**
>
> **The 0.8333 is not a lie about the arithmetic. It is a lie about what the number *means*.** **It is the score of the k that happened to suit those particular 60 patients**, and on the next 60 it will not repeat.
>
> **You had 19 chances to peek here, and the manual loop used all of them.**

## When manual search is still the right thing

> **For understanding, not for choosing.** **Sweeping k and plotting train against test is how you *see* overfitting** — the `k=1` result above teaches more than any grid search output. **Just do not let the plot pick your final model.**

---

# 21. Grid search

> **Define a set of values for each hyperparameter, try every combination, and score each one by cross-validation on the training data only.**

```python
param_grid = {"n_neighbors": range(1, 20)}

knn_pipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
grid_search = GridSearchCV(
    knn_pipe, {"kneighborsclassifier__n_neighbors": list(range(1, 20))}, cv=5)
grid_search.fit(X_bal, y_bal)

print("Best hyperparameters:", grid_search.best_params_)
print("Best CV score       :", round(grid_search.best_score_, 4))
print("Test score          :", round(grid_search.score(X_test, y_test), 4))
print("Model fits          :", len(grid_search.cv_results_["params"]) * 5)
```

**Output:**

```text
Best hyperparameters: {'kneighborsclassifier__n_neighbors': 17}
Best CV score       : 0.7195
Test score          : 0.7833
Model fits          : 95
```

## What `GridSearchCV` gives you afterwards

| Attribute | What it is |
|---|---|
| `best_params_` | The winning combination |
| `best_score_` | **Its mean cross-validated score** |
| `best_estimator_` | **The model, already refitted on all the training data** |
| `cv_results_` | Every combination's score — **a DataFrame waiting to happen** |

```python
results = pd.DataFrame(grid_search.cv_results_)
print(results[["param_kneighborsclassifier__n_neighbors", "mean_test_score", "std_test_score"]]
      .sort_values("mean_test_score", ascending=False).head().to_string(index=False))
```

> **`grid_search.predict(X_test)` uses `best_estimator_` automatically** — you do not need to refit anything by hand.

## ⚠️ The cost grows multiplicatively

| Parameters | Values each | Combinations | × 5-fold CV |
|---|---|---|---|
| 1 | 19 | 19 | **95 fits** |
| 2 | 5 | 25 | 125 fits |
| 3 | 5 | 125 | **625 fits** |
| 4 | 10 | 10,000 | **50,000 fits** |
| 6 | 10 | 1,000,000 | **5,000,000 fits** |

> **Grid search is exhaustive, which is its strength and its fatal weakness.** **Past three or four hyperparameters it becomes unusable** — and that is what random search is for.

---

# 22. Random search

> **Instead of every combination, sample a fixed number of random ones.**

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    knn_pipe, {"kneighborsclassifier__n_neighbors": list(range(1, 20))},
    n_iter=10, cv=5, random_state=42)
random_search.fit(X_bal, y_bal)

print("Best hyperparameters:", random_search.best_params_)
print("Best CV score       :", round(random_search.best_score_, 4))
print("Test score          :", round(random_search.score(X_test, y_test), 4))
```

**Output:**

```text
Best hyperparameters: {'kneighborsclassifier__n_neighbors': 17}
Best CV score       : 0.7195
Test score          : 0.7833
```

> **Identical to the grid, with 10 of 19 candidates.**

## Why random search usually wins

![Grid versus random coverage](images/s8-grid-vs-random.png)

> **With 25 trials, grid search tests only 5 distinct values of each parameter. Random search tests 25 distinct values of each.**
>
> **In almost every real problem, one or two hyperparameters matter and the rest barely do.** **Grid search spends its budget uniformly. Random search, by accident, samples the important parameter far more finely** — and that is why it usually finds a better setting for the same cost.

## Random search over ranges, not lists

**Random search can sample from a *distribution*, which grid search cannot do at all.**

```python
# illustrative: a syntax reference, not runnable as written.
from scipy.stats import uniform, randint

param_dist = {
    "svc__C": uniform(0.1, 1000),        # any real number in the range
    "svc__gamma": uniform(0.001, 1),     # not just the seven you thought of
}
RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5)
```

> **This is the real advantage.** **A grid can only ever return a value you typed in. A distribution can return `C = 37.4`.**

---

# 23. Tuning an SVM

**One hyperparameter is a warm-up. An SVM has three, and they interact.**

| Hyperparameter | What it controls |
|---|---|
| **`C`** | **How hard the model tries to classify every training point correctly.** Low C = a smoother, more forgiving boundary |
| **`kernel`** | **The shape of the boundary.** `linear` = a straight line; `rbf` = a curved one |
| **`gamma`** | **How far a single training point's influence reaches** — used only by `rbf` |

## Step 1 — one setting, by hand

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svm_classifier = make_pipeline(MinMaxScaler(), SVC(kernel="linear", random_state=42))
svm_classifier.fit(X_bal, y_bal)
y_pred = svm_classifier.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

**Output:**

```text
Accuracy: 0.78
Confusion Matrix:
[[35  6]
 [ 7 12]]
Classification Report:
              precision    recall  f1-score   support
           0       0.83      0.85      0.84        41
           1       0.67      0.63      0.65        19
    accuracy                           0.78        60
   macro avg       0.75      0.74      0.75        60
weighted avg       0.78      0.78      0.78        60
```

> **Recall on the patients who died is 0.63 — the model found 12 of 19.** **Better than §13's untuned Random Forest, which found 10.** **SMOTE is already earning its place.**

## Step 2 — grid search over all three

```python
grid = {
    "svc__C": [1, 10, 100, 1000],                            # regularisation
    "svc__kernel": ["linear", "rbf"],                        # boundary shape
    "svc__gamma": ["scale", "auto", 0.1, 0.2, 0.3, 0.5, 0.6],  # kernel reach
}

svm_pipe = make_pipeline(MinMaxScaler(), SVC(random_state=42))
grid_search = GridSearchCV(svm_pipe, param_grid=grid, cv=5,
                           scoring="accuracy", n_jobs=-1)
grid_search.fit(X_bal, y_bal)

print("Combinations tried  :", len(grid_search.cv_results_["params"]))
print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV score       :", round(grid_search.best_score_, 4))
print("Test score          :", round(grid_search.score(X_test, y_test), 4))
```

**Output:**

```text
Combinations tried  : 56
Best Hyperparameters: {'svc__C': 10, 'svc__gamma': 'auto', 'svc__kernel': 'rbf'}
Best CV score       : 0.8363
Test score          : 0.8167
```

## Step 3 — random search over the same grid

```python
random_search = RandomizedSearchCV(svm_pipe, param_distributions=grid,
                                   cv=5, n_iter=10, random_state=42, n_jobs=-1)
random_search.fit(X_bal, y_bal)

print("Best Hyperparameters:", random_search.best_params_)
print("Best CV score       :", round(random_search.best_score_, 4))
print("Test score          :", round(random_search.score(X_test, y_test), 4))
```

**Output:**

```text
Best Hyperparameters: {'svc__kernel': 'rbf', 'svc__gamma': 0.1, 'svc__C': 10}
Best CV score       : 0.8271
Test score          : 0.8000
```

## The comparison

| | Combinations | Time | Best CV | Test |
|---|---|---|---|---|
| **Manual (linear, C=1)** | **1** | instant | — | 0.7833 |
| **Grid search** | **56** | 4.0 s | **0.8363** | **0.8167** |
| **Random search** | **10** | **0.4 s** | 0.8271 | 0.8000 |

> **Random search reached within 1 point of the grid using 18% of the combinations and 10% of the time.**
>
> **It did not match it exactly, and that is the honest picture.** **Random search buys you most of the answer for a fraction of the cost** — which is the right trade when the grid is large, and the wrong one when the grid is 56 combinations and finishes in four seconds.
>
> **The professional recipe: random search to find the neighbourhood, then a narrow grid inside it.**

## What did tuning actually buy?

```python
y_pred_tuned = grid_search.best_estimator_.predict(X_test)
print(confusion_matrix(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))
```

**Output:**

```text
[[35  6]
 [ 5 14]]
              precision    recall  f1-score   support
           0       0.88      0.85      0.86        41
           1       0.70      0.74      0.72        19
    accuracy                           0.82        60
   macro avg       0.79      0.80      0.79        60
weighted avg       0.82      0.82      0.82        60
```

| | Untuned linear SVM | **Tuned SVM** |
|---|---|---|
| Accuracy | 0.78 | **0.82** |
| **Recall on deaths** | 0.63 | **0.74** |
| Deaths found | 12 of 19 | **14 of 19** |

> **Accuracy rose by 4 points. Recall on the class that matters rose by 11.**
>
> **Report the second number.** **Two more patients correctly flagged is what tuning actually bought, and "accuracy 0.82" hides it.**

---

# 24. Tuning inside a pipeline

**Everything in §17 applies with double force during a search: a grid search fits hundreds of models, and a leak in the setup contaminates every one of them.**

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("pca", PCA()),
    ("random_forest", RandomForestClassifier(random_state=42)),
])

param_grid = {
    "pca__n_components": np.arange(5, 11),
    "random_forest__n_estimators": np.arange(100, 600, 100),
    "random_forest__max_depth": np.arange(1, 20),
    "random_forest__criterion": ["gini", "entropy"],
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid,
                                   cv=5, n_iter=10, random_state=42, n_jobs=-1)
random_search.fit(X_bal, y_bal)

print("Best CV score:", round(random_search.best_score_, 4))
print("Best Hyperparameters:", random_search.best_params_)
print("Test score:", round(random_search.score(X_test, y_test), 4))
```

**Output:**

```text
Best CV score: 0.8519
Best Hyperparameters: {'random_forest__n_estimators': 200, 'random_forest__max_depth': 17,
                       'random_forest__criterion': 'entropy', 'pca__n_components': 10}
Test score: 0.85
```

> **The scaler and the PCA are refitted inside every fold, on that fold's training rows only.** **Doing either by hand outside the search would leak test information into all 50 fits.**

## Always compare against the untuned baseline

```python
baseline = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state=42))
baseline.fit(X_bal, y_bal)
print("untuned Random Forest, no PCA:", round(baseline.score(X_test, y_test), 4))
print("tuned PCA + Random Forest    :", round(random_search.score(X_test, y_test), 4))
```

**Output:**

```text
untuned Random Forest, no PCA: 0.8333
tuned PCA + Random Forest    : 0.85
```

> **A gain of 1.67 points for 50 model fits and a more complicated pipeline.**
>
> **Real, but small — and worth stating plainly.** **Tuning frequently buys less than people expect**, and a tuning run that lands *below* the baseline is telling you something about your pipeline rather than about the search.

## The naming rule that trips everyone up

```text
Pipeline([("scaler", ...), ("pca", ...), ("random_forest", ...)])
                                          └──────┬──────┘
param_grid = {"random_forest__max_depth": [...]}  │
              └──────┬──────┘  └───┬────┘         │
                 step name    parameter    must match EXACTLY
```

```python
try:
    RandomizedSearchCV(pipeline, {"rf__max_depth": [3]}, n_iter=1, cv=5).fit(X_bal, y_bal)
except ValueError as e:
    print("ERROR:", str(e)[:70])
```

**Output:** `ERROR: Invalid parameter 'rf' for estimator Pipeline(steps=[('scaler', Min`

> ⚠️ **Two underscores, and the prefix must be the step's name.** **A step called `random_forest` needs `random_forest__n_estimators`, not `rf__n_estimators`.** **This is the single most common error in the whole topic, and it is easy to make when copying a grid between projects.**
>
> **With `make_pipeline` the names are generated for you in lowercase** — `SVC()` becomes `svc`, hence `svc__C` in §22.

---

# 25. Bayesian optimization

> **Grid and random search have no memory. Every trial is chosen without reference to what the previous trials found. Bayesian optimization uses the results so far to decide what to try next.**

🧠 **Analogy: searching a hillside for the highest point in fog.**
>
> - **Grid search** walks a fixed lattice, measuring everywhere, learning nothing as it goes.
> - **Random search** wanders and measures at random.
> - **Bayesian optimization** measures, notices the ground rising to the north, **and spends its remaining measurements to the north.**

## How it works, in three parts

| Part | What it is |
|---|---|
| **Objective function** | **What you want to minimise** — usually `1 − accuracy`, or the error |
| **Search space** | The range each hyperparameter may take |
| **Surrogate model** | **A cheap model of "which settings look promising"**, updated after every trial |

**The loop:** try a setting → record the score → update the belief about where good settings live → **pick the most promising untried setting** → repeat.

## What it buys you

| | |
|---|---|
| ✅ **Far fewer trials** for the same quality | Typically 20–50 where random search needs hundreds |
| ✅ **Handles expensive models** | Where each fit takes minutes, trial count is everything |
| ❌ **Extra library**, extra complexity | `optuna`, `hyperopt` or `scikit-optimize` |
| ❌ **Trials must be sequential** | **It cannot use all 8 cores the way random search can** |
| ❌ **Overkill for small problems** | On §23's 56-combination grid, which ran in four seconds, it would save nothing |

## What the code looks like

**Two libraries dominate. Neither is a course dependency — these are reference snippets, not exercises.**

```python
# needs-install: pip install optuna
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective(trial):
    C = trial.suggest_float("C", 0.01, 1000, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    model = make_pipeline(MinMaxScaler(), SVC(C=C, kernel=kernel, gamma=gamma))
    score = cross_val_score(model, X_bal, y_bal,
                            cv=StratifiedKFold(5), scoring="accuracy").mean()
    return 1 - score                      # Optuna MINIMISES, so return the error

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print("Best Hyperparameters:", study.best_params)
```

```python
# needs-install: pip install hyperopt
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

space = {
    "C": hp.choice("C", [1, 10, 100, 1000]),
    "kernel": hp.choice("kernel", ["linear", "rbf"]),
}

def objective(params):
    model = make_pipeline(MinMaxScaler(), SVC(**params))
    score = cross_val_score(model, X_bal, y_bal, cv=5).mean()
    return {"loss": 1 - score, "status": STATUS_OK}

best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=30, trials=Trials())
```

## ⚠️ The mistake to avoid, whichever library you use

> **The objective function must score on cross-validated *training* data — never on the test set.**
>
> **An objective that returns `accuracy_score(y_test, model.predict(X_test))` will run happily, report an excellent number, and mean nothing.** **You have simply run 30 experiments on your test set and kept the best.** **This is §20's trap wearing a more sophisticated hat.**

## The four methods, compared

| Method | Trials needed | Uses past results? | Parallel? | Use when |
|---|---|---|---|---|
| **Manual** | Whatever you type | **You do** | — | **Learning what a parameter does** |
| **Grid** | All combinations | No | **Yes** | ≤ 3 parameters, small ranges |
| **Random** | Your budget | No | **Yes** | **The everyday default** |
| **Bayesian** | **Fewest** | **Yes** | Poorly | **Each fit is expensive** |

## The correct order of operations

```text
1. Preprocess          -> Session 3's sequence
2. Split off the test set                      -> and do not touch it again
3. Balance the TRAINING half only              -> SMOTE, if needed
4. Cross-validate several models on TRAIN      -> shortlist
5. Tune the shortlist's hyperparameters on TRAIN, with CV, inside a pipeline
6. Refit the winner on all of TRAIN            -> the search does this for you
7. Evaluate ONCE on TEST                       -> this is the number you report
```

> **Step 7 happens once.** **If you look at the test score, dislike it, change something and look again, you have quietly turned your test set into a validation set** — and you no longer have an unbiased estimate of anything.

## ✏️ Practice — tuning

1. Set up the tuning data in the correct order: split, then SMOTE on train only. **Print the class balance before and after, and say why the order matters.**
2. Sweep `n_neighbors` from 1 to 19 and print train and test accuracy. **Why is `k=1`'s training score exactly 1.0?**
3. Choose k with `GridSearchCV` instead. **Does it pick the same k as the test-score sweep? Which number would you report, and why?**
4. Fit a linear SVM and print the confusion matrix and classification report. **What is the recall on the patients who died?**
5. Grid-search the SVM over `C`, `kernel` and `gamma`. **Report combinations, CV score, test score — and the new recall on deaths.**
6. Repeat with `RandomizedSearchCV` and `n_iter=10`. **Compare score and time. Was anything lost?**
7. Tune a scaler → PCA → Random Forest pipeline. **Compare against the untuned forest, and deliberately misname a grid key to read the error.**

<details><summary>Solutions</summary>

```python
import time, multiprocessing
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

url = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/classification/heart_failure_raw.csv")
heart = pd.read_csv(url)
heart.loc[heart["age"] > 120, "age"] = np.nan
for c in ["age", "ejection_fraction", "serum_creatinine"]:
    heart[c] = heart[c].fillna(heart[c].median())
le = LabelEncoder()
for c in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
          "treatment_type", "DEATH_EVENT"]:
    heart[c] = le.fit_transform(heart[c])
X = heart.drop(columns=["DEATH_EVENT"]); y = heart["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(                   # 1
    X, y, test_size=.2, random_state=42, stratify=y)
print("before SMOTE:", y_train.value_counts().to_dict())
X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
print("after  SMOTE:", y_bal.value_counts().to_dict())
# Split FIRST. SMOTE builds synthetic rows by interpolating between real
# ones; run it before the split and 23% of the test set is invented data
# derived from training rows. The test score would measure nothing.

sc = MinMaxScaler().fit(X_bal)                                         # 2
Xb, Xt = sc.transform(X_bal), sc.transform(X_test)
tr, te = [], []
for k in range(1, 20):
    kn = KNeighborsClassifier(n_neighbors=k).fit(Xb, y_bal)
    tr.append(kn.score(Xb, y_bal)); te.append(kn.score(Xt, y_test))
    if k in (1, 3, 11, 17): print(f"k={k:>2}  train {tr[-1]:.4f}  test {te[-1]:.4f}")
print("best by test: k =", te.index(max(te)) + 1, round(max(te), 4))
# k=1 scores exactly 1.0 on train because the nearest neighbour of a
# training point IS that point. It has memorised the training set - and
# its test score is among the worst. Pure overfitting.

gs = GridSearchCV(make_pipeline(MinMaxScaler(), KNeighborsClassifier()),   # 3
                  {"kneighborsclassifier__n_neighbors": list(range(1, 20))},
                  cv=5).fit(X_bal, y_bal)
print("CV picks:", gs.best_params_, "CV", round(gs.best_score_, 4),
      "| honest test", round(gs.score(X_test, y_test), 4))
# CV picks k=17, not the k=11 that maximised the TEST score. Report the
# CV-chosen model's test score (0.7833), not the peeked 0.8333: the
# higher number is the score of the k that suited these 60 patients.

svm = make_pipeline(MinMaxScaler(), SVC(kernel="linear",               # 4
                                        random_state=42)).fit(X_bal, y_bal)
p = svm.predict(X_test)
print(confusion_matrix(y_test, p)); print(classification_report(y_test, p))
# Recall on class 1 (died) is 0.63 - it found 12 of the 19.

grid = {"svc__C": [1, 10, 100, 1000], "svc__kernel": ["linear", "rbf"],  # 5
        "svc__gamma": ["scale", "auto", 0.1, 0.2, 0.3, 0.5, 0.6]}
pipe = make_pipeline(MinMaxScaler(), SVC(random_state=42))
t0 = time.time(); g = GridSearchCV(pipe, grid, cv=5, n_jobs=-1).fit(X_bal, y_bal)
tg = time.time() - t0
print(f"GRID {len(g.cv_results_['params'])} combos  CV {g.best_score_:.4f}"
      f"  test {g.score(X_test, y_test):.4f}  {tg:.1f}s  {g.best_params_}")
print(classification_report(y_test, g.best_estimator_.predict(X_test)))
# 56 combos, CV 0.8363, test 0.8167 - and recall on deaths rises from
# 0.63 to 0.74. Two more patients found. THAT is what tuning bought.

t0 = time.time()                                                       # 6
r = RandomizedSearchCV(pipe, grid, n_iter=10, cv=5, random_state=42,
                       n_jobs=-1).fit(X_bal, y_bal)
print(f"RAND 10 combos  CV {r.best_score_:.4f}  test "
      f"{r.score(X_test, y_test):.4f}  {time.time()-t0:.1f}s")
# CV 0.8271 against the grid's 0.8363 - one point short, for 18% of the
# combinations and 10% of the time. On a 56-combination grid that runs
# in 4 seconds the grid is worth it; on a 50,000-fit grid it is not.

pl = Pipeline([("scaler", MinMaxScaler()), ("pca", PCA()),             # 7
               ("random_forest", RandomForestClassifier(random_state=42))])
pg = {"pca__n_components": np.arange(5, 11),
      "random_forest__n_estimators": np.arange(100, 600, 100),
      "random_forest__max_depth": np.arange(1, 20),
      "random_forest__criterion": ["gini", "entropy"]}
s = RandomizedSearchCV(pl, pg, cv=5, n_iter=10, random_state=42,
                       n_jobs=-1).fit(X_bal, y_bal)
base = make_pipeline(MinMaxScaler(),
                     RandomForestClassifier(random_state=42)).fit(X_bal, y_bal)
print("tuned  ", round(s.score(X_test, y_test), 4),
      "| untuned baseline", round(base.score(X_test, y_test), 4))
try:
    RandomizedSearchCV(pl, {"rf__max_depth": [3]}, n_iter=1, cv=5).fit(X_bal, y_bal)
except ValueError as e:
    print("MISNAMED KEY ->", str(e)[:80])
# Tuned 0.85 vs untuned 0.8333: a 1.67-point gain for 50 model fits.
# Real but small - always report the comparison. The misnamed key raises
# "Invalid parameter 'rf'": the step is called random_forest, so the key
# must be random_forest__max_depth. Two underscores, exact step name.
```
</details>

---

# ❓ Session 8 — 20 MCQs

**Answer from memory first, then check.**

### Preprocessing

**Q1.** `describe()` showed the heart data had a maximum age of 160, and `info()` showed 45 missing values. The correct order is…
- (a) Impute first, then look for impossible values  (b) **Turn the impossible ages into `NaN` first, then impute — otherwise the errors survive untouched**  (c) Drop both sets of rows  (d) It makes no difference

**Q2.** `LabelEncoder` on `treatment_type` gives Lifestyle=0, Medication=1, Other=2, Surgery=3. This is defensible here mainly because…
- (a) The categories are ordered  (b) **The models used are mostly trees, which can split anywhere and barely notice a false order**  (c) There are only four categories  (d) Dummies never work

### Underfitting and overfitting

**Q3.** On the heart data the depth-1 tree had the best test accuracy. This means…
- (a) The labels in the table were wrong to call it "underfitting" — **it is the right size for 239 rows**  (b) The test set is broken  (c) Deeper is always better  (d) Accuracy is the wrong metric

**Q4.** The depth-1 tree splits on `time`, and removing that column drops every model from ~0.84 to 0.69. `time` is…
- (a) A useful feature  (b) **An outcome artefact — follow-up was short *because* the patient died, so it partly records the answer**  (c) A duplicate column  (d) An outlier

**Q5.** The unrestricted tree on 15,240 cars scored train 0.9993 and CV 0.8276 — the *highest* CV of the three models. This means…
- (a) It is not overfitting  (b) **The gap of +0.17 correctly says it is memorising; that is a different question from whether it generalises worst**  (c) The CV is broken  (d) Trees never overfit

**Q6.** The same unrestricted tree on 1,000 rows scored train 1.0000 and CV −0.2062. The lesson is…
- (a) Trees are unreliable  (b) **Overfitting is capacity relative to data volume — the same model is fine at 15,000 rows and useless at 1,000**  (c) R² is broken  (d) Use more features

**Q7.** Four rows out of 15,244 (2 cars with `seats=0`, 2 driven over a million km) changed the deepest model's CV R² from 0.6612 to 0.8276. The deepest model suffered most because…
- (a) It uses more features  (b) **With no depth limit it builds a branch for a single extreme car, and R² punishes the resulting wild prediction**  (c) It trains longer  (d) It is unstable by nature

### Model validation

**Q8.** The same SVM on the same 299 rows scored 0.7333 and 0.8500 depending only on `random_state`. This shows…
- (a) The code is buggy  (b) **A single split is an unreliable estimate on small data**  (c) SVM is a bad model  (d) The data is corrupt

**Q9.** You should report cross-validation results as…
- (a) The best fold  (b) **The mean and the standard deviation**  (c) The mean only  (d) The worst fold

**Q10.** On `X_train`, plain `KFold` produced fold scores from 0.6667 to 0.8958 while `StratifiedKFold` ranged 0.7292 to 0.8723. That extra variation was measuring…
- (a) The model  (b) **The split — the 162/77 class balance drifted between folds**  (c) The scaler  (d) Random noise in the SVM

**Q11.** LOOCV scored highest of the four methods (0.7860) because…
- (a) It is more accurate  (b) **Each of its 299 models trains on 298 rows, more than any other method gives**  (c) It uses stratification  (d) It has no test set

**Q12.** The thing bootstrapping gives you that k-fold does not is…
- (a) Higher accuracy  (b) **A confidence interval**  (c) Faster training  (d) Stratification

**Q13.** The Random Forest chosen by cross-validation scored 0.80 accuracy on test but 0.53 recall on the patients who died. The right conclusion is…
- (a) Ship it  (b) **Accuracy is the wrong headline: it missed 9 of 19 deaths, and for a clinical tool a miss is the expensive error**  (c) The CV was wrong  (d) Use a different scaler

**Q14.** The `max_depth` validation curve on raw `selling_price` zigzagged; on `log(selling_price)` it was smooth. The zigzag was caused by…
- (a) Too few folds  (b) **A heavily skewed target — R² squares errors, so a handful of ₹39,500,000 cars decide each fold**  (c) A bad `random_state`  (d) Not scaling

**Q15.** A learning curve whose validation score has flattened tells you…
- (a) Collect more data  (b) **More data will not help; change the model instead**  (c) The model is broken  (d) Reduce the folds

### The two leaks

**Q16.** `MinMaxScaler().fit_transform(X)` before cross-validation is wrong because…
- (a) MinMax is the wrong scaler  (b) **The scaler sees the test rows in every fold, so each fold's test data influenced the transform**  (c) It is slow  (d) It needs the target

**Q17.** Running SMOTE on the full dataset before `train_test_split` produced a test set that was…
- (a) Too small  (b) **23% synthetic — rows interpolated from patients now sitting in the training set**  (c) Unbalanced  (d) Correctly balanced

### Tuning

**Q18.** Sweeping k from 1 to 19 and keeping the k with the best **test** score is wrong because…
- (a) It is slow  (b) **The test set has then chosen a hyperparameter, so its score is no longer an unbiased estimate**  (c) k should be odd  (d) 19 is too many

**Q19.** Tuning the SVM raised accuracy from 0.78 to 0.82 and recall on deaths from 0.63 to 0.74. The number to report is…
- (a) Accuracy, it is the standard  (b) **Recall — it is the class that matters, and "accuracy 0.82" hides the two extra patients found**  (c) Neither  (d) The CV score only

**Q20.** Bayesian optimization differs from grid and random search because…
- (a) It is always more accurate  (b) **It uses the results of previous trials to decide what to try next**  (c) It needs no objective function  (d) It parallelises better

<details><summary>Answers</summary>

**A1 — (b) Impossible values first.** **An impossible value is an error, not an extreme case.** Convert it to `NaN` and let imputation handle it — and note the other thirteen measurements for those two patients were perfectly good, so deleting the rows would have been wasteful.

**A2 — (b) The models are mostly trees.** **Session 3's table is explicit:** trees barely notice a false ordering because they can split anywhere; kNN and SVM are hurt, because they measure distance. **It is a defensible choice, not a free one, and you should be able to say why you made it.**

**A3 — (a) It is the right size for 239 rows.** **The labels in that table are hypotheses, not verdicts.** A 1.6-point gap and the best test score is not underfitting.

**A4 — (b) An outcome artefact.** **No amount of cross-validation catches this** — every method in Part A was scrupulously correct and every one was measuring a column that would not exist at prediction time. **Only knowing what your columns mean catches it.**

**A5 — (b) Two different questions.** **The gap measures memorising. The CV score measures usefulness.** With 15,240 rows a deep tree can do both.

**A6 — (b) Capacity relative to data volume.** **CV R² of −0.21 is worse than predicting the average price for every car**, from the same algorithm that scored 0.83 with fifteen times the data.

**A7 — (b) It builds a branch for a single extreme car.** **And its fold-to-fold spread fell from 0.227 to 0.065 once those four rows went.** You cannot evaluate honestly on data you have not checked.

**A8 — (b) A single split is unreliable on small data.** **The test set is 60 patients, so one patient is worth 1.67 points.** Both numbers came from correct code.

**A9 — (b) Mean and standard deviation.** **"0.75 ± 0.04" is a statement; "0.77" is a claim you cannot support.** And if two models' means differ by less than the spread, you have not shown a difference.

**A10 — (b) The split.** **A 23-point range between folds, on the same model and the same data.** Stratifying removed most of it.

**A11 — (b) More training data.** LOOCV's models see 298 rows; the 5-fold models see 239. **More data, better score — which is also why holdout, which trains on the least, is unreliable.**

**A12 — (b) A confidence interval.** **"0.76, and 95% of the time between 0.68 and 0.82"** is far more honest than a single figure — and it covered almost exactly the range the ten random seeds produced.

**A13 — (b) Accuracy is the wrong headline.** **It found 10 of 19 deaths and missed 9.** Session 5B's lesson, arriving in a real workflow.

**A14 — (b) A heavily skewed target.** **The median car is ₹559,000 and the maximum is 70× that.** An unreadable curve is telling you about your target, not your hyperparameter.

**A15 — (b) More data will not help.** Here CV climbed from 0.793 to 0.874 and the last 2,200 cars bought 0.004.

**A16 — (b) The scaler sees the test rows.** **MinMax uses each column's minimum and maximum, so one extreme test patient shifts the scaling of every training row.** Note it made the score *higher* here — a leak does not reliably inflate or deflate, which is why you cannot spot one by looking at the number.

**A17 — (b) 23% synthetic.** **You are testing the model on rows built out of its own training data.** Split first, resample the training half only.

**A18 — (b) The test set has chosen a hyperparameter.** **The peeked answer was 0.8333; the honest one was 0.7833.** You had 19 chances to peek and the loop used all of them.

**A19 — (b) Recall.** **Accuracy rose 4 points; recall on the class that matters rose 11.** Two more patients correctly flagged is what tuning actually bought.

**A20 — (b) It uses previous trials.** Grid and random search have no memory; Bayesian optimization builds a model of where good settings live and spends its remaining trials there — **at the cost of being hard to parallelise.**

</details>

---
# 🎯 Session 8 — Tasks

## Preprocessing

**Task 1 — Preprocess before you evaluate.** On `heart_failure_raw.csv`, run Session 3's sequence: explore, duplicates, impossible values, missing values, outliers, `LabelEncoder`. **Print proof at each step and state which output revealed each problem.**

**Task 2 — Clean the car data too.** Find the impossible values in `cardekho_preprocessed.csv`. **How many rows, and what fraction of the dataset?**

**Task 3 — The seed experiment.** Run a holdout split with 20 different `random_state` values. **Plot the 20 accuracies and report the range.** Write the one sentence you would say to someone quoting a single split's score.

**Task 4 — Fold count.** Compare 3-, 5-, 10-fold and LOOCV: mean, standard deviation and wall-clock time. **Produce a four-row table and recommend one.**

**Task 5 — Stratification matters.** Run plain `KFold` and `StratifiedKFold` on `X_train`, printing each fold's class balance alongside its score. **Show the connection between the two.**

**Task 6 — Bootstrap a confidence interval.** Run 500 bootstrap resamples and report a 95% interval. **Write the sentence you would put in a report for a non-technical stakeholder.**

**Task 7 — Build the leak, then measure it.** Scale before splitting, record the score; scale inside a pipeline, record it again. **Then do the same with SMOTE, and count how many test rows are synthetic.** Say which leak is more dangerous and why.

**Task 8 — Model selection, end to end.** Cross-validate four models on `X_train`, pick the best, refit, and evaluate once on test. **Report the full classification report and say whether you would deploy it.**

## Overfitting and underfitting

**Task 9 — Build all three.** Construct an underfitting, a good-fit and an overfitting model with cross-validation. **Report train, CV and gap in one table.**

**Task 10 — Capacity against data.** Run one unrestricted model on 300, 1,000, 3,000 and all rows. **Plot CV score against training size and write two sentences on what overfitting actually is.**

**Task 11 — What dirty rows cost.** Compare CV scores on raw and cleaned data for all your models. **Which model suffered most, and why that one?**

**Task 12 — The tree-choice table.** Reproduce the heart-failure comparison table and the train-vs-test plot. **Does the best model match its label?**

**Task 13 — Hunt for a leaky column.** For any dataset you use, print `export_text` of a depth-1 tree and investigate the column it chooses. **Could that value have been known at prediction time?** *(Run this check on every project.)*

**Task 14 — An unreadable curve.** Draw a validation curve that zigzags, diagnose the cause, fix it, and draw it again. **Report both.**

**Task 15 — Learning curve.** Draw one for your model. **Answer in a sentence: would collecting more data help?** Justify from the shape.

## Tuning

**Task 16 — Manual sweep.** Sweep one hyperparameter and plot train, test and cross-validated score together. **Mark the value each of the three would choose. Which is the honest one, and how big is the difference?**

**Task 17 — Grid against random.** Run both on the same SVM grid. **Report combinations, time, CV score and test score for each, and say when you would accept random search's answer.**

**Task 18 — Report what tuning bought.** Print the classification report before and after tuning. **Report the change in recall on the minority class, not just accuracy.**

**Task 19 — Pipeline tuning.** Tune a scaler → reducer → model pipeline against the untuned baseline. **Deliberately misname one grid key, read the error, and write down the naming rule in your own words.**

**Task 20 — The complete honest workflow.** On one dataset, run all seven steps from §25: preprocess, split, balance the training half, shortlist by CV, tune inside a pipeline, refit, evaluate **once**. **Write it up as you would for a stakeholder — including the uncertainty and anything you could not verify.**

---

## ✅ Session 8 checklist

- [ ] I run **Session 3's sequence** before evaluating anything
- [ ] I fix **impossible values before imputing**, not after
- [ ] I print `le.classes_` after every `LabelEncoder`
- [ ] I never report a single split's score as *the* accuracy on small data
- [ ] I report cross-validation as **mean ± standard deviation**
- [ ] I use **`StratifiedKFold` for classification**, always
- [ ] **Every step that learns from data goes inside a `Pipeline`**
- [ ] I balance and augment **after** the split, never before
- [ ] I print the **train score next to the validation score**, every time
- [ ] I know the gap measures *memorising*, which is not the same as *generalising worse*
- [ ] I can read a validation curve and a learning curve — **and diagnose an unreadable one**
- [ ] **I check whether my best feature could have been known at prediction time**
- [ ] **I never let the test set choose a hyperparameter**
- [ ] I know when to use grid, random and Bayesian search
- [ ] **I compare a tuned model against the untuned baseline**, and report honestly when tuning bought little

---

| | |
|---|---|
| **Previous** | [Session 7 — Unsupervised Learning: Clustering](session-07-unsupervised.md) |
| **Next** | [Session 9 — Deep Learning](session-09-deep-learning.md) |
| **Notebook** | [session-08-evaluation-tuning.ipynb](../notebooks/session-08-evaluation-tuning.ipynb) |
| **More practice** | [Exercises & assignments](../exercises-assignments.md) |
