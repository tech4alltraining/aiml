# Session 5 — Supervised Learning, Deployment & Streamlit

**Regression & Classification · Linear Regression + metrics · Logistic Regression, kNN, SVM, Gaussian Naive Bayes, Decision Trees, Random Forest + metrics · Model saving, loading & inference · Streamlit app development**

| | |
|---|---|
| **Notebook** | [session-05-supervised-learning.ipynb](../notebooks/session-05-supervised-learning.ipynb) |
| **Previous** | [Session 4 — Introduction to ML & AI](session-04-intro-ml-ai.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **This is the longest session in the course, and the most useful.** By the end you will have trained six different classifiers, saved a model to a file, and put it behind a web app that anyone can use.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Split any supervised problem into regression or classification, and pick the right metrics
2. Fit a Linear Regression and read its coefficients out loud
3. Report MAE, MSE, RMSE and R² — and say when each one misleads
4. Train and compare six classifiers with the same four lines of code
5. Read a confusion matrix, and explain why **accuracy alone is dangerous**
6. Save a trained model to a file and load it in a completely separate program
7. Build a Streamlit web app that makes live predictions

---

## The five topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Supervised learning](#1-supervised-learning-regression-and-classification) | Number → regression. Category → classification |
| 2 | [Regression & metrics](#2-linear-regression-and-its-metrics) | RMSE is in your target's units. Read it out loud |
| 3 | [Classification & metrics](#3-classification-six-algorithms) | Accuracy alone lies on imbalanced data |
| 4 | [Saving & inference](#4-saving-loading-and-inference) | Save the **pipeline**, not the bare model |
| 5 | [Streamlit](#5-streamlit-app-development) | Cache the model, or it reloads on every click |

---

# 1. Supervised learning: regression and classification

**Supervised** means you have the answers. You show the model inputs *and* correct outputs, and it learns the mapping between them.

🧠 **Analogy: past exam papers with the answer key.** You study a hundred solved problems. Nobody explained the underlying theory — you inferred it from worked examples. Then you sit a new paper. **The final exam is the test set.**

| | Regression | Classification |
|---|---|---|
| Answer is | A **number** | A **category** |
| Question | *How much? How many?* | *Which one? Yes or no?* |
| Example | Salary from experience | Loan approved or rejected |
| Typical metric | RMSE, R² | Accuracy, F1, ROC-AUC |
| Bad prediction | Off by ₹4,000 | Simply wrong |

> **A regression prediction can be nearly right. A classification prediction is right or wrong.** That difference is why the two families need completely different metrics.

## 📘 Examples

**Example 1 — the same four lines, either way**

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

reg = LinearRegression().fit(X_train, y_salary)     # y = 45000, 62000, ...
clf = LogisticRegression().fit(X_train, y_approved) # y = 1, 0, 1, ...

reg.predict(X_test)   # -> [51234.7, 48901.2, ...]   numbers
clf.predict(X_test)   # -> [1, 0, 1, ...]            categories
```

**Example 2 — the target column decides everything**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

ads = pd.read_csv(BASE + "regression/advertising.csv")
loans = pd.read_csv(BASE + "loan_data_10k.csv")

print("Sales     :", ads["Sales"].dtype, "unique:", ads["Sales"].nunique())
print("loan_status:", loans["loan_status"].dtype, "unique:", loans["loan_status"].nunique())
```

`Sales` has 121 distinct float values → **regression**. `loan_status` has 2 → **classification**.

**Example 3 — the trap: numbers that are really categories**

```python
# Postcode 682024 is a NUMBER in the file, but it is a CATEGORY in meaning.
# Postcode 682024 is not "twice" postcode 341012.
#
# Rule of thumb: if the ARITHMETIC is meaningless, it is a category.
#   age 30 + age 30 = 60   -> meaningful, so age is a number
#   pincode + pincode      -> nonsense,   so pincode is a category
```

## ✏️ Practice

1. `person_income` in the loan data — regression target or classification target?
2. A column `rating` with values 1–5. Argue for **both** answers.
3. Load the advertising data and confirm `Sales` is continuous.
4. Which metric family applies to *"how many days until this machine fails?"*
5. Name a column that looks numeric but is really a category.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

# 1 - regression. Income is a number and arithmetic on it is meaningful.
# 2 - Regression: ratings are ordered and 4 is genuinely closer to 5 than 1 is.
#     Classification: only five values exist and you cannot predict 3.7 stars.
#     Real teams argue about this. Try both and compare.

ads = pd.read_csv(BASE + "regression/advertising.csv")
print(ads["Sales"].nunique(), "distinct values of", len(ads))   # 3
print(ads["Sales"].describe())

# 4 - regression. "How many days" is a number.
# 5 - pincode, phone number, student roll number, year-as-a-label,
#     and any ID column. Arithmetic on them is meaningless.
```
</details>

## ❓ MCQs

**Q1.** What makes a problem *supervised*?
- (a) A human watches the training
- (b) The training data includes the correct answers
- (c) It uses a neural network
- (d) The data is clean

**Q2.** Predicting a house price is…
- (a) Classification  (b) Regression  (c) Clustering  (d) Generation

**Q3.** Why do regression and classification need different metrics?
- (a) They use different libraries
- (b) A regression prediction can be *nearly* right; a classification one is right or wrong
- (c) Classification is harder
- (d) They do not — accuracy works for both

**Q4.** A `pincode` column contains numbers. It should be treated as…
- (a) A numeric feature — it is stored as a number
- (b) A category, because arithmetic on it is meaningless
- (c) The target
- (d) Dropped always

**Q5.** Your target has exactly two distinct values. This is…
- (a) Regression  (b) Binary classification  (c) Clustering  (d) Impossible

**Q6.** Which of these makes it *impossible* to do supervised learning?
- (a) Missing values  (b) No labelled target column  (c) Too few columns  (d) Categorical features

<details><summary>Answers</summary>

**A1 — (b).** You supply the answer key; the model learns the mapping.

**A2 — (b) Regression.** A price is a number.

**A3 — (b).** "Off by ₹4,000" is a meaningful statement; "40% wrong on this row" is not.

**A4 — (b).** Pincode 682024 is not twice pincode 341012. **If the arithmetic is meaningless, it is a category.**

**A5 — (b) Binary classification.**

**A6 — (b).** Without labels you are in unsupervised territory (Session 7). The others are inconveniences, not blockers.
</details>

## 🎯 Tasks

**Task 1 — Target audit.** Take three datasets from [`datasets/`](../../../datasets/). For each column write: numeric or categorical, *and* whether it could serve as a target. **Flag every column that is numeric in storage but categorical in meaning** — this is where beginners lose most accuracy.

**Task 2 — The rating argument.** Model a 1–5 rating column as **both** regression and classification. Report both sets of metrics and write a paragraph on which you would ship. There is no single right answer; the reasoning is the deliverable.

---

# 2. Linear Regression and its metrics

Linear regression fits a straight line — or, with several inputs, a flat surface:

```text
prediction = w1*feature1 + w2*feature2 + ... + intercept
```

"Training" means finding the `w` values that make the errors smallest.

🧠 **Analogy: the price of a used car.** You have a hunch that price falls by roughly ₹40,000 per year of age and rises by ₹15,000 per unit of engine power. Those two hunches are the **coefficients**. Linear regression works them out from data instead of guessing.

## The four metrics

| Metric | Reads as | Units | Watch out |
|---|---|---|---|
| **MAE** | Average size of error | Same as target | Treats all errors equally |
| **MSE** | Average *squared* error | Target **squared** | Punishes big misses hard; unreadable units |
| **RMSE** | √MSE | Same as target | **Report this one out loud** |
| **R²** | Fraction of variation explained | None (0–1) | 1.0 is suspicious, not excellent |

> **RMSE is in your target's units.** "RMSE 1.7" on a sales target measured in thousands of units means *typically off by about 1,700 units*. That sentence is what a stakeholder actually needs.

## 📘 Examples

**Example 1 — simple regression, one input**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

sal = pd.read_csv(BASE + "regression/salary_data.csv").dropna()
sal.columns = [c.strip().lstrip("﻿") for c in sal.columns]   # strip a stray BOM

m = LinearRegression().fit(sal[["Experience"]], sal["Salary"])
print(f"Salary = {m.coef_[0]:,.0f} * Experience + {m.intercept_:,.0f}")
# Salary = 6,845 * Experience + 31,921
```

**Read that out loud:** *"Each extra year of experience is worth about ₹6,845, starting from a base of about ₹31,921."* **A model you can say in a sentence is a model you can defend in a meeting.**

**Example 2 — multiple regression, and all four metrics**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ads = pd.read_csv(BASE + "regression/advertising.csv")
X, y = ads[["TV", "Radio", "Newspaper"]], ads["Sales"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)

print(f"MAE  {mean_absolute_error(yte, pred):.3f}")            # 1.275
print(f"MSE  {mean_squared_error(yte, pred):.3f}")             # 2.908
print(f"RMSE {np.sqrt(mean_squared_error(yte, pred)):.3f}")    # 1.705
print(f"R2   {r2_score(yte, pred):.4f}")                       # 0.9059
```

**Example 3 — the coefficients tell a story**

```python
for name, coef in zip(X.columns, model.coef_):
    print(f"{name:<10} {coef:+.4f}")
# TV         +0.0545
# Radio      +0.1009
# Newspaper  +0.0043
```

**Radio is worth about twice TV per unit spent, and newspaper is worth almost nothing.** That is a budget recommendation, not just a number.

> ⚠️ Coefficients are only comparable like this when the features are on **similar scales**. Here all three are spend in the same units. If one were in rupees and another in lakhs, you would have to scale first.

## ✏️ Practice

1. Fit the salary model. What does the model predict for 7 years of experience?
2. Compute all four metrics on the advertising data.
3. Drop `Newspaper` and refit. Does R² fall meaningfully? What does that tell you?
4. Why is MSE 2.908 while RMSE is 1.705? Which would you put in a report?
5. Fit a model, then score it on the **training** data. Is it higher? Why does that not count?

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

sal = pd.read_csv(BASE + "regression/salary_data.csv").dropna()
sal.columns = [c.strip().lstrip("﻿") for c in sal.columns]
m = LinearRegression().fit(sal[["Experience"]], sal["Salary"])
print("7 years ->", round(m.predict([[7]])[0]))                        # 1

ads = pd.read_csv(BASE + "regression/advertising.csv")
X, y = ads[["TV", "Radio", "Newspaper"]], ads["Sales"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
mo = LinearRegression().fit(Xtr, ytr); p = mo.predict(Xte)
print("R2 with Newspaper   :", round(r2_score(yte, p), 4))             # 2

X2 = ads[["TV", "Radio"]]                                              # 3
X2tr, X2te, _, _ = train_test_split(X2, y, test_size=.2, random_state=42)
p2 = LinearRegression().fit(X2tr, ytr).predict(X2te)
print("R2 without Newspaper:", round(r2_score(yte, p2), 4))
# Barely changes. Newspaper carries almost no information -> drop it.

# 4 - MSE squares the errors, so it is in "sales units squared", which nobody
#     can interpret. RMSE takes the square root and lands back in sales units.
#     REPORT RMSE.

print("train R2:", round(mo.score(Xtr, ytr), 4))                       # 5
print("test  R2:", round(mo.score(Xte, yte), 4))
# Training score measures MEMORISATION. Only unseen data measures learning.
```
</details>

## ❓ MCQs

**Q1.** Which metric is in the same units as the target and safest to quote?
- (a) MSE  (b) RMSE  (c) R²  (d) Accuracy

**Q2.** R² = 1.0 on your test set most likely means…
- (a) A perfect model  (b) Data leakage or a bug  (c) Too little data  (d) Wrong metric

**Q3.** Why does MSE punish large errors more than MAE?
- (a) It uses absolute values  (b) It squares each error before averaging  (c) It divides by n−1  (d) It ignores small errors

**Q4.** `Radio` coefficient +0.1009 and `TV` +0.0545, both spend in the same units. This suggests…
- (a) TV is twice as effective  (b) Radio is roughly twice as effective per unit spent  (c) Nothing  (d) Radio should be dropped

**Q5.** Dropping a feature barely changes R². That feature is…
- (a) Essential  (b) Carrying little information for this target  (c) The target  (d) Miscoded

**Q6.** Why can you not compare raw coefficients across features on very different scales?
- (a) You can, always
- (b) A coefficient is "per one unit" — one rupee and one lakh are not comparable units
- (c) scikit-learn forbids it
- (d) Coefficients are random

<details><summary>Answers</summary>

**A1 — (b) RMSE.** MSE is in squared units; R² has no units at all. RMSE gives you "typically off by about X".

**A2 — (b).** Real data has noise. A perfect score almost always means the target leaked into the features.

**A3 — (b).** Squaring makes an error of 10 count 100× an error of 1.

**A4 — (b).** Per unit of spend, radio moves sales about twice as much — a real budget recommendation.

**A5 — (b).** Consider dropping it: fewer features means a simpler, faster, more explainable model.

**A6 — (b).** Scale the features first if you want to compare coefficients directly.
</details>

## 🎯 Tasks

**Task 1 — The advertising budget memo.** Using the fitted coefficients, write a one-page recommendation for a marketing head: where should the next ₹100,000 go, and what do you expect to gain? **Include RMSE as an honest error bar** and one sentence on what the model cannot tell you (hint: this is correlation, not a controlled experiment).

**Task 2 — Metric intuition.** Take the advertising test set and manually inflate **one** prediction by 50 units. Recompute MAE, MSE, RMSE and R². Report how much each metric moved. **Which metric was most sensitive to the single outlier, and when would you want that?**

---

# 3. Classification: six algorithms

Every one of these uses the **same four lines**. What changes is *how* they decide.

| Algorithm | How it decides | Needs scaling? | Good for |
|---|---|---|---|
| **Logistic Regression** | A weighted sum, squashed into a probability | Yes | A fast, explainable baseline |
| **kNN** | Asks the *k* nearest neighbours to vote | **Essential** | Simple, no real training |
| **SVM** | Finds the widest boundary between classes | **Essential** | Clean margins, medium data |
| **Gaussian Naive Bayes** | Bayes' rule, assuming features are independent | No | Very fast, text, tiny data |
| **Decision Tree** | A flowchart of yes/no questions | No | Explaining a decision to a human |
| **Random Forest** | Hundreds of trees, voting | No | **The default. Usually wins on tables** |

🧠 **Analogies:**
- **kNN** — *asking your five nearest neighbours* what they think. Move house and the answer changes.
- **Decision Tree** — a *doctor's triage flowchart*: fever? cough? travel history?
- **Random Forest** — *a panel of 200 doctors*, each seeing a slightly different subset of the notes, then voting. One doctor's odd opinion gets outvoted.
- **SVM** — drawing the *widest possible road* between two neighbourhoods; the houses on the kerb are the "support vectors".

> ⚠️ **kNN and SVM measure distance, so unscaled features wreck them.** A salary column in the tens of thousands drowns out an age column under 100. **Always scale before kNN and SVM.**

## Metrics — and why accuracy alone is dangerous

The **confusion matrix** is the source of all of them:

```text
                     predicted NO    predicted YES
   actually NO     True Negative     False Positive   <- false alarm
   actually YES    False Negative    True Positive    <- MISS
```

| Metric | Formula | Answers |
|---|---|---|
| **Accuracy** | (TP+TN) / all | *What fraction did I get right?* |
| **Precision** | TP / (TP+FP) | *When I say YES, how often am I right?* |
| **Recall** | TP / (TP+FN) | *Of all real YESes, how many did I catch?* |
| **F1** | harmonic mean | *One number balancing both* |
| **ROC-AUC** | area under ROC | *How well do I rank YES above NO?* |

**The trade-off is real and unavoidable:**

- **Cancer screening** → maximise **recall**. A missed case is fatal; a false alarm means one more test.
- **Spam filter** → maximise **precision**. Junk in the inbox is annoying; a job offer in the spam folder is a disaster.

## 📘 Examples

**Example 1 — the accuracy trap, on real data**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

d = pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv").dropna()
for c in d.select_dtypes(include="object").columns:
    d[c] = LabelEncoder().fit_transform(d[c])

X, y = d.drop(columns=["diabetes"]), d["diabetes"]
print("only", f"{y.mean():.1%}", "of patients have diabetes")   # 8.5%

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
dumb = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
pred = dumb.predict(Xte)

print("accuracy:", round(accuracy_score(yte, pred), 4))   # 0.9150
print("recall  :", round(recall_score(yte, pred), 4))     # 0.0000
print(confusion_matrix(yte, pred))
# [[18300     0]
#  [ 1700     0]]     <- 1,700 diabetics, ZERO detected
```

**91.5% accuracy, and it never found a single patient.** This is the single most important slide in the session.

**Example 2 — all six classifiers, measured**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score, roc_auc_score

loans = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in loans.select_dtypes(include="object").columns:
    loans[c] = LabelEncoder().fit_transform(loans[c])
X, y = loans.drop(columns=["loan_status"]), loans["loan_status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

sc = StandardScaler()
Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)   # fit on TRAIN only

models = {
    "LogisticRegression": (LogisticRegression(max_iter=2000), True),
    "kNN (k=5)":          (KNeighborsClassifier(n_neighbors=5), True),
    "SVM (rbf)":          (SVC(kernel="rbf", probability=True, random_state=42), True),
    "GaussianNB":         (GaussianNB(), True),
    "DecisionTree":       (DecisionTreeClassifier(max_depth=6, random_state=42), False),
    "RandomForest":       (RandomForestClassifier(n_estimators=200, random_state=42), False),
}
for name, (m, needs_scaling) in models.items():
    a, b = (Xtr_s, Xte_s) if needs_scaling else (Xtr, Xte)
    m.fit(a, ytr)
    p, prob = m.predict(b), m.predict_proba(b)[:, 1]
    print(f"{name:<20} acc {accuracy_score(yte,p):.4f}  prec {precision_score(yte,p):.4f}"
          f"  rec {recall_score(yte,p):.4f}  f1 {f1_score(yte,p):.4f}  auc {roc_auc_score(yte,prob):.4f}")
```

Measured results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8650 | 0.8405 | 0.9010 | 0.8697 | 0.9401 |
| kNN (k=5) | 0.8490 | 0.8196 | 0.8950 | 0.8556 | 0.9246 |
| SVM (rbf) | 0.8780 | 0.8430 | 0.9290 | 0.8839 | 0.9479 |
| Gaussian Naive Bayes | 0.8245 | **0.7409** | **0.9980** | 0.8504 | 0.9316 |
| Decision Tree | 0.8545 | **0.8857** | **0.8140** | 0.8484 | 0.9467 |
| **Random Forest** | **0.8910** | 0.8710 | 0.9180 | **0.8939** | **0.9638** |

**Look at the two bolded rows.** Naive Bayes catches 99.8% of approvals — because it says "approve" to almost everyone, which is why its precision is the worst. The Decision Tree is the opposite: most trustworthy when it says yes, but it misses the most. **They have similar F1 scores and completely different personalities.** That is why you never report a single number.

**Example 3 — the decision tree, in plain English**

```python
from sklearn.tree import export_text

tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr, ytr)
print(export_text(tree, feature_names=list(X.columns), max_depth=3))
```

**You can hand this printout to a bank manager.** No other model on the list gives you that.

## ✏️ Practice

1. Train all six on the loan data. Which wins on F1?
2. Run kNN **without** scaling. How much accuracy do you lose?
3. Try `k = 1, 5, 25, 101`. Plot accuracy against k. What shape do you get?
4. Print the confusion matrix for Gaussian Naive Bayes. Explain its 99.8% recall from the matrix alone.
5. Set `DecisionTreeClassifier(max_depth=None)`. Compare train and test accuracy. What has happened?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

loans = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in loans.select_dtypes(include="object").columns:
    loans[c] = LabelEncoder().fit_transform(loans[c])
X, y = loans.drop(columns=["loan_status"]), loans["loan_status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
sc = StandardScaler(); Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

# 1 - Random Forest, F1 0.8939.

kn = KNeighborsClassifier().fit(Xtr, ytr)                              # 2
print("kNN unscaled:", round(kn.score(Xte, yte), 4))
kn_s = KNeighborsClassifier().fit(Xtr_s, ytr)
print("kNN scaled  :", round(kn_s.score(Xte_s, yte), 4))
# Unscaled, person_income (tens of thousands) drowns out every other column.

for k in [1, 5, 25, 101]:                                              # 3
    m = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, ytr)
    print(f"k={k:<4} acc {m.score(Xte_s, yte):.4f}")
# Low k = jumpy and overfitted; very high k = smoothed into mush.
# The curve rises then falls: there is a sweet spot in the middle.

nb = GaussianNB().fit(Xtr_s, ytr)                                      # 4
print(confusion_matrix(yte, nb.predict(Xte_s)))
# The bottom row is almost all True Positives -> recall near 1.
# But the TOP-RIGHT cell is huge: it approves masses of bad loans too.
# It says YES to nearly everyone. High recall, poor precision.

deep = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)           # 5
print("train:", round(deep.score(Xtr, ytr), 4))   # ~1.0
print("test :", round(deep.score(Xte, yte), 4))   # much lower
# It MEMORISED the training set. That is overfitting -- see Session 8.
```
</details>

## ❓ MCQs

**Q1.** A model scores 91.5% accuracy but finds none of the 1,700 positive cases. What happened?
- (a) A bug  (b) The classes are imbalanced and accuracy is hiding it  (c) Too few features  (d) Wrong library

**Q2.** Which two algorithms **must** have scaled features?
- (a) Decision Tree and Random Forest  (b) kNN and SVM  (c) Naive Bayes and Logistic Regression  (d) None

**Q3.** For cancer screening you should maximise…
- (a) Precision  (b) Recall  (c) Accuracy  (d) Training speed

**Q4.** For a spam filter you should maximise…
- (a) Precision  (b) Recall  (c) Training speed  (d) Number of features

**Q5.** Gaussian Naive Bayes gets recall 0.998 and precision 0.741. This means it…
- (a) Is the best model  (b) Says YES to almost everything  (c) Says NO to almost everything  (d) Is broken

**Q6.** Which model can you print out and hand to a non-technical manager?
- (a) SVM  (b) Random Forest  (c) Decision Tree  (d) kNN

**Q7.** Why does a Random Forest usually beat a single Decision Tree?
- (a) It is bigger  (b) Many trees vote, so one tree's quirks get outvoted  (c) It scales features  (d) It uses more data

**Q8.** `predict_proba` returns…
- (a) The accuracy  (b) The predicted class  (c) The probability of each class  (d) The confusion matrix

<details><summary>Answers</summary>

**A1 — (b).** Only 8.5% of patients have diabetes, so "always say no" scores 91.5%. **Always check your class balance, and always look at recall.**

**A2 — (b) kNN and SVM.** Both measure distance, so a large-range column drowns out the rest. Trees split one column at a time and do not care.

**A3 — (b) Recall.** A missed cancer is fatal; a false alarm means one more test.

**A4 — (a) Precision.** A job offer in the spam folder is far worse than junk in the inbox.

**A5 — (b).** Approving nearly everyone catches nearly every real approval (high recall) while being wrong often (low precision).

**A6 — (c) Decision Tree**, via `export_text` or a plot. Explainability is its real advantage.

**A7 — (b).** Each tree sees a different slice; averaging cancels their individual mistakes.

**A8 — (c).** It gives you confidence, so you can move the decision threshold instead of always using 0.5.
</details>

## 🎯 Tasks

**Task 1 — The six-model bake-off.** On a dataset from [`datasets/classification/`](../../../datasets/classification/) that was not used in class, train all six models and produce the comparison table with all five metrics. **Then write a recommendation naming one model — and state which metric drove your choice and why that metric matters for that particular problem.**

**Task 2 — Choose the metric before you look.** For each of these, decide **before training** whether precision or recall matters more, and write one sentence on the cost of each error type: (a) screening for a rare disease, (b) filtering job applications, (c) detecting credit card fraud, (d) recommending a film. Then train a model for one of them and tune to favour your chosen metric.

**Task 3 — Scaling proof.** Show, with numbers, how much kNN loses when features are unscaled. Then explain in one paragraph why the Decision Tree's score barely moves. **It does wiggle in the last decimal — say why that is not the tree learning anything.**

---

# 4. Saving, loading and inference

A model that only exists inside a notebook is useless. **Training happens once; predicting happens thousands of times.**

🧠 **Analogy: a cooked meal versus the recipe.** Training is cooking the meal — slow, and you only do it once. Saving the model is putting it in the fridge. Inference is reheating a portion whenever someone is hungry. **You do not re-cook the whole meal for every guest.**

```python
import joblib

joblib.dump(model, "model.joblib")     # save
loaded = joblib.load("model.joblib")   # load, in a totally different program
loaded.predict(new_data)               # inference
```

> ⚠️ **The number one deployment bug:** saving the model but not the scaler. Your app then feeds raw values into a model that expects scaled ones, and the predictions are quietly wrong. **The fix is to save a `Pipeline`.**

## The Pipeline: one object, everything inside

```python
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

## ✏️ Practice

1. Train a pipeline, save it, load it, and confirm the scores match exactly.
2. Save the model **without** the scaler, then predict on raw unscaled values. How wrong is it?
3. How large is a `RandomForestClassifier(n_estimators=200)` file compared with a Logistic Regression pipeline?
4. Build a single-row `DataFrame` and predict on it with `predict_proba`.
5. Why should model files be added to `.gitignore`?

<details><summary>Solutions</summary>

```python
import joblib, os, tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

d = pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv").dropna()
for c in d.select_dtypes(include="object").columns:
    d[c] = LabelEncoder().fit_transform(d[c])
X, y = d.drop(columns=["diabetes"]), d["diabetes"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

OUT = tempfile.mkdtemp()          # keep model files OUT of your repo folder
pp = os.path.join(OUT, "pipeline.joblib")

pipe = Pipeline([("scaler", StandardScaler()),                         # 1
                 ("model", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
joblib.dump(pipe, pp)
print("scores match after reload:", pipe.score(Xte, yte) == joblib.load(pp).score(Xte, yte))

sc = StandardScaler().fit(Xtr)                                         # 2
bare = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
print("fed SCALED values :", round(bare.score(sc.transform(Xte), yte), 4))
print("fed RAW values    :", round(bare.score(Xte, yte), 4))
# Worse -- and in a real app it fails SILENTLY. No error, just bad predictions.

rp = os.path.join(OUT, "rf.joblib")                                    # 3
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
joblib.dump(rf, rp)
print("logreg pipeline KB:", round(os.path.getsize(pp) / 1024, 1))
print("random forest   KB:", round(os.path.getsize(rp) / 1024))
# The forest stores 200 whole trees, so it is thousands of times larger.

row = pd.DataFrame([{                                                  # 4
    "gender": 1, "age": 54.0, "hypertension": 1, "heart_disease": 0,
    "smoking_history": 4, "bmi": 31.2, "HbA1c_level": 6.8,
    "blood_glucose_level": 160,
}])[list(X.columns)]              # same names AND same order as training
print("prediction :", pipe.predict(row)[0])
print("probability:", round(pipe.predict_proba(row)[0][1], 4))

# 5 - They are large binaries that change completely on every retrain.
#     Git cannot diff them and the repository bloats fast.
#     Commit the TRAINING SCRIPT; regenerate the model.
```
</details>

## ❓ MCQs

**Q1.** What is the most common model-deployment bug?
- (a) Wrong Python version  (b) Saving the model but not the scaler  (c) Too little RAM  (d) Wrong metric

**Q2.** Why save a `Pipeline` rather than a bare model?
- (a) Smaller file  (b) It carries the preprocessing inside, so raw input works  (c) It trains faster  (d) It is required

**Q3.** `joblib.load()` on an untrusted file is dangerous because…
- (a) It is slow  (b) The file can execute arbitrary code  (c) It corrupts the model  (d) It needs internet

**Q4.** Predicting on one new person needs a DataFrame with…
- (a) Any columns  (b) Exactly the training columns, same names and order  (c) Only numeric columns  (d) The target included

**Q5.** Model files should be gitignored because…
- (a) They are secret  (b) They are large binaries that change entirely on each retrain  (c) Git rejects them  (d) They are temporary

**Q6.** A pipeline prevents the Session 3 leakage bug because…
- (a) It is faster
- (b) `fit` runs the scaler on the training fold only, automatically
- (c) It removes outliers
- (d) It does not — you must still be careful

<details><summary>Answers</summary>

**A1 — (b).** The app then feeds raw values into a model expecting scaled ones and is **quietly wrong** — no error, just bad predictions.

**A2 — (b).** One object, one file, no mismatch possible.

**A3 — (b).** Treat a `.joblib` like an executable. Only load your own.

**A4 — (b).** Names and order both matter. Build it as a `DataFrame`.

**A5 — (b).** Commit the training script and regenerate the model instead.

**A6 — (b).** The pipeline fits every preprocessing step inside the training split. **Structure beats discipline.**
</details>

## 🎯 Tasks

**Task 1 — Two separate programs.** Write `train.py` that trains a pipeline and saves it, and `predict.py` that loads it and predicts on three new rows entered as a DataFrame. **`predict.py` must not import anything from `train.py`** — that separation is exactly what deployment means.

**Task 2 — The silent failure.** Deliberately save a scaler-less model, feed it raw values, and record the accuracy drop. **Write a short paragraph on why this bug is more dangerous than a crash.**

---

# 5. Streamlit App Development

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

## ✏️ Practice

1. Build the minimum app and run it locally.
2. Remove `@st.cache_resource`. What do you notice when moving a slider?
3. Add a sidebar showing the model's test accuracy.
4. Add `st.session_state` history and chart every prediction made in the session.
5. Add a `st.file_uploader` and score a whole CSV at once.

<details><summary>Solutions</summary>

> These are fragments of a Streamlit app. They only run inside `streamlit run app.py` — paste them into your `app.py` rather than into a notebook cell.

```python
# streamlit-only: run with `streamlit run app.py`, not as a plain script
# 2 - Every slider move reloads the model file. With a Random Forest this is
#     visibly slow; a spinner appears on every single interaction.
#     Streamlit reruns the WHOLE script on every interaction. Caching is
#     not an optimisation here -- it is what makes the app usable.

# 3
with st.sidebar:
    st.metric("Test accuracy", "89.1%")
    st.caption("Random Forest, 200 trees, held-out test set")

# 4
if "history" not in st.session_state:
    st.session_state.history = []
if st.button("Predict"):
    st.session_state.history.append(float(prob))
if st.session_state.history:
    st.line_chart(st.session_state.history)

# 5
up = st.file_uploader("Score a CSV", type="csv")
if up:
    new = pd.read_csv(up)
    new["risk"] = model.predict_proba(new)[:, 1]
    st.dataframe(new)
    st.download_button("Download", new.to_csv(index=False), "scored.csv")
```
</details>

## ❓ MCQs

**Q1.** What happens when a user moves a slider in Streamlit?
- (a) Only that widget updates  (b) The whole script reruns top to bottom  (c) Nothing until you press a button  (d) The page reloads from the server cache

**Q2.** `@st.cache_resource` is for…
- (a) DataFrames  (b) Loaded models and connections  (c) Charts  (d) User input

**Q3.** Which value survives a rerun?
- (a) A plain Python variable  (b) Anything in `st.session_state`  (c) Anything global  (d) Nothing

**Q4.** How should an API key reach a Streamlit app?
- (a) Hard-coded at the top  (b) `st.secrets`, with the secrets file gitignored  (c) In the URL  (d) As a slider default

**Q5.** Which command runs the app?
- (a) `python app.py`  (b) `streamlit run app.py`  (c) `streamlit app.py`  (d) `run streamlit app.py`

**Q6.** Your app is slow on every click. First thing to check?
- (a) The internet connection
- (b) Whether the model load is wrapped in `@st.cache_resource`
- (c) The CSS
- (d) The Python version

<details><summary>Answers</summary>

**A1 — (b).** The whiteboard is wiped and rewritten. Understanding this explains almost every Streamlit surprise.

**A2 — (b).** Use `@st.cache_data` for DataFrames and computed values.

**A3 — (b).** Everything else is recreated from scratch on each rerun.

**A4 — (b).** Never commit a key. See [setup-guide.md](../setup-guide.md).

**A5 — (b) `streamlit run app.py`.** Running it with `python` produces warnings and no app.

**A6 — (b).** Reloading a 22MB model on every keypress is the usual cause.
</details>

## 🎯 Tasks

**Task 1 — Ship your model.** Take the pipeline you saved in Topic 4 and build a Streamlit app around it: one input widget per feature, a Predict button, a probability shown with `st.metric`, and a sidebar stating the model's test score and its limitations. **Include an honest disclaimer** — this matters more than the code.

**Task 2 — The batch tool.** Extend it to accept a CSV upload, score every row, sort by risk, and offer a download button. **Handle the case where the uploaded file has the wrong columns** — show a clear message instead of a stack trace.

**Task 3 — Explain your app.** Write a one-page README for your app: what it predicts, what data it was trained on, its measured performance, and **three situations where it should not be trusted.** Every deployed model needs this page.

---

# ✅ Before you move on

- [ ] I can tell regression from classification from the target column alone
- [ ] I can read a Linear Regression's coefficients out loud as a sentence
- [ ] I know why RMSE is the metric to quote, and why R² = 1.0 is suspicious
- [ ] I have trained all six classifiers and compared them on five metrics
- [ ] I know kNN and SVM need scaling, and trees do not
- [ ] I can read a confusion matrix and explain why 91.5% accuracy can be worthless
- [ ] I can name a problem needing high recall and one needing high precision
- [ ] I have saved a **pipeline** and loaded it in a separate program
- [ ] I have built and run a Streamlit app that makes live predictions

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-05-supervised-learning.ipynb) | Every example above, runnable |
| [Streamlit tutorial (simple)](../tutorials/apps/streamlit-app-simple.md) | Step-by-step first app |
| [Streamlit tutorial (advanced)](../tutorials/apps/streamlit-app-advanced.md) | Layout, caching, state |
| [Loan app walkthrough](../tutorials/apps/loan-app.md) | An end-to-end ML app |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
