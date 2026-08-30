# Session 6 — Data Augmentation & Feature Engineering

**Data Augmentation Techniques · Feature Engineering Techniques · Feature Reduction Techniques**

| | |
|---|---|
| **Notebook** | [session-06-augmentation-features.ipynb](../notebooks/session-06-augmentation-features.ipynb) |
| **Previous** | [Session 5 — Supervised Learning](session-05-regression.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **This session is about the data, not the model.** In Session 5 you swapped models and gained a few points. Here you change the *columns and rows* — which in real projects moves the needle far more.
>
> **Every technique in this session is a trade, not a free win.** You will measure the cost of each one. That honesty is the point of the session.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Explain what data augmentation is, and why it is different for tables, images and text
2. Balance an imbalanced dataset three different ways — and **measure what it costs you**
3. Say why `class_weight="balanced"` should be your first attempt, not SMOTE
4. Create new features from existing columns, and check whether they actually helped
5. Reduce features with `SelectKBest`, RFE and PCA
6. Explain why PCA can *hurt* a Random Forest
7. Decide, with evidence, whether a technique is worth keeping

---

## The three topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Data Augmentation](#1-data-augmentation-techniques) | Balancing buys recall and **pays in precision** |
| 2 | [Feature Engineering](#2-feature-engineering-techniques) | A ratio often beats both columns it came from |
| 3 | [Feature Reduction](#3-feature-reduction-techniques) | Fewer features is a trade, not an upgrade |

---

# 1. Data Augmentation Techniques

**Augmentation means creating more training data from the data you already have.**

🧠 **Analogy: revising from one photo of a face.** If you only ever see one photo of your friend, you might not recognise them in a hat, or in dim light, or side-on. Show yourself the same photo flipped, darkened and rotated and you learn *the friend*, not *the photo*. **Augmentation teaches the model the thing, not the example.**

| Data type | Techniques | Why it works |
|---|---|---|
| **Images** | Flip, rotate, crop, brightness, noise | A cat rotated 10° is still a cat |
| **Text** | Synonym swap, back-translation, random deletion | Meaning survives small rewording |
| **Tables** | Oversampling, SMOTE, noise injection, class weights | Rebalances rare classes |

> ⚠️ **Augment the training set only — never the test set.** Augmenting before splitting puts near-copies of the same row on both sides, and your score becomes fiction. This is the Session 3 leakage lesson wearing a different hat.

## The tabular case: imbalanced classes

The diabetes data is 91.5% negative. A model trained on it plays safe and under-predicts the rare class. Three fixes:

| Method | What it does | Cost |
|---|---|---|
| **Random oversampling** | Duplicates minority rows | Duplicates encourage overfitting |
| **SMOTE** | Creates *synthetic* rows between real neighbours | Can invent impossible combinations |
| **`class_weight="balanced"`** | Tells the model rare mistakes cost more | None — **and it is one keyword** |

## 📘 Examples

**Example 1 — the imbalance, and the baseline**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

d = pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv").dropna()
for c in d.select_dtypes(include="object").columns:
    d[c] = LabelEncoder().fit_transform(d[c])

X, y = d.drop(columns=["diabetes"]), d["diabetes"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
print(ytr.value_counts().to_dict())      # {0: 73200, 1: 6800}
```

**Example 2 — SMOTE, written out so you can see it is not magic**

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X_min, n_new, k=5, seed=42):
    """Make n_new synthetic rows on the lines BETWEEN real minority rows."""
    rng = np.random.default_rng(seed)
    A = X_min.to_numpy(float)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
    idx = nn.kneighbors(A, return_distance=False)[:, 1:]      # drop self
    base = rng.integers(0, len(A), n_new)                     # pick a real row
    nbr = idx[base, rng.integers(0, k, n_new)]                # pick a neighbour
    lam = rng.random((n_new, 1))                              # step 0..1 along
    return pd.DataFrame(A[base] + lam * (A[nbr] - A[base]), columns=X_min.columns)
```

**That is the entire idea:** pick a real minority row, pick one of its neighbours, and place a new point somewhere on the line between them. The production version is `pip install imbalanced-learn`, then `SMOTE().fit_resample(X, y)` — but now you know what it does.

**Example 3 — the measured cost of balancing**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (imbalanced) | **0.8588** | 0.6371 | **0.7315** |
| Random oversampling | 0.4238 | 0.8888 | 0.5740 |
| SMOTE | 0.4294 | 0.8835 | 0.5779 |
| `class_weight="balanced"` | 0.4226 | 0.8871 | 0.5725 |

**Read that table carefully, because it is not what students expect.**

Balancing took recall from **0.64 to 0.89** — it now finds far more diabetic patients. But precision collapsed from **0.86 to 0.42**, and **F1 got worse**. The balanced model shouts "diabetes" far more often; most of those extra shouts are wrong.

**So was it worth it?** For a *screening* test, yes — missing a diabetic patient is much worse than sending a healthy one for a confirmatory blood test. For a system that automatically starts medication, absolutely not.

> **This is the real lesson of the session: there is no technique that just makes things better.** There is only "what am I buying, and what am I paying?" Anyone who tells you SMOTE improves models has not measured it.

Notice too that all three methods land in the same place — and `class_weight="balanced"` is **one keyword** against twenty lines of SMOTE. **Try the keyword first.**

## ✏️ Practice

1. Compute the class balance of the diabetes data. What accuracy does "always predict no" get?
2. Train with `class_weight="balanced"`. Report precision, recall and F1 against the baseline.
3. Which changed more, recall or precision? Was the trade worth it? Argue your case.
4. Apply SMOTE **before** the train/test split. Compare the test score with doing it after. Why is the first number a lie?
5. Name an augmentation for images that would **destroy** the label. (Hint: think about digits.)

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

d = pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv").dropna()
for c in d.select_dtypes(include="object").columns:
    d[c] = LabelEncoder().fit_transform(d[c])
X, y = d.drop(columns=["diabetes"]), d["diabetes"]
print("positive rate:", round(y.mean(), 4))                            # 1
print("always-no accuracy:", round(1 - y.mean(), 4))                   # 0.915

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
for label, m in [("baseline", LogisticRegression(max_iter=2000)),      # 2
                 ("balanced", LogisticRegression(max_iter=2000, class_weight="balanced"))]:
    p = m.fit(Xtr, ytr).predict(Xte)
    print(f"{label:<10} prec {precision_score(yte,p):.4f} "
          f"rec {recall_score(yte,p):.4f} f1 {f1_score(yte,p):.4f}")

# 3 - Recall rose ~0.25; precision FELL ~0.44. Precision moved more.
#     Worth it for screening (a miss is far costlier than a false alarm);
#     not worth it if a positive triggers treatment automatically.

# 4 - Synthetic rows are built from real rows. Split afterwards and a
#     synthetic row derived from training data lands in the test set.
#     The model is then graded on rows it has effectively already seen.
#     ALWAYS: split first, augment the TRAINING half only.

# 5 - Horizontal flip on handwritten digits. A flipped 2 is not a 2.
#     Same for rotating 6 by 180 degrees -> it becomes 9.
#     The augmentation must preserve the LABEL, and that depends on the task.
```
</details>

## ❓ MCQs

**Q1.** Data augmentation should be applied to…
- (a) The whole dataset before splitting  (b) The training set only  (c) The test set only  (d) Both halves equally

**Q2.** SMOTE creates new rows by…
- (a) Copying minority rows  (b) Interpolating between a minority row and one of its neighbours  (c) Deleting majority rows  (d) Random noise from scratch

**Q3.** After balancing, recall rose from 0.64 to 0.89 while precision fell from 0.86 to 0.42. This means the model now…
- (a) Is strictly better  (b) Predicts the rare class far more often, and is often wrong when it does  (c) Is broken  (d) Overfits the test set

**Q4.** Which should you try **first** on an imbalanced table?
- (a) SMOTE  (b) `class_weight="balanced"`  (c) Collecting 10× more data  (d) Dropping the majority class

**Q5.** Horizontally flipping images of handwritten digits is a bad augmentation because…
- (a) It is slow  (b) It changes the label — a flipped 2 is not a 2  (c) It needs more memory  (d) Flipping is never valid

**Q6.** F1 got *worse* after balancing. You should…
- (a) Always revert  (b) Decide using the real cost of a miss versus a false alarm  (c) Ignore F1  (d) Use accuracy instead

<details><summary>Answers</summary>

**A1 — (b) The training set only.** Augmenting first puts near-copies on both sides of the split — leakage, and a fictional score.

**A2 — (b).** It places a new point on the line between a real minority row and a real neighbour.

**A3 — (b).** It shouts the rare class more often. Most extra shouts are wrong — that is precisely what falling precision means.

**A4 — (b).** One keyword, no synthetic data, and here it matched SMOTE almost exactly.

**A5 — (b).** Augmentation must **preserve the label**. Flipping is fine for cats and wrong for digits.

**A6 — (b).** F1 weights precision and recall equally, and your problem may not. **Screening wants recall; automated treatment wants precision.**
</details>

## 🎯 Tasks

**Task 1 — The augmentation trade, costed.** Take an imbalanced dataset and produce the four-row table above. Then write a short memo answering: *if this model screens patients, which row do you ship?* **Put a number on the cost of a miss and the cost of a false alarm** — even a made-up number forces the reasoning into the open.

**Task 2 — Leak it on purpose.** Apply SMOTE before splitting, record the test score, then do it correctly and record it again. Report the gap. **Write one paragraph on why the first number would have survived a code review.**

**Task 3 — Image augmentation by hand.** Load an image with PIL and produce five augmented versions (flip, rotate, crop, brightness, noise). **For each, state whether it would preserve the label for (a) cat-vs-dog and (b) handwritten digits.** They are not the same answer.

---

# 2. Feature Engineering Techniques

**Feature engineering is creating new columns that make the pattern easier to see.**

🧠 **Analogy: judging whether a loan is affordable.** You are told "the loan is ₹800,000" and "the income is ₹400,000". You *could* eyeball both. But the number you actually care about is **the ratio: 2.0**. That ratio is a new column — and it can be more informative than either column it came from.

| Technique | Example | When |
|---|---|---|
| **Ratios** | `loan_amnt / person_income` | Two columns only matter *relative* to each other |
| **Differences** | `end_date - start_date` | Duration matters more than the dates |
| **Date parts** | month, weekday, is_weekend | Seasonality and weekly cycles |
| **Binning** | age → child / adult / senior | The effect is by band, not smooth |
| **Interactions** | `rate × amount` | Two features matter *together* |
| **Aggregations** | customer's average past order | Per-group history |

> **Domain knowledge beats algorithms here.** A banker knows loan-to-income matters. No amount of hyperparameter tuning discovers that for you.

## 📘 Examples

**Example 1 — four engineered features on the loan data**

```python
loans["loan_to_income"]      = loans.loan_amnt / loans.person_income
loans["income_per_year_exp"] = loans.person_income / (loans.person_emp_exp + 1)
loans["credit_per_hist"]     = loans.credit_score / (loans.cb_person_cred_hist_length + 1)
loans["rate_x_amount"]       = loans.loan_int_rate * loans.loan_amnt
```

> ⚠️ **Note the `+ 1` in two denominators.** Someone with 0 years of experience would give division by zero. **Every ratio you build needs this check.**

**Example 2 — did it help? Measure, do not assume**

| Model | Accuracy | F1 |
|---|---|---|
| Random Forest, 13 original features | 0.8910 | 0.8939 |
| Random Forest, + 4 engineered features | 0.8935 | 0.8955 |

**A gain of about 0.25 percentage points.** Real, but small — and honestly reported. Feature engineering is not always dramatic, especially when a Random Forest can already approximate ratios by splitting twice. **It matters much more for linear models, which cannot build a ratio at all.**

**Example 3 — dates into features**

```python
df["order_date"] = pd.to_datetime(df["order_date"])

df["year"]       = df.order_date.dt.year
df["month"]      = df.order_date.dt.month
df["weekday"]    = df.order_date.dt.dayofweek       # 0 = Monday
df["is_weekend"] = df.weekday.isin([5, 6]).astype(int)
df["days_ago"]   = (pd.Timestamp("today") - df.order_date).dt.days
```

**A raw date is almost useless to a model. Its parts are gold.** No model can work out "Saturdays are busy" from a timestamp; every model can learn it from `is_weekend`.

## ✏️ Practice

1. Create `loan_to_income` and check its correlation with `loan_status`. Is it stronger than either original column?
2. Add all four engineered features and compare Random Forest F1 before and after.
3. Add the same four features to a **Logistic Regression**. Is the gain bigger or smaller than for the forest? Why?
4. Bin `person_age` into four bands and compare with using raw age.
5. Build a feature that would cause **leakage** on the loan data, and explain why it is illegal.

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])

L["loan_to_income"] = L.loan_amnt / L.person_income                    # 1
print(L[["loan_amnt", "person_income", "loan_to_income"]]
      .corrwith(L.loan_status).round(4))
# The ratio correlates more strongly than either column alone.

def run(frame, model, scale=False):                                    # 2, 3
    X, y = frame.drop(columns=["loan_status"]), frame["loan_status"]
    a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    if scale:
        s = StandardScaler().fit(a); a, b = s.transform(a), s.transform(b)
    return f1_score(d, model.fit(a, c).predict(b))

L2 = L.copy()
L2["income_per_year_exp"] = L2.person_income / (L2.person_emp_exp + 1)
L2["credit_per_hist"]     = L2.credit_score / (L2.cb_person_cred_hist_length + 1)
L2["rate_x_amount"]       = L2.loan_int_rate * L2.loan_amnt
base = L.drop(columns=["loan_to_income"])

rf = lambda: RandomForestClassifier(n_estimators=200, random_state=42)
lg = lambda: LogisticRegression(max_iter=2000)
print(f"RF  before {run(base, rf()):.4f}   after {run(L2, rf()):.4f}")
print(f"LR  before {run(base, lg(), True):.4f}   after {run(L2, lg(), True):.4f}")
# 3 - The LINEAR model gains more. A forest can approximate a ratio by
#     splitting on both columns; a linear model literally cannot represent
#     division, so you must hand it the ratio.

L3 = L.copy()                                                          # 4
L3["age_band"] = pd.cut(L3.person_age, [0, 25, 40, 60, 200], labels=False)
print(f"raw age {run(L, rf()):.4f}   binned {run(L3.drop(columns=['person_age']), rf()):.4f}")
# Binning usually LOSES information for a tree. It helps linear models,
# where it lets a straight line bend.

# 5 - Anything recorded AFTER the decision. For example a column
#     "amount_repaid_so_far": only approved loans have repayments, so it
#     gives the answer away. The test score would look superb and the
#     model would be useless in production, where that column does not
#     exist yet at decision time.
#     THE TEST: "would I know this value at the moment I must predict?"
```
</details>

## ❓ MCQs

**Q1.** Why can `loan_amnt / person_income` beat both columns it came from?
- (a) It is smaller  (b) Affordability depends on the two *relative* to each other  (c) It removes outliers  (d) It is normalised

**Q2.** Why add `+ 1` to a denominator?
- (a) Style  (b) To avoid division by zero when the column is 0  (c) To scale it  (d) It is not needed

**Q3.** Which model type gains **most** from a hand-built ratio feature?
- (a) Random Forest  (b) Logistic Regression  (c) Decision Tree  (d) kNN

**Q4.** A raw timestamp column is usually…
- (a) The best feature  (b) Near-useless until you split it into parts  (c) Dropped always  (d) The target

**Q5.** You add a column `amount_repaid_so_far` to a loan-approval model. This is…
- (a) Good feature engineering  (b) Leakage — it is unknown at decision time  (c) Binning  (d) Augmentation

**Q6.** Your engineered features gained 0.25 percentage points. You should…
- (a) Discard them as noise  (b) Report the real number honestly and decide if the extra complexity is worth it  (c) Claim a large improvement  (d) Re-run until the number is bigger

<details><summary>Answers</summary>

**A1 — (b).** ₹800,000 is a small loan for a large income and an impossible one for a small income. **The ratio carries what neither column carries alone.**

**A2 — (b).** Zero experience or zero history is a real value in real data, and it will crash your pipeline in production.

**A3 — (b) Logistic Regression.** A linear model cannot represent division; a tree can approximate it with two splits. **Hand the linear model the ratio.**

**A4 — (b).** Split it into month, weekday, is_weekend, days_ago.

**A5 — (b) Leakage.** **The test: would I know this value at the moment I must predict?**

**A6 — (b).** A small honest gain is a real result. Re-running until the number looks good is how you fool yourself.
</details>

## 🎯 Tasks

**Task 1 — Five features from domain knowledge.** Pick a dataset and invent five features from **thinking about the domain**, not from a list. Write one sentence per feature on *why a human expert would care*. Measure before and after. **Report the honest number even if it is small or negative.**

**Task 2 — The leakage hunt.** For a dataset of your choice, list every column and mark each with "known at prediction time: yes/no". **Any 'no' is a leak.** Write down how you would have discovered each one if it had not been obvious.

**Task 3 — Linear versus tree.** Show, with numbers, that a hand-built ratio helps Logistic Regression more than it helps a Random Forest. **Explain the result in terms of what each model can represent.**

---

# 3. Feature Reduction Techniques

**Fewer columns means faster training, simpler explanations, and less overfitting — but usually a little less accuracy.** Once again: a trade.

🧠 **Analogy: packing for a trip.** You could take everything you own. A smaller bag is faster to carry and easier to search — but you will occasionally miss something you wanted. **The skill is knowing what you can leave behind.**

| Method | How it chooses | Type |
|---|---|---|
| `VarianceThreshold` | Drops near-constant columns | Filter |
| `SelectKBest` | Keeps the k with the strongest statistical link to the target | Filter |
| `RFE` | Repeatedly trains and drops the weakest | Wrapper |
| `feature_importances_` | Asks a trained forest which it used | Embedded |
| `PCA` | Builds new combined axes capturing the most variance | Projection |

> **PCA is different from the others.** The first four *keep some of your original columns*. PCA **replaces them all** with new mathematical combinations. You gain compression and lose the ability to say "the model used credit score".

## 📘 Examples

**Example 1 — how many features do you actually need?**

| Features kept (`SelectKBest`) | Accuracy |
|---|---|
| 3 | 0.8415 |
| 5 | 0.8785 |
| 8 | 0.8810 |
| 13 (all) | 0.8910 |

**Going from 13 features to 8 costs one percentage point and drops five columns.** That may be an excellent trade if those five columns are expensive to collect, or slow, or legally awkward. It is a bad trade if they are free.

**Example 2 — PCA, and the surprise**

| PCA components | Variance captured | Accuracy |
|---|---|---|
| 2 | 36.7% | 0.7570 |
| 5 | 64.5% | 0.8350 |
| 8 | 86.9% | 0.8570 |
| 13 | **100%** | 0.8755 |

**Look at the last row.** PCA with 13 components keeps **100% of the variance** — it throws nothing away — yet accuracy is 0.8755 against 0.8910 for the raw 13 columns.

**Why?** PCA *rotates* the data into new diagonal axes. A Random Forest splits on **one column at a time**, along the original axes. Rotating the data means a boundary that was one clean vertical cut now needs a staircase of many cuts to approximate.

> **PCA is not a free upgrade, and on tree models it is often a downgrade.** It shines for linear models, distance-based models like kNN, and visualisation — not for forests.

**Example 3 — asking the model what it used**

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X_train.columns)
print(imp.sort_values(ascending=False).round(4))
```

**This is usually the most useful reduction method you have**, because it uses the model you actually intend to ship.

## ✏️ Practice

1. Run `SelectKBest` with k = 3, 5, 8, 13 and record accuracy for each.
2. Which features does `SelectKBest(k=5)` keep? Do they match the forest's importances?
3. Run PCA with 2, 5, 8, 13 components. Plot variance captured against accuracy.
4. Explain, in your own words, why PCA with 100% variance still loses accuracy for a forest.
5. Repeat the PCA comparison with **kNN** instead of a forest. Is the picture different?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])
X, y = L.drop(columns=["loan_status"]), L["loan_status"]
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
s = StandardScaler().fit(a); a_s, b_s = s.transform(a), s.transform(b)

for k in [3, 5, 8, 13]:                                                # 1
    sel = SelectKBest(f_classif, k=k).fit(a, c)
    m = RandomForestClassifier(n_estimators=200, random_state=42).fit(sel.transform(a), c)
    print(f"SelectKBest k={k:<3} acc {m.score(sel.transform(b), d):.4f}")

sel5 = SelectKBest(f_classif, k=5).fit(a, c)                           # 2
print("\\nkept:", list(X.columns[sel5.get_support()]))
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(a, c)
print("forest top 5:", list(pd.Series(rf.feature_importances_, index=X.columns)
                            .nlargest(5).index))
# They overlap but are NOT identical. SelectKBest scores each column ALONE;
# the forest sees columns in combination.

for n in [2, 5, 8, 13]:                                                # 3
    p = PCA(n_components=n, random_state=42).fit(a_s)
    m = RandomForestClassifier(n_estimators=200, random_state=42).fit(p.transform(a_s), c)
    print(f"PCA n={n:<3} var {p.explained_variance_ratio_.sum():.3f} "
          f"acc {m.score(p.transform(b_s), d):.4f}")

# 4 - PCA ROTATES the data onto diagonal axes. A forest splits one column
#     at a time, along the ORIGINAL axes. After rotation, a boundary that
#     was one clean vertical cut needs a staircase of cuts to approximate.
#     No information is lost - but it is now in a shape the forest
#     finds harder to use.

for n in [2, 5, 8, 13]:                                                # 5
    p = PCA(n_components=n, random_state=42).fit(a_s)
    m = KNeighborsClassifier().fit(p.transform(a_s), c)
    print(f"kNN + PCA n={n:<3} acc {m.score(p.transform(b_s), d):.4f}")
print("kNN raw:", round(KNeighborsClassifier().fit(a_s, c).score(b_s, d), 4))
# kNN measures DISTANCE, and a rotation does not change distances.
# It is not hurt the way the forest is.
```
</details>

## ❓ MCQs

**Q1.** PCA with enough components to keep 100% of the variance still lowered Random Forest accuracy. Why?
- (a) PCA loses information  (b) PCA rotates the axes, and trees split along the original axes  (c) A bug  (d) PCA needs scaling

**Q2.** Which reduction method lets you still say "the model used credit score"?
- (a) PCA  (b) `SelectKBest`  (c) Neither  (d) Both

**Q3.** `VarianceThreshold` removes columns that are…
- (a) Correlated with the target  (b) Nearly constant  (c) Categorical  (d) Missing values

**Q4.** Cutting 13 features to 8 cost about one percentage point. This is a good trade when…
- (a) Always  (b) Those five columns are expensive, slow or legally awkward to collect  (c) Never  (d) The model is linear

**Q5.** Which reduction method uses the model you actually intend to ship?
- (a) `VarianceThreshold`  (b) `SelectKBest`  (c) `feature_importances_` from a trained model  (d) PCA

**Q6.** `SelectKBest` and a forest's importances disagree. The most likely reason is…
- (a) One is broken  (b) `SelectKBest` scores each column alone; the forest sees them in combination  (c) Different random seeds  (d) Missing values

**Q7.** PCA is a better fit for which of these?
- (a) Random Forest  (b) Decision Tree  (c) kNN and linear models  (d) Naive Bayes only

<details><summary>Answers</summary>

**A1 — (b).** No information is lost; it is just in a shape the forest finds harder to cut.

**A2 — (b) `SelectKBest`.** It keeps real columns. PCA replaces them with combinations no one can name.

**A3 — (b).** A column that is the same value for almost every row carries almost no information.

**A4 — (b).** Accuracy is not the only cost in a real system. Collection cost, latency and privacy all count.

**A5 — (c).** Its judgement reflects the model you will actually deploy.

**A6 — (b).** A column can be useless alone and valuable alongside another.

**A7 — (c).** Distance-based and linear models are unaffected by rotation — or helped by it.
</details>

## 🎯 Tasks

**Task 1 — The reduction curve.** For a dataset of your choice, plot accuracy against number of features kept, for both `SelectKBest` and PCA on the same axes. **Mark the point you would ship and justify it in two sentences** — including something other than accuracy.

**Task 2 — Reproduce the PCA surprise.** Show PCA at 100% variance underperforming raw features for a tree model, then show the effect shrinking or vanishing for kNN. **Explain both results with one sentence about how each model draws its boundary.**

**Task 3 — The expensive column.** Suppose one feature costs ₹500 per patient to measure. Compute the accuracy with and without it, and write the decision memo: **at what accuracy gain would you pay for it?** There is no correct answer — the reasoning is the deliverable.

---

# ✅ Before you move on

- [ ] I can explain augmentation for tables, images and text
- [ ] I augment the **training set only**, and I can say why
- [ ] I have measured that balancing buys recall and pays in precision
- [ ] I try `class_weight="balanced"` before reaching for SMOTE
- [ ] I can build a ratio feature and check whether it actually helped
- [ ] I know why a ratio helps a linear model more than a forest
- [ ] I can test any feature with *"would I know this at prediction time?"*
- [ ] I can reduce features three ways and explain what each costs
- [ ] I know why PCA can hurt a tree model and not hurt kNN
- [ ] **I report the honest number, even when it is small**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-06-augmentation-features.ipynb) | Every example above, runnable |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
