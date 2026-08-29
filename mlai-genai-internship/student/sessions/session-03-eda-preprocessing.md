# Session 3 — EDA & Data Preprocessing

**Exploratory Data Analysis · Handling Missing Values · Removing Duplicates · Outlier Detection & Removal · Transformation & Scaling · Encoding · Train-Test Split**

| | |
|---|---|
| **Notebook** | [session-03-eda-preprocessing.ipynb](../notebooks/session-03-eda-preprocessing.ipynb) |
| **Previous** | [Session 2 — NumPy, Pandas & Visualisation](session-02-numpy-pandas.md) |
| **Next** | [Session 4 — Introduction to ML & AI](session-04-intro-ml-ai.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Session 2 taught you to clean data so a human can read it. This session prepares it so a model can learn from it.** Those are different jobs.
>
> By the end you will have `X_train`, `X_test`, `y_train`, `y_test` — four variables that every model in Sessions 5 to 9 expects.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Explore a new dataset systematically and say what is in it
2. Choose a strategy for missing values, and defend it
3. Decide whether a duplicate is an error or a real repeat
4. Detect outliers with the IQR rule and Z-scores — **and know when both are wrong**
5. Scale features with `MinMaxScaler` and `StandardScaler`, and pick between them
6. Encode categories with Label Encoding and dummy variables, and know when each is wrong
7. Split data into `X_train`, `X_test`, `y_train`, `y_test` correctly
8. **Explain data leakage, and show what it costs**

---

## The seven topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Exploratory Data Analysis](#1-exploratory-data-analysis) | Look before you touch anything |
| 2 | [Handling Missing Values](#2-handling-missing-values) | Filling is a guess — say which guess you made |
| 3 | [Removing Duplicates](#3-removing-duplicates) | Not every duplicate is an error |
| 4 | [Outlier Detection & Removal](#4-outlier-detection--removal) | The IQR rule flags tails, not just errors |
| 5 | [Transformation & Scaling](#5-data-transformation--scaling) | Distance-based models need it; trees do not |
| 6 | [Data Encoding](#6-data-encoding-techniques) | Label Encoding invents an order that is not there |
| 7 | [Train-Test Split](#7-train-test-split) | **Fit on train only. Always.** |

**Five checkpoint problems** sit between the topics:

| After topic | Problem |
|---|---|
| 1 | [⭐ The dataset report card](#-checkpoint-problem-1--the-dataset-report-card) |
| 3 | [⭐ The clean-up audit](#-checkpoint-problem-2--the-clean-up-audit) |
| 4 | [⭐ The outlier decision](#-checkpoint-problem-3--the-outlier-decision) |
| 6 | [⭐ Make it all numeric](#-checkpoint-problem-4--make-it-all-numeric) |
| 7 | [⭐ The complete preprocessing pipeline](#-checkpoint-problem-5--the-complete-preprocessing-pipeline) |

**Every topic has the same shape:**

```text
📘 Examples      3-4 short examples of the new idea
🌍 Scenarios     3 examples from real situations
✏️ Tasks         5 scenario-based tasks, with solutions
❓ MCQs          5 questions, with answers and why
```

---

# 1. Exploratory Data Analysis

**EDA is looking at your data before you do anything to it.**

🧠 **Analogy: a doctor before prescribing.** They do not open with medication. They ask questions, take your temperature, listen. **Only then do they decide what to do.** Preprocessing without EDA is prescribing without examining.

## The order to look in

```python
# illustrative: a syntax reference, not runnable as written.
df.shape                      # 1. how big?
df.head()                     # 2. what does a row look like?
df.info()                     # 3. types, and where are the gaps?
df.describe()                 # 4. numeric spread - and impossible values
df.isna().sum()               # 5. missing per column
df.duplicated().sum()         # 6. repeated rows
df["target"].value_counts()   # 7. is the target balanced?
df.corr(numeric_only=True)    # 8. what moves with what?
```

> **Steps 1–7 take ten seconds and decide everything you do afterwards.** Skipping them is how people end up scaling a column that turns out to be an ID number.

## The five questions EDA must answer

| Question | Answered by |
|---|---|
| How much data is there? | `.shape` |
| What is my target, and is it balanced? | `value_counts(normalize=True)` |
| Which columns have gaps? | `.isna().sum()` |
| Are there impossible values? | `.describe()` — look at min and max |
| Which columns are text and need encoding? | `.select_dtypes("object")` |

## 📘 Examples

**Example 1 — the first look**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv")
print("shape:", df.shape)
print(df.head(3))
print("\ntext columns:", df.select_dtypes("object").columns.tolist())
```

**Example 2 — the target, first**

```python
print(df["loan_status"].value_counts())
print(df["loan_status"].value_counts(normalize=True).round(3))
```

**A 50/50 split means accuracy will be a meaningful metric.** If it were 90/10 you would need recall and F1 instead — Session 5 explains why. **Check this before you build anything.**

**Example 3 — `describe()`, read properly**

```python
print(df[["person_age", "person_income", "credit_score"]].describe().round(2))
```

**Two things to notice, and both matter:**

- `person_age` max is **144**. Nobody is 144 — that is Topic 4's problem.
- `person_income` mean is **72,290** against a median of **60,954**. The mean sits well above the middle, so a few very large incomes are dragging it up.

**Example 4 — a reusable EDA function**

```python
def explore(df, target=None):
    print(f"Rows {len(df):,}   Columns {len(df.columns)}")
    print(f"Duplicates: {df.duplicated().sum()}")
    miss = df.isna().sum()
    print("Missing:", miss[miss > 0].to_dict() if miss.any() else "none")
    print("Text columns:", df.select_dtypes("object").columns.tolist())
    if target:
        print(f"\nTarget '{target}':")
        print(df[target].value_counts(normalize=True).round(3).to_string())

explore(df, target="loan_status")
```

**Write it once, run it on every dataset for the rest of the course.**

## 🌍 Scenarios

**Scenario 1 — deciding what preprocessing this dataset needs**

```python
plan = []
if df.isna().sum().any():                        plan.append("handle missing values (Topic 2)")
if df.duplicated().sum():                        plan.append("remove duplicates (Topic 3)")
if (df["person_age"] > 100).any():               plan.append("fix impossible ages (Topic 4)")
if len(df.select_dtypes("object").columns):      plan.append("encode text columns (Topic 6)")
ranges = df.select_dtypes("number").max() - df.select_dtypes("number").min()
if ranges.max() / max(ranges.min(), 1) > 100:    plan.append("scale features (Topic 5)")

print("PREPROCESSING PLAN")
for i, step in enumerate(plan, 1):
    print(f"  {i}. {step}")
```

**EDA does not just describe the data — it produces your to-do list.**

**Scenario 2 — checking a column is what you think it is**

```python
for col in df.select_dtypes("object").columns:
    print(f"{col:<32}{df[col].nunique():>4} distinct  {df[col].unique()[:3]}")
```

**A text column with 9,000 distinct values in 10,000 rows is an ID, not a category.** Encoding it would be a mistake — you would create 9,000 columns.

**Scenario 3 — the group comparison that suggests a signal**

```python
print(df.groupby("loan_status")[["person_income", "loan_percent_income",
                                 "loan_int_rate"]].mean().round(2))
```

**Where the two groups differ most is where your signal is.** `loan_percent_income` separates them clearly — which matches its 0.405 correlation from Session 2.

## ✏️ Tasks

1. Load the loan dataset and print shape, dtypes and the first three rows.
2. Print the target balance as proportions. Is accuracy a fair metric here?
3. Run `describe()` and name two columns with suspicious values. Say why.
4. List every text column with its number of distinct values. Flag any that look like IDs.
5. Write an `explore()` function and run it on a second dataset of your choice.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

print(df.shape); print(df.dtypes); print(df.head(3))                   # 1

print(df["loan_status"].value_counts(normalize=True).round(3))         # 2
# 50/50 - balanced, so accuracy IS meaningful here. At 90/10 it would
# not be, and you would report recall and F1 instead.

print(df.describe().round(2).to_string())                              # 3
print("\nperson_age max =", df.person_age.max(), "-> impossible")
print("person_income mean", round(df.person_income.mean()),
      "vs median", round(df.person_income.median()), "-> skewed")

for c in df.select_dtypes("object").columns:                           # 4
    n = df[c].nunique()
    flag = "  <- looks like an ID" if n > len(df) * 0.5 else ""
    print(f"{c:<32}{n:>5} distinct{flag}")

def explore(d, target=None):                                           # 5
    print(f"Rows {len(d):,}  Columns {len(d.columns)}  Duplicates {d.duplicated().sum()}")
    m = d.isna().sum()
    print("Missing:", m[m > 0].to_dict() if m.any() else "none")
    print("Text:", d.select_dtypes("object").columns.tolist())
    if target:
        print(d[target].value_counts(normalize=True).round(3).to_string())

explore(pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv"),
        target="diabetes")
```
</details>

## ❓ MCQs

**Q1.** What should you do first with a new dataset?
- (a) Train a model  (b) Explore it — shape, head, info, describe  (c) Scale it  (d) Split it

**Q2.** Why check the target balance before anything else?
- (a) Curiosity  (b) It decides whether accuracy is a fair metric  (c) It is required  (d) It speeds training

**Q3.** A text column has 9,000 distinct values in 10,000 rows. It is probably…
- (a) A useful category  (b) An ID, which should not be encoded as a category  (c) Missing data  (d) The target

**Q4.** The mean is far above the median. This suggests…
- (a) Missing values  (b) The column is skewed by some large values  (c) An error  (d) Nothing

**Q5.** What does EDA produce, besides understanding?
- (a) A model  (b) Your preprocessing to-do list  (c) A chart  (d) A report

<details><summary>Answers</summary>

**A1 — (b) Explore it.** **Preprocessing without EDA is prescribing without examining.**

**A2 — (b).** At 90/10, a model that predicts "no" every time scores 90% and is useless.

**A3 — (b) An ID.** Encoding it would create 9,000 columns.

**A4 — (b) Skew.** The fastest outlier detector you have.

**A5 — (b) Your to-do list.** Each thing you find maps onto a topic in this session.
</details>

---

## ⭐ Checkpoint Problem 1 — The dataset report card

> **Uses only:** EDA. Topic 1.

**The problem.** Write a function that takes any DataFrame and a target column and prints a complete report card: size, duplicates, missing values, target balance, impossible-looking values, and a preprocessing plan.

<details><summary>Solution</summary>

```python
import pandas as pd

def report_card(df, target=None):
    print("=" * 60)
    print(f"  {len(df):,} rows  x  {len(df.columns)} columns")
    print("=" * 60)

    # --- structure
    num = df.select_dtypes("number").columns.tolist()
    txt = df.select_dtypes("object").columns.tolist()
    print(f"\nNumeric ({len(num)}): {num}")
    print(f"Text    ({len(txt)}): {txt}")

    # --- problems
    print("\nPROBLEMS FOUND")
    problems = []

    miss = df.isna().sum()
    if miss.any():
        problems.append(f"missing values in {(miss > 0).sum()} column(s): "
                        f"{miss[miss > 0].to_dict()}")
    dups = df.duplicated().sum()
    if dups:
        problems.append(f"{dups} duplicate row(s)")

    for c in num:
        if df[c].min() < 0 and "score" not in c.lower():
            problems.append(f"{c} has negative values (min {df[c].min()})")
        if df[c].mean() > df[c].median() * 1.5:
            problems.append(f"{c} is heavily skewed "
                            f"(mean {df[c].mean():,.0f} vs median {df[c].median():,.0f})")
    for c in txt:
        if df[c].nunique() > len(df) * 0.5:
            problems.append(f"{c} has {df[c].nunique()} distinct values - likely an ID")

    print("  none" if not problems else
          "\n".join(f"  - {p}" for p in problems))

    # --- target
    if target:
        bal = df[target].value_counts(normalize=True)
        print(f"\nTARGET '{target}'")
        print(bal.round(3).to_string())
        minority = bal.min()
        print("  balanced - accuracy is fair" if minority > 0.4 else
              f"  IMBALANCED ({minority:.1%} minority) - report recall and F1")

    # --- the plan
    print("\nPREPROCESSING PLAN")
    steps = []
    if miss.any():                  steps.append("handle missing values")
    if dups:                        steps.append("remove duplicates")
    if any("skewed" in p or "negative" in p for p in problems):
        steps.append("investigate outliers")
    if txt:                         steps.append("encode text columns")
    steps.append("scale features (if using a distance-based model)")
    steps.append("split into X_train / X_test / y_train / y_test")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")


BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
report_card(pd.read_csv(BASE + "loan_data_10k.csv"), target="loan_status")
```

**The value is not the printing — it is that the function forces you to look at every one of these things.** Run it on a dataset and you cannot accidentally skip the target balance.
</details>

**Make it harder:**

1. Add a check for columns where a single value covers over 95% of rows (near-constant, and usually useless).
2. Add the correlation of every numeric column with the target, sorted.
3. Run it on three datasets and compare the plans it produces.

---

# 2. Handling Missing Values

**Session 2 showed you `fillna` and `dropna`. This topic is about choosing between them, and being honest about what you chose.**

🧠 **Analogy: a survey with unanswered questions.** You can throw away every incomplete form (you lose data), write in a typical answer (you invent information), or note that it was skipped and carry on. **All three are defensible. Silently doing one without saying so is not.**

## Why values go missing matters

| Reason | Example | What to do |
|---|---|---|
| **Random** | The scanner glitched | Fill — it will not bias anything |
| **Not random** | High earners skip the income question | **Filling here biases your data.** Consider a "missing" flag |
| **Not applicable** | `spouse_name` for unmarried people | Not really missing — it is a valid absence |

> **The third case is the one people get wrong.** Filling `spouse_name` with the most common name is nonsense. **Ask why it is missing before deciding how to fill it.**

## The strategies

```python
# illustrative: a syntax reference, not runnable as written.
df.dropna()                                  # drop rows with any gap
df.dropna(subset=["income"])                 # only if income is missing
df = df.drop(columns=["mostly_empty_col"])   # drop the column instead

df["age"] = df["age"].fillna(df["age"].median())        # numeric, skewed
df["age"] = df["age"].fillna(df["age"].mean())          # numeric, symmetric
df["city"] = df["city"].fillna(df["city"].mode()[0])    # categorical
df["city"] = df["city"].fillna("Unknown")               # make it a category
df["reading"] = df["reading"].ffill()                   # time series
```

## Choosing by how much is missing

| Missing | Usually |
|---|---|
| **0–5%** | Drop those rows — you can spare them |
| **5–40%** | Fill, and say how |
| **Over 40%** | Consider dropping the column; ask whether it is worth keeping |
| **Over 70%** | Almost always drop the column |

> ⚠️ **`SimpleImputer` from scikit-learn does the same job but fits on training data only** — which is what Topic 7 requires. Use `fillna` while exploring; use `SimpleImputer` inside a pipeline when you build a model.

## 📘 Examples

**Example 1 — find them, and their share**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

d = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")
miss = d.isna().sum()
pct = (miss / len(d) * 100).round(1)
print(pd.DataFrame({"missing": miss, "percent": pct}))
```

**Example 2 — the cost of dropping**

```python
print(f"all rows        : {len(d)}")
print(f"after dropna()  : {len(d.dropna())}")
print(f"lost            : {len(d) - len(d.dropna())} "
      f"({(1 - len(d.dropna()) / len(d)):.0%})")
```

**Always compute this before you call `dropna()`.** On a small dataset it can cost you a quarter of your rows.

**Example 3 — filling, column by column**

```python
clean = d.copy()
clean["Country"] = clean["Country"].fillna(clean["Country"].mode()[0])
clean["Age"] = clean["Age"].fillna(clean["Age"].median())
clean["Salary"] = clean["Salary"].fillna(clean["Salary"].median())
print("gaps remaining:", clean.isna().sum().sum())
```

**Different columns need different treatment.** One blanket `fillna(0)` would put a salary of zero on a real person.

**Example 4 — the median-versus-mean choice, measured**

```python
salary = d["Salary"]
print(f"mean   {salary.mean():,.0f}")
print(f"median {salary.median():,.0f}")
print(f"filled with mean   -> new mean {salary.fillna(salary.mean()).mean():,.0f}")
print(f"filled with median -> new mean {salary.fillna(salary.median()).mean():,.0f}")
```

**Filling with the mean leaves the mean unchanged by construction** — which sounds neat but understates the spread. **The median is the safer default on skewed data.**

## 🌍 Scenarios

**Scenario 1 — a missing-value flag, when absence is informative**

```python
df = pd.read_csv(BASE + "prepreprocessing/pre_data.csv").copy()

df["salary_was_missing"] = df["Salary"].isna().astype(int)   # keep the fact
df["Salary"] = df["Salary"].fillna(df["Salary"].median())    # then fill

print(df[["Salary", "salary_was_missing"]])
```

**If high earners skip the income question, "this was missing" is itself a signal.** The flag preserves it; filling alone throws it away.

**Scenario 2 — deciding per column, automatically**

```python
d = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")
for col in d.columns:
    pct = d[col].isna().mean() * 100
    if pct == 0:
        continue
    if pct > 40:
        action = "consider dropping the COLUMN"
    elif d[col].dtype == "object":
        action = f"fill with mode ('{d[col].mode()[0]}')"
    else:
        action = f"fill with median ({d[col].median():,.1f})"
    print(f"{col:<12}{pct:>6.1f}% missing   -> {action}")
```

**Scenario 3 — writing it down**

```text
MISSING VALUE TREATMENT

  Country : 1 of 12 (8.3%)  filled with the mode, "France"
  Age     : 1 of 12 (8.3%)  filled with the median, 38.0
  Salary  : 1 of 12 (8.3%)  filled with the median, 61,000
                            a salary_was_missing flag was kept

These three values are ASSUMPTIONS, not measurements. If a conclusion
depends on those rows, it should be checked against the raw data.
```

**That last sentence is what separates careful work from a script that ran.**

## ✏️ Tasks

1. Print the count and percentage missing for every column of `pre_data.csv`.
2. Compute how many rows you would lose to `dropna()`, as a count and a percentage.
3. Fill each column appropriately by type, and confirm no gaps remain.
4. Add a `was_missing` flag for one column before filling it. Explain when that is worth doing.
5. Write the missing-value section of a report for the dataset, stating what you assumed.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
d = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")

miss = d.isna().sum()                                                  # 1
print(pd.DataFrame({"missing": miss, "percent": (miss / len(d) * 100).round(1)}))

lost = len(d) - len(d.dropna())                                        # 2
print(f"dropna() costs {lost} of {len(d)} rows ({lost / len(d):.0%})")
# ALWAYS compute this before calling dropna().

clean = d.copy()                                                       # 3
for col in clean.columns:
    if clean[col].isna().any():
        fill = (clean[col].mode()[0] if clean[col].dtype == "object"
                else clean[col].median())
        clean[col] = clean[col].fillna(fill)
print("gaps remaining:", clean.isna().sum().sum())      # 0

flagged = d.copy()                                                     # 4
flagged["salary_was_missing"] = flagged["Salary"].isna().astype(int)
flagged["Salary"] = flagged["Salary"].fillna(flagged["Salary"].median())
# Worth doing when the absence is INFORMATIVE - if high earners skip the
# income question, "this was missing" is itself a signal. Filling alone
# throws that signal away.

# 5 - State, for each column: how many were missing, what you filled with,
#     and that the filled values are ASSUMPTIONS, not measurements.
```
</details>

## ❓ MCQs

**Q1.** A column is 60% missing. You should usually…
- (a) Fill it with the mean  (b) Consider dropping the column  (c) Fill with 0  (d) Ignore it

**Q2.** Why does the *reason* a value is missing matter?
- (a) It does not  (b) If it is missing for a reason related to its value, filling biases the data  (c) It changes the dtype  (d) It affects speed

**Q3.** For a skewed numeric column, the safer filler is…
- (a) The mean  (b) The median  (c) Zero  (d) The mode

**Q4.** When is a `was_missing` flag worth adding?
- (a) Always  (b) When the absence itself carries information  (c) Never  (d) Only for text

**Q5.** Why use `SimpleImputer` rather than `fillna` when building a model?
- (a) It is faster  (b) It fits on the training data only, which Topic 7 requires  (c) It is more accurate  (d) `fillna` is deprecated

<details><summary>Answers</summary>

**A1 — (b).** Over 40% missing, ask whether the column earns its place.

**A2 — (b).** If high earners skip the income question, filling with the median pulls your data toward a false picture.

**A3 — (b) The median.** One extreme value drags the mean a long way.

**A4 — (b).** It preserves a signal that filling would erase.

**A5 — (b).** Computing the median from the test set too is leakage — Topic 7.
</details>

---

# 3. Removing Duplicates

**Session 2 showed you `drop_duplicates()`. This topic is about deciding whether you should.**

🧠 **Analogy: two identical entries in a shop's till roll.** Either the cashier scanned the same item twice by mistake, **or two customers each bought one.** The till roll alone cannot tell you which — and deleting the wrong one loses a real sale.

## Finding them

```python
# illustrative: a syntax reference, not runnable as written.
df.duplicated().sum()                    # how many
df[df.duplicated(keep=False)]            # every copy, including the first
df.duplicated(subset=["id"]).sum()       # duplicates on a key column only
```

## The decision

| Ask | If yes | If no |
|---|---|---|
| Is there an ID column? | De-duplicate on **that** | Continue |
| Could two rows legitimately be identical? | **Keep them** — you need a distinguishing column | Drop them |
| Are they identical except for one column? | Investigate — it may be a correction | — |

## Removing

```python
# illustrative: a syntax reference, not runnable as written.
df = df.drop_duplicates().reset_index(drop=True)
df = df.drop_duplicates(subset=["txn_id"])         # on a key
df = df.drop_duplicates(subset=["student"], keep="last")   # keep corrections
```

## 📘 Examples

**Example 1 — how many, and where**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

d = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")
print("duplicates:", d.duplicated().sum())
print(d[d.duplicated(keep=False)])
```

**Example 2 — on the big dataset**

```python
loans = pd.read_csv(BASE + "loan_data_10k.csv")
print(f"rows {len(loans):,}  duplicates {loans.duplicated().sum()}")
```

**Example 3 — the duplicate that is real**

```python
sales = pd.DataFrame({"item": ["tea", "tea", "tea", "coffee"],
                      "price": [15, 15, 15, 25]})
print(f"revenue as recorded : {sales['price'].sum()}")            # 70
print(f"after drop_duplicates: {sales.drop_duplicates()['price'].sum()}")  # 40
```

**You just deleted 30 rupees of real sales.** Three people bought tea; `drop_duplicates` cannot tell that from an error.

**Example 4 — the fix: de-duplicate on a key**

```python
sales = pd.DataFrame({"txn_id": [101, 102, 103, 103],
                      "item": ["tea", "tea", "coffee", "coffee"],
                      "price": [15, 15, 25, 25]})
clean = sales.drop_duplicates(subset=["txn_id"])
print(f"revenue: {clean['price'].sum()}")      # 55 - correct
```

**The two genuine tea sales survive; the genuinely duplicated transaction goes.**

## 🌍 Scenarios

**Scenario 1 — duplicates from a merge**

```python
jan = pd.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
feb = pd.DataFrame({"id": [3, 4, 5], "amount": [300, 400, 500]})

merged = pd.concat([jan, feb], ignore_index=True)
print(f"merged {len(merged)} rows, {merged.duplicated().sum()} duplicate(s)")
clean = merged.drop_duplicates(subset=["id"]).reset_index(drop=True)
print(f"clean  {len(clean)} rows")
```

**Overlapping periods are the classic way duplicates appear.**

**Scenario 2 — near-duplicates, which `drop_duplicates` will miss**

```python
people = pd.DataFrame({
    "name": ["Priya Sharma", "priya sharma", "PRIYA SHARMA "],
    "city": ["Kochi", "Kochi", "Kochi"],
})
print("exact duplicates:", people.duplicated().sum())      # 0

people["name_key"] = people["name"].str.strip().str.lower()
print("after normalising:", people.duplicated(subset=["name_key", "city"]).sum())  # 2
```

**Three rows for one person, and `drop_duplicates` found none of them.** Normalise the text first — the wrong-format lesson from Session 2, doing real work.

**Scenario 3 — the honest write-up**

```text
DUPLICATE HANDLING

  1 exact duplicate row found and removed (12 rows -> 11).

  Checked whether it could be a legitimate repeat: the row carries a
  date and full measurements, and no two sessions on the same date
  appear elsewhere in the file. Treated as a data-entry error.

  No ID column exists in this dataset. If one is added later, this
  check should be redone on the ID instead of the whole row.
```

## ✏️ Tasks

1. Count the duplicates in `pre_data.csv` and print every copy.
2. Check the loan dataset for duplicates. How many are there?
3. Build a sales table where two identical rows are both genuine, and show what dropping costs you.
4. Fix task 3 by adding a transaction ID and de-duplicating on that.
5. Find near-duplicates that differ only by case or spacing, and count them properly.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

d = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")                # 1
print("duplicates:", d.duplicated().sum())
print(d[d.duplicated(keep=False)])

loans = pd.read_csv(BASE + "loan_data_10k.csv")                        # 2
print(f"{len(loans):,} rows, {loans.duplicated().sum()} duplicates")

sales = pd.DataFrame({"item": ["tea"] * 3 + ["coffee"],                # 3
                      "price": [15, 15, 15, 25]})
print(sales["price"].sum(), "->", sales.drop_duplicates()["price"].sum())
# 70 -> 40. Thirty rupees of REAL SALES deleted.

sales = pd.DataFrame({"txn_id": [101, 102, 103, 103],                  # 4
                      "item": ["tea", "tea", "coffee", "coffee"],
                      "price": [15, 15, 25, 25]})
print(sales.drop_duplicates(subset=["txn_id"])["price"].sum())    # 55

people = pd.DataFrame({"name": ["Priya Sharma", "priya sharma",        # 5
                                "PRIYA SHARMA "], "city": ["Kochi"] * 3})
print("exact:", people.duplicated().sum())                        # 0
people["key"] = people["name"].str.strip().str.lower()
print("normalised:", people.duplicated(subset=["key", "city"]).sum())   # 2
# drop_duplicates compares EXACT values. Normalise the text first.
```
</details>

## ❓ MCQs

**Q1.** Two rows record tea at 15. Dropping one…
- (a) Is always correct  (b) May delete a real second sale  (c) Is required  (d) Does nothing

**Q2.** What is the best way to de-duplicate when an ID column exists?
- (a) `drop_duplicates()`  (b) `drop_duplicates(subset=["id"])`  (c) `dropna()`  (d) Sort first

**Q3.** `"Priya Sharma"` and `"priya sharma "` are…
- (a) Found by `drop_duplicates()`  (b) Not found — you must normalise the text first  (c) Always dropped  (d) An error

**Q4.** When is `keep="last"` right?
- (a) Never  (b) When later rows are corrections of earlier ones  (c) Always  (d) For text only

**Q5.** Why `reset_index(drop=True)` after dropping rows?
- (a) Speed  (b) The index has gaps; this renumbers it  (c) It is required  (d) It sorts the data

<details><summary>Answers</summary>

**A1 — (b).** **The data alone cannot tell an error from a real repeat.**

**A2 — (b).** The ID makes it unambiguous.

**A3 — (b).** Exact comparison misses them entirely.

**A4 — (b).** Keeping the corrected value rather than the original.

**A5 — (b).** Without `drop=True` the old index is kept as a new column.
</details>

---

## ⭐ Checkpoint Problem 2 — The clean-up audit

> **Uses:** EDA, missing values, duplicates. Topics 1–3.

**The problem.** Write a function that cleans any DataFrame's missing values and duplicates, and returns both the cleaned frame **and a log of every decision it made**.

<details><summary>Solution</summary>

```python
import pandas as pd

def clean_basic(df, drop_threshold=0.4):
    """Handle duplicates and missing values, returning (clean_df, log)."""
    log, clean = [], df.copy()

    # --- duplicates first, so they cannot skew the medians we compute next
    n = clean.duplicated().sum()
    if n:
        clean = clean.drop_duplicates().reset_index(drop=True)
        log.append(f"removed {n} exact duplicate row(s)")

    # --- columns that are mostly empty
    for col in list(clean.columns):
        share = clean[col].isna().mean()
        if share > drop_threshold:
            clean = clean.drop(columns=[col])
            log.append(f"dropped column '{col}' ({share:.0%} missing)")

    # --- fill what remains, by type
    for col in clean.columns:
        n = clean[col].isna().sum()
        if not n:
            continue
        if clean[col].dtype == "object":
            fill = clean[col].mode()[0]
            log.append(f"filled {n} missing '{col}' with mode '{fill}'")
        else:
            fill = clean[col].median()
            log.append(f"filled {n} missing '{col}' with median {fill:,.2f}")
        clean[col] = clean[col].fillna(fill)

    return clean, log


BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
raw = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")
clean, log = clean_basic(raw)

print("CLEANING LOG")
for i, step in enumerate(log, 1):
    print(f"  {i}. {step}")
print(f"\n{len(raw)} rows -> {len(clean)} rows")
print(f"gaps remaining: {clean.isna().sum().sum()}")
print(f"duplicates remaining: {clean.duplicated().sum()}")
```

**Two design decisions worth defending:**

1. **Duplicates go first.** A duplicated row would otherwise be counted twice in every median computed afterwards, so removing it first makes those medians honest.
2. **The function returns the log, it does not just print it.** That means the caller can store it in a report — which is what makes the cleaning defensible later.
</details>

**Make it harder:**

1. Add a `was_missing` flag for any column over 10% missing.
2. Add near-duplicate detection on normalised text columns.
3. Return a third value: a dictionary of the fill values used, so the same fills can be applied to new data later. *(This is exactly what `SimpleImputer` stores — and Topic 7 explains why it matters.)*

---

# 4. Outlier Detection & Removal

**An outlier is a value far from the rest.** Some are errors. Some are real and important. **Telling them apart is judgement, not code.**

🧠 **Analogy: a class where everyone is 1.6–1.8 m tall and one student is 2.1 m.** Is that a typing error, or is there a very tall student? **The data cannot tell you. You have to look.** And if you are studying basketball recruitment, that student is the entire point.

## The two standard rules

### The IQR rule

```python
# illustrative: a syntax reference, not runnable as written.
q1, q3 = df["col"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["col"] < low) | (df["col"] > high)]
```

**The middle 50% of the data spans `iqr`. Anything more than 1.5 × that beyond the quartiles is flagged.** It is what a box plot draws.

### The Z-score rule

```python
# illustrative: a syntax reference, not runnable as written.
z = (df["col"] - df["col"].mean()) / df["col"].std()
outliers = df[z.abs() > 3]
```

**More than 3 standard deviations from the mean.** Assumes a roughly bell-shaped column — **on a skewed column it under-reports**, because the outliers themselves inflate the standard deviation.

## ⚠️ Both rules are mechanical, and both can be wrong

Measured on the loan dataset:

| Column | Q1 | Q3 | IQR bounds | Flagged by IQR | Flagged by Z |
|---|---|---|---|---|---|
| `person_age` | 24 | 30 | **[15, 39]** | **496 (5.0%)** | 170 |
| `person_income` | 41,622 | 87,422 | **[−27,078, 156,121]** | 501 (5.0%) | 107 |
| `loan_amnt` | 5,000 | 14,000 | [−8,500, 27,500] | 157 (1.6%) | — |

**Read those bounds carefully, because both are nonsense in different ways.**

1. **`person_age` bounds are [15, 39].** By this rule, **being 45 makes you an outlier.** Nearly 500 people are flagged, and almost none of them are errors — they are just older borrowers. The rule found the *tail of a skewed distribution*, not mistakes.

2. **`person_income` has a lower bound of −27,078.** A negative income is impossible, so that half of the rule can never trigger. **The rule is not wrong — it is simply not meaningful here.**

> **The IQR rule flags tails, not errors.** On a skewed column it will always flag around 5% of your data, whatever that data is. **Never delete rows just because a formula flagged them.**
>
> The real error in this dataset — the single person aged **144** — is caught by *knowing what an age can be*, not by either rule.

## What to actually do

| Option | When | Code |
|---|---|---|
| **Investigate** | Always first | `df[df["col"] > limit]` |
| **Fix** | A typo you can reason about | `df.loc[cond, "col"] = value` |
| **Cap (winsorise)** | Extreme but plausible | `df["col"].clip(low, high)` |
| **Remove** | Impossible, and you have data to spare | `df = df[df["col"] <= limit]` |
| **Keep** | It is real and it matters | do nothing, and say so |
| **Use a robust model** | Many outliers | trees, or `RobustScaler` (Topic 5) |

## 📘 Examples

**Example 1 — the IQR rule, and reading its bounds**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ["person_age", "person_income", "loan_amnt"]:
    lo, hi = iqr_bounds(df[col])
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:<16} bounds [{lo:>12,.0f}, {hi:>10,.0f}]   flagged {n:>4} ({n/len(df):.1%})")
```

**Example 2 — Z-scores, and why they differ**

```python
for col in ["person_age", "person_income"]:
    z = (df[col] - df[col].mean()) / df[col].std()
    print(f"{col:<16} |z| > 3: {(z.abs() > 3).sum()}")
```

**Z flags far fewer than IQR here.** On a skewed column the extreme values inflate the standard deviation, which widens the threshold and hides them. **The two rules disagree, and neither is authoritative.**

**Example 3 — the outlier that is a genuine error**

```python
print("age range:", df["person_age"].min(), "to", df["person_age"].max())
print("over 100:", (df["person_age"] > 100).sum())
print(df[df["person_age"] > 100][["person_age", "person_income", "loan_amnt"]])
```

**One row, age 144.** No formula found it — **domain knowledge did.**

**Example 4 — the three treatments compared**

```python
col = "person_income"
lo, hi = iqr_bounds(df[col])

removed = df[(df[col] >= lo) & (df[col] <= hi)]
capped = df.copy(); capped[col] = capped[col].clip(lo, hi)

print(f"{'':<12}{'rows':>8}{'mean':>12}{'max':>12}")
print(f"{'original':<12}{len(df):>8}{df[col].mean():>12,.0f}{df[col].max():>12,.0f}")
print(f"{'removed':<12}{len(removed):>8}{removed[col].mean():>12,.0f}{removed[col].max():>12,.0f}")
print(f"{'capped':<12}{len(capped):>8}{capped[col].mean():>12,.0f}{capped[col].max():>12,.0f}")
```

**Capping keeps every row and pulls the extremes in. Removing costs you 501 rows** — and, importantly, every other column in them.

## 🌍 Scenarios

**Scenario 1 — domain limits beat statistical rules**

```python
limits = {
    "person_age": (18, 100),          # a borrower is an adult, and mortal
    "person_income": (0, 5_000_000),  # income cannot be negative
    "loan_amnt": (500, 500_000),
    "credit_score": (300, 850),       # the scale's own bounds
}

for col, (lo, hi) in limits.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:<16}{bad:>4} outside {lo:,}-{hi:,}")
```

**Writing limits down forces you to think about what each column means.** That is the actual work — and it found the age-144 row, which the IQR rule buried among 496 false alarms.

**Scenario 2 — when the outlier is the point**

```python
big = df[df["person_income"] > df["person_income"].quantile(0.99)]
print(f"top 1% of earners: {len(big)} people")
print(f"their approval rate: {big['loan_status'].mean():.1%}")
print(f"everyone else:      {df['loan_status'].mean():.1%}")
```

**If you are studying high-value lending, deleting the high earners deletes your subject.** In fraud detection the outliers *are* the fraud. **Ask what you are studying before you remove anything.**

**Scenario 3 — the decision, written down**

```text
OUTLIER TREATMENT

  person_age: 1 row with age 144. Impossible - removed.
              The IQR rule also flagged 496 rows (bounds [15, 39]), but
              those are simply older borrowers, not errors. NOT removed.

  person_income: max 2,448,661. Extreme but possible. KEPT, because
              high-value applications are part of what we are modelling.
              A RobustScaler is used in Topic 5 so these do not distort
              the scaling.

  Rows removed in total: 1 of 10,000.
```

**Notice how little was removed.** A good outlier policy usually deletes very little and explains a lot.

## ✏️ Tasks

1. Compute IQR bounds and outlier counts for three numeric columns. Comment on whether the bounds make sense.
2. Compute Z-score outliers for the same columns. Do the two rules agree?
3. Explain why `person_income` gets a negative IQR lower bound, and what that tells you.
4. Compare removing versus capping `person_income`: rows kept, mean, and maximum.
5. Write a domain-limits dictionary for the loan dataset and report violations.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

def iqr_bounds(s):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ["person_age", "person_income", "loan_amnt"]:               # 1
    lo, hi = iqr_bounds(df[col])
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:<16}[{lo:>12,.0f}, {hi:>10,.0f}]  {n:>4} flagged")
# person_age bounds are [15, 39] - so being 45 makes you an "outlier".
# Nearly 500 rows flagged, and almost none are errors.

for col in ["person_age", "person_income"]:                            # 2
    z = (df[col] - df[col].mean()) / df[col].std()
    print(f"{col:<16}|z|>3: {(z.abs() > 3).sum()}")
# The two rules DISAGREE, and neither is authoritative. On a skewed
# column the extremes inflate the std, which hides them from the Z rule.

# 3 - Q1 is 41,622 and the IQR is 45,800, so Q1 - 1.5*IQR is -27,078.
#     Income cannot be negative, so that half of the rule can never fire.
#     The rule is not wrong - it is simply not MEANINGFUL for this column.

col = "person_income"; lo, hi = iqr_bounds(df[col])                    # 4
removed = df[(df[col] >= lo) & (df[col] <= hi)]
capped = df.copy(); capped[col] = capped[col].clip(lo, hi)
print(f"original {len(df):>6} mean {df[col].mean():>10,.0f} max {df[col].max():>10,.0f}")
print(f"removed  {len(removed):>6} mean {removed[col].mean():>10,.0f} max {removed[col].max():>10,.0f}")
print(f"capped   {len(capped):>6} mean {capped[col].mean():>10,.0f} max {capped[col].max():>10,.0f}")

limits = {"person_age": (18, 100), "person_income": (0, 5_000_000),    # 5
          "loan_amnt": (500, 500_000), "credit_score": (300, 850)}
for col, (lo, hi) in limits.items():
    print(f"{col:<16}{((df[col] < lo) | (df[col] > hi)).sum():>4} outside {lo:,}-{hi:,}")
# THIS found the age-144 row. The IQR rule buried it among 496 false alarms.
```
</details>

## ❓ MCQs

**Q1.** The IQR rule gives `person_age` bounds of [15, 39]. This means…
- (a) Everyone over 39 is a data error  (b) The rule is flagging the tail of a skewed column, not errors  (c) The data is corrupt  (d) Ages are wrong

**Q2.** `person_income` gets an IQR lower bound of −27,078. This tells you…
- (a) Some incomes are negative  (b) That half of the rule can never fire — it is not meaningful here  (c) A bug  (d) To remove the column

**Q3.** Which found the impossible age of 144?
- (a) The IQR rule  (b) The Z-score rule  (c) Domain knowledge about what an age can be  (d) `dropna()`

**Q4.** In fraud detection, outliers should be…
- (a) Removed  (b) Kept — they are what you are looking for  (c) Capped  (d) Averaged

**Q5.** Why does the Z-score rule under-report on a skewed column?
- (a) It is broken  (b) The extreme values inflate the standard deviation, widening the threshold  (c) It needs more data  (d) It only works on integers

<details><summary>Answers</summary>

**A1 — (b).** **The IQR rule flags tails, not errors.** On a skewed column it flags roughly 5% of anything.

**A2 — (b).** The rule is mechanical; it does not know income cannot be negative.

**A3 — (c) Domain knowledge.** No formula found it.

**A4 — (b).** **The outliers are the subject.** Deleting them deletes the problem.

**A5 — (b).** The outliers hide themselves by widening the very threshold meant to catch them.
</details>

---

## ⭐ Checkpoint Problem 3 — The outlier decision

> **Uses:** EDA and outlier detection. Topics 1–4.

**The problem.** For each numeric column of the loan dataset, report the IQR bounds, the count flagged, and a **recommendation** — investigate, cap, remove or keep — with a one-line reason. Then apply your recommendations and report how many rows you actually removed.

<details><summary>Solution</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

# What each column can PLAUSIBLY be. This is the domain knowledge that
# no statistical rule contains.
DOMAIN = {
    "person_age": (18, 100),
    "person_income": (0, 5_000_000),
    "person_emp_exp": (0, 60),
    "loan_amnt": (500, 500_000),
    "loan_int_rate": (0, 40),
    "loan_percent_income": (0, 2),
    "cb_person_cred_hist_length": (0, 60),
    "credit_score": (300, 850),
}

print(f"{'column':<28}{'IQR bounds':>26}{'IQR flag':>10}{'impossible':>12}  recommendation")
print("-" * 108)

impossible_mask = pd.Series(False, index=df.index)

for col, (dlo, dhi) in DOMAIN.items():
    q1, q3 = df[col].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr = ((df[col] < lo) | (df[col] > hi)).sum()

    bad = (df[col] < dlo) | (df[col] > dhi)
    n_bad = bad.sum()
    impossible_mask |= bad

    if n_bad:
        rec = f"REMOVE {n_bad} impossible row(s)"
    elif n_iqr / len(df) > 0.03:
        rec = "KEEP - a skewed tail, not errors"
    elif n_iqr:
        rec = "KEEP - few, and plausible"
    else:
        rec = "nothing to do"

    print(f"{col:<28}[{lo:>10,.0f},{hi:>11,.0f}]{n_iqr:>10}{n_bad:>12}  {rec}")

clean = df[~impossible_mask].reset_index(drop=True)
print(f"\nRemoved {impossible_mask.sum()} row(s) of {len(df):,} "
      f"({impossible_mask.sum() / len(df):.2%})")
print(f"Kept {len(clean):,} rows")
```

**The point of this exercise is the gap between the two count columns.**

The IQR rule flags hundreds of rows. Domain limits flag **one**. Deleting on the IQR rule would have thrown away roughly 5% of the dataset — including every older borrower and every high earner — to remove a single genuine error.

**A good outlier policy deletes very little and explains a lot.**
</details>

**Make it harder:**

1. Add a Z-score column to the table and note where the two rules disagree most.
2. Instead of removing, cap each column at its domain limits with `clip()` and compare the means.
3. Compare the correlation of each column with `loan_status` before and after your treatment. Did removing outliers change the signal?

---

# 5. Data Transformation & Scaling

**Scaling puts every column on a comparable range.** Some models need it badly; others do not care at all.

🧠 **Analogy: comparing a person's height in millimetres with their age in years.** Height gives numbers around 1,700; age gives numbers around 25. **Any method that measures distance will treat height as almost the whole story — not because it matters more, but because its numbers are bigger.** Scaling removes the unfairness of the units.

## The two you were asked for

### `MinMaxScaler` — squash into 0 to 1

```python
# illustrative: a syntax reference, not runnable as written.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

```text
value - min
-----------      ->  every column ends up in [0, 1]
 max - min
```

### `StandardScaler` — centre on 0, spread of 1

```python
# illustrative: a syntax reference, not runnable as written.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```text
value - mean
------------     ->  mean 0, standard deviation 1
    std
```

### And a third worth knowing: `RobustScaler`

```python
from sklearn.preprocessing import RobustScaler   # uses the median and IQR
```

**Because it uses the median rather than the mean, outliers barely move it** — which matters on the skewed income column from Topic 4.

## Choosing

| Scaler | Output | Use when | Outliers |
|---|---|---|---|
| **MinMaxScaler** | [0, 1] | You need a bounded range; neural networks | **Badly affected** — one extreme value squashes everything else |
| **StandardScaler** | mean 0, std 1 | **The default** for most models | Affected, but less |
| **RobustScaler** | median 0 | The column has real outliers you are keeping | **Barely affected** |

Measured on the loan data (first 1,000 rows, three columns):

| Scaler | Means | Minimums | Maximums |
|---|---|---|---|
| StandardScaler | `[-0.0, 0.0, 0.0]` | `[-1.14, -1.39, -3.69]` | `[7.16, 6.46, 2.37]` |
| MinMaxScaler | `[0.14, 0.18, 0.61]` | `[0.0, 0.0, 0.0]` | `[1.0, 1.0, 1.0]` |
| RobustScaler | `[0.29, 0.20, -0.09]` | `[-0.83, -1.09, -2.88]` | `[7.33, 6.25, 1.71]` |

**Look at the MinMax means: 0.14 and 0.18.** Those columns are skewed, so almost every value is crushed into the bottom fifth of the range while one extreme value holds the top. **That is MinMax's weakness, visible in one number.**

## ⚠️ Which models actually need this

| Needs scaling | Does not care |
|---|---|
| kNN, SVM, K-Means (**they measure distance**) | Decision Tree |
| Logistic / Linear Regression (for stable fitting) | Random Forest |
| Neural networks | Gradient boosting |
| PCA | Naive Bayes |

> **Trees split one column at a time, so the relative scale of columns is irrelevant to them.** You will measure exactly this in Session 5.

## The rule that matters most

```python
# illustrative: a syntax reference, not runnable as written.
scaler.fit(X_train)                       # learn min/max/mean from TRAIN ONLY
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # apply the SAME numbers
```

> ⚠️ **Never call `fit` or `fit_transform` on the test set.** Fitting on all the data lets information from the test set influence your training — that is leakage, and Topic 7 measures what it costs.

## 📘 Examples

**Example 1 — the same column, three ways**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv")
sub = df[["person_age", "person_income", "credit_score"]].head(1000)

for name, scaler in [("StandardScaler", StandardScaler()),
                     ("MinMaxScaler", MinMaxScaler()),
                     ("RobustScaler", RobustScaler())]:
    out = pd.DataFrame(scaler.fit_transform(sub), columns=sub.columns)
    print(f"{name:<16} mean {out.mean().round(2).tolist()}  "
          f"max {out.max().round(2).tolist()}")
```

**Example 2 — why the units were unfair**

```python
print(sub.describe().loc[["min", "max"]].round(0))
# person_income spans tens of thousands; credit_score spans hundreds.
# Before scaling, income dominates any distance calculation entirely.
```

**Example 3 — scaling does not change the shape**

```python
scaled = pd.Series(StandardScaler().fit_transform(sub[["person_income"]]).ravel())
print(f"original correlation with age: {sub['person_income'].corr(sub['person_age']):.4f}")
print(f"scaled   correlation with age: {scaled.corr(sub['person_age']):.4f}")
```

**Identical.** Scaling moves and stretches a column; it does not change how it relates to anything else. **It is a change of units, not of information.**

**Example 4 — reversing it**

```python
scaler = StandardScaler().fit(sub)
scaled = scaler.transform(sub)
back = scaler.inverse_transform(scaled)
print("recovered original:", pd.DataFrame(back, columns=sub.columns).head(2).round(1).to_dict("records"))
```

**`inverse_transform` gets your real units back** — which you need whenever you show a prediction to a person.

## 🌍 Scenarios

**Scenario 1 — the correct fit/transform pattern**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

d = df.dropna().copy()
for c in d.select_dtypes("object").columns:
    d[c] = LabelEncoder().fit_transform(d[c])

X = d.drop(columns=["loan_status"])
y = d["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # FIT on train
X_test_scaled = scaler.transform(X_test)          # TRANSFORM test with the same numbers

print("train mean ~0:", X_train_scaled.mean().round(4))
print("test mean is NOT exactly 0:", X_test_scaled.mean().round(4))
```

> **The test mean is not exactly zero, and that is correct.** The test set was scaled using the *training* set's mean — which is precisely the point. If it came out at exactly zero, you would have fitted on the test data.

**Scenario 2 — a pipeline makes the mistake impossible**

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])
pipe.fit(X_train, y_train)            # the scaler fits on the training fold only
print("accuracy:", round(pipe.score(X_test, y_test), 4))
```

**A pipeline removes the chance of getting it wrong.** You will use this everywhere from Session 5 onwards.

**Scenario 3 — choosing when outliers are real**

```python
income = df[["person_income"]]
for name, sc in [("MinMax", MinMaxScaler()), ("Robust", RobustScaler())]:
    out = sc.fit_transform(income).ravel()
    print(f"{name:<8} median {pd.Series(out).median():>8.4f}   max {out.max():>8.2f}")
```

**MinMax pushes the median down near zero** because one 2.4-million income owns the top of the range. **RobustScaler keeps the median at zero where it belongs.**

## ✏️ Tasks

1. Scale three columns with all three scalers and compare their means, minimums and maximums.
2. Show that scaling does not change the correlation between two columns.
3. Fit a scaler on the training set and transform both sets. Confirm the test mean is not exactly 0, and explain why that is right.
4. Use `inverse_transform` to recover the original values.
5. Compare MinMax and Robust on the skewed income column. Which keeps the median sensible, and why?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")
sub = df[["person_age", "person_income", "credit_score"]].head(1000)

for name, sc in [("Standard", StandardScaler()), ("MinMax", MinMaxScaler()),  # 1
                 ("Robust", RobustScaler())]:
    o = pd.DataFrame(sc.fit_transform(sub), columns=sub.columns)
    print(f"{name:<10}mean {o.mean().round(2).tolist()}  max {o.max().round(2).tolist()}")

s = pd.Series(StandardScaler().fit_transform(sub[["person_income"]]).ravel())  # 2
print(f"before {sub['person_income'].corr(sub['person_age']):.4f}  "
      f"after {s.corr(sub['person_age']):.4f}")
# IDENTICAL. Scaling is a change of UNITS, not of information.

d = df.dropna().copy()                                                 # 3
for c in d.select_dtypes("object").columns:
    d[c] = LabelEncoder().fit_transform(d[c])
X, y = d.drop(columns=["loan_status"]), d["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_train)
print("train mean", sc.transform(X_train).mean().round(4))
print("test  mean", sc.transform(X_test).mean().round(4))
# The test mean is NOT exactly 0, and that is CORRECT - it was scaled with
# the TRAINING set's mean. Exactly 0 would mean you fitted on the test set.

scaler = StandardScaler().fit(sub)                                     # 4
back = scaler.inverse_transform(scaler.transform(sub))
print(pd.DataFrame(back, columns=sub.columns).head(2).round(1))

for name, s2 in [("MinMax", MinMaxScaler()), ("Robust", RobustScaler())]:  # 5
    o = s2.fit_transform(df[["person_income"]]).ravel()
    print(f"{name:<8}median {pd.Series(o).median():.4f}  max {o.max():.2f}")
# MinMax pushes the median near zero because ONE 2.4-million income owns
# the top of the range. RobustScaler uses the median and IQR, so it is
# barely moved by that value.
```
</details>

## ❓ MCQs

**Q1.** What does `MinMaxScaler` produce?
- (a) Mean 0, std 1  (b) Values in [0, 1]  (c) Median 0  (d) Integers

**Q2.** Which models do **not** need scaling?
- (a) kNN and SVM  (b) Decision Trees and Random Forests  (c) Neural networks  (d) K-Means

**Q3.** Why is `MinMaxScaler` badly affected by outliers?
- (a) It uses the mean  (b) One extreme value defines the maximum, crushing everything else toward 0  (c) It is slow  (d) It is not affected

**Q4.** After correct scaling, the test set's mean is not exactly 0. This means…
- (a) A bug  (b) It is right — the test set was scaled with the training set's numbers  (c) You must re-fit  (d) The data is bad

**Q5.** Does scaling change the correlation between two columns?
- (a) Yes, a lot  (b) No — it is a change of units, not of information  (c) Only for MinMax  (d) Only for skewed data

<details><summary>Answers</summary>

**A1 — (b) [0, 1].**

**A2 — (b) Trees.** They split one column at a time, so relative scale is irrelevant.

**A3 — (b).** On the income column, MinMax pushes the median down to 0.18.

**A4 — (b).** **Exactly zero would mean you had fitted on the test set.**

**A5 — (b) No.** It moves and stretches; it does not reorder or re-relate.
</details>

---

# 6. Data Encoding Techniques

**Models do arithmetic. Text has none.** Encoding turns categories into numbers.

🧠 **Analogy: seat numbers versus jersey numbers.** Seat 5 really is between seat 4 and seat 6 — the number carries order. **A footballer wearing number 10 is not "more" than number 7.** Encoding goes wrong when you give a jersey number to something and the model reads it as a seat number.

## Label Encoding — one column of integers

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["intent_encoded"] = le.fit_transform(df["loan_intent"])
```

On the loan data this produces:

```text
DEBTCONSOLIDATION -> 0
EDUCATION         -> 1
HOMEIMPROVEMENT   -> 2
MEDICAL           -> 3
PERSONAL          -> 4
VENTURE           -> 5
```

> ⚠️ **Those numbers imply an order that does not exist.** The model now believes `VENTURE (5)` is five times `EDUCATION (1)`, and that `MEDICAL (3)` sits neatly between `HOMEIMPROVEMENT` and `PERSONAL`. **None of that is true — the codes are alphabetical.**
>
> **A linear model, kNN or SVM will act on that false order.** A tree will not care much, because it can split anywhere.

## Dummy Variable Encoding — one column per category

```python
dummies = pd.get_dummies(df["loan_intent"], prefix="intent")
```

**Six categories become six columns of 0 and 1**, with no order implied at all.

```python
pd.get_dummies(df["loan_intent"], drop_first=True)     # 6 categories -> 5 columns
```

**`drop_first=True` removes one column.** The dropped category is still fully represented — it is the case where all the others are 0. This avoids the "dummy variable trap", where the columns are perfectly predictable from each other and confuse linear models.

## Which to use

| Situation | Use |
|---|---|
| The target column (`yes`/`no`) | **Label Encoding** |
| Genuinely ordered categories (`Low` < `Medium` < `High`) | **Label Encoding**, or `OrdinalEncoder` with the order stated |
| Unordered categories, few of them | **Dummy variables** |
| Unordered categories, very many | Dummies would explode — group them, or use target encoding |
| Tree-based model | Either — trees cope with label encoding |

> **`person_education` in the loan data is genuinely ordered:** `High School` < `Associate` < `Bachelor` < `Master` < `Doctorate`. **But `LabelEncoder` sorts alphabetically**, which gives `Associate=0, Bachelor=1, Doctorate=2, High School=3, Master=4` — a nonsense order. **For ordered categories you must state the order yourself.**

## 📘 Examples

**Example 1 — Label Encoding, and the order it invents**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

le = LabelEncoder()
codes = le.fit_transform(df["loan_intent"])
print(dict(zip(le.classes_, le.transform(le.classes_))))
```

**Example 2 — dummy variables**

```python
dummies = pd.get_dummies(df["loan_intent"], prefix="intent")
print(dummies.shape)          # 6 columns
print(dummies.head(3))

fewer = pd.get_dummies(df["loan_intent"], prefix="intent", drop_first=True)
print(fewer.shape)            # 5 columns
```

**Example 3 — an ordered category done properly**

```python
order = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
mapping = {level: i for i, level in enumerate(order)}
df["education_ordered"] = df["person_education"].map(mapping)

print(mapping)
print(df[["person_education", "education_ordered"]].drop_duplicates()
        .sort_values("education_ordered"))
```

**Compare that with what `LabelEncoder` would have given you** — alphabetical, putting `Doctorate` at 2 and `High School` at 3. **The `.map()` version encodes the real order.**

**Example 4 — encoding everything at once**

```python
d = df.copy()
text_cols = d.select_dtypes("object").columns.tolist()
print("to encode:", text_cols)

for c in text_cols:
    d[c] = LabelEncoder().fit_transform(d[c])

print("all numeric now:", d.select_dtypes("object").empty)
```

**This is the quick approach used throughout Sessions 5–8** — acceptable there because those examples use tree models, which tolerate the false ordering. **For a linear model you would use dummies instead.**

## 🌍 Scenarios

**Scenario 1 — choosing per column**

```python
d = df.copy()
for col in d.select_dtypes("object").columns:
    n = d[col].nunique()
    if n == 2:
        advice = "Label Encoding (binary - no false order possible)"
    elif n <= 10:
        advice = f"dummies ({n} columns, or {n-1} with drop_first)"
    else:
        advice = f"{n} categories - too many for dummies; group them first"
    print(f"{col:<32}{n:>3}  {advice}")
```

> **With two categories, Label Encoding is safe** — 0 and 1 cannot imply a wrong order, because there is only one gap.

**Scenario 2 — the cost of dummies**

```python
before = df.select_dtypes("object")
after = pd.get_dummies(df, drop_first=True)
print(f"{len(df.columns)} columns -> {len(after.columns)} after dummy encoding")
```

**Dummies grow your table.** With a 50-category column you would gain 49 columns — which is why the guidance above caps it.

**Scenario 3 — the mistake, made visible**

```python
le = LabelEncoder()
edu_wrong = le.fit_transform(df["person_education"])
print("LabelEncoder order:", dict(zip(le.classes_, le.transform(le.classes_))))
print("\nThis says Doctorate (2) < High School (3), which is backwards.")
print("A linear model would act on that. Use an explicit mapping instead.")
```

## ✏️ Tasks

1. Label-encode `loan_intent` and print the code assigned to each category. What order does it imply?
2. Create dummy variables for `loan_intent`, with and without `drop_first`. How many columns each?
3. Encode `person_education` with `LabelEncoder`, then with a correct explicit order. Compare.
4. For every text column, print the number of categories and recommend an encoding.
5. Encode every text column and confirm the DataFrame is fully numeric.

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

le = LabelEncoder(); le.fit(df["loan_intent"])                         # 1
print(dict(zip(le.classes_, le.transform(le.classes_))))
# Alphabetical. It implies VENTURE (5) > EDUCATION (1), which is meaningless.

print(pd.get_dummies(df["loan_intent"]).shape[1], "columns")           # 2
print(pd.get_dummies(df["loan_intent"], drop_first=True).shape[1], "with drop_first")

le2 = LabelEncoder()                                                   # 3
print("LabelEncoder:", dict(zip(le2.fit(df.person_education).classes_,
                                le2.transform(le2.classes_))))
order = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
print("correct order:", {lv: i for i, lv in enumerate(order)})
# LabelEncoder puts Doctorate at 2 and High School at 3 - BACKWARDS.

for col in df.select_dtypes("object").columns:                         # 4
    n = df[col].nunique()
    rec = ("Label (binary)" if n == 2 else
           f"dummies ({n-1} cols with drop_first)" if n <= 10 else
           "too many - group first")
    print(f"{col:<32}{n:>3}  {rec}")

d = df.copy()                                                          # 5
for c in d.select_dtypes("object").columns:
    d[c] = LabelEncoder().fit_transform(d[c])
print("fully numeric:", d.select_dtypes("object").empty)      # True
```
</details>

## ❓ MCQs

**Q1.** What is wrong with Label Encoding an unordered category?
- (a) It is slow  (b) The integers imply an order the categories do not have  (c) It creates too many columns  (d) Nothing

**Q2.** Six categories with `pd.get_dummies(drop_first=True)` gives…
- (a) 6 columns  (b) 5 columns  (c) 1 column  (d) 7 columns

**Q3.** Why is `drop_first=True` used?
- (a) To save memory only  (b) The dropped category is still represented (all others 0), and it avoids the dummy variable trap  (c) It improves accuracy  (d) It is required

**Q4.** `LabelEncoder` on `person_education` gives `Doctorate=2, High School=3`. This is…
- (a) Correct  (b) Backwards — it sorted alphabetically, not by level  (c) Random  (d) An error

**Q5.** Which model type tolerates Label Encoding of unordered categories best?
- (a) Linear regression  (b) kNN  (c) Decision trees and forests  (d) SVM

<details><summary>Answers</summary>

**A1 — (b).** Jersey numbers read as seat numbers.

**A2 — (b) 5 columns.**

**A3 — (b).** The information is not lost, and linear models are not confused by perfectly predictable columns.

**A4 — (b) Backwards.** **For ordered categories you must state the order yourself.**

**A5 — (c) Trees.** They can split anywhere, so a false ordering costs them little.
</details>

---

## ⭐ Checkpoint Problem 4 — Make it all numeric

> **Uses:** EDA, cleaning, encoding. Topics 1–6.

**The problem.** Take the loan dataset and produce a fully numeric DataFrame ready for a model — choosing the encoding for each text column deliberately, and reporting what you did.

<details><summary>Solution</summary>

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

raw = pd.read_csv(BASE + "loan_data_10k.csv")
df = raw.copy()
log = []

# --- clean first (Topics 2-4)
n = df.duplicated().sum()
if n:
    df = df.drop_duplicates().reset_index(drop=True); log.append(f"removed {n} duplicate(s)")

n = (df["person_age"] > 100).sum()
df = df[df["person_age"] <= 100].reset_index(drop=True)
log.append(f"removed {n} row(s) with an impossible age")

n = df.isna().sum().sum()
if n:
    df = df.dropna().reset_index(drop=True); log.append(f"dropped {n} row(s) with gaps")

# --- encode, choosing per column (Topic 6)
EDU_ORDER = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]

for col in df.select_dtypes("object").columns.tolist():
    n_cat = df[col].nunique()

    if col == "person_education":                    # genuinely ORDERED
        df[col] = df[col].map({lv: i for i, lv in enumerate(EDU_ORDER)})
        log.append(f"{col}: ordinal mapping, {EDU_ORDER}")

    elif n_cat == 2:                                 # binary - Label is safe
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        log.append(f"{col}: label encoded {dict(zip(le.classes_, le.transform(le.classes_)))}")

    else:                                            # unordered - dummies
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        log.append(f"{col}: {n_cat} categories -> {dummies.shape[1]} dummy columns")

print("PREPROCESSING LOG")
for i, s in enumerate(log, 1):
    print(f"  {i}. {s}")

print(f"\n{raw.shape} -> {df.shape}")
print("fully numeric:", df.select_dtypes("object").empty)
print("any missing  :", df.isna().any().any())
print("\ncolumns:", df.columns.tolist())
```

**Three deliberate choices in that code:**

1. **`person_education` gets an explicit ordinal mapping**, because the levels genuinely have an order and `LabelEncoder` would have sorted them alphabetically into nonsense.
2. **Binary columns get Label Encoding**, which is safe — with two categories there is only one gap, so no false order can be implied.
3. **Everything else gets dummies with `drop_first=True`**, so no order is invented and linear models are not confused.

**The row count barely changes and the column count grows.** That is what correct encoding looks like.
</details>

**Make it harder:**

1. Print the correlation of each new dummy column with `loan_status`. Which category matters most?
2. Compare the shape you get using Label Encoding everywhere against your deliberate version.
3. Wrap the whole thing in a function that takes a DataFrame and returns `(encoded_df, log)`.

---

# 7. Train-Test Split

**You must test a model on data it has never seen.** Everything in Sessions 5 to 9 depends on getting this right.

🧠 **Analogy: past papers and the real exam.** You revise from past papers. If the exam contained the *same questions*, your mark would measure memory, not understanding. **The test set is the real exam — and it only works if the model has never seen it.**

## The four variables

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["loan_status"])     # the FEATURES  - everything you know
y = df["loan_status"]                    # the TARGET    - what you predict

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

| Variable | Is | Used for |
|---|---|---|
| `X_train` | Features the model learns from | `model.fit(X_train, y_train)` |
| `y_train` | Answers the model learns from | " |
| `X_test` | Features it has never seen | `model.predict(X_test)` |
| `y_test` | The true answers, kept hidden | `model.score(X_test, y_test)` |

> **These four names are a convention, not a rule — but everyone uses them.** Every example in this course, and almost every tutorial you will ever read, calls them exactly this. **Use the same names and your code reads like everyone else's.**

## The three arguments that matter

| Argument | Does | Use |
|---|---|---|
| `test_size=0.2` | Holds back 20% | 0.2 is the usual default; 0.3 on small data |
| `random_state=42` | Makes the split reproducible | **Always set it** |
| `stratify=y` | Keeps the class balance in both halves | **Always, for classification** |

## Why `stratify` matters

Measured on the diabetes dataset, which is **8.5% positive**. Splitting eight times without stratifying:

```text
test positive-rate ranges from 8.11% to 8.74%   (true rate 8.50%)
```

**With `stratify=y`, every split gives exactly the true rate.** On this dataset the drift is small; on a rarer class or a smaller dataset it is not — and a test set with the wrong balance gives you a score that means nothing.

## ⚠️ Data leakage — the mistake that flatters you

**Leakage is when information from the test set reaches the model during training.** Your score goes up and your model gets worse in the real world.

### The rule

```python
# ❌ WRONG - the scaler sees the test data
scaler = StandardScaler().fit(X)
X_train_scaled = scaler.transform(X_train)

# ✅ RIGHT - the scaler only ever sees training data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Split first. Then fit everything — scalers, imputers, encoders, feature selectors — on the training half only.**

### How much does it actually cost?

**This depends entirely on what leaked, and the honest answer surprises people.**

**Scaler leakage, measured over 20 seeds on the loan data:**

```text
mean difference: -0.00005   std 0.00063
```

**Essentially nothing.** A scaler learns a mean and a standard deviation; on ten thousand rows those barely change when you add the test set. **Scaler leakage is a bad habit rather than a catastrophe** — it matters most on small datasets.

**Target-based leakage is a completely different story.** Here is feature selection done on all the data, on **100 rows of pure random noise with random labels** — where the true accuracy is 50% by construction:

```text
LEAKY   accuracy: 0.800
CORRECT accuracy: 0.433
(the labels are random, so the truth is 0.500)
```

**The leaky version reports 80% on data containing no signal whatsoever.** Choosing the 20 "best" columns out of 5,000 using the labels — *including the test labels* — finds columns that happen to fit the test set by chance.

> **Anything that looks at `y` must happen after the split.** Scaling is forgiving; feature selection, target encoding and resampling are not. **If a result looks too good, suspect leakage before you celebrate.**

## The complete correct order

```text
1. Load
2. Clean          (missing values, duplicates, impossible values)
3. Encode         (text -> numbers)
4. SPLIT          <- everything after this fits on TRAIN ONLY
5. Scale          fit on X_train, transform both
6. Train          model.fit(X_train, y_train)
7. Evaluate       model.score(X_test, y_test)
```

> **Cleaning and encoding before the split is acceptable** when they do not use the target and do not depend on the data's distribution — dropping an impossible age, or mapping `yes`/`no` to 1/0. **Anything that computes a statistic — a median, a mean, a scale, a selection — belongs after step 4.**

## 📘 Examples

**Example 1 — the standard split**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv").dropna()
for c in df.select_dtypes("object").columns:
    df[c] = LabelEncoder().fit_transform(df[c])

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train {X_train.shape}   y_train {y_train.shape}")
print(f"X_test  {X_test.shape}    y_test  {y_test.shape}")
print(f"train balance {y_train.mean():.4f}   test balance {y_test.mean():.4f}")
```

**Example 2 — what `stratify` protects you from**

```python
d = pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv").dropna()
Xd, yd = d.drop(columns=["diabetes"]), d["diabetes"]

rates = []
for seed in range(8):
    _, _, _, yt = train_test_split(Xd, yd, test_size=0.2, random_state=seed)
    rates.append(yt.mean())

print(f"true positive rate      : {yd.mean():.4f}")
print(f"unstratified test range : {min(rates):.4f} to {max(rates):.4f}")

_, _, _, yt = train_test_split(Xd, yd, test_size=0.2, random_state=0, stratify=yd)
print(f"stratified              : {yt.mean():.4f}   <- matches exactly")
```

**Example 3 — scaling after the split, correctly**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)     # FIT on train
X_test_scaled = scaler.transform(X_test)           # TRANSFORM test

print("train mean:", X_train_scaled.mean().round(6))
print("test mean :", X_test_scaled.mean().round(6), " <- not exactly 0, and that is CORRECT")
```

**Example 4 — a pipeline, which makes leakage structurally impossible**

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])
pipe.fit(X_train, y_train)
print("accuracy:", round(pipe.score(X_test, y_test), 4))
```

**Everything inside a pipeline fits on the training fold only, automatically.** **Structure beats discipline** — you cannot forget a rule the tool enforces for you.

## 🌍 Scenarios

**Scenario 1 — the leakage that actually costs you**

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(0)
X_noise = pd.DataFrame(rng.normal(size=(100, 5000)))    # PURE NOISE
y_random = pd.Series(rng.integers(0, 2, 100))           # RANDOM labels

# WRONG: choose the "best" columns using ALL the data, including test labels
sel = SelectKBest(f_classif, k=20).fit(X_noise, y_random)
a, b, c, d = train_test_split(pd.DataFrame(sel.transform(X_noise)), y_random,
                              test_size=0.3, random_state=1, stratify=y_random)
print("LEAKY  :", round(LogisticRegression(max_iter=2000).fit(a, c).score(b, d), 3))

# RIGHT: select inside a pipeline, so it only ever sees the training fold
a, b, c, d = train_test_split(X_noise, y_random, test_size=0.3,
                              random_state=1, stratify=y_random)
pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)),
                 ("m", LogisticRegression(max_iter=2000))])
print("CORRECT:", round(pipe.fit(a, c).score(b, d), 3))
print("truth  : 0.500  (the labels are random)")
```

**80% on data with no signal in it.** That is what leakage buys you — and it is why a surprisingly good score should make you suspicious, not pleased.

**Scenario 2 — a three-way split, when you will tune**

```python
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"train {len(X_train):,}  validation {len(X_val):,}  test {len(X_test):,}")
```

**Use the validation set to make choices; touch the test set once, at the very end.** Every time you look at the test set and change something, you leak a little.

**Scenario 3 — when a random split is the wrong split**

```text
TIME SERIES: predicting tomorrow from yesterday.

A random split puts future rows in the training set and past rows in the
test set - so the model "predicts" the past having already seen the
future. Your score is meaningless.

Split by TIME instead:
    train = df[df["date"] <  "2026-01-01"]
    test  = df[df["date"] >= "2026-01-01"]

The same applies to GROUPED data - all rows for one patient, or one
customer, must land on the SAME side of the split.
```

## ✏️ Tasks

1. Split the loan dataset into the four standard variables and print all four shapes.
2. Confirm that `stratify=y` gives the same class balance in train and test.
3. Split the diabetes data eight times without stratifying and report the range of test positive rates.
4. Scale correctly after splitting, and explain why the test mean is not exactly 0.
5. Reproduce the leakage demonstration and explain in one paragraph what went wrong.

<details><summary>Solutions</summary>

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv").dropna()
for c in df.select_dtypes("object").columns:
    df[c] = LabelEncoder().fit_transform(df[c])
X, y = df.drop(columns=["loan_status"]), df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(                   # 1
    X, y, test_size=.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(f"train {y_train.mean():.4f}  test {y_test.mean():.4f}")         # 2

d = pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv").dropna()
Xd, yd = d.drop(columns=["diabetes"]), d["diabetes"]                   # 3
rates = [train_test_split(Xd, yd, test_size=.2, random_state=s)[3].mean()
         for s in range(8)]
print(f"true {yd.mean():.4f}  unstratified range {min(rates):.4f}-{max(rates):.4f}")

sc = StandardScaler().fit(X_train)                                     # 4
print("train mean", sc.transform(X_train).mean().round(6))
print("test  mean", sc.transform(X_test).mean().round(6))
# The test set was scaled with the TRAINING set's mean and std. If its
# mean came out at exactly 0, you would have fitted on the test data.

rng = np.random.default_rng(0)                                         # 5
Xn = pd.DataFrame(rng.normal(size=(100, 5000)))
yr = pd.Series(rng.integers(0, 2, 100))
sel = SelectKBest(f_classif, k=20).fit(Xn, yr)          # sees ALL labels
a, b, c_, d_ = train_test_split(pd.DataFrame(sel.transform(Xn)), yr,
                                test_size=.3, random_state=1, stratify=yr)
print("LEAKY  ", round(LogisticRegression(max_iter=2000).fit(a, c_).score(b, d_), 3))
a, b, c_, d_ = train_test_split(Xn, yr, test_size=.3, random_state=1, stratify=yr)
pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)),
                 ("m", LogisticRegression(max_iter=2000))])
print("CORRECT", round(pipe.fit(a, c_).score(b, d_), 3))
# The data is PURE NOISE and the labels are RANDOM, so the truth is 0.500.
# Choosing the 20 "best" of 5,000 columns using the labels - including the
# TEST labels - finds columns that fit the test set by chance. The reported
# 80% measures nothing but that coincidence.
```
</details>

## ❓ MCQs

**Q1.** What are the four standard variables from a split?
- (a) `train`, `test`, `x`, `y`  (b) `X_train`, `X_test`, `y_train`, `y_test`  (c) `a`, `b`, `c`, `d`  (d) `features`, `target`, `train`, `test`

**Q2.** What does `stratify=y` do?
- (a) Sorts the data  (b) Keeps the class balance the same in both halves  (c) Scales it  (d) Shuffles it

**Q3.** When must a scaler be fitted?
- (a) On all the data  (b) On the training set only, after the split  (c) On the test set  (d) It does not matter

**Q4.** Feature selection on all the data gave 80% accuracy on pure random noise. Why?
- (a) The model is good  (b) Choosing columns using the test labels finds ones that fit the test set by chance  (c) Not enough data  (d) A bug

**Q5.** For time-series data, a random split is wrong because…
- (a) It is slow  (b) It puts future rows in training and past rows in test, so the model predicts the past knowing the future  (c) Dates cannot be split  (d) It is not wrong

<details><summary>Answers</summary>

**A1 — (b).** A convention everyone follows — use it and your code reads like everyone else's.

**A2 — (b).** Essential for classification, especially on an imbalanced target.

**A3 — (b) Training set only.** Everything after the split fits on train.

**A4 — (b).** **The truth was 50%.** If a result looks too good, suspect leakage before you celebrate.

**A5 — (b).** Split by time instead. The same applies to grouped data — all rows for one patient must land on the same side.
</details>

---

## ⭐ Checkpoint Problem 5 — The complete preprocessing pipeline

> **Uses everything in this session.**

**The problem.** Take the raw loan dataset and produce `X_train`, `X_test`, `y_train`, `y_test`, fully cleaned, encoded and scaled, with **no leakage** — and print a log of every step. This is the code you will reuse for the rest of the course.

<details><summary>Solution</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

TARGET = "loan_status"
EDU_ORDER = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
DOMAIN = {"person_age": (18, 100), "person_income": (0, 5_000_000),
          "credit_score": (300, 850)}

log = []

# ---------- 1. LOAD
raw = pd.read_csv(BASE + "loan_data_10k.csv")
df = raw.copy()
log.append(f"loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns")

# ---------- 2. CLEAN  (safe before the split: no statistics, no target)
n = df.duplicated().sum()
if n:
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"removed {n} duplicate row(s)")

for col, (lo, hi) in DOMAIN.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    if bad:
        df = df[(df[col] >= lo) & (df[col] <= hi)].reset_index(drop=True)
        log.append(f"removed {bad} row(s) with impossible {col}")

n = df.isna().sum().sum()
if n:
    df = df.dropna().reset_index(drop=True)
    log.append(f"dropped {n} row(s) containing gaps")

# ---------- 3. ENCODE  (safe before the split: a fixed mapping, no target used)
for col in df.select_dtypes("object").columns.tolist():
    if col == "person_education":
        df[col] = df[col].map({lv: i for i, lv in enumerate(EDU_ORDER)})
        log.append(f"{col}: ordinal mapping")
    elif df[col].nunique() == 2:
        df[col] = LabelEncoder().fit_transform(df[col])
        log.append(f"{col}: label encoded (binary)")
    else:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        log.append(f"{col}: {dummies.shape[1]} dummy column(s)")

# ---------- 4. SPLIT  <- everything after this line fits on TRAIN ONLY
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
log.append(f"split 80/20, stratified: train {len(X_train):,}, test {len(X_test):,}")

# ---------- 5. SCALE  (fit on TRAIN, transform BOTH)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                             columns=X_test.columns, index=X_test.index)
log.append("scaled with StandardScaler, fitted on X_train only")

# ---------- REPORT
print("PREPROCESSING LOG")
for i, step in enumerate(log, 1):
    print(f"  {i}. {step}")

print(f"""
READY FOR MODELLING
  X_train {X_train_scaled.shape}   y_train {y_train.shape}
  X_test  {X_test_scaled.shape}    y_test  {y_test.shape}

  class balance   train {y_train.mean():.4f}   test {y_test.mean():.4f}
  train mean      {X_train_scaled.values.mean():+.6f}   (should be ~0)
  test mean       {X_test_scaled.values.mean():+.6f}   (should NOT be exactly 0)
  any missing     {X_train_scaled.isna().any().any()}
  all numeric     {X_train_scaled.select_dtypes('object').empty}
""")
```

**The comment on step 4 is the whole point of this session.**

Steps 2 and 3 run before the split because they use **no statistics and no target** — removing an impossible age and mapping education levels to integers would give the same answer whether or not the test rows were present.

Step 5 runs after, because `StandardScaler` computes a **mean and a standard deviation**, and computing those from the test set too is exactly the leakage Topic 7 measured.

**If you are ever unsure which side of the split something belongs on, ask: does this step look at the data as a whole, or at the target?** If yes, it goes after.
</details>

**Make it harder:**

1. Rewrite steps 3 and 5 as a scikit-learn `Pipeline` with `ColumnTransformer`, so leakage becomes structurally impossible.
2. Add a `SimpleImputer` inside the pipeline instead of dropping rows with gaps.
3. Wrap the whole thing in `prepare(path, target)` returning the four variables plus the log, and run it on the diabetes dataset without changing the function.

---

# ✅ Before you move on

**EDA**

- [ ] I explore before I touch anything: shape, head, info, describe
- [ ] I check the target balance first, because it decides my metric
- [ ] I compare mean and median to spot skew
- [ ] I can turn my EDA findings into a preprocessing to-do list

**Cleaning**

- [ ] I choose a missing-value strategy per column and write down what I assumed
- [ ] I know a duplicate might be a real repeat, and de-duplicate on an ID when one exists
- [ ] **I know the IQR rule flags tails, not errors**
- [ ] I use domain limits to find impossible values, because no formula will

**Preparing for a model**

- [ ] I can scale with MinMax, Standard and Robust, and pick between them
- [ ] I know trees do not need scaling and kNN and SVM do
- [ ] I know Label Encoding invents an order, and when that is harmful
- [ ] I use dummy variables for unordered categories, with `drop_first=True`
- [ ] I can state an ordered category's order myself rather than letting the encoder guess
- [ ] I produce `X_train`, `X_test`, `y_train`, `y_test` with `stratify=y`
- [ ] **I fit scalers and selectors on the training set only**
- [ ] **I know a surprisingly good score means suspect leakage, not celebrate**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-03-eda-preprocessing.ipynb) | Every example above, runnable |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
| [Session 4 — Introduction to ML & AI](session-04-intro-ml-ai.md) | Where this data finally meets a model |
