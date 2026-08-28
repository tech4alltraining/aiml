# Session 2 — Python Libraries: NumPy & Pandas

**NumPy Fundamentals · Pandas for Data Handling · Data Loading & Exploration · Exploratory Data Analysis · Practice**

| | |
|---|---|
| **Notebook** | [session-02-numpy-pandas.ipynb](../notebooks/session-02-numpy-pandas.ipynb) |
| **Previous** | [Session 1 — Python Refresher](session-01-python-refresher.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Explain why NumPy exists and when an array beats a list
2. Create, reshape and index NumPy arrays
3. Use boolean masking to filter without a loop
4. Load a CSV into a Pandas DataFrame from a file **or** a URL
5. Run the five commands you should run on *any* new dataset
6. Select, filter, sort and group a DataFrame
7. Create new columns from existing ones
8. Read a `describe()` table and spot an outlier from it
9. Read a correlation column and say what it does **not** prove

---

## The five topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [NumPy fundamentals](#1-numpy-fundamentals) | An array acts on every element at once |
| 2 | [Pandas for data handling](#2-pandas-for-data-handling) | A DataFrame is a spreadsheet you can command |
| 3 | [Data loading and exploration](#3-data-loading-and-exploration) | Five commands, every single time |
| 4 | [Exploratory Data Analysis](#4-exploratory-data-analysis) | Look before you model |
| 5 | [Putting it together](#5-putting-it-together) | Answer questions, not just run commands |

---

# 1. NumPy fundamentals

🧠 **Analogy: the shopping bag and the egg tray.**

A Python **list** is a shopping bag — throw anything in, but to act on everything you must pull items out one at a time.

A NumPy **array** is an egg tray — every slot holds the same kind of thing, in a fixed grid. Because the computer knows every slot is identical, it can act on **all of them at once**.

On a million numbers the egg tray is roughly **fifty times faster**. That is the entire reason NumPy exists.

## 📘 Examples

**Example 1 — the difference that matters**

```python
import numpy as np

prices_list  = [100, 200, 300]
prices_array = np.array([100, 200, 300])

print(prices_list * 2)     # [100, 200, 300, 100, 200, 300]  the BAG duplicated
print(prices_array * 2)    # [200 400 600]                   every ELEMENT doubled
```

**Example 2 — shape, dtype, and whole-array maths**

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)      # (5,)     5 elements, 1 dimension
print(b.shape)      # (2, 3)   2 rows, 3 columns
print(a.dtype)      # int64    every element is this type

print(a + 10)       # no loop anywhere
print(a ** 2)
print(b.T)          # transpose: rows become columns
print(np.arange(12).reshape(3, 4))
```

**Example 3 — statistics and boolean masking**

```python
data = np.array([23, 45, 12, 67, 34, 89, 21])

print(data.mean().round(2), np.median(data), data.std().round(2))
print(data.min(), data.max())

mask = data > 30              # an array of True/False, one per element
print(mask)                   # [False True False True True True False]
print(data[mask])             # [45 67 34 89]  only the True positions
print(mask.sum(), "above 30") # True counts as 1

# Reproducible randomness - fix the seed so results can be compared
rng = np.random.default_rng(seed=42)
print(rng.integers(1, 100, size=5))
```

## ✏️ Practice

`marks = np.array([78, 92, 65, 88, 45, 97, 55, 71, 83, 60])` — **no `for` loops**.

1. Print the mean to 1 decimal place.
2. Print the highest and lowest mark.
3. Count how many scored **75 or above** using a boolean mask.
4. Print only the marks **below 60**.
5. Add 5 to every mark but **cap at 100** (hint: `np.minimum`).

<details><summary>Solutions</summary>

```python
import numpy as np
marks = np.array([78, 92, 65, 88, 45, 97, 55, 71, 83, 60])

print(round(marks.mean(), 1))          # 1 -> 73.4
print(marks.max(), marks.min())        # 2 -> 97 45
print((marks >= 75).sum())             # 3 -> 4
print(marks[marks < 60])               # 4 -> [45 55]
print(np.minimum(marks + 5, 100))      # 5
```
</details>

## ❓ MCQs

**Q1.** `[1, 2, 3] * 2` and `np.array([1, 2, 3]) * 2` differ. What do they give?
- (a) Both give `[2, 4, 6]`
- (b) The list gives `[1,2,3,1,2,3]`; the array gives `[2,4,6]`
- (c) The list gives `[2,4,6]`; the array gives `[1,2,3,1,2,3]`
- (d) The array raises an error

**Q2.** An array has `.shape` of `(2, 3)`. How many elements does it hold?
- (a) 2  (b) 3  (c) 5  (d) 6

**Q3.** What does `(data > 30).sum()` count?
- (a) The sum of values above 30
- (b) How many values are above 30
- (c) The index of the first value above 30
- (d) Nothing — you cannot sum booleans

**Q4.** Why do ML examples set a random seed?
- (a) It makes the code faster
- (b) So the "random" numbers repeat, making results reproducible
- (c) It is required by NumPy
- (d) To get better random numbers

**Q5.** Why does this course discourage `for` loops over arrays?
- (a) Loops are not allowed in Python
- (b) NumPy acts on the whole array at once, which is far faster on real data
- (c) Loops give wrong answers
- (d) It is only a style preference

<details><summary>Answers</summary>

**A1 — (b).** A list is a bag, so `* 2` duplicates the container. An array is an egg tray where every slot is the same type, so maths applies element-wise.

**A2 — (d) 6.** Two rows × three columns.

**A3 — (b).** `True` counts as 1 and `False` as 0, so summing a boolean array counts the `True`s.

**A4 — (b).** You and your classmate then get identical numbers, and your work is reproducible when someone re-runs it.

**A5 — (b).** Real datasets have a million rows, not ten. Thinking in whole arrays is the habit you are building.
</details>

## 🎯 Tasks

**Task 1 — Class statistics.** Given a 2-D array where each **row** is a student and each **column** a subject, print with **no `for` loop**: each student's average (`axis=1`), each subject's average (`axis=0`), the highest single score and who it belongs to, and how many scores are below 60.

**Task 2 — Prove the speed claim.** Create a Python list and a NumPy array of one million numbers. Square every element, timing both. **Report the ratio.** Was it close to 50×?

---

# 2. Pandas for data handling

🧠 **Analogy: a spreadsheet that takes orders.**

A `DataFrame` is a spreadsheet — rows, columns, headers. Instead of clicking and dragging, you give it instructions.

| In Excel you would | In Pandas you type |
|---|---|
| Scroll to the top to check the data | `df.head()` |
| Read the row count off the status bar | `df.shape` |
| Filter → "Price greater than 100000" | `df[df["price"] > 100000]` |
| Sort descending by Price | `df.sort_values("price", ascending=False)` |
| Insert a PivotTable | `df.groupby("fuel")["price"].mean()` |
| Find blanks with conditional formatting | `df.isnull().sum()` |
| Add a formula column | `df["new"] = df["a"] / df["b"]` |

**You are not learning a new idea — you are learning new words for one you already have.**

## 📘 Examples

**Example 1 — selecting**

```python
import pandas as pd

BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "regression/cardekho_dataset.csv")

df["selling_price"]                    # one name  -> a Series
df[["brand", "selling_price"]]         # a LIST of names -> a DataFrame
df.iloc[0]                             # first row, by position
df.iloc[0:5, 0:3]                      # first 5 rows, first 3 columns
```

**Example 2 — filtering**

```python
expensive = df[df["selling_price"] > 1000000]

# Use & for AND and | for OR, and wrap EACH condition in brackets.
# Plain `and` / `or` do not work on whole columns.
recent_petrol = df[(df["vehicle_age"] < 5) & (df["fuel_type"] == "Petrol")]

print(len(expensive), len(recent_petrol))
```

**Example 3 — grouping, sorting, new columns**

```python
# GROUPING is the single most useful Pandas skill
print(df.groupby("fuel_type")["selling_price"].mean().round(0))

# agg() asks several questions at once
print(df.groupby("brand")["selling_price"].agg(["count", "mean", "max"]).head())

# Sorting
print(df.sort_values("selling_price", ascending=False).head(3))

# A new column from existing ones
df["price_lakhs"] = (df["selling_price"] / 100000).round(2)

# Counting categories
print(df["fuel_type"].value_counts())
print(df["fuel_type"].value_counts(normalize=True).round(3))   # proportions
```

## ✏️ Practice

Load the Titanic dataset: `pd.read_csv(BASE + "classification/archive/titanic.csv")`.

1. Print its **shape**, then the first 3 rows, then the column names.
2. How many passengers **survived**? (The column is 1/0, so `.sum()` works.)
3. What is the **survival rate by `sex`**? Use `groupby`.
4. Filter to `class == 'First'` passengers who survived. How many?
5. Add a column `is_child` that is `True` when `age < 12`. How many children were aboard?

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
t = pd.read_csv(BASE + "classification/archive/titanic.csv")

print(t.shape); print(t.head(3)); print(t.columns.tolist())      # 1
print("survived:", t["survived"].sum())                          # 2
print(t.groupby("sex")["survived"].mean().round(3))              # 3
print(len(t[(t["class"] == "First") & (t["survived"] == 1)]))    # 4
t["is_child"] = t["age"] < 12                                    # 5
print("children:", t["is_child"].sum())
```
</details>

## ❓ MCQs

**Q1.** What is the difference between `df["price"]` and `df[["price"]]`?
- (a) Nothing
- (b) The first returns a Series; the second returns a DataFrame
- (c) The first returns a DataFrame; the second a Series
- (d) The second is invalid

**Q2.** Why write `df[(a > 1) & (b < 2)]` rather than `df[a > 1 and b < 2]`?
- (a) `and` is deprecated
- (b) `and` works on single values; `&` compares element by element
- (c) Brackets are just style
- (d) `&` is faster

**Q3.** This prints missing values you thought you removed. Why?
```python
df.dropna()
print(df.isnull().sum())
```
- (a) `dropna()` is broken
- (b) `dropna()` returns a **new** DataFrame and does not modify `df`
- (c) You must call it twice
- (d) `isnull()` counts differently

**Q4.** What does `df.groupby("fuel")["price"].mean()` do?
- (a) Sorts by fuel then price
- (b) For each fuel type, computes the average price
- (c) Filters to rows with an average price
- (d) Creates a new column

**Q5.** What does `value_counts(normalize=True)` return?
- (a) Counts, sorted
- (b) Counts scaled to proportions of the total
- (c) The normalised values of the column
- (d) An error

<details><summary>Answers</summary>

**A1 — (b).** scikit-learn wants a DataFrame for `X`, which is why you see double brackets there.

**A2 — (b).** The brackets are required because `&` binds more tightly than `>`.

**A3 — (b).** Most Pandas methods return a copy. You need `df = df.dropna()`. This catches almost everybody once.

**A4 — (b).** It is the Pandas version of a spreadsheet PivotTable.

**A5 — (b) proportions.** Useful for checking class balance at a glance.
</details>

## 🎯 Tasks

**Task 1 — The Titanic investigation.** Answer each **in one sentence of plain English**, backed by code: overall survival fraction; by `sex`, and by how much they differ; by `class`; average `age` and how many rows are missing it; which combined sex-and-class group fared best and worst. Finish with one sentence describing who was most likely to survive.

**Task 2 — Car market report.** Which brand has the highest average price **among brands with at least 50 cars** — and why does that minimum matter? How does average price change with `vehicle_age`? Which `seller_type` sells the most expensive cars? Every answer needs a number **and** a sentence.

---

# 3. Data loading and exploration

## 📘 Examples

**Example 1 — two ways to load**

```python
# From a URL - works anywhere, nothing to download
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

# From a local file, if you cloned the repository
# df = pd.read_csv("../../../datasets/loan_data_10k.csv")
```

**Example 2 — the five commands, in this order**

```python
df.shape            # 1. How big is it?              (rows, columns)
df.head()           # 2. Do the values look sensible?
df.info()           # 3. Column names, types, non-null counts
df.describe()       # 4. Ranges, quartiles, outliers
df.isnull().sum()   # 5. What is missing, and where?
```

**Example 3 — twelve rows you can check by eye**

Before working with 10,000 rows, work with one you can hold in your head.

```python
small = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")   # 12 rows
print(small)

print(small.isnull().sum())                 # 3 columns have gaps
print(small[small.duplicated(keep=False)])  # rows 8 and 10 are identical
print(small.describe())                     # one salary is 18x the next
```

## ✏️ Practice

1. Load `classification/diabetes_prediction_dataset.csv`. Print its shape.
2. How many missing values, and in which columns?
3. Is the target (`diabetes`) balanced? Print the proportions.
4. What is the **average `bmi`** for people with diabetes versus without?
5. Which numeric column correlates most strongly with `diabetes`?

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
dia = pd.read_csv(BASE + "classification/diabetes_prediction_dataset.csv")

print(dia.shape)                                                    # 1
miss = dia.isnull().sum(); print(miss[miss > 0] if miss.sum() else "none")  # 2
print(dia["diabetes"].value_counts(normalize=True).round(3))        # 3
print(dia.groupby("diabetes")["bmi"].mean().round(2))               # 4
print(dia.corr(numeric_only=True)["diabetes"].drop("diabetes")
        .abs().sort_values(ascending=False).head(3))                # 5
```
</details>

## ❓ MCQs

**Q1.** Which command tells you the **number of rows and columns**?
- (a) `df.info()`  (b) `df.shape`  (c) `df.describe()`  (d) `df.head()`

**Q2.** A column shows as `object` in `df.info()` but you expected numbers. What does that mean?
- (a) The data is corrupt
- (b) It is being stored as text — something non-numeric is in it
- (c) `info()` always shows `object`
- (d) It has missing values

**Q3.** Which command shows how many missing values each column has?
- (a) `df.dropna()`  (b) `df.isnull().sum()`  (c) `df.describe()`  (d) `df.count()`

**Q4.** Why start with a 12-row dataset before a 10,000-row one?
- (a) It loads faster
- (b) You can verify with your eyes that the code agrees with the data
- (c) Pandas works better on small data
- (d) There is no reason

**Q5.** `df.head()` shows 5 rows by default. How do you see 20?
- (a) `df.head(20)`  (b) `df.head[20]`  (c) `df.top(20)`  (d) `df.head(rows=20)`

<details><summary>Answers</summary>

**A1 — (b) `df.shape`.** It returns `(rows, columns)`.

**A2 — (b).** A stray `"N/A"`, a currency symbol, or a thousands comma will do it. Find and fix it before modelling.

**A3 — (b).** `isnull()` gives True/False per cell; `.sum()` counts the Trues per column.

**A4 — (b).** When you later run `isnull().sum()` on 100,000 rows you will trust it, because you checked it once on data you could read.

**A5 — (a) `df.head(20)`.**
</details>

## 🎯 Tasks

**Task 1 — The data quality report.** Pick a dataset from [`datasets/`](../../../datasets/) nobody used in class. One page covering: shape and what one row represents; every column's type and role; missing values with a recommended fix **for each**; outliers checked properly with your judgement; duplicates; target balance and what it means for your metric; three questions the data could answer; and **one thing that surprised you**.

**Task 2 — The suspicious spreadsheet.** *A colleague sends `sales_data.csv` saying "it's clean, just build the model". You find 3% negative revenue, `"North"` and `"north"` both present, 12% duplicated IDs, and dates in three formats.*

Write a memo under 400 words: what you can fix yourself; what needs their input and the **exact questions** you would ask; what you do if they say "just drop the bad rows"; and whether you would proceed.

> **What this tests:** whether you push back. Dropping bad rows discards 12% of the data, and negative revenue might be genuine refunds. **You cannot tell from the data alone.**

---

# 4. Exploratory Data Analysis

🧠 **Analogy: the doctor's check-up.** A doctor does not prescribe the moment you walk in. First the check-up, **only then** a treatment. Skipping EDA and jumping to `model.fit()` is prescribing without examining the patient.

## 📘 Examples

**Example 1 — how to read a `describe()` table**

| Row | What it tells you | The check to make |
|---|---|---|
| `count` | How many **non-missing** values | Lower than the row count? You have gaps |
| `mean` | The average | Compare with `50%` — a big gap means outliers |
| `std` | How spread out | Larger than the mean? Very wide spread |
| `min` | Smallest value | Negative where it should not be? A data error |
| `25/50/75%` | The quartiles. `50%` is the **median** | Where the bulk of your data lives |
| `max` | Largest value | **Many times bigger than `75%`? An outlier** |

**Example 2 — the number that explains why we use the median**

```python
small = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")
print(small["Salary"].describe())

# mean   ~196,091   <- dragged up by ONE row
# 50%      67,000   <- unaffected
# 75%      81,000
# max   1,500,000   <- the outlier
```

One row moved the average by nearly 130,000. **That is the concrete reason we fill missing numbers with the median, not the mean.**

**Example 3 — how to read a correlation column**

```python
loans = pd.read_csv(BASE + "loan_data_10k.csv")
print(loans.corr(numeric_only=True)["loan_status"].sort_values().round(3))
```

| Value | What it means |
|---|---|
| $+0.7$ to $+1.0$ | Strong positive — as this rises, the target rises |
| $+0.3$ to $+0.7$ | Moderate positive |
| $-0.3$ to $+0.3$ | Weak *linear* relationship. **Not the same as "useless"** |
| $-0.7$ to $-0.3$ | Moderate negative |
| $-1.0$ to $-0.7$ | Strong negative |

> ⚠️ **Two warnings.** A correlation of −0.3 is **not** weak in real data — do not discard a column because it scored 0.2. And **correlation is not causation**: ice cream sales correlate with drowning deaths, because hot weather causes both.

## ✏️ Practice

On the loan dataset:

1. Run the five commands and note anything surprising.
2. Is `loan_status` balanced? What does that mean for your choice of metric?
3. Which three columns correlate most strongly (in absolute terms) with `loan_status`?
4. What is the average `credit_score` for approved versus rejected loans?
5. Compare `mean` and `50%` for `person_income`. What does the gap tell you?

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv")

print(loans.shape); loans.info(); print(loans.describe())        # 1
print(loans["loan_status"].value_counts(normalize=True).round(3))# 2  ~50/50 -> accuracy is safe here
c = loans.corr(numeric_only=True)["loan_status"].drop("loan_status")
print(c.abs().sort_values(ascending=False).head(3))              # 3
print(loans.groupby("loan_status")["credit_score"].mean().round(1))  # 4
print(loans["person_income"].describe()[["mean", "50%"]])        # 5
```
</details>

## ❓ MCQs

**Q1.** Why do EDA rather than going straight to `model.fit()`?
- (a) It is required by scikit-learn
- (b) A model cannot tell you your data is broken — it will train on it and be confidently wrong
- (c) It makes training faster
- (d) To choose a random seed

**Q2.** `mean` is 196,091 and `50%` is 67,000. What does that gap tell you?
- (a) The data is normally distributed
- (b) A few very large values are dragging the average up — there are outliers
- (c) Half the data is missing
- (d) The column is categorical

**Q3.** Your target is 92% class 0 and 8% class 1. What follows?
- (a) Accuracy is a safe metric
- (b) Accuracy misleads — always predicting class 0 scores 92% and catches nothing
- (c) You must delete the minority class
- (d) The dataset is unusable

**Q4.** Two columns correlate at 0.85. Does one cause the other?
- (a) Yes  (b) No — correlation measures that they move together, not why  (c) Only if positive  (d) Only above 0.9

**Q5.** A column correlates with the target at 0.15. Should you delete it?
- (a) Yes, anything below 0.3 is useless
- (b) Not on that basis — correlation only measures *linear* relationship on its own
- (c) Yes, if it is categorical
- (d) Only if it also has missing values

<details><summary>Answers</summary>

**A1 — (b).** It will happily train on duplicated rows, mis-scaled units and leaked columns.

**A2 — (b).** The median is the more honest summary when this happens.

**A3 — (b).** Report recall, precision and F1 for the minority class instead.

**A4 — (b).** A third factor may drive both. Causation needs an experiment or domain knowledge.

**A5 — (b).** The relationship may be curved, or the column may matter only in combination with another. Test it in a model first.
</details>

## 🎯 Tasks

**Task 1 — Compare two datasets.** Run the five-question sweep on **both** the loan and diabetes datasets. Produce a side-by-side table: rows, columns, missing values, target balance, strongest correlated feature. Then three sentences on which you would rather model, and why.

**Task 2 — Data detective, in pairs.** Each pair takes **one** dataset nobody else has. Run the five questions and present it to the group in a minute: dataset name, shape, the target, what is missing, your biggest surprise, and one question the data could answer.

---

# 5. Putting it together

## 🎯 Session task

Using the **cardekho** dataset, answer these. Write each as **one sentence in plain English**, not a raw code output.

1. How many cars, and how many columns describe each one?
2. Which fuel type has the highest average selling price?
3. Which brand appears most often?
4. Are there missing values? In which columns?
5. What is the most expensive car — plausible, or a data-entry error?

> "The dataset has 15,411 cars described by 14 columns" is an answer. A screenshot of `df.shape` is not.

---

# ✅ Before you move on

- [ ] I can explain why an array beats a list, with the egg-tray analogy
- [ ] I can filter an array with a boolean mask, no loop
- [ ] I can load a CSV from a URL and from a file
- [ ] I run the five commands on every new dataset
- [ ] I can filter a DataFrame on two conditions with `&`
- [ ] I can `groupby` and explain the result in a sentence
- [ ] I know why `df.dropna()` alone changes nothing
- [ ] I can spot an outlier from `describe()` by comparing `75%` and `max`
- [ ] I can say why the median beats the mean for filling gaps
- [ ] I know that correlation is not causation, and can give the ice-cream example

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-02-numpy-pandas.ipynb) | Every example above, runnable |
| [NumPy exercises](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/numpy-exercises.ipynb) | Drill problems |
| [Pandas exercises](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/pandas-exercises.ipynb) | Drill problems |
