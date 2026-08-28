# Session 3 — Visualisation & Data Preprocessing

**Matplotlib & Seaborn · Handling Missing Values · Remove Duplicates · Outlier Detection & Removal · Data Transformation & Scaling · Data Encoding · Train-Test Split · Practice**

| | |
|---|---|
| **Notebook** | [session-03-visualisation-preprocessing.ipynb](../notebooks/session-03-visualisation-preprocessing.ipynb) |
| **Previous** | [Session 2 — NumPy & Pandas](session-02-numpy-pandas.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Choose a chart from the question you are asking — and **read** the chart you drew
2. Decide, per column, whether to drop or fill missing values
3. Find duplicate rows and judge whether they are genuine
4. Detect outliers with the IQR rule and decide what to do about them
5. Scale features, and say which scaler suits which situation
6. Encode categories, and say when `LabelEncoder` would mislead a model
7. Split data correctly — and explain **data leakage** to someone else

---

## The seven topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Visualisation](#1-visualisation) | A chart is a question, not decoration |
| 2 | [Missing values](#2-handling-missing-values) | Median, not mean — outliers drag the mean |
| 3 | [Duplicates](#3-removing-duplicates) | A duplicate may be genuine. Check before deleting |
| 4 | [Outliers](#4-outlier-detection--removal) | Detection finds candidates; **you** decide |
| 5 | [Transformation & scaling](#5-data-transformation--scaling) | Distance-based models need scaling; trees do not |
| 6 | [Encoding](#6-data-encoding-techniques) | Ordered → Label. Unordered → One-hot |
| 7 | [Train-test split](#7-train-test-split) | **Split before you scale** |

---

# 1. Visualisation

🧠 **Analogy: charts are camera lenses.** You do not pick a lens because it looks nice — you pick it for what you are photographing. **If you cannot say what question your chart answers, delete it.**

| Your question | Chart |
|---|---|
| How is one numeric column spread out? | Histogram |
| Are there outliers? | Box plot |
| Do two numeric columns move together? | Scatter plot |
| How do categories compare on average? | Bar plot |
| How many rows in each category? | Count plot |
| Which columns relate to which? | Heatmap |

## 📘 Examples

**Example 1 — histogram, and how to read it**

```python
import seaborn as sns, matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

sns.histplot(df["selling_price"], bins=40, kde=True)
plt.title("Distribution of selling price")
plt.show()
```

| What you look at | What it means |
|---|---|
| The x-axis | The value being measured |
| The y-axis | How many rows fall in that range |
| Each bar (a "bin") | A value range. 40 bins = the range cut into 40 slices |
| Tall bars on the left | Most cars are cheap — where the bulk of your data lives |
| A long flat tail right | A few very expensive cars. **Right-skewed** |

> **Why skew matters:** a linear model gets dragged towards those few expensive cars, making it worse at the many cheap ones.

**Example 2 — box plot, and the IQR it draws**

```python
sns.boxplot(x=df["km_driven"])
```

| What you look at | What it means |
|---|---|
| The box | The middle 50% — from the 25th to the 75th percentile |
| The line inside | The **median** |
| The whiskers | Reach 1.5 × IQR beyond the box — the "normal" range |
| Dots past the whisker | **Outliers.** Worth a second look |

**Example 3 — the two charts everyone confuses**

```python
sns.countplot(x="fuel_type", data=df)                     # how MANY rows
sns.barplot(x="fuel_type", y="selling_price", data=df)    # the AVERAGE price
```

> ⚠️ **The trap.** A category can be **rare** (short count bar) yet **expensive** (tall bar-plot bar). If one fuel type has the highest average price but appears 12 times in 15,000 rows, that average is unreliable. **Always check the count plot before believing the bar plot.**

## ✏️ Practice

1. Histogram of `km_driven` with 30 bins. Is it skewed? Which way?
2. Box plot of `selling_price`. Roughly how many outlier dots?
3. Scatter of `km_driven` against `selling_price`. Describe the direction in one sentence.
4. Count plot of `transmission_type`. Which is more common?
5. Bar plot of average price by `seller_type` — **then check the counts**. Is the top category reliable?

<details><summary>Solutions</summary>

```python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "regression/cardekho_dataset.csv")

sns.histplot(df["km_driven"], bins=30); plt.show()        # 1 right-skewed
sns.boxplot(x=df["selling_price"]); plt.show()            # 2
sns.scatterplot(x="km_driven", y="selling_price", data=df, alpha=.4); plt.show()  # 3 downward
sns.countplot(x="transmission_type", data=df); plt.show() # 4
print(df["transmission_type"].value_counts())
sns.barplot(x="seller_type", y="selling_price", data=df, errorbar=None); plt.show()  # 5
print(df["seller_type"].value_counts())   # ALWAYS check the counts too
```
</details>

## ❓ MCQs

**Q1.** You want to know whether study hours relate to exam marks. Which chart?
- (a) Histogram  (b) Box plot  (c) Scatter plot  (d) Count plot

**Q2.** What is the difference between a count plot and a bar plot?
- (a) None
- (b) Count shows rows per category; bar shows the average of another column
- (c) Bar shows rows per category; count shows an average
- (d) Count plots only work on numbers

**Q3.** In a box plot, what do the dots beyond the whiskers mean?
- (a) Errors in the data
- (b) Outliers — candidates for investigation
- (c) Missing values
- (d) The mean

**Q4.** Your histogram has a long tail to the right. What is that called?
- (a) Left-skewed  (b) Right-skewed  (c) Normal  (d) Bimodal

**Q5.** A bar plot shows one category with by far the highest average. What do you check first?
- (a) Whether the chart is colourful enough
- (b) How many rows that category has
- (c) The y-axis label
- (d) Nothing — the chart is the answer

<details><summary>Answers</summary>

**A1 — (c) Scatter plot.** It is the only chart showing two numeric variables at once.

**A2 — (b).** A category can be rare yet have a high average.

**A3 — (b).** They are *candidates for investigation*, not errors. Check whether each is genuine or a typo.

**A4 — (b) Right-skewed.** The mean is pulled above the median.

**A5 — (b).** An impressive average built on twelve rows is not a finding.
</details>

## 🎯 Tasks

**Task 1 — The four-panel story.** Build a 2×2 figure for the Titanic dataset. For each panel write **one sentence saying what it tells you**, not what it is. *"It shows the distribution of age"* is not an answer; *"Most passengers were 20–40, with a noticeable group of young children"* is.

**Task 2 — The misleading chart.** Deliberately make a chart mislead using only honest data: change the bin count, cut the y-axis so it does not start at zero, or plot an average over a category with very few rows. Then write a paragraph on how a reader could have caught you. **You will read charts made by other people for the rest of your career.**

---

# 2. Handling missing values

🧠 **Analogy: gaps in a form.** Someone left a field blank. You can throw the whole form away, guess a sensible value, or mark it as "not given". Each choice changes what you can conclude.

| Strategy | When | How |
|---|---|---|
| **Drop rows** | Very few missing, and they look random | `df.dropna()` |
| **Drop the column** | Most of the column is missing | `df.drop(columns=[...])` |
| **Fill with median** | Numeric, and outliers are present | `df[c].fillna(df[c].median())` |
| **Fill with mean** | Numeric, roughly symmetric, no outliers | `df[c].fillna(df[c].mean())` |
| **Fill with mode** | Categorical | `df[c].fillna(df[c].mode()[0])` |
| **Fill with a marker** | Missing is *itself* meaningful | `df[c].fillna("Unknown")` |

## 📘 Examples

**Example 1 — find them first**

```python
missing = df.isnull().sum()
print(missing[missing > 0])
print((df.isnull().sum() / len(df) * 100).round(1))   # as a percentage
```

**Example 2 — median, not mean, and why**

```python
small = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")

print("mean  :", small["Salary"].mean())     # ~196,091 - dragged by one row
print("median:", small["Salary"].median())   #   67,000 - unaffected

small["Salary"] = small["Salary"].fillna(small["Salary"].median())
```

**Example 3 — categorical, and "missing means something"**

```python
small["Country"] = small["Country"].fillna(small["Country"].mode()[0])

# Sometimes a blank is information. "Did not disclose income" is a fact
# about the applicant, not an absence of one.
df["income_missing"] = df["income"].isnull().astype(int)   # keep the signal
df["income"] = df["income"].fillna(df["income"].median())  # then fill
```

## ✏️ Practice

On the Titanic dataset:

1. How many missing values are in `age`? What percentage is that?
2. Fill missing `age` with the **median**. Confirm none remain.
3. Fill missing `embarked` with the most common value.
4. Compare the mean `age` before and after filling. Did it move? Why so little?
5. Create `age_missing` **before** filling, so the information is not lost.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
t = pd.read_csv(BASE + "classification/archive/titanic.csv")

print(t["age"].isnull().sum(), f"{t['age'].isnull().mean()*100:.1f}%")   # 1
before = t["age"].mean()
t["age_missing"] = t["age"].isnull().astype(int)                         # 5
t["age"] = t["age"].fillna(t["age"].median())                            # 2
print("remaining:", t["age"].isnull().sum())
t["embarked"] = t["embarked"].fillna(t["embarked"].mode()[0])            # 3
print(f"mean before {before:.2f}, after {t['age'].mean():.2f}")          # 4
# It barely moved: we filled with the median, which sits near the centre already.
```
</details>

## ❓ MCQs

**Q1.** Why fill missing numbers with the median rather than the mean?
- (a) The median is faster to compute
- (b) The mean is dragged by extreme values; the median is not
- (c) The mean cannot handle missing data
- (d) There is no difference

**Q2.** A column is 80% missing. What is usually the best move?
- (a) Fill it all with the median
- (b) Drop the column
- (c) Fill with zero
- (d) Drop every row that has a gap

**Q3.** What does `df[c].mode()[0]` give you?
- (a) The average  (b) The middle value  (c) The most common value  (d) The first row

**Q4.** When is "missing" itself worth keeping as information?
- (a) Never
- (b) When the fact of not answering is meaningful, e.g. "did not disclose income"
- (c) Only for numeric columns
- (d) Only when more than half is missing

**Q5.** You fill `age` with the median and the column mean barely moves. Why?
- (a) The fill did not work
- (b) The median sits near the centre already, so adding copies of it shifts little
- (c) Pandas caches the mean
- (d) The column had no gaps

<details><summary>Answers</summary>

**A1 — (b).** In the twelve-row file one salary of 1,500,000 pushed the mean to 196,091 while the median stayed at 67,000.

**A2 — (b).** Filling 80% of a column invents most of it. Whatever you fill with becomes the column.

**A3 — (c) the most common value.** `mode()` returns a Series because there can be ties, so you take `[0]`.

**A4 — (b).** Create a `_missing` flag **before** filling, so the model can still use that signal.

**A5 — (b).** Filling with a central value by design barely moves the centre.
</details>

## 🎯 Tasks

**Task 1 — Missing-value policy.** For `classification/heart_failure_raw.csv`, write a table: every column, how much is missing, your chosen strategy, and **one sentence of justification**. Then implement it and confirm nothing is left.

**Task 2 — Does the strategy matter?** Take one numeric column with gaps. Fill it three ways — drop the rows, fill with mean, fill with median — and compare the column's mean, median and standard deviation each time. **Which choice changed the data most, and would that change a model's conclusion?**

---

# 3. Removing duplicates

🧠 **Analogy: the same form submitted twice.** Someone pressed Submit twice. Or two different people genuinely gave the same answers. **You cannot tell which from the data alone** — and the two need opposite treatment.

## 📘 Examples

**Example 1 — find them before you delete them**

```python
print("duplicate rows:", df.duplicated().sum())

# keep=False shows ALL copies, so you can look at them side by side
print(df[df.duplicated(keep=False)].sort_values(list(df.columns)))
```

**Example 2 — removing, and checking what you removed**

```python
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"removed {before - len(df)} rows, {len(df)} remain")
```

**Example 3 — duplicates on a subset of columns**

```python
# Two records with the SAME customer id but different addresses are
# not identical rows - but they are still a problem.
print(df.duplicated(subset=["customer_id"]).sum())

# keep="last" keeps the most recent record rather than the first
df = df.drop_duplicates(subset=["customer_id"], keep="last")
```

> ⚠️ **A duplicate is not automatically an error.** Two customers can genuinely have the same age, city and spend. **Exact duplicates across *every* column are suspicious; duplicates on an ID column are almost always a real problem.**

## ✏️ Practice

On `prepreprocessing/pre_data.csv` (twelve rows):

1. How many duplicate rows are there?
2. Print them with `keep=False` so you can see both copies.
3. Remove them and confirm the new row count.
4. Now check for duplicates on `Country` alone. How many? Are those a problem?
5. Explain in one sentence why the answers to 1 and 4 are so different.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
d = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")

print("duplicates:", d.duplicated().sum())                    # 1 -> 1
print(d[d.duplicated(keep=False)])                            # 2
d2 = d.drop_duplicates().reset_index(drop=True)               # 3
print(len(d), "->", len(d2))
print("by Country:", d2.duplicated(subset=["Country"]).sum()) # 4 -> several
# 5: repeated COUNTRIES are expected - many people share a country. A repeated
#    ENTIRE ROW is suspicious, because every field matching is unlikely by chance.
```
</details>

## ❓ MCQs

**Q1.** What does `df.duplicated().sum()` return?
- (a) The duplicate rows  (b) How many rows are duplicates of an earlier row  (c) All repeated values  (d) The number of unique rows

**Q2.** Why use `keep=False` when inspecting duplicates?
- (a) It deletes them
- (b) It shows **all** copies, not just the later ones, so you can compare them
- (c) It keeps the first copy
- (d) It is faster

**Q3.** Repeated values in a single `Country` column — a problem?
- (a) Yes, always remove them
- (b) No — many rows sharing a country is expected
- (c) Only if there are more than ten
- (d) Only for numeric columns

**Q4.** When is a duplicate almost certainly a real problem?
- (a) When two rows share one column
- (b) When an **ID** column repeats
- (c) When two rows have the same age
- (d) Duplicates are never a problem

**Q5.** After `drop_duplicates()`, why call `.reset_index(drop=True)`?
- (a) It removes more duplicates
- (b) The old index now has gaps; this renumbers rows from 0
- (c) It sorts the data
- (d) It is required by Pandas

<details><summary>Answers</summary>

**A1 — (b).** It marks each row that has appeared before, so the first copy is not counted.

**A2 — (b).** By default the *first* occurrence is not flagged, so you would only see one of each pair.

**A3 — (b).** Repeated categories are normal. An entire row matching across every column is the suspicious case.

**A4 — (b) a repeated ID.** IDs are meant to be unique; a repeat means double entry or a bad join.

**A5 — (b).** Without it your index skips numbers, which is confusing later and can trip up positional indexing.
</details>

## 🎯 Tasks

**Task 1 — Duplicate audit.** On a dataset of your choice: count exact duplicates, count duplicates on each single column, and count duplicates on plausible *combinations* (e.g. name + date of birth). For each, say whether it is expected or suspicious, and why.

**Task 2 — The double-submission scenario.** *A survey system had a bug: some people's responses were saved twice, seconds apart. There is a `submitted_at` timestamp.* Write code that removes only the accidental doubles — same answers, timestamps within 10 seconds — while keeping genuine repeat respondents who filled it in weeks apart. **What assumption are you making, and how would you check it?**

---

# 4. Outlier detection & removal

🧠 **Analogy: the one very tall person in a class photo.** They are real. They are not a mistake. But if you use the class average height to order uniforms, they will get the wrong size — **and so will everyone else.**

An outlier is a value far from the rest. It might be:
- **a genuine extreme** — a taxi with 400,000 km
- **a data-entry error** — an extra zero
- **a different population** — a commercial vehicle in a car dataset

**Detection finds candidates. You decide which is which.**

## 📘 Examples

**Example 1 — the IQR rule, by hand**

```python
q1 = df["km_driven"].quantile(0.25)
q3 = df["km_driven"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[(df["km_driven"] < lower) | (df["km_driven"] > upper)]
print(f"IQR {iqr:,.0f} | fences {lower:,.0f} to {upper:,.0f}")
print(f"{len(outliers)} outliers, {len(outliers)/len(df)*100:.1f}% of rows")
```

**Example 2 — the z-score rule**

```python
z = (df["km_driven"] - df["km_driven"].mean()) / df["km_driven"].std()
print("beyond 3 standard deviations:", (z.abs() > 3).sum())
```

> **IQR vs z-score.** The z-score uses the mean and standard deviation, **both of which the outliers themselves distort**. IQR uses quartiles, which they do not. **Prefer IQR when you suspect strong outliers** — which is exactly when you are looking.

**Example 3 — three things you can do about them**

```python
# 1. REMOVE - only when you believe they are errors
clean = df[(df["km_driven"] >= lower) & (df["km_driven"] <= upper)]

# 2. CAP (winsorise) - keep the row, pull the value to the fence
capped = df.copy()
capped["km_driven"] = capped["km_driven"].clip(lower=lower, upper=upper)

# 3. TRANSFORM - compress the scale so extremes matter less
import numpy as np
df["km_log"] = np.log1p(df["km_driven"])
```

## ✏️ Practice

On `selling_price` in the cardekho dataset:

1. Compute Q1, Q3 and the IQR.
2. Compute the lower and upper fences.
3. How many rows fall outside them? What percentage?
4. Print the five most extreme values. Do they look like errors or genuine?
5. Cap the column at the fences and compare `describe()` before and after.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "regression/cardekho_dataset.csv")
col = "selling_price"

q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)       # 1
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr                 # 2
out = df[(df[col] < lower) | (df[col] > upper)]
print(f"{len(out)} outliers, {len(out)/len(df)*100:.1f}%")     # 3
print(df.nlargest(5, col)[["car_name", col]])                  # 4 - luxury cars, genuine
print(df[col].describe().round(0))                             # 5
print(df[col].clip(lower, upper).describe().round(0))
```
</details>

## ❓ MCQs

**Q1.** What is the IQR?
- (a) The range from min to max
- (b) The distance from the 25th to the 75th percentile
- (c) The standard deviation
- (d) The mean minus the median

**Q2.** Why prefer IQR over the z-score when you suspect strong outliers?
- (a) IQR is faster
- (b) The z-score uses the mean and std, which the outliers themselves distort
- (c) The z-score only works on small data
- (d) They are identical

**Q3.** A car shows 400,000 km. What should you do?
- (a) Delete it immediately
- (b) Investigate — it may be a genuine taxi or a typo
- (c) Replace it with the mean
- (d) Ignore outliers entirely

**Q4.** What does "winsorising" mean?
- (a) Deleting outlier rows
- (b) Capping extreme values at a chosen fence, keeping the row
- (c) Taking the logarithm
- (d) Splitting the data

**Q5.** You cap a column but its scaled values still look flattened, with one value at 1.0. What does that tell you?
- (a) The capping failed silently
- (b) The cap was too loose — the extreme value still dominates the scale
- (c) Capping never works
- (d) You should use the mean instead

<details><summary>Answers</summary>

**A1 — (b).** The middle 50% of the data — exactly the box in a box plot.

**A2 — (b).** Quartiles are not distorted by the extremes, so IQR stays reliable precisely when you need it.

**A3 — (b).** Detection produces candidates. Deleting without checking throws away genuine data.

**A4 — (b).** You keep the row (and all its other columns) but pull the extreme value in.

**A5 — (b).** **A preprocessing step that runs without error is not one that worked.** Always look at the output.
</details>

## 🎯 Tasks

**Task 1 — Outlier decision log.** For three numeric columns in a dataset of your choice, produce a table: column, IQR fences, how many outliers, a look at the extreme rows, your decision (keep / cap / remove / transform), and **one sentence of justification each**.

**Task 2 — Does it change the model?** Train a simple regression twice — once on raw data, once with outliers capped. Report R² for both. **Did capping help, hurt, or do nothing? Would you keep the change?** A change that does nothing is still a result worth recording.

---

# 5. Data transformation & scaling

🧠 **Analogy: converting to a common currency.** One column runs 0–1, another runs 0–1,500,000. To a model that measures distance, the big column shouts and the small one whispers — **not because it matters more, but because its numbers are bigger.**

| Scaler | What it does | Use when |
|---|---|---|
| **StandardScaler** | Mean 0, standard deviation 1 | The default. Most models, roughly bell-shaped data |
| **MinMaxScaler** | Squashes to 0–1 | You need a bounded range, e.g. for neural networks |
| **RobustScaler** | Uses median and IQR | **Outliers are present** and you are keeping them |
| **log transform** | Compresses a long right tail | Strongly skewed data like income or price |

> **Which models need it?** Distance- and gradient-based ones: kNN, SVM, logistic regression, neural networks, PCA, k-Means. **Tree-based models do not care** — a decision tree splits on "is x > 5", and rescaling does not change the ordering.

## 📘 Examples

**Example 1 — the three scalers side by side**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

cols = ["person_age", "person_income", "credit_score"]
X = loans[cols]

for name, scaler in [("Standard", StandardScaler()),
                     ("MinMax", MinMaxScaler()),
                     ("Robust", RobustScaler())]:
    scaled = scaler.fit_transform(X)
    print(f"{name:<9} mean {scaled.mean():.3f}  min {scaled.min():.2f}  max {scaled.max():.2f}")
```

**Example 2 — a log transform for a skewed column**

```python
import numpy as np

print("raw skew :", round(df["selling_price"].skew(), 2))
df["price_log"] = np.log1p(df["selling_price"])   # log1p handles zeros safely
print("log skew :", round(df["price_log"].skew(), 2))
```

**Example 3 — why trees do not care**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# same model, unscaled vs scaled
for name, model in [("DecisionTree", DecisionTreeClassifier(max_depth=4, random_state=42)),
                    ("LogisticReg ", LogisticRegression(max_iter=2000))]:
    raw = model.fit(X_train, y_train).score(X_test, y_test)
    sc  = model.fit(X_train_scaled, y_train).score(X_test_scaled, y_test)
    print(f"{name}  unscaled {raw:.4f}  scaled {sc:.4f}")
```

## ✏️ Practice

1. Apply `StandardScaler` to three numeric loan columns. Print the mean and std after scaling.
2. Apply `MinMaxScaler` to the same columns. What are the min and max now?
3. Print the skew of `person_income`, then of `log1p(person_income)`. Did it improve?
4. Which of these need scaling: kNN, decision tree, logistic regression, random forest?
5. Scale a column that contains a large outlier with `StandardScaler`, then `RobustScaler`. Compare the results.

<details><summary>Solutions</summary>

```python
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv").dropna()
cols = ["person_age", "person_income", "credit_score"]

s = StandardScaler().fit_transform(loans[cols])                    # 1
print(s.mean(axis=0).round(3), s.std(axis=0).round(3))
m = MinMaxScaler().fit_transform(loans[cols])                      # 2
print(m.min(axis=0), m.max(axis=0))
print(round(loans["person_income"].skew(), 2),                     # 3
      round(np.log1p(loans["person_income"]).skew(), 2))
# 4: kNN YES, logistic regression YES. Decision tree NO, random forest NO.
r = RobustScaler().fit_transform(loans[["person_income"]])         # 5
print("Standard max:", s[:, 1].max().round(2), " Robust max:", r.max().round(2))
```
</details>

## ❓ MCQs

**Q1.** Why does kNN need scaled features but a decision tree does not?
- (a) kNN is newer
- (b) kNN measures distance, so a large-numbered column dominates; a tree splits on thresholds
- (c) Trees cannot handle floats
- (d) kNN requires values between 0 and 1 by definition

**Q2.** What does `StandardScaler` produce?
- (a) Values between 0 and 1
- (b) Mean 0 and standard deviation 1
- (c) Integers
- (d) Ranked values

**Q3.** When would you choose `RobustScaler`?
- (a) When data is perfectly normal
- (b) When outliers are present and you are keeping them
- (c) When you need values in 0–1
- (d) For categorical data

**Q4.** Why `np.log1p(x)` rather than `np.log(x)`?
- (a) It is faster
- (b) `log1p` computes `log(1+x)`, so a zero does not become `-inf`
- (c) `log` is deprecated
- (d) They are identical

**Q5.** Which of these does **not** need feature scaling?
- (a) Logistic regression  (b) Random forest  (c) k-Means  (d) Neural network

<details><summary>Answers</summary>

**A1 — (b).** With income 0–1,500,000 and age 18–80, income dominates every distance purely because its numbers are bigger.

**A2 — (b).** Hence the name — it standardises to a common scale.

**A3 — (b).** It uses the median and IQR, which outliers do not distort.

**A4 — (b).** Income and price columns often contain zeros, and `log(0)` is `-inf`.

**A5 — (b) Random forest.** Trees split on thresholds, and rescaling does not change the ordering of values.
</details>

## 🎯 Tasks

**Task 1 — Scaler comparison.** Train the same model on the same data with no scaling, `StandardScaler`, `MinMaxScaler` and `RobustScaler`. Report all four scores in a table. Then repeat with a **decision tree**. **Which model was affected, which was not, and does that match what you expected?**

**Task 2 — Fix the skew.** Find the most skewed numeric column in a dataset. Plot it, apply a log transform, plot it again. Then train a linear model on both versions and compare R². **Did fixing the skew help?**

---

# 6. Data encoding techniques

🧠 **Analogy: numbering the exits on a roundabout versus numbering your friends.**

Exit 1, 2, 3 have a real order — exit 3 comes after exit 2. Numbering your friends 1, 2, 3 does **not** mean friend 3 is more than friend 2. **If you hand a model the second kind of number, it will believe the first kind of meaning.**

| Method | Use for | Produces |
|---|---|---|
| **LabelEncoder** | Ordered categories, or **tree models** | One column of integers |
| **OneHotEncoder / `get_dummies`** | Unordered categories with linear models | One 0/1 column per category |
| **Ordinal mapping (by hand)** | Ordered categories where **you** set the order | One column, your order |

## 📘 Examples

**Example 1 — LabelEncoder, and printing the mapping**

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
loans["loan_intent"] = encoder.fit_transform(loans["loan_intent"])

# LabelEncoder assigns 0,1,2... in ALPHABETICAL order.
# ALWAYS print the mapping - any app serving this model must reuse it.
print(dict(zip(encoder.classes_, range(len(encoder.classes_)))))
```

**Example 2 — one-hot, for unordered categories**

```python
demo = pd.DataFrame({"city": ["Delhi", "Mumbai", "Chennai", "Delhi"]})
print(pd.get_dummies(demo, columns=["city"], dtype=int))

#    city_Chennai  city_Delhi  city_Mumbai
# 0             0           1            0
# 1             0           0            1
```

**Example 3 — an ordered category, mapped by hand**

```python
# Alphabetical order would give High=0, Low=1, Medium=2 - nonsense.
order = {"Low": 0, "Medium": 1, "High": 2}
df["risk_encoded"] = df["risk"].map(order)
```

> ⚠️ **The trap.** `LabelEncoder` on `["Chennai", "Delhi", "Mumbai"]` gives 0, 1, 2. A linear model then believes **Mumbai > Delhi > Chennai**, and that Mumbai is *twice* Delhi. It will act on that. **A tree model would not care** — it only asks "is x > 1.5?".

## ✏️ Practice

1. Label-encode `person_education` and **print the mapping**.
2. One-hot encode `person_home_ownership` with `get_dummies`. How many columns did you get?
3. `sizes = ["Small","Medium","Large","Medium"]`. Encode with the correct order by hand.
4. What would `LabelEncoder` give for those sizes, and why is it wrong?
5. You have 500 distinct city names. Why is one-hot encoding a problem here, and what could you do instead?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv").dropna()

e = LabelEncoder(); loans["person_education"] = e.fit_transform(loans["person_education"])  # 1
print(dict(zip(e.classes_, range(len(e.classes_)))))

d = pd.get_dummies(loans["person_home_ownership"], prefix="home", dtype=int)  # 2
print(d.shape[1], "columns:", list(d.columns))

sizes = pd.Series(["Small","Medium","Large","Medium"])                       # 3
print(sizes.map({"Small":0, "Medium":1, "Large":2}).tolist())

# 4: alphabetical -> Large=0, Medium=1, Small=2. That says Small > Large.
# 5: 500 columns of mostly zeros - slow, sparse, and easy to overfit.
#    Alternatives: group rare cities into "Other", or use target/frequency encoding.
```
</details>

## ❓ MCQs

**Q1.** When is `LabelEncoder` a poor choice?
- (a) With tree models
- (b) With unordered categories fed to a linear model
- (c) With ordered categories
- (d) It is never a poor choice

**Q2.** In what order does `LabelEncoder` assign its numbers?
- (a) Order of appearance  (b) Alphabetical  (c) By frequency  (d) Random

**Q3.** `get_dummies` on a column with 4 categories produces how many columns?
- (a) 1  (b) 2  (c) 4  (d) 16

**Q4.** Why print the encoder's mapping?
- (a) To check for typos
- (b) Because any app serving the model must apply the **same** mapping
- (c) It is required by scikit-learn
- (d) To count the categories

**Q5.** You have 500 distinct cities. What is the problem with one-hot encoding?
- (a) It is not allowed
- (b) 500 mostly-zero columns — slow, sparse, and easy to overfit
- (c) It loses the city names
- (d) There is no problem

<details><summary>Answers</summary>

**A1 — (b).** The model reads the numbers as magnitudes and concludes Mumbai > Delhi.

**A2 — (b) Alphabetical.** Which is why it gives `Large=0, Medium=1, Small=2` — an order that means nothing.

**A3 — (c) 4**, one per category (some settings drop one to avoid redundancy).

**A4 — (b).** Send "Master" where the model expects "Doctorate" and every prediction is quietly wrong — with no error.

**A5 — (b).** Group rare categories into "Other", or use frequency/target encoding instead.
</details>

## 🎯 Tasks

**Task 1 — Encode a full dataset.** Take `heart_failure_raw.csv`. For each categorical column decide label vs one-hot vs manual order, **write one sentence of justification**, implement it, and print every mapping you used.

**Task 2 — Prove the trap.** Take an unordered category. Train a **logistic regression** twice: once label-encoded, once one-hot. Compare the scores. Then do the same with a **decision tree**. **Which model was affected, and does the result match the theory?**

---

# 7. Train-test split

🧠 **Analogy: studying with the answer key.**

Practise for an exam with the answer key open and you score 100% every time. You walk into the real exam confident — **and fail.** Your practice score was meaningless because information from the answers leaked into your practice.

**That is data leakage.** The commonest way students cause it is **scaling before splitting**: `fit_transform` on the whole dataset learns the mean and range of the test rows, and bakes them into training.

## 📘 Examples

**Example 1 — split first, always**

```python
from sklearn.model_selection import train_test_split

X = loans.drop(columns=["loan_status"])
y = loans["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # hold back 20% the model never sees
    random_state=42,    # same split every run, so results compare
    stratify=y,         # keep the class balance in both halves
)
print(X_train.shape, X_test.shape)
```

**Example 2 — then scale: fit on train, transform both**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform on TRAIN
X_test_scaled  = scaler.transform(X_test)        # transform ONLY on test
```

**Example 3 — a Pipeline makes leakage much harder**

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
pipe.fit(X_train, y_train)      # the scaler is fitted INSIDE, on train only
print(pipe.score(X_test, y_test))
```

> **Why `stratify=y` matters.** Without it, a random split of an imbalanced dataset can leave your test set with almost none of the minority class — and your recall figure becomes meaningless.

## How badly does leakage bite? It depends what leaked

**Scaler leakage is real but mild.** On the 10,000-row loan dataset, scaling before splitting changes the test score by well under a percentage point — within run-to-run noise. It is still wrong, and a `Pipeline` makes avoiding it free, but it will not fool you on its own.

**Target-based leakage is the one that ruins projects.** Take 100 rows of pure random noise with random labels — genuinely no signal at all. Select the 20 "best" features using the whole dataset, then split:

| Pipeline | Reported accuracy |
|---|---|
| Selected features on **all** data, then split | **90%** |
| Split first, selected on **training** data only | **50%** |

The honest answer is 50% — the labels are random. The leaky pipeline reports 90% because it picked the columns that happened to match the labels **including the test rows**. That is not a model; it is memorising the answer key.

The same applies to fitting an imputer, or oversampling with SMOTE, before the split.

> **The symptom is identical in both cases: a test score that looks too good.** When you see one, suspect leakage before you celebrate.

## ✏️ Practice

1. Split the loan data 80/20 with `random_state=42` and `stratify=y`. Print both shapes.
2. Print the class balance in `y_train` and `y_test`. Are they similar? Why?
3. Repeat **without** `stratify`. Did the balance shift?
4. Scale correctly (fit on train only) and print the train mean — it should be near 0.
5. Now deliberately scale **before** splitting. Train a model both ways and compare test scores.

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in loans.select_dtypes(include="object").columns:
    loans[c] = LabelEncoder().fit_transform(loans[c])
X, y = loans.drop(columns=["loan_status"]), loans["loan_status"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)  # 1
print(Xtr.shape, Xte.shape)
print(ytr.value_counts(normalize=True).round(3).tolist(),                                # 2
      yte.value_counts(normalize=True).round(3).tolist())   # similar - that IS stratify
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42)                        # 3
print(d.value_counts(normalize=True).round(3).tolist())
sc = StandardScaler(); print(sc.fit_transform(Xtr).mean().round(6))                       # 4
# 5: on 10,000 rows the difference is tiny - under a percentage point, within
#    noise. Scaler leakage is real but mild. The dangerous kind is target-based:
#    see the notebook's noise demonstration, where it turns 50% into 90%.
```
</details>

## ❓ MCQs

**Q1.** Why must the scaler be fitted on the training set only?
- (a) It is faster
- (b) Fitting on everything leaks the test set's mean and range into training
- (c) scikit-learn requires it
- (d) The test set has no mean

**Q2.** What does `stratify=y` do?
- (a) Sorts the data by y
- (b) Keeps the same class proportions in train and test
- (c) Removes the minority class
- (d) Shuffles more thoroughly

**Q3.** Why set `random_state=42`?
- (a) 42 gives the best split
- (b) So the split is the same every run, making results comparable
- (c) It improves accuracy
- (d) It is required

**Q4.** You select features using the whole dataset, then split. On data with **no real signal**, what score might you report?
- (a) About 50%, correctly
- (b) Far above 50% — the selection saw the test rows' labels
- (c) Exactly 0%
- (d) It would raise an error

**Q5.** Why does a `Pipeline` help prevent leakage?
- (a) It is faster
- (b) The scaler is fitted inside, on the training portion only — including inside cross-validation
- (c) It removes outliers
- (d) It does not help

<details><summary>Answers</summary>

**A1 — (b).** Your test score becomes inflated and meaningless — you were studying with the answer key.

**A2 — (b).** Essential for classification, especially imbalanced data.

**A3 — (b).** Any fixed number works; 42 is a convention. The point is reproducibility.

**A4 — (b).** In the notebook's demonstration this turns a true 50% into a reported 90% on pure noise. **Target-based leakage is far more dangerous than scaler leakage**, which is usually under a percentage point.

**A5 — (b).** This matters most inside cross-validation, where the scaler must be refitted on every fold.
</details>

## 🎯 Tasks

**Task 1 — Preprocess a dataset end to end.** Take `heart_failure_raw.csv` and write a complete, commented pipeline covering all six earlier topics: missing values, duplicates, outliers, scaling, encoding, split. **For every decision add a comment saying why.** Print the data after every step and confirm at the end: no gaps, all numeric, correctly split.

**Task 2 — Cause leakage on purpose, twice.** First the mild kind: scale the whole dataset then split, and compare against the correct order. Record the difference — you should find it small.

Then the dangerous kind: generate 100 rows of pure random noise with random labels, select the 20 "best" features **on all the data**, split, and score. Then do it correctly. **Report both.**

**Write one sentence on why the second is so much worse than the first** — and what that tells you about which preprocessing steps must go inside a `Pipeline`.

---

# ✅ Before you move on

- [ ] I pick a chart from the question, and can say what the chart tells me
- [ ] I check the count plot before believing a bar plot
- [ ] I can justify drop vs mean vs median vs mode per column
- [ ] I check whether a duplicate is genuine before deleting it
- [ ] I can compute IQR fences and decide keep / cap / remove / transform
- [ ] I know which models need scaling and which do not
- [ ] I know when `LabelEncoder` would mislead a model
- [ ] I **split before I scale**, and can explain leakage with the answer-key analogy

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-03-visualisation-preprocessing.ipynb) | Every example above, runnable |
| [Matplotlib & Seaborn exercises](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/matplotlib-seaborn-exercises.ipynb) | Drill problems |
