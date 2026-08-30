# Session 3 — EDA & Data Preprocessing

**Exploratory Data Analysis · Handling Missing Values · Removing Duplicates · Outlier Detection & Removal · Transformation & Scaling · Encoding · Train-Test Split**

| | |
|---|---|
| **Notebook** | [session-03-eda-preprocessing.ipynb](../notebooks/session-03-eda-preprocessing.ipynb) |
| **Previous** | [Session 2 — NumPy, Pandas & Visualisation](session-02-numpy-pandas.md) |
| **Next** | [Session 4 — Introduction to AI & ML](session-04-intro-ml-ai.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Session 2 taught you to clean data so a human can read it. This session prepares it so a model can learn from it.** Those are different jobs.
>
> By the end you will have `X_train`, `X_test`, `y_train`, `y_test` — the four variables every model in Sessions 5 to 9 expects.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Explore a new dataset systematically and say exactly what is wrong with it
2. Handle missing values by **dropping** or by **imputation**, and defend the choice
3. Find and remove duplicate rows
4. Detect outliers with a **box plot** and with the **IQR rule**, and decide what to do
5. Encode text with **Label Encoding** and **dummy variables**, and know when each is wrong
6. Scale with **MinMaxScaler** and **StandardScaler**
7. Split into `X_train`, `X_test`, `y_train`, `y_test` without leakage
8. **Run the whole sequence on a dataset you have never seen**

---

## How this session works

**Two complete walkthroughs, start to finish.**

| | Dataset | Why |
|---|---|---|
| **Use case 1** | `pre_data.csv` — 12 rows | **Small enough to print at every step.** It contains one of every problem, so you can watch each fix happen |
| **Use case 2** | `loan_data_10k.csv` — 10,000 rows | The same sequence at realistic scale, where you cannot see the rows |

**Each concept is explained where you first meet it, then applied immediately.** The walkthroughs run as one continuous sequence, the way you would actually do the work.

**At the end of the session:** [20 MCQs](#-20-mcqs) and [20 preprocessing tasks](#-preprocessing-tasks).

---

## The sequence

```text
1. LOAD                 get the data in
2. EXPLORE (EDA)        head, tail, info, describe - before touching anything
3. MISSING VALUES       drop them, or fill them
4. DUPLICATES           find and remove repeated rows
5. OUTLIERS             box plot to see them, IQR rule to measure them
6. ENCODING             text -> numbers
7. SCALING              put the numeric columns on a comparable range
8. TRAIN-TEST SPLIT     the bridge into Session 5
```

> **The order is not arbitrary — each step changes the numbers the next one uses.** You will see this happen for real in use case 1, where a choice made at step 3 quietly costs a whole row at step 5.

---

# Use case 1 — `pre_data.csv`

**Twelve rows, four columns, and one of every problem in the session.** Because it is so small, you can print the whole table at every step and *see* each change.

---

## Step 1 — Import the libraries and load the data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/prepreprocessing/pre_data.csv"

df = pd.read_csv(dataset_url)
df
```

```text
    Country   Age     Salary Purchased
0    France  44.0    72000.0        No
1     Spain  27.0    48000.0       Yes
2       NaN  30.0    54000.0        No
3     Spain  38.0    61000.0        No
4   Germany  40.0        NaN       Yes
5    France  35.0    58000.0       Yes
6     Spain   NaN    52000.0        No
7    France  48.0    79000.0       Yes
8   Germany  50.0    83000.0        No
9    France  37.0    67000.0       Yes
10  Germany  50.0    83000.0        No
11    Spain  50.0  1500000.0        No
```

> **Reading from a URL means nothing to download.** The same line works on your laptop and in Colab.

**Four columns:**

| Column | Holds |
|---|---|
| `Country` | Which country the person is from — **text** |
| `Age` | Their age — a number |
| `Salary` | Their salary — a number |
| `Purchased` | Whether they bought the product — **text**, and this is our **target** |

---

## Step 2 — Exploratory Data Analysis

### What EDA is

**EDA means looking at your data before you change anything.**

🧠 **Analogy: a doctor before prescribing.** They do not open with medication. They ask questions, take your temperature, listen. **Only then do they decide what to do.** Preprocessing without EDA is prescribing without examining.

**EDA is not decoration — it produces your to-do list.** Every problem you find maps onto one of the steps that follow.

### `head()` and `tail()` — what a row looks like

```python
df.head()
```

**`head()` shows the first five rows.** On a 10,000-row file you cannot print everything, so this is how you check the data loaded correctly and see what a row actually contains.

```python
df.tail()
```

**`tail()` shows the last five.** **Worth running as well as `head()`** — files sometimes have a summary row, a blank row, or corrupted records at the bottom that the first five rows would never reveal.

### `info()` — types and gaps

```python
df.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12 entries, 0 to 11
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   Country    11 non-null     object
 1   Age        11 non-null     float64
 2   Salary     11 non-null     float64
 3   Purchased  12 non-null     object
dtypes: float64(2), object(2)
```

**This one output answers two questions at once.**

**First, where are the gaps?** Read `Non-Null Count` against the row count. There are **12 rows**, so:

| Column | Non-null | Missing |
|---|---|---|
| `Country` | 11 | **1** |
| `Age` | 11 | **1** |
| `Salary` | 11 | **1** |
| `Purchased` | 12 | 0 |

**Second, what type is each column?** `object` means **text**. `Country` and `Purchased` are text, so **they will need encoding** at step 6.

### `describe(include='all')` — the summary of every column

```python
df.describe(include='all')
```

```text
       Country        Age        Salary Purchased
count       11  11.000000  1.100000e+01        12
unique       3        NaN           NaN         2
top     France        NaN           NaN        No
freq         4        NaN           NaN         7
mean       NaN  40.818182  1.960909e+05       NaN
std        NaN   8.244006  4.326353e+05       NaN
min        NaN  27.000000  4.800000e+04       NaN
25%        NaN  36.000000  5.600000e+04       NaN
50%        NaN  40.000000  6.700000e+04       NaN
75%        NaN  49.000000  8.100000e+04       NaN
max        NaN  50.000000  1.500000e+06       NaN
```

> **`include='all'` is what makes this useful.** Plain `describe()` shows only the numeric columns. With `include='all'` you get the text columns too — and the `NaN`s simply mean "that statistic does not apply to this kind of column".

**For the text columns:**

| Row | Means | Reading it |
|---|---|---|
| `unique` | How many distinct values | `Country` has **3**, `Purchased` has **2** |
| `top` | The most common value | `France`, and `No` |
| `freq` | How often the top value appears | `France` 4 times, `No` 7 times |

**For the numeric columns, look hard at `Salary`:**

| | Value |
|---|---|
| `mean` | **196,091** |
| `50%` (the median) | **67,000** |
| `max` | **1,500,000** |

**The mean is nearly three times the median.** That only happens when a few very large values drag the average up — and the `max` of 1,500,000 tells you exactly which value is doing it.

> **Comparing the mean with the median is the fastest outlier detector you have.** Remember this number: **the mean of `Salary` is 196,091.** It will matter at step 3.

### What EDA found

| Problem | Where | Fixed at |
|---|---|---|
| **Missing values** | `Country` row 2, `Salary` row 4, `Age` row 6 | Step 3 |
| **A suspicious salary** | 1,500,000 in row 11 | Step 5 |
| **Text columns** | `Country` and `Purchased` | Step 6 |
| **Possible duplicates** | Rows 8 and 10 look identical | Step 4 |

**That list is the plan for the rest of this walkthrough.**

---

## Step 3 — Handling missing values

### What a missing value is

**A missing value is a cell where no data was recorded.** Pandas shows it as `NaN` — *Not a Number*.

🧠 **Analogy: a survey with unanswered questions.** You can throw away every incomplete form (you lose data), write in a typical answer (you invent information), or note that it was skipped. **All three are defensible. Silently doing one without saying so is not.**

### Checking for them

```python
df.isnull().sum()
```

```text
Country      1
Age          1
Salary       1
Purchased    0
```

```python
df.isna().sum()
```

```text
Country      1
Age          1
Salary       1
Purchased    0
```

> **`isnull()` and `isna()` are exactly the same function** — Pandas provides both names because R users expect `isna()` and SQL users expect `isnull()`. **Use whichever you prefer; you will see both in other people's code.**

**How it works:** `isna()` returns `True`/`False` for every cell, and `.sum()` counts the `True`s per column — because `True` counts as 1.

### The two approaches

**We will try both, side by side.** Keep a copy so you can compare:

```python
df2 = df.copy()      # df  -> we will DROP on this one
                     # df2 -> we will IMPUTE on this one
```

### Approach A — dropping

**`dropna()` removes every row containing any missing value.**

```python
df.dropna(inplace=True)
df
```

```text
    Country   Age     Salary Purchased
0    France  44.0    72000.0        No
1     Spain  27.0    48000.0       Yes
3     Spain  38.0    61000.0        No
5    France  35.0    58000.0       Yes
7    France  48.0    79000.0       Yes
8   Germany  50.0    83000.0        No
9    France  37.0    67000.0       Yes
10  Germany  50.0    83000.0        No
11    Spain  50.0  1500000.0        No
```

**Twelve rows became nine.** Rows 2, 4 and 6 are gone — **and notice you lost every other column in those rows too.** Row 4 had a perfectly good country, age and purchase decision; all of it went because one cell was blank.

> **`inplace=True` modifies `df` directly instead of returning a new DataFrame.** Without it you would need `df = df.dropna()`. **Be careful with `inplace` — the original is gone, and on a small dataset like this you often want it back.**

**Losing 3 of 12 rows is 25% of the data.** On a dataset this small, that is far too expensive.

### Approach B — imputation

**Imputation means filling the gap with a sensible substitute.**

| Column type | Fill with | Why |
|---|---|---|
| Numeric | **Mean** or **median** | A typical value for that column |
| Text / category | **Mode** — the most common value | There is no "average" text |

```python
df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Salary'] = df2['Salary'].fillna(df2['Salary'].mean())
df2['Country'] = df2['Country'].fillna(df2['Country'].mode()[0])
df2
```

```text
    Country        Age        Salary Purchased
0    France  44.000000  7.200000e+04        No
1     Spain  27.000000  4.800000e+04       Yes
2    France  30.000000  5.400000e+04        No
3     Spain  38.000000  6.100000e+04        No
4   Germany  40.000000  1.960909e+05       Yes
5    France  35.000000  5.800000e+04       Yes
6     Spain  40.818182  5.200000e+04        No
7    France  48.000000  7.900000e+04       Yes
8   Germany  50.000000  8.300000e+04        No
9    France  37.000000  6.700000e+04       Yes
10  Germany  50.000000  8.300000e+04        No
11    Spain  50.000000  1.500000e+06        No
```

**All twelve rows survive.** Look at what was filled:

- **Row 2** `Country` → `France`, the mode
- **Row 6** `Age` → `40.818182`, the mean age
- **Row 4** `Salary` → `196,090.91`, the mean salary

> **`.mode()` returns a Series, not a single value**, because there can be a tie for most-common. That is why you take `[0]`.

### ⚠️ Look carefully at row 4

**The filled salary is 196,091.** Every other salary in the table is between 48,000 and 83,000.

**We filled it with the mean — and the mean was inflated by the 1,500,000 outlier sitting in row 11.** So the substitute we invented is more than double any real salary in the data.

> **Remember this. It will cost us a row at step 5.**

### Drop or impute?

| | Dropping | Imputation |
|---|---|---|
| Rows kept | 9 of 12 | **12 of 12** |
| You lose | Good data in other columns | Nothing |
| You invent | Nothing | Three values |
| Best when | Few gaps and plenty of data | Gaps matter, or data is scarce |

**We continue with `df2`, the imputed version**, because losing a quarter of a 12-row dataset is not acceptable.

---

## Step 4 — Removing duplicates

### What a duplicate is

**A duplicate is the same row recorded twice** — usually a data-entry slip or a botched file merge.

🧠 **Analogy: two identical entries on a shop's till roll.** Either the cashier scanned the same item twice by mistake, **or two customers each bought one.** The till roll alone cannot tell you which.

**Why they matter:** a duplicated row is counted twice in every average and every training step. **It quietly gives one row double the influence of every other row.**

### Checking

```python
df2.duplicated().sum()
```

```text
1
```

**`duplicated()` returns `True` for every row that repeats an earlier one.** It marks the *second and later* copies, not the first — which is why two identical rows give a count of **1**, not 2.

**To see every copy including the first:**

```python
df2[df2.duplicated(keep=False)]
```

```text
    Country   Age   Salary Purchased
8   Germany  50.0  83000.0        No
10  Germany  50.0  83000.0        No
```

**Rows 8 and 10 are identical in all four columns.**

### Removing

```python
df2.drop_duplicates(inplace=True)
df2
```

**Twelve rows become eleven.** Row 10 is gone; row 8 stays.

> **`drop_duplicates()` keeps the first copy by default.** Use `keep="last"` when later rows are corrections of earlier ones.

### ⚠️ When not to remove duplicates

**Not every duplicate is an error.** Two customers who each bought tea at 15 produce identical rows — **dropping one deletes a real sale.**

**Before dropping, ask: could two rows legitimately be identical?** If yes, you need a column that distinguishes them — a transaction ID, a timestamp — and you should de-duplicate on *that* instead.

**Here, each row is a different person, and two people with identical country, age, salary and decision is implausible in a 12-row file. Treating it as an error is reasonable — and you should say so.**

---

## Step 5 — Outlier detection and removal

### What an outlier is

**An outlier is a value far away from the rest.** Some are errors; some are real and important. **Telling them apart is judgement, not code.**

🧠 **Analogy: a class where everyone is 1.6–1.8 m tall and one student is 2.1 m.** Is that a typing error, or is there a very tall student? **The data cannot tell you. You have to look.**

### Seeing them — the box plot

**A box plot is the fastest way to see whether a column has outliers.**

```python
sns.boxplot(df2['Age'])
plt.show()
```

**How to read a box plot:**

```text
        |------[=====|=====]------|      o     o
        |         box   |         |      outliers
      lower       median         upper
      whisker                    whisker

  the BOX    spans the middle 50% of the values (Q1 to Q3)
  the LINE   inside the box is the median
  the WHISKERS reach the furthest value within 1.5 x IQR
  the DOTS   beyond the whiskers are flagged as OUTLIERS
```

**For `Age`, there are no dots.** Every age sits comfortably inside the whiskers — the values run from 27 to 50, with nothing unusual.

```python
sns.boxplot(df2['Salary'])
plt.show()
```

**For `Salary`, the picture is completely different.** The box is squashed almost flat near the bottom of the chart, and there are dots far above it. **That squashing is itself the signal** — one enormous value has stretched the axis so far that all the real salaries are compressed into a sliver.

### Measuring them — the IQR rule

**The box plot shows you outliers exist. The IQR rule tells you exactly which rows.**

```text
Q1  = the 25th percentile   (a quarter of values are below it)
Q3  = the 75th percentile   (three quarters are below it)
IQR = Q3 - Q1               (the range of the middle half)

Anything below  Q1 - 1.5 x IQR   or above  Q3 + 1.5 x IQR   is an outlier.
```

**In words: take the middle half of the data, and flag anything more than one-and-a-half of that width beyond either end.** **This is exactly what the box plot draws** — the whiskers stop at those bounds and everything past them becomes a dot.

```python
# Identify the quartiles
q1, q3 = np.percentile(df2['Salary'], [25, 75])

# Calculate the interquartile range
iqr = q3 - q1

# Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
print(lower_bound, upper_bound)
```

```text
18500.0 118500.0
```

**Any salary below 18,500 or above 118,500 is flagged.**

```python
df2[(df2['Salary'] < lower_bound) | (df2['Salary'] > upper_bound)]
```

```text
    Country   Age        Salary Purchased
4   Germany  40.0  1.960909e+05       Yes
11    Spain  50.0  1.500000e+06        No
```

### ⚠️ Two rows flagged, not one — and this matters

**Row 11 is the genuine outlier**, the 1,500,000 salary you spotted in `describe()` at step 2.

**Row 4 is the row we filled ourselves at step 3.**

> **We filled row 4's missing salary with the mean, 196,091. That mean was inflated by row 11's 1,500,000. And now the value we invented is itself being flagged as an outlier — and deleted.**
>
> **A perfectly good row is about to be thrown away because of a choice we made two steps earlier.**

**This is why the order of preprocessing steps matters, and it is not a hypothetical.**

**How to avoid it — two ways:**

| Fix | Effect |
|---|---|
| **Fill with the median instead of the mean** | The median is 67,000, comfortably inside the bounds. Row 4 survives |
| **Handle outliers before imputing** | Remove row 11 first, then the mean of what remains is sensible |

```python
# The median would have been a safe filler:
print("mean  :", df['Salary'].mean())      # 196,090.91  <- outside the bounds
print("median:", df['Salary'].median())    #  67,000.00  <- comfortably inside
```

**Both fixes work. The lesson is the same either way: an extreme value contaminates any average computed from it, including one you use as a filler.**

### Removing the outliers

```python
df2 = df2[(df2['Salary'] >= lower_bound) & (df2['Salary'] <= upper_bound)]
df2
```

```text
   Country        Age   Salary Purchased
0   France  44.000000  72000.0        No
1    Spain  27.000000  48000.0       Yes
2   France  30.000000  54000.0        No
3    Spain  38.000000  61000.0        No
5   France  35.000000  58000.0       Yes
6    Spain  40.818182  52000.0        No
7   France  48.000000  79000.0       Yes
8  Germany  50.000000  83000.0        No
9   France  37.000000  67000.0       Yes
```

**Nine rows remain, from an original twelve.**

> ⚠️ **Do not delete rows just because a formula flagged them.** The IQR rule is mechanical. **Look at what was flagged and decide.** Here one row was a real error and one was our own doing — and the rule could not tell them apart.

---

## Step 6 — Encoding

### Why encoding is needed

**A model does arithmetic. Text has none.** `"France"` cannot be multiplied by a weight, so every text column must become numbers.

🧠 **Analogy: seat numbers versus jersey numbers.** Seat 5 really is between seat 4 and seat 6 — the number carries order. **A footballer wearing number 10 is not "more" than number 7.** Encoding goes wrong when you give something a jersey number and the model reads it as a seat number.

### First, see what you are encoding

```python
df2['Country'].unique()
```

```text
array(['France', 'Spain', 'Germany'], dtype=object)
```

```python
df2['Purchased'].unique()
```

```text
array(['No', 'Yes'], dtype=object)
```

> **Always run `.unique()` before encoding.** It tells you how many categories you are about to create, and it catches problems — inconsistent spellings, stray whitespace, an unexpected `nan`. **Three countries and two purchase values, both clean.**

### Label Encoding

**`LabelEncoder` assigns each distinct value an integer, in alphabetical order.**

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df2['Country'] = le.fit_transform(df2['Country'])
print(le.classes_)

df2['Purchased'] = le.fit_transform(df2['Purchased'])
print(le.classes_)

df2
```

```text
['France' 'Germany' 'Spain']
['No' 'Yes']

   Country        Age   Salary  Purchased
0        0  44.000000  72000.0          0
1        2  27.000000  48000.0          1
2        0  30.000000  54000.0          0
3        2  38.000000  61000.0          0
5        0  35.000000  58000.0          1
6        2  40.818182  52000.0          0
7        0  48.000000  79000.0          1
8        1  50.000000  83000.0          0
9        0  37.000000  67000.0          1
```

**`le.classes_` tells you the mapping.** Reading it in order gives the codes:

```text
Country :  France = 0,  Germany = 1,  Spain = 2
Purchased: No     = 0,  Yes     = 1
```

**Every column is a number now.**

> **`fit_transform` does two things:** `fit` learns which categories exist, and `transform` replaces the text with the codes. **Re-using the same `le` object for a second column is fine here — it simply re-fits.** In production you would keep a separate encoder per column, so you can decode each one later.

### ⚠️ What Label Encoding implies about `Country`

**`France=0, Germany=1, Spain=2` says that Spain is twice Germany, and that Germany sits neatly between France and Spain.**

**None of that is true.** The codes are alphabetical, and countries have no order.

| Model type | Does the false order hurt? |
|---|---|
| Decision Tree, Random Forest | **Barely** — they can split anywhere, so they cope |
| Logistic / Linear Regression | **Yes** — they multiply the code by a weight |
| kNN, SVM | **Yes** — they measure distance, so Spain looks "far" from France |

**`Purchased` is safe** — with only two categories there is a single gap, so no false ordering is possible. **It is also the target, which is exactly what Label Encoding is for.**

### The alternative — dummy variables

**For an unordered category with more than two values, dummy variables carry no implied order at all.**

```python
pd.get_dummies(df2, columns=['Country'], drop_first=True)
```

**Each country becomes its own 0/1 column.** `drop_first=True` removes one — **the dropped category is still fully represented as the row where all the others are 0**, and leaving it out avoids the "dummy variable trap" that confuses linear models.

| Approach | Columns | Implies an order? | Use when |
|---|---|---|---|
| **Label Encoding** | 1 | **Yes** | Binary columns, the target, or genuinely ordered categories |
| **Dummy variables** | one per category (−1) | No | Unordered categories, especially with a linear model |

> **Label Encoding is what this walkthrough uses, and it is fine for the tree models you will meet first.** **Use case 2 shows all three treatments** — ordered, binary and unordered — on a dataset that needs each of them.

---

## Step 7 — Feature scaling

### Why scaling is needed

**Look at the two numeric columns:**

```text
Age     runs from 27 to 50
Salary  runs from 48,000 to 83,000
```

🧠 **Analogy: comparing a person's height in millimetres with their age in years.** Height gives numbers around 1,700; age gives numbers around 25. **Any method that measures distance will treat height as almost the whole story — not because it matters more, but because its numbers are bigger.** Scaling removes the unfairness of the units.

**Keep a copy so you can compare the two scalers:**

```python
df3 = df2.copy()
```

### MinMaxScaler — squash into 0 to 1

```text
value - min
-----------      ->  every column ends up between 0 and 1
 max - min
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df3[['Age', 'Salary']] = scaler.fit_transform(df3[['Age', 'Salary']])
df3
```

```text
   Country     Age  Salary  Purchased
0        0  0.7391  0.6857          0
1        2  0.0000  0.0000          1
2        0  0.1304  0.1714          0
3        2  0.4783  0.3714          0
5        0  0.3478  0.2857          1
6        2  0.6008  0.1143          0
7        0  0.9130  0.8857          1
8        1  1.0000  1.0000          0
9        0  0.4348  0.5429          1
```

**Every value now sits between 0 and 1.** The smallest age (27) became **0.0000** and the largest (50) became **1.0000**; the same for salary.

**Use it when you need a bounded range** — neural networks often expect it. ⚠️ **It is badly affected by outliers**: one extreme value defines the maximum and crushes everything else toward 0. **We removed our outlier at step 5, so this is safe here.**

### StandardScaler — centre on 0, spread of 1

```text
value - mean
------------     ->  mean 0, standard deviation 1
    std
```

```python
from sklearn.preprocessing import StandardScaler

ssc = StandardScaler()
df2[['Age', 'Salary']] = ssc.fit_transform(df2[['Age', 'Salary']])
df2
```

```text
   Country     Age  Salary  Purchased
0        0  0.7055  0.7110          0
1        2 -1.6317 -1.3644          1
2        0 -1.2193 -0.8455          0
3        2 -0.1194 -0.2402          0
5        0 -0.5319 -0.4996          1
6        2  0.2680 -1.0185          0
7        0  1.2554  1.3163          1
8        1  1.5304  1.6622          0
9        0 -0.2569  0.2786          1
```

**Values now centre on 0**, with negatives below average and positives above. **Row 1 has the lowest age and the lowest salary, so both are strongly negative. Row 8 has the highest of both.**

### Choosing between them

| | MinMaxScaler | StandardScaler |
|---|---|---|
| Output range | Exactly 0 to 1 | Centred on 0, typically −3 to +3 |
| Outliers | **Badly affected** | Affected, but less |
| Use for | Neural networks, bounded inputs | **The usual default** |

### Which models need scaling

| Needs scaling | Does not care |
|---|---|
| kNN, SVM, K-Means (**they measure distance**) | Decision Tree |
| Logistic / Linear Regression | Random Forest |
| Neural networks | Gradient boosting |
| PCA | Naive Bayes |

**Trees split one column at a time, so the relative scale of columns is irrelevant to them.**

> **Notice we scaled only `Age` and `Salary`.** `Country` and `Purchased` are already small integers, and `Purchased` is the target — **you never scale the target of a classification problem.**

---

## Step 8 — Train-test split

**This step is not in the preprocessing notebook, but it is where preprocessing leads.** Everything in Sessions 5 to 9 begins with these four variables.

### Why you must split

**You must test a model on data it has never seen.**

🧠 **Analogy: past papers and the real exam.** You revise from past papers. **If the exam contained the same questions, your mark would measure memory, not understanding.** The test set is the real exam, and it only works if the model has never seen it.

### Features and target

```text
X  the FEATURES - everything the model is allowed to look at
y  the TARGET   - the thing it must predict
```

```python
from sklearn.model_selection import train_test_split

X = df2.drop(columns=['Purchased'])     # features
y = df2['Purchased']                    # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print("X_train:", X_train.shape, "  y_train:", y_train.shape)
print("X_test :", X_test.shape, "  y_test :", y_test.shape)
```

```text
X_train: (6, 3)   y_train: (6,)
X_test : (3, 3)   y_test : (3,)
```

| Variable | Is | Used for |
|---|---|---|
| `X_train` | Features the model learns from | `model.fit(X_train, y_train)` |
| `y_train` | Answers it learns from | " |
| `X_test` | Features it has never seen | `model.predict(X_test)` |
| `y_test` | The true answers, kept hidden | `model.score(X_test, y_test)` |

> **These four names are a convention everyone follows.** Use them and your code reads like every tutorial and every colleague's code.

### The three arguments

| Argument | Does |
|---|---|
| `test_size=0.3` | Holds back 30%. 0.2 is the usual default; use more on small data |
| `random_state=42` | Makes the split reproducible. **Always set it** |
| `stratify=y` | Keeps the class balance in both halves. **Always, for classification** |

### ⚠️ One correction to the order above

**We scaled at step 7 and split at step 8. For a preprocessing demonstration that is fine — but when you build a real model, the split must come first.**

```python
# illustrative: a syntax reference, not runnable as written.
# The correct order for modelling:
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)   # SPLIT first

scaler = StandardScaler()
scaler.fit(X_train)                    # FIT on training data only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)   # apply the SAME numbers
```

> **Fitting a scaler on all the data lets the test set influence the mean and standard deviation used in training. That is data leakage, and it makes your score look better than the model really is.**
>
> **Split first. Then fit every scaler, imputer and selector on the training half only.** Use case 2 does it in this order.

> ⚠️ **A 3-row test set tells you almost nothing.** One wrong prediction moves the score by 33 percentage points. **This dataset is for learning the mechanics, not for drawing conclusions** — and being clear about that is part of doing the work honestly.

---

## The complete pipeline for use case 1

**Everything above, in one block:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/prepreprocessing/pre_data.csv"

# 1. LOAD
df2 = pd.read_csv(dataset_url)

# 2. EDA (findings: 3 gaps, 1 duplicate, 1 extreme salary, 2 text columns)

# 3. MISSING VALUES - impute rather than drop, to keep all 12 rows
df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Salary'] = df2['Salary'].fillna(df2['Salary'].median())   # MEDIAN - see step 5
df2['Country'] = df2['Country'].fillna(df2['Country'].mode()[0])

# 4. DUPLICATES
df2 = df2.drop_duplicates()

# 5. OUTLIERS - IQR rule on Salary
q1, q3 = np.percentile(df2['Salary'], [25, 75])
iqr = q3 - q1
lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df2 = df2[(df2['Salary'] >= lower_bound) & (df2['Salary'] <= upper_bound)]

# 6. ENCODING
df2['Country'] = LabelEncoder().fit_transform(df2['Country'])
df2['Purchased'] = LabelEncoder().fit_transform(df2['Purchased'])

# 7. SPLIT  <- before scaling, to avoid leakage
X = df2.drop(columns=['Purchased'])
y = df2['Purchased']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 8. SCALING - fit on train, transform both
ssc = StandardScaler().fit(X_train[['Age', 'Salary']])
X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
X_train_scaled[['Age', 'Salary']] = ssc.transform(X_train[['Age', 'Salary']])
X_test_scaled[['Age', 'Salary']] = ssc.transform(X_test[['Age', 'Salary']])

print(f"X_train {X_train_scaled.shape}, X_test {X_test_scaled.shape}")
print("no missing values:", not X_train_scaled.isna().any().any())
print("all numeric      :", X_train_scaled.select_dtypes('object').empty)
```

```text
X_train (7, 3), X_test (3, 3)
no missing values: True
all numeric      : True
```

> **Two deliberate changes from the step-by-step walkthrough above:**
>
> 1. **`Salary` is filled with the median, not the mean** — so the filled row is not later deleted as an outlier. **Ten rows survive instead of nine.**
> 2. **The split happens before the scaling** — so no test-set information reaches the scaler.
>
> **Both changes came from watching the step-by-step version go wrong.** That is what a walkthrough is for.

---

# Use case 2 — `loan_data_10k.csv`

**The same eight steps, on 10,000 rows.** Everything you learned in use case 1 still applies — but you can no longer look at the table, so **you have to trust the summaries.**

**What changes at this scale:**

| | Use case 1 | Use case 2 |
|---|---|---|
| Rows | 12 | 10,000 |
| Can you read every row? | **Yes** | No — you rely on `describe()` and counts |
| Missing values | 3 of 48 cells (6%) | 3 of 140,000 cells (0.002%) |
| Duplicates | 1 | **0** |
| Text columns | 2 | **5**, one of them genuinely ordered |
| Dropping bad rows costs | 25% of the data | 0.01% — effectively nothing |

---

## Step 1 — Load

```python
import pandas as pd

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"

df_raw = pd.read_csv(dataset_url)
df = df_raw.copy()

print("shape:", df.shape)
print(df.head(3))
```

```text
shape: (10000, 14)
```

---

## Step 2 — EDA

**Same six questions, but now you read summaries instead of rows.**

```python
df.info()
```

```text
 #   Column                          Non-Null Count  Dtype
---  ------                          --------------  -----
 0   person_age                      10000 non-null  int64
 1   person_gender                    9999 non-null  object
 2   person_education                10000 non-null  object
 3   person_income                   10000 non-null  float64
 4   person_emp_exp                  10000 non-null  int64
 5   person_home_ownership           10000 non-null  object
 6   loan_amnt                       10000 non-null  float64
 7   loan_intent                     10000 non-null  object
 8   loan_int_rate                    9999 non-null  float64
 9   loan_percent_income              9999 non-null  float64
 10  cb_person_cred_hist_length      10000 non-null  int64
 11  credit_score                    10000 non-null  int64
 12  previous_loan_defaults_on_file  10000 non-null  object
 13  loan_status                     10000 non-null  int64
```

**With 14 columns, scanning `Non-Null Count` by eye is slow. Ask directly:**

```python
missing = df.isna().sum()
print(missing[missing > 0])
print("\nduplicates:", df.duplicated().sum())
```

```text
person_gender          1
loan_int_rate          1
loan_percent_income    1

duplicates: 0
```

**Three missing values out of 140,000 cells, and no duplicates at all.** A far cleaner dataset than use case 1 — **but the three gaps still have to be dealt with, and you will see shortly why one of them is dangerous.**

### The target

```python
print(df["loan_status"].value_counts(normalize=True).round(4))
```

```text
1    0.5
0    0.5
```

**Exactly balanced.** Accuracy will be a fair metric.

### `describe()` — and the impossible value

```python
print(df[["person_age", "person_income", "credit_score"]].describe().round(2))
```

```text
       person_age  person_income  credit_score
count    10000.00       10000.00      10000.00
mean        27.70       72289.64        632.36
std          6.04       58462.43         50.69
min         20.00        8000.00        418.00
25%         24.00       41621.75        601.00
50%         26.00       60954.00        639.00
75%         30.00       87421.50        670.00
max        144.00     2448661.00        807.00
```

**Two things to notice, exactly as in use case 1:**

1. **`person_age` has a maximum of 144.** Nobody is 144. **That is a data error, and no statistical rule will tell you so — only knowing what an age can be.**
2. **`person_income` mean is 72,290 against a median of 60,954.** Skewed, just like `Salary` was.

```python
print("people over 100:", (df["person_age"] > 100).sum())
print(df[df["person_age"] > 100][["person_age", "person_income", "loan_amnt"]])
```

```text
people over 100: 1

      person_age  person_income  loan_amnt
9372         144        15162.0     3000.0
```

### Text columns

```python
for col in df.select_dtypes("object").columns:
    print(f"{col:<34}{df[col].nunique():>3} distinct  {list(df[col].dropna().unique())[:4]}")
```

```text
person_gender                       2 distinct  ['female', 'male']
person_education                    5 distinct  ['Master', 'High School', 'Bachelor', 'Associate']
person_home_ownership               4 distinct  ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
loan_intent                         6 distinct  ['MEDICAL', 'PERSONAL', 'EDUCATION', 'HOMEIMPROVEMENT']
previous_loan_defaults_on_file      2 distinct  ['No', 'Yes']
```

**Five text columns, and they are not all the same kind:**

| Column | Categories | Ordered? |
|---|---|---|
| `person_gender` | 2 | No — binary |
| `person_education` | 5 | **Yes** — High School < Associate < Bachelor < Master < Doctorate |
| `person_home_ownership` | 4 | No |
| `loan_intent` | 6 | No |
| `previous_loan_defaults_on_file` | 2 | No — binary |

**`person_education` is the interesting one.** Those five levels genuinely have an order, and step 6 must preserve it.

### What EDA found

| Problem | Detail | Fixed in |
|---|---|---|
| **An impossible value** | One person aged 144 | Step 5 |
| **Missing values** | 3 cells, in 3 different columns | Step 4 |
| **No duplicates** | Nothing to do | Step 3 |
| **Text columns** | 5, needing three different treatments | Step 6 |
| **Skewed income** | Mean far above median | Step 8 (scaling choice) |

---

## Step 3 — Duplicates

```python
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"{before} -> {len(df)}  ({before - len(df)} removed)")
```

```text
10000 -> 10000  (0 removed)
```

**None found.** **Run the check anyway** — it costs one line, and finding out later that your data was duplicated is far more expensive.

---

## Step 4 — Impossible values

> **This step comes before missing values here, and that is a deliberate change from use case 1.**
>
> The age-144 row would otherwise contribute to the medians we are about to compute. **It is the same reasoning as removing duplicates first: get the obviously wrong rows out before you calculate anything from them.**

```python
n_bad = (df["person_age"] > 100).sum()
df = df[df["person_age"] <= 100].reset_index(drop=True)
print(f"removed {n_bad} row with an impossible age -> {len(df)} rows")
print(f"that is {n_bad / len(df_raw):.2%} of the data")
```

```text
removed 1 row with an impossible age -> 9999 rows
that is 0.01% of the data
```

> **Compare this with use case 1**, where removing one outlier cost 10% of the dataset. **At 10,000 rows, deleting a bad row is free. At 12 rows it is a real decision.** The right action depends on how much data you have.

**Setting sensible limits per column is worth doing explicitly:**

```python
limits = {
    "person_age": (18, 100),           # a borrower is an adult, and mortal
    "person_income": (0, 5_000_000),   # income cannot be negative
    "credit_score": (300, 850),        # the scale's own bounds
    "loan_int_rate": (0, 40),
}
for col, (lo, hi) in limits.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:<20}{bad:>4} outside {lo:,}-{hi:,}")
```

```text
person_age             0 outside 18-100
person_income          0 outside 0-5,000,000
credit_score           0 outside 300-850
loan_int_rate          0 outside 0-40
```

**All clear now.** **Writing the limits down forces you to think about what each column means** — which is the real work of this step.

---

## Step 5 — Missing values

```python
missing = df.isna().sum()
print(missing[missing > 0])
print("\nas a share of all cells:", f"{df.isna().sum().sum() / df.size:.4%}")
```

```text
person_gender          1
loan_int_rate          1
loan_percent_income    1

as a share of all cells: 0.0021%
```

### Could we just drop these rows?

```python
print(f"rows now       : {len(df)}")
print(f"after dropna() : {len(df.dropna())}")
print(f"cost           : {len(df) - len(df.dropna())} rows ({(len(df)-len(df.dropna()))/len(df):.3%})")
```

```text
rows now       : 9999
after dropna() : 9996
cost           : 3 rows (0.030%)
```

> **Here, dropping is completely reasonable** — three rows out of ten thousand. **In use case 1, the same choice cost a quarter of the data.** Same technique, opposite decision, and the reason is entirely the size of the dataset.

**We will fill instead, to keep every row and to practise the choice.**

```python
gender_mode = df["person_gender"].mode()[0]
df["person_gender"] = df["person_gender"].fillna(gender_mode)

for col in ["loan_int_rate", "loan_percent_income"]:
    median = df[col].median()
    df[col] = df[col].fillna(median)
    print(f"{col:<22} filled with median {median}")

print(f"person_gender          filled with mode '{gender_mode}'")
print("\nmissing remaining:", df.isna().sum().sum())
```

```text
loan_int_rate          filled with median 11.01
loan_percent_income    filled with median 0.12
person_gender          filled with mode 'male'

missing remaining: 0
```

### ⚠️ Why this step must come before encoding

**Here is what happens if you encode first.** Watch closely:

```python
from sklearn.preprocessing import LabelEncoder

demo = df_raw.copy()                              # the RAW data, gap still present
le_demo = LabelEncoder()
demo["person_gender"] = le_demo.fit_transform(demo["person_gender"])
print(dict(zip([str(c) for c in le_demo.classes_], le_demo.transform(le_demo.classes_))))
```

```text
{'female': 0, 'male': 1, 'nan': 2}
```

> **`LabelEncoder` turned the missing value into a third gender category, numbered 2.**
>
> **It does not warn you.** The column now looks perfectly numeric, contains a category that does not exist, and the model will happily learn from it. On a bigger gap this quietly corrupts the whole column.
>
> **Fill first, then encode.** Always.

**The same mistake in the numeric columns causes a different failure:** an unfilled `NaN` passes silently through the split and turns your entire scaled array into `NaN`. **You usually discover it when a model refuses to train, several steps later.**

---

## Step 6 — Encoding

**Five text columns, and three different treatments — because they are three different kinds of column.**

### Ordered categories: `person_education`

**`LabelEncoder` would sort these alphabetically:**

```text
Associate=0, Bachelor=1, Doctorate=2, High School=3, Master=4
```

> **That puts Doctorate below High School.** The order is real, and the encoder got it backwards. **When a category genuinely has an order, state the order yourself.**

```python
EDUCATION_ORDER = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
education_map = {level: i for i, level in enumerate(EDUCATION_ORDER)}

df["person_education"] = df["person_education"].map(education_map)
print(education_map)
```

```text
{'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
```

### Binary categories: Label Encoding is safe

```python
for col in ["person_gender", "previous_loan_defaults_on_file"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"{col:<34}{dict(zip(le.classes_, le.transform(le.classes_)))}")
```

```text
person_gender                     {'female': 0, 'male': 1}
previous_loan_defaults_on_file    {'No': 0, 'Yes': 1}
```

**With two categories there is only one gap, so no false ordering is possible.**

### Unordered categories: dummy variables

```python
unordered = ["person_home_ownership", "loan_intent"]
dummies = pd.get_dummies(df[unordered], drop_first=True).astype(int)
df = pd.concat([df.drop(columns=unordered), dummies], axis=1)

print(f"{len(unordered)} columns -> {dummies.shape[1]} dummy columns")
print(dummies.columns.tolist())
print("\nshape now:", df.shape)
print("text columns left:", df.select_dtypes("object").shape[1])
```

```text
2 columns -> 8 dummy columns
['person_home_ownership_OTHER', 'person_home_ownership_OWN',
 'person_home_ownership_RENT', 'loan_intent_EDUCATION',
 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
 'loan_intent_PERSONAL', 'loan_intent_VENTURE']

shape now: (9999, 20)
text columns left: 0
```

**14 columns became 20.** Four home-ownership categories became 3 columns and six loan intents became 5, because `drop_first=True` leaves one out of each — **and that dropped category is the case where all the others are 0.**

**Everything is numeric now.**

---

## Step 7 — Train-test split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train:", X_train.shape, "  y_train:", y_train.shape)
print("X_test :", X_test.shape, "  y_test :", y_test.shape)
print(f"\nbalance  full {y.mean():.4f}   train {y_train.mean():.4f}   test {y_test.mean():.4f}")
```

```text
X_train: (7999, 19)   y_train: (7999,)
X_test : (2000, 19)   y_test : (2000,)

balance  full 0.5000   train 0.5001   test 0.5000
```

> **Compare this with use case 1**, where the train and test balances came out at 0.571 and 0.333. **With 10,000 rows, `stratify=y` lands almost exactly on the true rate.** Stratification works properly when there are enough rows to divide.

**Note `test_size=0.2` here against `0.3` in use case 1.** With plenty of data, 20% is a large enough test set; on tiny data you hold back more so the test set is not absurdly small.

---

## Step 8 — Scaling

```python
from sklearn.preprocessing import StandardScaler

numeric_cols = ["person_age", "person_income", "person_emp_exp", "loan_amnt",
                "loan_int_rate", "loan_percent_income",
                "cb_person_cred_hist_length", "credit_score"]

scaler = StandardScaler()
scaler.fit(X_train[numeric_cols])                 # FIT on train only

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("before scaling:")
print(X_train[["person_age", "person_income"]].describe().loc[["mean", "min", "max"]].round(1))
print("\nafter scaling:")
print(X_train_scaled[["person_age", "person_income"]].describe().loc[["mean", "min", "max"]].round(3))
```

```text
before scaling:
       person_age  person_income
mean         27.7        72337.5
min          20.0         8000.0
max          94.0      2448661.0

after scaling:
       person_age  person_income
mean          0.0          0.000
min          -1.3         -1.098
max          11.1         40.575
```

**Both columns now centre on 0.** But look at the maximums: **`person_income` reaches 40.6 standard deviations above the mean.** That is the skew you spotted at step 2, and scaling did not remove it — **it is a change of units, not of shape.**

```python
print(f"train mean: {X_train_scaled[numeric_cols].values.mean():+.6f}")
print(f"test  mean: {X_test_scaled[numeric_cols].values.mean():+.6f}   <- not exactly 0, and that is CORRECT")
```

```text
train mean: -0.000000
test  mean: +0.003057   <- not exactly 0, and that is CORRECT
```

**The test set was scaled with the training set's numbers.** Exactly zero would mean you had fitted on the test data.

> **On a column this skewed, `RobustScaler` is worth considering.** It uses the median and IQR instead of the mean and standard deviation, so a 2.4-million income barely moves it. **`MinMaxScaler` would be the worst choice here** — that one extreme value would define the maximum and crush every other income toward 0.

---

## The complete pipeline for use case 2

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"

EDUCATION_ORDER = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
DOMAIN_LIMITS = {"person_age": (18, 100), "person_income": (0, 5_000_000),
                 "credit_score": (300, 850), "loan_int_rate": (0, 40)}

# 1. LOAD
df_raw = pd.read_csv(dataset_url)
df = df_raw.copy()

# 2. EDA (findings: 1 impossible age, 3 gaps, 0 duplicates, 5 text columns)

# 3. DUPLICATES
df = df.drop_duplicates().reset_index(drop=True)

# 4. IMPOSSIBLE VALUES - before computing any medians
for col, (lo, hi) in DOMAIN_LIMITS.items():
    df = df[(df[col] >= lo) & (df[col] <= hi)]
df = df.reset_index(drop=True)

# 5. MISSING VALUES - mode for text, median for numbers
df["person_gender"] = df["person_gender"].fillna(df["person_gender"].mode()[0])
for col in ["loan_int_rate", "loan_percent_income"]:
    df[col] = df[col].fillna(df[col].median())

# 6. ENCODING - three different treatments for three kinds of column
df["person_education"] = df["person_education"].map(
    {level: i for i, level in enumerate(EDUCATION_ORDER)})          # ordered
for col in ["person_gender", "previous_loan_defaults_on_file"]:
    df[col] = LabelEncoder().fit_transform(df[col])                 # binary
unordered = ["person_home_ownership", "loan_intent"]
df = pd.concat([df.drop(columns=unordered),
                pd.get_dummies(df[unordered], drop_first=True).astype(int)], axis=1)

# 7. SPLIT  <- everything after this fits on TRAIN ONLY
X = df.drop(columns=["loan_status"])
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 8. SCALING - fit on train, transform both
numeric_cols = ["person_age", "person_income", "person_emp_exp", "loan_amnt",
                "loan_int_rate", "loan_percent_income",
                "cb_person_cred_hist_length", "credit_score"]
scaler = StandardScaler().fit(X_train[numeric_cols])
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"{df_raw.shape} raw  ->  X_train {X_train_scaled.shape}, X_test {X_test_scaled.shape}")
print("no missing values :", not X_train_scaled.isna().any().any())
print("all numeric       :", X_train_scaled.select_dtypes("object").empty)
print(f"balance  train {y_train.mean():.4f}  test {y_test.mean():.4f}")
```

```text
(10000, 14) raw  ->  X_train (7999, 19), X_test (2000, 19)
no missing values : True
all numeric       : True
balance  train 0.5001  test 0.5000
```

---

## What the two use cases showed

**The same eight steps, and different decisions at almost every one.**

| Step | `pre_data.csv` (12 rows) | `loan_data_10k.csv` (10,000 rows) |
|---|---|---|
| **Duplicates** | 1 found, removed | 0 found — check anyway |
| **Missing** | 3 of 48 cells. Dropping would cost **25%** → filled | 3 of 140,000. Dropping would cost **0.03%** → could go either way |
| **Outliers** | 1 salary of 1.5M → removed, at a cost of 10% of the data | 1 impossible age → removed, at a cost of 0.01% |
| **Encoding** | 2 columns, 2 treatments | 5 columns, **3** treatments — including a genuinely ordered one |
| **Split** | `test_size=0.3`; stratify cannot balance 7 rows | `test_size=0.2`; stratify lands on 0.5001 / 0.5000 |
| **Scaling** | `StandardScaler` is fine | Income is so skewed that `RobustScaler` deserves consideration |

> **The sequence is fixed. The judgements are not.** How much data you have changes what you can afford to throw away, and what kind of columns you have changes how you encode them.
>
> **That is why every step in this session asked you to look before deciding.**

---

# ❓ 20 MCQs

**Answer from memory first, then check.** Anything you get wrong points at the step you should re-read.

### Exploratory Data Analysis

**Q1.** You load a new dataset. What should you do first?
- (a) Train a model  (b) Explore it — shape, head, info, describe  (c) Scale it  (d) Split it

**Q2.** `df.info()` shows a column with 9,999 non-null values in a 10,000-row DataFrame. This means…
- (a) The column is fine  (b) One value is missing  (c) There is a duplicate  (d) The dtype is wrong

**Q3.** `describe()` shows `Salary` with a mean of 196,091 and a median of 67,000. This tells you…
- (a) Nothing  (b) The column is skewed by one or more very large values  (c) A quarter of the data is missing  (d) The column is text

**Q4.** A column shows dtype `object`. Before a model can use it you must…
- (a) Scale it  (b) Encode it into numbers  (c) Drop it  (d) Nothing

**Q5.** Why check the target's class balance during EDA?
- (a) Curiosity  (b) It decides whether accuracy is a fair metric later  (c) It speeds up training  (d) It is required by sklearn

### Missing values and duplicates

**Q6.** For a **skewed** numeric column, the safer filler is…
- (a) The mean  (b) The median  (c) Zero  (d) The mode

**Q7.** In `pre_data.csv`, filling the missing `Salary` with the **mean** (196,091) caused that row to be…
- (a) Kept unchanged  (b) Flagged and deleted as an outlier two steps later  (c) Duplicated  (d) Encoded incorrectly

**Q8.** For a missing **text** category, you fill with…
- (a) The mean  (b) The median  (c) The mode  (d) Zero

**Q9.** `df["Country"].mode()` returns a Series rather than one value because…
- (a) A bug  (b) There can be a tie for the most common value  (c) It counts rows  (d) It is deprecated

**Q10.** Why does the order of preprocessing steps matter?
- (a) It does not  (b) Each step changes the numbers the next one uses  (c) sklearn requires a fixed order  (d) Only for speed

**Q11.** `df.isnull().sum()` and `df.isna().sum()` are…
- (a) Different functions  (b) Exactly the same function under two names  (c) Only for text  (d) Deprecated

**Q12.** Two customers each buy tea at 15, producing identical rows. Calling `drop_duplicates()`…
- (a) Is always correct  (b) Would delete a real sale  (c) Does nothing  (d) Raises an error

### Outliers

**Q13.** On a box plot, the dots beyond the whiskers represent…
- (a) The median  (b) Outliers — values beyond 1.5 × IQR from the quartiles  (c) Missing values  (d) Duplicates

**Q14.** In use case 1, the `Salary` IQR bounds were [18,500, 118,500] and **two** rows were flagged. They were…
- (a) The two lowest salaries  (b) The genuine 1,500,000 outlier, and the row we had filled with the mean  (c) Two duplicates  (d) Two missing values

**Q15.** The loan dataset's impossible age of 144 was found by…
- (a) The IQR rule  (b) Knowing what an age can be  (c) `dropna()`  (d) The Z-score rule

### Encoding

**Q16.** Label Encoding an **unordered** category is a problem because…
- (a) It is slow  (b) The integers imply an order the categories do not have  (c) It creates too many columns  (d) It loses data

**Q17.** `LabelEncoder` on `person_education` produces `Doctorate=2, High School=3`. This is…
- (a) Correct  (b) Backwards — it sorted alphabetically, not by level  (c) Random  (d) An error

**Q18.** `pd.get_dummies(df["Country"], drop_first=True)` on three countries produces…
- (a) 3 columns  (b) 2 columns  (c) 1 column  (d) 4 columns

### Splitting and scaling

**Q19.** What does `stratify=y` do?
- (a) Sorts the data  (b) Keeps the class balance the same in train and test  (c) Scales the features  (d) Removes outliers

**Q20.** After correctly scaling, the training mean is 0.000000 and the test mean is +0.003057. This means…
- (a) A bug — refit the scaler  (b) It is correct: the test set was scaled using the training set's mean and std  (c) The test set is too small  (d) You must scale the test set separately

<details><summary>Answers</summary>

**A1 — (b) Explore it.** Preprocessing without EDA is prescribing without examining. **Every problem you find becomes a step in the plan.**

**A2 — (b) One value is missing.** Compare `Non-Null Count` against the row count — the difference is the number of gaps.

**A3 — (b) Skewed.** The mean is dragged up by extreme values while the median stays with the bulk of the data. **Comparing the two is the fastest outlier detector you have.**

**A4 — (b) Encode it.** `object` means text, and a model does arithmetic.

**A5 — (b).** At 90/10, a model that predicts the majority class every time scores 90% and is useless. You need to know that *before* you choose a metric.

**A6 — (b) The median.** One extreme value drags the mean a long way and barely moves the median.

**A7 — (b) Deleted as an outlier.** The mean was inflated by the 1,500,000 row, so the value we invented (196,091) landed outside the IQR upper bound of 118,500. **A choice at step 3 cost us a whole row at step 5.**

**A8 — (c) The mode**, the most common value. There is no "average" text.

**A9 — (b) There can be a tie.** Which is why you take `.mode()[0]`.

**A10 — (b).** Demonstrated in use case 1: imputing with the mean before removing the outlier created a value that was then deleted as an outlier itself. **Preprocessing steps are not independent.**

**A11 — (b) The same function.** Pandas provides both because R users expect `isna()` and SQL users expect `isnull()`.

**A12 — (b) It would delete a real sale.** The data alone cannot tell an error from a genuine repeat. **Add a transaction ID and de-duplicate on that instead.**

**A13 — (b) Outliers.** The whiskers stop at Q1 − 1.5×IQR and Q3 + 1.5×IQR, and anything past them becomes a dot. **The box plot and the IQR rule are the same calculation, drawn and printed.**

**A14 — (b).** One was a real error; the other was our own doing. **The rule could not tell them apart** — which is why you look at what was flagged before deleting it.

**A15 — (b) Domain knowledge.** No statistical rule knows that people do not live to 144. **This is where judgement beats formulas.**

**A16 — (b).** Jersey numbers read as seat numbers. A linear model, kNN or SVM will act on the false order.

**A17 — (b) Backwards.** It sorted the words alphabetically. **When a category genuinely has an order, state the order yourself with an explicit map.**

**A18 — (b) 2 columns.** The dropped category is the row where both are 0 — no information is lost, and the dummy variable trap is avoided.

**A19 — (b).** Essential for classification, and especially on an imbalanced target.

**A20 — (b) It is correct.** The test set was transformed with the *training* set's numbers. **Exactly 0.000000 on the test set would mean you had fitted on the test data — which is leakage.**
</details>

---

# 🎯 Preprocessing tasks

**Work through these on your own.** They are ordered from short to substantial.

---

## Warm-up — one step each

**Task 1 — The EDA report.**
Load `pre_data.csv` and answer all six EDA questions in writing: how big, what a row looks like, types and gaps, numeric spread, duplicates, and target balance. **Finish with a list of the problems you found and which step fixes each.**

**Task 2 — The cost of dropping.**
For both datasets, compute how many rows `dropna()` would cost, as a count and a percentage. **Then state, for each, whether dropping or filling is the better choice and why.**

**Task 3 — Mean versus median, measured.**
On `pre_data.csv`, fill the missing `Salary` twice — once with the mean and once with the median. **Print both filled tables and write one sentence on why the mean version is wrong here.**

**Task 4 — The duplicate's influence.**
Show that removing the duplicate row changes the `Age` and `Salary` medians. **Then explain in your own words why this makes the order of the steps matter.**

**Task 5 — IQR by hand.**
Compute Q1, Q3, IQR and both bounds for `Salary` in `pre_data.csv` **without** using a helper function — just `quantile()` and arithmetic. Confirm you get [18,500, 118,500].

---

## Applying a choice

**Task 6 — Cap instead of remove.**
Redo step 5 of use case 1 using `clip()` to cap the outlier instead of deleting the row. **Compare the resulting `Salary` column, the row count, and the mean against the removal version. Which would you ship, and why?**

**Task 7 — The encoding comparison.**
Encode `Country` twice: once with `LabelEncoder` and once with `get_dummies`. **Print both results side by side and write two sentences on what the Label Encoding version implies that is not true.**

**Task 8 — Preserve the order.**
On the loan dataset, encode `person_education` both ways — with `LabelEncoder` and with an explicit ordered map. **Print both mappings and state exactly which pairs of levels the encoder puts in the wrong order.**

**Task 9 — Scaler comparison.**
Scale the loan dataset's `person_income` with `MinMaxScaler`, `StandardScaler` and `RobustScaler`. **Print the median and maximum of each result and say which scaler you would choose for this column, with a reason.**

**Task 10 — Prove the split rule.**
Fit a `StandardScaler` on all the data, then on the training set only. **Print the test-set mean under each. Explain which is correct and what the other one is called.**

---

## Whole pipelines

**Task 11 — A third dataset.**
Take `classification/diabetes_prediction_dataset.csv` and run all eight steps on it, using `dataset_url` as in this session. **Report the shapes of `X_train`, `X_test`, `y_train`, `y_test`, and the class balance in each.**

> ⚠️ **This dataset is only about 8.5% positive.** Say explicitly what that means for `stratify=y` and for the metric you would report later.

**Task 12 — The reusable function.**
Turn the eight steps into `preprocess(dataset_url, target_column)` that returns `X_train, X_test, y_train, y_test` **plus a log of every decision it made**. Run it unchanged on both datasets from this session.

**Task 13 — The order experiment.**
Run the pipeline on `pre_data.csv` twice: once with duplicates removed first, and once with missing values filled first. **Print the final `X_train` from each and identify every value that differs. Explain why.**

**Task 14 — The wrong-order bug.**
Reproduce the `LabelEncoder` bug from use case 2: encode `person_gender` before filling its gap, and show the third category appearing. **Then write three sentences on why this bug is more dangerous than a crash.**

**Task 15 — Sensible limits.**
Write a domain-limits dictionary for `classification/diabetes_prediction_dataset.csv` — plausible ranges for age, BMI, HbA1c and blood glucose. **Report how many rows fall outside each, and decide what to do about them.**

---

## Going further

**Task 16 — The scikit-learn way.**
Rewrite use case 2's pipeline using `ColumnTransformer` and `Pipeline`, with `SimpleImputer` for the gaps and `StandardScaler` for the numbers. **Explain in one paragraph why this version makes leakage structurally impossible rather than merely avoided.**

**Task 17 — Missing-value flags.**
On any dataset, add a `was_missing` indicator column before filling a gap. **Then argue, in one paragraph, when this is worth doing and when it is noise.**

**Task 18 — The preprocessing report.**
For one dataset, write the preprocessing section of a project report: what you found, what you did, what you assumed, and **which conclusions would be weak because they depend on values you invented.**

**Task 19 — Break it deliberately.**
Take the finished use case 1 pipeline and introduce one error of each kind: fill before de-duplicating, encode before filling, and scale before splitting. **For each, show a number that changes, and say how you would have caught it.**

**Task 20 — Your own messy dataset.**
Find a dataset with genuine problems — from Kaggle, or data.gov.in — and run the full sequence. **Produce a before-and-after summary table and a written note on every judgement call you made.**

---

# ✅ Before you move on

**EDA**

- [ ] I explore before I change anything: shape, head, info, describe
- [ ] I read `Non-Null Count` against the row count to find gaps
- [ ] I compare mean and median to spot skew
- [ ] I check the target balance, because it decides my metric
- [ ] I turn EDA findings into a numbered preprocessing plan

**Cleaning**

- [ ] I remove duplicates **first**, and I can say why with numbers
- [ ] I fill text with the mode and skewed numbers with the median
- [ ] I compute what `dropna()` would cost before calling it
- [ ] I know the IQR rule, and that it flags tails as well as errors
- [ ] I use domain limits to find impossible values, because no formula will

**Preparing for a model**

- [ ] I fill missing values **before** encoding, so no phantom category appears
- [ ] I use dummy variables for unordered categories, with `drop_first=True`
- [ ] I state an ordered category's order myself rather than letting the encoder guess
- [ ] I produce `X_train`, `X_test`, `y_train`, `y_test` with `stratify=y`
- [ ] **I fit the scaler on the training set only**
- [ ] I know the test mean should *not* be exactly zero
- [ ] **I can run all eight steps, in order, on a dataset I have never seen**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-03-eda-preprocessing.ipynb) | Both walkthroughs, runnable |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
| [Session 4 — Introduction to AI & ML](session-04-intro-ml-ai.md) | Where this data finally meets a model |
