# Session 5 — Regression

**Supervised learning · Linear Regression · Evaluation metrics · Three complete use cases**

| | |
|---|---|
| **Notebook** | [session-05-regression.ipynb](../notebooks/session-05-regression.ipynb) |
| **Previous** | [Session 4 — Introduction to AI & ML](session-04-intro-ml-ai.md) |
| **Next** | [Session 5B — Classification & Deployment](session-05b-classification.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Sessions 1–4 got you ready. This is where you train your first model.**
>
> Regression predicts a **number**. By the end you will have built three of them, and — more importantly — **you will know what every evaluation metric actually means.**

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Tell a regression problem from a classification one, from the target column alone
2. Run the full workflow: load, explore, clean, split, train, predict, evaluate
3. **Explain MAE, MSE, RMSE and R² properly — including what each one hides**
4. Read a linear model's coefficients out loud as a plain-English sentence
5. Judge whether a model is good enough by comparing RMSE against the target's scale
6. Recognise when the limit is the data rather than the model

---

## How this session is organised

**Three complete use cases**, each chosen to show something the previous one could not.

| | Dataset | Features | What it teaches |
|---|---|---|---|
| **Use case 1** | `salary_data.csv` | **1** — years of experience | The whole workflow, and **what every metric means** |
| **Use case 2** | `advertising.csv` | **3** — TV, radio, newspaper | Reading coefficients, and dropping a useless feature |
| **Use case 3** | `cardekho_dataset.csv` | **10** — a real used-car market | Encoding, scaling, and **what to do when the model is not good enough** |

| # | Topic |
|---|---|
| 1 | [Supervised learning: regression and classification](#1-supervised-learning-regression-and-classification) |
| 2 | [Use case 1 — Salary from experience](#2-use-case-1--salary-from-experience) |
| 3 | [Use case 2 — Advertising spend and sales](#3-use-case-2--advertising-spend-and-sales) |
| 4 | [Use case 3 — Used car prices](#4-use-case-3--used-car-prices) |
| 5 | [What the three use cases showed](#5-what-the-three-use-cases-showed) |
| | [❓ 20 MCQs](#-regression--20-mcqs) · [🎯 Tasks](#-regression--tasks) |

**Practices sit between the use cases.** The MCQs and tasks are at the end.

---

# 1. Supervised learning: regression and classification

**Supervised** means you have the answers. You show the model inputs *and* correct outputs, and it learns the mapping between them.

🧠 **Analogy: past exam papers with the answer key.** You study a hundred solved problems. Nobody explained the underlying theory — you inferred it from worked examples. Then you sit a new paper. **The final exam is the test set.**

## The two kinds

| | Regression | Classification |
|---|---|---|
| The answer is | A **number** | A **category** |
| Question | *How much? How many?* | *Which one? Yes or no?* |
| Example | Salary from experience | Loan approved or rejected |
| A bad prediction is | Off by ₹4,000 | Simply wrong |
| Typical metric | RMSE, R² | Accuracy, F1 |

> **A regression prediction can be nearly right. A classification prediction is right or wrong.** That single difference is why the two families need completely different metrics — and it is why this session is split into Part A and Part B.

## The trap: numbers that are really categories

```text
age 30 + age 30 = 60        -> meaningful  -> age is a NUMBER
pincode + pincode           -> nonsense    -> pincode is a CATEGORY
```

**If the arithmetic is meaningless, it is a category** — however it is stored. Phone numbers, roll numbers, postcodes and any ID column all look numeric and are not.

## ✏️ Practice

**Number (N) or category (C) — or neither?**

1. How much will this used car sell for?
2. Will this student pass the semester?
3. What natural groups exist among our customers?
4. How many minutes until the bus arrives?
5. A pincode column, stored as an integer

<details><summary>Solutions</summary>

```text
1. N - regression
2. C - classification
3. Neither - unsupervised clustering; there is no answer to predict
4. N - regression
5. C - a CATEGORY. Adding two pincodes is nonsense, so the arithmetic
       test says it is not really a number.
```
</details>

---

# 2. Use case 1 — Salary from experience

**One feature, one target, and no complications.** That is exactly why we start here: **every step is visible, and you can concentrate on what the metrics mean.**

**The question:** given someone's years of experience, predict their salary.

---

## Step 1 — Import the libraries and load the data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/salary_data.csv"

df = pd.read_csv(dataset_url)
df.head()
```

**Output:**

```text
   Experience    Salary
0         5.0   90000.0
1         3.0   65000.0
2        15.0  150000.0
3         7.0   60000.0
4        20.0  200000.0
```

**Two columns.** `Experience` is the **feature** — the information we are allowed to use. `Salary` is the **target** — the number we must predict.

---

## Step 2 — Exploratory Data Analysis

**Exactly the routine from Session 3.** Look before you touch anything.

```python
df.head()
df.tail()
df.info()
df.describe()
```

**Output of `info()`:**

```text
RangeIndex: 375 entries, 0 to 374
Data columns (total 2 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   Experience  373 non-null    float64
 1   Salary      373 non-null    float64
```

**375 rows, but only 373 non-null in each column — so there are two gaps in each.**

**Output of `describe()`:**

```text
       Experience         Salary
count  373.000000     373.000000
mean    10.030831  100577.345845
std      6.557007   48240.013482
min      0.000000     350.000000
25%      4.000000   55000.000000
50%      9.000000   95000.000000
75%     15.000000  140000.000000
max     25.000000  250000.000000
```

**Experience runs 0 to 25 years; salary runs 350 to 250,000.** The mean and median salary are close (100,577 against 95,000), so this column is **not badly skewed** — unlike the salary column you met in Session 3.

### Correlation — does experience actually relate to salary?

**Before building anything, check that there is a relationship to find.**

```python
df.corr()
```

**Output:**

```text
            Experience    Salary
Experience    1.000000  0.930338
Salary        0.930338  1.000000
```

**0.93 is a very strong positive correlation.** As experience rises, salary rises with it, closely. **This is the number that tells you a linear model is worth trying at all.**

```python
sns.heatmap(df.corr(), annot=True)
plt.show()
```

**A heatmap is overkill for two columns**, but on a dataset with fifteen it is the fastest way to see which pairs move together. **Learn the habit here where it is easy to read.**

> **If the correlation had been 0.05, a straight line would have been the wrong tool** — and you would have found that out in ten seconds rather than after building a model.

---

## Step 3 — Preprocessing

### Missing values

```python
df.isnull().sum()
```

**Output:**

```text
Experience    2
Salary        2
```

**Two gaps in each column.** With 375 rows, dropping them costs well under 1% of the data — **so dropping is the sensible choice here.**

```python
df.dropna(inplace=True)
print(df.shape)
```

**Output:**

```text
(373, 2)
```

> **Compare this with Session 3's `pre_data.csv`**, where dropping three rows cost 25% of the dataset and imputation was the right answer. **Same technique, opposite decision — and the deciding factor is how much data you have.**

### Duplicates

```python
df.duplicated().sum()
```

**Output:**

```text
219
```

**219 duplicates out of 373 rows — that is 59% of the dataset.**

> ⚠️ **Do not remove these.**
>
> **Look at what the columns are.** `Experience` is a whole number of years from 0 to 25. `Salary` is a round figure. **With only 26 possible experience values and a few dozen common salary figures, two different people having identical rows is completely ordinary.**
>
> **Two people with 5 years of experience both earning 90,000 is not a data-entry error. It is two people.**
>
> Removing them would delete 59% of a legitimate dataset — and it would systematically delete the *most common* combinations, which are exactly the ones the model most needs to learn.

**This is Session 3's lesson made concrete:** *before dropping duplicates, ask whether two rows could legitimately be identical.* **Here, obviously yes.**

```python
# df.drop_duplicates(inplace=True)   <- deliberately NOT done
```

### Outliers

```python
plt.boxplot(df['Experience'])
plt.show()

plt.boxplot(df['Salary'])
plt.show()
```

**Neither box plot shows any dots beyond the whiskers.** The IQR bounds work out at −12 to 32 years and −72,500 to 267,500 — **and every value falls comfortably inside.**

**No outliers to handle.**

### Encoding and scaling

**Neither is needed here:**

- **Encoding** converts text to numbers. **Both columns are already numeric**, so there is nothing to encode.
- **Scaling** puts columns on a comparable range. **Linear regression does not require it** — and with a single feature there is nothing to compare against anyway.

> **Recognising that a step is unnecessary is as important as knowing how to do it.** A pipeline that scales a column for no reason is not more careful; it is just longer.

---

## Step 4 — Train-test split

```python
from sklearn.model_selection import train_test_split

X = df.drop('Salary', axis=1)      # features - everything except the target
y = df['Salary']                   # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(df.shape, X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

**Output:**

```text
(373, 2) (373, 1) (373,)
(298, 1) (298,)
(75, 1) (75,)
```

**Read those shapes carefully — they tell you the split worked.**

| | Shape | Meaning |
|---|---|---|
| `X` | `(373, 1)` | 373 rows, **1 feature** |
| `y` | `(373,)` | 373 targets, and no second dimension |
| `X_train` | `(298, 1)` | 80% of the rows |
| `X_test` | `(75, 1)` | the remaining 20% |

**298 + 75 = 373.** Nothing was lost.

> **`X` is a DataFrame (two-dimensional) and `y` is a Series (one-dimensional).** That is not an accident — scikit-learn expects a table of features and a single column of answers. **`df.drop('Salary', axis=1)` keeps `X` two-dimensional even with one feature**, which is what the model needs.

> **No `stratify` here.** That argument keeps class proportions balanced, and a regression target has no classes to balance. **It is for classification only.**

---

## Step 5 — Model selection and training

**Linear regression fits a straight line through the data.**

```text
Salary = slope × Experience + intercept
```

**"Training" means finding the slope and intercept that make the errors smallest.**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

**Two lines.** `LinearRegression()` creates the model; `.fit()` learns from the training data.

> **This is the pattern from Session 4's Topic 19** — create, train, predict, measure. **Every model in scikit-learn works this way**, so learning it once is enough.

---

## Step 6 — Making predictions

```python
y_pred = model.predict(X_test)
y_pred[:5]
```

**Output:**

```text
[174795.47,  99746.98, 140682.52,  72456.62, 147505.11]
```

```python
y_test[:5]
```

**Output:**

```text
[180000.0, 65000.0, 125000.0, 80000.0, 140000.0]
```

**Now compare them, row by row:**

| | Actual | Predicted | Difference |
|---|---|---|---|
| 1 | 180,000 | 174,795 | **−5,205** |
| 2 | 65,000 | 99,747 | **+34,747** |
| 3 | 125,000 | 140,683 | +15,683 |
| 4 | 80,000 | 72,457 | −7,543 |
| 5 | 140,000 | 147,505 | +7,505 |

**Some predictions are close; one is out by nearly 35,000.** **The whole point of the next step is to turn this column of differences into a single number you can report.**

---

## Step 7 — Evaluation metrics, properly

**This is the most important section in Part A.** Four metrics, and each answers a different question.

### The problem with just adding up the errors

**Take four predictions with these errors:**

```text
prediction 1:  actual 10, predicted 11  ->  error = 10 - 11 = -1
prediction 2:  actual 15, predicted 14  ->  error = 15 - 14 = +1
prediction 3:  actual  9, predicted 11  ->  error =  9 - 11 = -2
prediction 4:  actual  8, predicted  6  ->  error =  8 -  6 = +2
```

**Add them up:**

```text
-1 + 1 + -2 + 2  =  0
```

> **Total error zero — and every single prediction was wrong.**
>
> **The positive and negative errors cancelled out.** This is why you can never simply sum the errors, and it is the reason all four metrics below do something to remove the sign.

**There are two ways to remove a sign: take the absolute value, or square it. Those two choices give you MAE and MSE.**

---

### MAE — Mean Absolute Error

**Take the absolute value of each error, then average them.**

```text
|-1| + |+1| + |-2| + |+2|  =  1 + 1 + 2 + 2  =  6
MAE = 6 / 4 = 1.5
```

**In words: on average, this model is wrong by 1.5.**

| | |
|---|---|
| **Units** | The same as your target |
| **Reads as** | "typically off by about this much" |
| **Treats all errors** | Equally — an error of 10 counts ten times an error of 1 |

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
```

**Output:**

```text
Mean Absolute Error: 12094.170266826043
```

> **Read that out loud: "the model is typically wrong by about ₹12,094".** **That sentence is what a non-technical person actually needs**, and MAE is the metric that gives it to you most directly.

---

### MSE — Mean Squared Error

**Square each error, then average them.**

```text
(-1)² + (+1)² + (-2)² + (+2)²  =  1 + 1 + 4 + 4  =  10
MSE = 10 / 4 = 2.5
```

**Squaring removes the sign too — but it does something else as well.**

> **Squaring punishes large errors far more than small ones.** An error of 10 becomes 100; an error of 1 becomes 1. **One big miss now counts a hundred times a small one.**
>
> **That is sometimes exactly what you want** — if being badly wrong occasionally is worse than being slightly wrong often.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

**Output:**

```text
Mean Squared Error: 241834883.89985102
```

> ⚠️ **241 million?** **The units are rupees *squared*.** Nobody can interpret that, and you should never put it in a report for a human. **MSE is useful for comparing two models, not for describing one.**

---

### RMSE — Root Mean Squared Error

**Take the square root of MSE.**

```text
RMSE = √2.5 = 1.58
```

**This undoes the squaring, so the number lands back in your target's units** — while keeping MSE's heavier punishment of large errors.

```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
```

**Output:**

```text
Root Mean Squared Error: 15551.041184437
```

> **RMSE is the metric to quote.** It is in rupees, it is interpretable, and it does not hide occasional large mistakes the way MAE can.

### MAE and RMSE together tell you something neither says alone

```text
MAE  = 12,094
RMSE = 15,551
```

**RMSE is always at least as large as MAE.** **How much larger tells you about the shape of your errors:**

| | Means |
|---|---|
| **RMSE ≈ MAE** | Errors are all roughly the same size |
| **RMSE ≫ MAE** | A few predictions are badly wrong, and dragging RMSE up |

**Here RMSE is about 29% above MAE — a moderate gap.** Most predictions are decent, and a handful are noticeably worse. **Looking at the two together is free information most people ignore.**

---

### R² — the coefficient of determination

**The first three metrics tell you *how wrong* you are. R² tells you *how much better than nothing* you are.**

🧠 **Analogy: comparing against a lazy guess.** Suppose you refuse to build a model and simply predict the *average salary* for everyone. That is the worst reasonable guess. **R² asks: how much of the error did the model remove, compared with that lazy guess?**

| R² | Meaning |
|---|---|
| **1.0** | Perfect — every prediction exact |
| **0.9** | The model explains 90% of the variation |
| **0.0** | No better than just guessing the average |
| **Negative** | **Worse than guessing the average** — yes, this is possible |

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
```

**Output:**

```text
R-squared: 0.8991335611
```

> **R² of 0.899 means the model explains about 90% of the variation in salary.** For a model with a single feature, that is a genuinely good result — **and it is what the 0.93 correlation at step 2 predicted.**

> ⚠️ **R² of 1.0 on a test set is not a triumph — it is a warning.** Real data has noise, so a perfect score almost always means the target leaked into your features.

---

### The four metrics side by side

| Metric | Value | Units | Use it to |
|---|---|---|---|
| **MAE** | 12,094 | Rupees | Say "typically off by about ₹12,000" |
| **MSE** | 241,834,884 | Rupees **squared** | Compare two models — never to describe one |
| **RMSE** | 15,551 | Rupees | **Report this one** |
| **R²** | 0.899 | None | Say "explains 90% of the variation" |

> **Report RMSE and R² together.** RMSE says how wrong you are in real units; R² says whether that is good relative to doing nothing. **Neither is enough alone.**

---

## Step 8 — Reading the model

**A linear model can be read out loud, and this is one of its great advantages.**

```python
slope = model.coef_
intercept = model.intercept_

print(f'Slope: {slope}')
print(f'Y-Intercept: {intercept}')
print(f'X-Intercept: {-intercept/slope}')
```

**Output:**

```text
Slope: [6822.59017499]
Y-Intercept: 31521.077629
X-Intercept: [-4.62012]
```

**The learned equation is:**

```text
Salary = 6,822.59 × Experience + 31,521.08
```

| Term | Value | What it means |
|---|---|---|
| **Slope** | 6,822.59 | **Each extra year of experience is worth about ₹6,823** |
| **Y-intercept** | 31,521.08 | The predicted salary at zero experience — a starting salary |
| **X-intercept** | −4.62 | Where the line crosses zero salary |

> **The x-intercept of −4.62 years is meaningless**, and worth saying so. It is where the mathematics puts the line, not a fact about the world — **you cannot have −4.62 years of experience.** **A model can be extended beyond its data; its answers there should not be trusted.**

> **"Each extra year of experience is worth about ₹6,823, starting from around ₹31,500."** **A model you can say in one sentence is a model you can defend in a meeting** — and no other family in this session gives you that as easily.

---

## Step 9 — Plotting the result

```python
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=1,
         label='Regression line', marker='*')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Linear Regression on Salary Data')
plt.legend()
plt.show()
```

![Scatter of actual salaries with the fitted regression line through them](images/s5-salary-regression-line.png)

**The line runs cleanly through the middle of the points**, and the points sit fairly evenly above and below it. **That is what a good linear fit looks like.**

**What to look for in this plot:**

| If you see | It means |
|---|---|
| Points scattered evenly above and below the line | **A good fit** |
| Points forming a curve around the line | A straight line is the wrong shape |
| The spread widening as x increases | Errors grow with the prediction — a real pattern worth handling |
| A few points very far from the line | Outliers, or something the features do not capture |

> **Always plot a regression, even when the metrics look good.** **R² of 0.899 would look identical whether the relationship is a clean line or a curve the model is cutting through** — and only the picture tells you which.

### Two diagnostic plots worth drawing every time

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

# Predicted against actual - points should hug the diagonal
ax1.scatter(y_test, y_pred, alpha=.6, edgecolor='k', linewidth=.4)
lims = [y_test.min() - 5000, y_test.max() + 5000]
ax1.plot(lims, lims, 'r--', label='perfect prediction')
ax1.set_xlabel('Actual salary'); ax1.set_ylabel('Predicted salary')
ax1.set_title('Predicted vs actual'); ax1.legend()

# Residuals - the errors, plotted against the prediction
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=.6, edgecolor='k', linewidth=.4)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('Predicted salary'); ax2.set_ylabel('Error (actual − predicted)')
ax2.set_title('Residuals')

plt.tight_layout()
plt.show()
```

![Predicted versus actual salaries, and a residual plot](images/s5-salary-diagnostics.png)

**Read them together:**

| Plot | What good looks like | What a problem looks like |
|---|---|---|
| **Predicted vs actual** | Points hugging the red diagonal | A curve, or points drifting off at one end |
| **Residuals** | A **shapeless cloud** around zero | A curve, a funnel, or any visible pattern |

> **The residual plot is the more sensitive of the two.** A pattern in the residuals means the model is missing something systematic — **and it will show a pattern long before R² drops enough to worry you.**

---

## ✏️ Practice — use case 1

1. Load the salary dataset and print its shape, `info()` and `describe()`.
2. Compute the correlation between `Experience` and `Salary`. What does the value tell you?
3. Count the duplicates. **Explain in one sentence why they should not be removed here.**
4. Split the data and print all four shapes. Confirm the train and test rows add up.
5. Train the model and print MAE, MSE, RMSE and R². **Write the one sentence you would say to a manager.**

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/salary_data.csv"
df = pd.read_csv(dataset_url)

print(df.shape)                       # 1  (375, 2)
df.info()
print(df.describe())

print(df.corr())                      # 2  0.930 - very strong positive
# As experience rises, salary rises closely with it. A straight line is
# worth trying. Had this been 0.05, it would not have been.

df.dropna(inplace=True)
print("duplicates:", df.duplicated().sum())                            # 3
# 219 of 373 - about 59%. Experience is a whole number of years and
# salary is a round figure, so two different people having identical
# rows is completely ordinary. They are real people, not data errors.

X = df.drop('Salary', axis=1); y = df['Salary']                        # 4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("rows add up:", len(X_train) + len(X_test) == len(X))

model = LinearRegression().fit(X_train, y_train)                       # 5
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE  {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"MSE  {mse:,.0f}")
print(f"RMSE {np.sqrt(mse):,.0f}")
print(f"R2   {r2_score(y_test, y_pred):.4f}")
# "The model predicts salary to within about 15,500 rupees typically,
#  and explains roughly 90% of the variation in pay."
```
</details>

---

# 3. Use case 2 — Advertising spend and sales

**One feature was enough to learn the mechanics. Real problems have several** — and that changes what you can read out of the model.

**The question:** given what was spent on TV, radio and newspaper advertising, predict the resulting sales.

---

## Steps 1–2 — Load and explore

```python
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/advertising.csv"

df = pd.read_csv(dataset_url)
print(df.shape)
df.head()
```

**Output:**

```text
(200, 4)

      TV  Radio  Newspaper  Sales
0  230.1   37.8       69.2   22.1
1   44.5   39.3       45.1   10.4
2   17.2   45.9       69.3   12.0
```

**Three features and one target.** Each row is one advertising campaign: what was spent on each channel, and what sales followed.

```python
print("missing:", df.isnull().sum().sum())
print("duplicates:", df.duplicated().sum())
```

**Output:**

```text
missing: 0
duplicates: 0
```

**Nothing to clean.** That is unusual for real data, and it lets us concentrate on the modelling.

### Correlation — now genuinely useful

**With one feature, correlation told you whether to bother. With three, it starts ranking them.**

```python
df.corr()['Sales'].sort_values(ascending=False)
```

**Output:**

```text
Sales        1.0000
TV           0.9012
Radio        0.3496
Newspaper    0.1580
```

**Read that ranking:**

- **TV at 0.90** — a very strong relationship with sales
- **Radio at 0.35** — moderate
- **Newspaper at 0.16** — weak, close to nothing

> **This is a prediction about what the model will find**, and it is worth writing down *before* you train, so you can check whether the model agrees.

### Seeing the three relationships

```python
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, channel in zip(axes, ['TV', 'Radio', 'Newspaper']):
    ax.scatter(df[channel], df['Sales'], alpha=.6, edgecolor='k', linewidth=.3)
    ax.set_title(f"{channel}   (r = {df[channel].corr(df['Sales']):.3f})")
    ax.set_xlabel(f'{channel} spend')
axes[0].set_ylabel('Sales')
plt.tight_layout()
plt.show()
```

![Three scatter plots: TV, Radio and Newspaper spend against Sales](images/s5-advertising-channels.png)

**The three panels show something the correlation numbers cannot:**

- **TV** — a clear upward band, but it **fans out** at higher spend. High TV budgets give less predictable returns.
- **Radio** — a looser upward trend, more scattered throughout.
- **Newspaper** — close to a shapeless cloud, which is what a correlation of 0.16 looks like.

> **That fan shape in the TV panel is exactly why you plot.** No single correlation number contains it, and it would change how you advise a marketing team.

---

## Steps 3–5 — Split, train, predict

**No preprocessing is needed, so we go straight to the split.**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop('Sales', axis=1)     # all three channels
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Output:**

```text
(160, 3) (40, 3)
```

**Notice `X_train` is now `(160, 3)` rather than `(298, 1)`.** **Three columns instead of one — and not a single line of the modelling code had to change.**

---

## Step 6 — Evaluate

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE  {mae:.4f}")
print(f"MSE  {mse:.4f}")
print(f"RMSE {rmse:.4f}")
print(f"R2   {r2:.4f}")
```

**Output:**

```text
MAE  1.2748
MSE  2.9078
RMSE 1.7052
R2   0.9059
```

**Sales are measured in thousands of units, so RMSE 1.71 means: typically off by about 1,710 units.**

> **R² of 0.906 with three features, against 0.899 with one in use case 1.** **Different datasets, so the two are not directly comparable** — but both are strong fits.

---

## Step 7 — Reading three coefficients

**This is what use case 1 could not show you.**

```python
for name, coef in zip(X.columns, model.coef_):
    print(f"{name:<12}{coef:+.6f}")
print(f"{'intercept':<12}{model.intercept_:+.6f}")
```

**Output:**

```text
TV          +0.054509
Radio       +0.100945
Newspaper   +0.004337
intercept   +4.714126
```

**The learned equation is:**

```text
Sales = 0.0545 × TV + 0.1009 × Radio + 0.0043 × Newspaper + 4.71
```

**Each coefficient says: how much does Sales change when this channel goes up by one unit, holding the others fixed?**

| Channel | Coefficient | Read as |
|---|---|---|
| **Radio** | +0.1009 | **The strongest per unit spent** |
| **TV** | +0.0545 | About half radio's effect per unit |
| **Newspaper** | +0.0043 | **Essentially nothing** |

> **Radio is worth roughly twice TV per unit of spend.** **That is a budget recommendation, not just a number** — and it is the kind of statement only a linear model hands you directly.

### ⚠️ The correlation and the coefficient disagree — and both are right

**Look back at the correlations:**

```text
correlation:   TV 0.90   >   Radio 0.35   >   Newspaper 0.16
coefficient:   Radio 0.10 >   TV 0.05     >   Newspaper 0.004
```

**TV has the higher correlation, but radio has the higher coefficient. How?**

> **They answer different questions.**
>
> **Correlation** asks: *as TV spend rises, does Sales rise?* TV budgets are large, so TV spend explains a lot of the total variation — hence 0.90.
>
> **The coefficient** asks: *if I spend one more unit on TV, what happens?* **Per unit spent, radio moves sales harder.**
>
> **Both are true.** TV explains more of what happened; radio gives more per rupee. **The first is a description of the past; the second is advice about the next rupee.**

> ⚠️ **Coefficients are only comparable like this when the features are on similar scales.** Here all three are spend in the same units, so the comparison is fair. **If one were in rupees and another in lakhs, you would have to scale first** — use case 3 shows exactly that.

---

## Step 8 — Should we drop Newspaper?

**Its coefficient is 0.004 and its correlation is 0.16. It looks useless. Test it rather than assuming.**

```python
X2 = df[['TV', 'Radio']]                  # drop Newspaper
X2_train, X2_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, random_state=42)

model2 = LinearRegression().fit(X2_train, y_train)
y_pred2 = model2.predict(X2_test)

print(f"with Newspaper   : R2 {r2_score(y_test, y_pred):.4f}  RMSE {rmse:.4f}")
print(f"without Newspaper: R2 {r2_score(y_test, y_pred2):.4f}  "
      f"RMSE {np.sqrt(mean_squared_error(y_test, y_pred2)):.4f}")
```

**Output:**

```text
with Newspaper   : R2 0.9059  RMSE 1.7052
without Newspaper: R2 0.9079  RMSE 1.6872
```

> **Removing a feature made the model slightly *better*.**
>
> **That surprises people, because more information ought to help.** But a feature carrying almost no signal still carries noise, and the model spends a little of its capacity fitting that noise. **Drop it and the model has one less distraction.**

**And a simpler model is better for reasons beyond the score:** fewer columns to collect, fewer to explain, fewer things to go wrong in production.

> **The gain here is small — 0.002 of R².** **Session 8 teaches you how to tell whether a difference that small is real** or just the particular rows that landed in the test set. **For now, note that the honest claim is "no worse, and simpler", not "better".**

---

## ✏️ Practice — use case 2

1. Load the advertising data and print the correlation of each channel with `Sales`.
2. Train the model and print all four metrics.
3. Print the three coefficients. **Which channel gives most per unit spent?**
4. Explain in two sentences why TV has the highest correlation but radio the highest coefficient.
5. Drop `Newspaper`, retrain, and compare. **Would you ship the simpler model? Say why.**

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/advertising.csv"
df = pd.read_csv(dataset_url)

print(df.corr()['Sales'].sort_values(ascending=False))                 # 1

X = df.drop('Sales', axis=1); y = df['Sales']                          # 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE {mean_absolute_error(y_test,y_pred):.4f}  MSE {mse:.4f}  "
      f"RMSE {np.sqrt(mse):.4f}  R2 {r2_score(y_test,y_pred):.4f}")

for n, c in zip(X.columns, model.coef_):                               # 3
    print(f"{n:<12}{c:+.6f}")
# Radio, at +0.1009 per unit - roughly twice TV's +0.0545.

# 4 - Correlation asks "as TV spend rises, does Sales rise?" TV budgets
#     are large, so TV explains much of the total variation.
#     The coefficient asks "what does ONE MORE unit of TV buy?" - and per
#     unit, radio moves sales harder. Both are true; one describes the
#     past, the other advises the next rupee.

X2 = df[['TV', 'Radio']]                                               # 5
a, b, c_, d_ = train_test_split(X2, y, test_size=0.2, random_state=42)
p2 = LinearRegression().fit(a, c_).predict(b)
print(f"with    R2 {r2_score(y_test, y_pred):.4f}")
print(f"without R2 {r2_score(d_, p2):.4f}")
# Yes - it is no worse, and it is simpler: one fewer column to collect,
# explain and maintain. The honest claim is "no worse and simpler",
# not "better" - the 0.002 gain is too small to call on one split.
```
</details>

---

# 4. Use case 3 — Used car prices

**The first two datasets were clean and cooperative. This one is neither** — and it is much more like what you will actually be handed.

**The question:** given a used car's age, mileage, engine and other details, predict its selling price.

**What is different here:**

| | Use cases 1 & 2 | Use case 3 |
|---|---|---|
| Rows | 200–375 | **15,411** |
| Text columns | None | **Six** |
| Target | Well behaved | **Heavily skewed** |
| Result | Good fit | **Not good enough — and we deal with that honestly** |

---

## Step 1 — Load and explore

```python
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/cardekho_dataset.csv"

df = pd.read_csv(dataset_url)
print(df.shape)
print(df.columns.tolist())
```

**Output:**

```text
(15411, 14)
['Unnamed: 0', 'car_name', 'brand', 'model', 'vehicle_age', 'km_driven',
 'seller_type', 'fuel_type', 'transmission_type', 'mileage', 'engine',
 'max_power', 'seats', 'selling_price']
```

```python
print("missing:", df.isnull().sum().sum())
print("duplicates:", df.duplicated().sum())
print(df['selling_price'].describe())
```

**Output:**

```text
missing: 0
duplicates: 0

mean     7.749713e+05
50%      5.560000e+05
max      3.950000e+07
```

> **Mean 774,971 against a median of 556,000, and a maximum of 39.5 million.** **The target is heavily right-skewed** — most cars are ordinary and a few are extremely expensive. **Remember this; it comes back at step 6.**

### Looking at the text columns

```python
for col in df.select_dtypes('object').columns:
    print(f"{col:<22}{df[col].nunique():>5} distinct")
```

**Output:**

```text
car_name                121 distinct
brand                    32 distinct
model                   120 distinct
seller_type               3 distinct
fuel_type                 5 distinct
transmission_type         2 distinct
```

**Six text columns, and they are not all the same kind of problem.**

---

## Step 2 — Preprocessing

### Dropping columns that cannot help

```python
df = df.drop(columns=['Unnamed: 0', 'car_name', 'model'])
print(df.shape)
```

**Output:**

```text
(15411, 11)
```

**Three columns removed, each for a different reason:**

| Column | Why it goes |
|---|---|
| `Unnamed: 0` | **A row number** left over from how the file was saved. It is an index, not information |
| `car_name` | **121 distinct values.** Dummy-encoding it would add 120 columns |
| `model` | **120 distinct values**, and it largely repeats `car_name` |

> **`brand` is kept at 32 categories.** That is still a lot, but it is genuine information — a Maruti and a BMW are different markets. **The line between "a useful category" and "too many to encode" is a judgement, and 121 is clearly over it.**

### Encoding

```python
from sklearn.preprocessing import LabelEncoder

for col in ['brand', 'seller_type', 'fuel_type', 'transmission_type']:
    df[col] = LabelEncoder().fit_transform(df[col])

print("all numeric:", df.select_dtypes('object').empty)
```

**Output:**

```text
all numeric: True
```

> ⚠️ **Label Encoding on `brand` implies an order that does not exist** — that brand 20 is somehow more than brand 5. **Session 3 warned about exactly this, and it matters more for linear regression than for a tree**, because a linear model multiplies the code by a weight.
>
> **Dummy variables would be more correct here.** They would also add 31 columns. **We use Label Encoding to keep this walkthrough readable, and note the cost honestly** — one of the tasks at the end asks you to try it the other way and measure the difference.

---

## Step 3 — Split and scale

```python
from sklearn.preprocessing import StandardScaler

X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)        # FIT on train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.shape, X_test.shape)
```

**Output:**

```text
(12328, 10) (3083, 10)
```

**Ten features, 12,328 training rows.**

### ⚠️ Does scaling actually change anything here?

**Test it rather than assuming.**

```python
model_scaled = LinearRegression().fit(X_train_scaled, y_train)
pred_scaled = model_scaled.predict(X_test_scaled)

model_raw = LinearRegression().fit(X_train, y_train)
pred_raw = model_raw.predict(X_test)

print(f"with scaling   : R2 {r2_score(y_test, pred_scaled):.4f}")
print(f"without scaling: R2 {r2_score(y_test, pred_raw):.4f}")
```

**Output:**

```text
with scaling   : R2 0.6639
without scaling: R2 0.6639
```

> **Identical, to four decimal places.**
>
> **Linear regression does not need scaling to fit well.** It simply adjusts each coefficient to suit that column's units.
>
> **So why scale at all?** **Because it makes the coefficients comparable with one another** — which is the whole point of step 5 below. **Scaling changes what you can read, not how well the model fits.**

---

## Step 4 — Train and evaluate

```python
model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print(f"MAE  {mean_absolute_error(y_test, y_pred):>12,.0f}")
print(f"RMSE {np.sqrt(mse):>12,.0f}")
print(f"R2   {r2_score(y_test, y_pred):>12.4f}")
```

**Output:**

```text
MAE       278,522
RMSE      503,024
R2         0.6639
```

### ⚠️ Read this honestly

**R² of 0.66 sounds acceptable. It is not.**

```python
print(f"median car price : {y.median():>12,.0f}")
print(f"RMSE             : {np.sqrt(mse):>12,.0f}")
print(f"RMSE as a share of the median price: {np.sqrt(mse)/y.median():.0%}")
```

**Output:**

```text
median car price :      556,000
RMSE             :      503,024
RMSE as a share of the median price: 90%
```

> **The typical error is 90% of the typical car's price.**
>
> **A model that says "this car is worth ₹556,000, give or take ₹503,000" is useless to a buyer or a seller.**
>
> **R² of 0.66 hid that completely.** R² is a ratio with no units, so it cannot tell you whether the remaining error matters. **Only putting RMSE next to the actual scale of the target reveals it.**

**This is the most important lesson in Part A: always compare your error against the size of the thing you are predicting.**

---

## Step 5 — Reading the coefficients

**Now that the features are scaled, the coefficients are comparable.**

```python
for name, coef in sorted(zip(X.columns, model.coef_), key=lambda t: -abs(t[1])):
    print(f"{name:<20}{coef:>14,.0f}")
```

**Output:**

```text
max_power                  641,806
vehicle_age               -181,644
mileage                     64,377
km_driven                  -50,734
engine                      50,421
transmission_type          -44,793
brand                       11,378
seats                        7,003
fuel_type                   -6,147
seller_type                -2,047
```

**These tell a story that matches common sense:**

| Feature | Sign | Reading |
|---|---|---|
| **`max_power`** | **+641,806** | **By far the strongest.** More powerful cars cost much more |
| **`vehicle_age`** | **−181,644** | **Negative** — older cars are worth less. Exactly as expected |
| `km_driven` | −50,734 | Negative — more mileage, lower price |
| `brand` | +11,378 | **Small, and meaningless anyway** — the label codes are arbitrary |

> **The `brand` coefficient is a good example of why Label Encoding on an unordered category is unsatisfying.** The model has fitted a straight line through arbitrary brand numbers. **The number is not wrong arithmetically; it just does not mean anything.**

---

## Step 6 — Improving it: the skewed target

**Step 1 noticed the target was heavily skewed. That is the problem worth attacking.**

**Linear regression assumes errors are roughly even across the range.** With prices from 100,000 to 39.5 million, they are not — **the model is being pulled around by a handful of very expensive cars.**

**A standard fix is to model the *logarithm* of the price instead.** Taking logs compresses the long tail, so the expensive cars stop dominating.

```python
y_log = np.log1p(y)          # log(1 + price), so a price of 0 is safe

X_train, X_test, y_log_train, y_log_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42)

model_log = LinearRegression().fit(X_train, y_log_train)
pred_log = model_log.predict(X_test)

print(f"R2 on the original target : 0.6639")
print(f"R2 on the log target      : {r2_score(y_log_test, pred_log):.4f}")
```

**Output:**

```text
R2 on the original target : 0.6639
R2 on the log target      : 0.8634
```

> **R² rises from 0.66 to 0.86** — a large improvement from one line of preprocessing, and **more than any change of model would have given here.**

> ⚠️ **Two honest caveats.**
>
> **First, the two R² values are not directly comparable** — one is measured on prices, the other on log-prices. **To compare fairly you must convert the predictions back with `np.expm1()` and recompute RMSE in rupees.**
>
> **Second, this did not make the model good — it made it better.** The remaining error is still substantial. **A real used-car model would need features this dataset does not have: condition, service history, accident record, and location.**

### Seeing the difference

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

ax1.scatter(y_test, y_pred, alpha=.25, s=12)
ax1.plot([0, 4_000_000], [0, 4_000_000], 'r--', label='perfect prediction')
ax1.set_xlim(0, 4_000_000); ax1.set_ylim(0, 4_000_000)
ax1.set_xlabel('Actual price'); ax1.set_ylabel('Predicted price')
ax1.set_title('Original target'); ax1.legend()

ax2.scatter(y_log_test, pred_log, alpha=.25, s=12, color='seagreen')
ax2.set_xlabel('Actual log(price)'); ax2.set_ylabel('Predicted log(price)')
ax2.set_title('Log-transformed target')

plt.tight_layout()
plt.show()
```

![Predicted versus actual car prices, before and after log-transforming the target](images/s5-car-log-transform.png)

**The left panel is what an unusable model looks like.** The points form a wide, shapeless spread rather than following the diagonal — and notice the model rarely predicts anything above about 1.5 million, however expensive the car really was. **It has learned to play safe near the middle.**

**The right panel is visibly tighter.** Taking logs compressed the long tail, so the expensive cars stopped dominating and the model could spread its predictions properly.

**That last point is the honest conclusion: sometimes the limit is the data, not the model.** **No algorithm invents information that was never collected.**

---

## ✏️ Practice — use case 3

1. Load the car dataset and print the shape, the number of distinct values in each text column, and `describe()` for `selling_price`.
2. Explain why `car_name` and `model` are dropped but `brand` is kept.
3. Train the model and compute RMSE. **Then express RMSE as a percentage of the median price and say what that means.**
4. Print the scaled coefficients in order of size. **Do the signs of `vehicle_age` and `km_driven` make sense?**
5. Apply `np.log1p` to the target, retrain, and report the new R². **Name the two caveats that stop this being a clean win.**

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/cardekho_dataset.csv"
df = pd.read_csv(dataset_url)

print(df.shape)                                                        # 1
for c in df.select_dtypes('object').columns:
    print(f"{c:<22}{df[c].nunique():>5}")
print(df['selling_price'].describe())

# 2 - car_name has 121 distinct values and model has 120; dummy-encoding
#     either would add over a hundred columns, and they largely repeat
#     each other. brand has 32 - still many, but it is real information:
#     a Maruti and a BMW are different markets.
#     Unnamed: 0 is a leftover row number, not information at all.

df = df.drop(columns=['Unnamed: 0', 'car_name', 'model'])              # 3
for c in ['brand', 'seller_type', 'fuel_type', 'transmission_type']:
    df[c] = LabelEncoder().fit_transform(df[c])
X, y = df.drop('selling_price', axis=1), df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(X_train)
model = LinearRegression().fit(sc.transform(X_train), y_train)
pred = model.predict(sc.transform(X_test))
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"RMSE {rmse:,.0f}  =  {rmse / y.median():.0%} of the median price")
# The typical error is 90% of the typical car's price. R2 of 0.66 hid
# that entirely, because R2 has no units.

for n, c in sorted(zip(X.columns, model.coef_), key=lambda t: -abs(t[1])):  # 4
    print(f"{n:<20}{c:>14,.0f}")
# vehicle_age is NEGATIVE - older cars are worth less. km_driven is
# NEGATIVE - more mileage, lower price. Both match common sense, which
# is a good sign the model has learned something real.

y_log = np.log1p(y)                                                    # 5
a, b, c_, d_ = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"log-target R2 {r2_score(d_, LinearRegression().fit(a, c_).predict(b)):.4f}")
# CAVEAT 1: the two R2 values are measured on different targets - one on
#   prices, one on log-prices - so they are not directly comparable. To
#   compare fairly, convert back with np.expm1() and recompute RMSE.
# CAVEAT 2: better is not good. The remaining error is still large, and
#   a real model would need condition, service history and location -
#   features this dataset does not contain.
```
</details>

---

# 5. What the three use cases showed

| | Use case 1 — Salary | Use case 2 — Advertising | Use case 3 — Cars |
|---|---|---|---|
| Rows / features | 373 / 1 | 200 / 3 | **15,411 / 10** |
| Missing values | 2, dropped | None | None |
| Duplicates | **219, deliberately kept** | None | None |
| Encoding | Not needed | Not needed | **4 columns** |
| Scaling | Not needed | Not needed | **For readable coefficients only** |
| R² | 0.899 | 0.906 | **0.664** |
| RMSE in context | ₹15,551 on ~₹95,000 | 1.7 on ~14 | **₹503,024 on ~₹556,000** |
| Verdict | Good | Good | **Not good enough** |

**Four things worth carrying forward:**

1. **The code barely changed.** One feature or ten, 200 rows or 15,000 — `fit`, `predict`, and the same four metrics. **The workflow is the skill; the dataset decides the judgements.**

2. **Not every duplicate is an error.** Use case 1 kept 219 of them, because two people with the same experience and salary are two people.

3. **A metric with no units can hide a bad model.** R² of 0.66 sounded acceptable until RMSE was placed next to the median price.

4. **Sometimes the limit is the data.** Use case 3 improved a lot from a log transform and still was not good — because the columns that would explain a used car's price were never collected.

---

# ❓ Regression — 20 MCQs

**Answer from memory first, then check.**

### The basics

**Q1.** Regression predicts…
- (a) A category  (b) A number  (c) A cluster  (d) A probability only

**Q2.** In `X = df.drop('Salary', axis=1)` and `y = df['Salary']`, what is `y`?
- (a) The features  (b) The target — the thing being predicted  (c) The test set  (d) The model

**Q3.** Why is `stratify=y` not used in a regression split?
- (a) It is a bug  (b) A regression target has no classes to balance  (c) It is only for large data  (d) It is optional everywhere

**Q4.** `X_train.shape` is `(298, 1)` and `X_test.shape` is `(75, 1)`. The original data had…
- (a) 298 rows  (b) 373 rows  (c) 75 rows  (d) 1 row

### Evaluation metrics

**Q5.** Four predictions have errors −1, +1, −2, +2. Their **sum** is…
- (a) 6  (b) 0  (c) 1.5  (d) 10

**Q6.** What does that sum tell you about the model?
- (a) It is perfect  (b) Nothing — positive and negative errors cancelled out  (c) It is broken  (d) The errors are small

**Q7.** For those same errors, MAE is…
- (a) 0  (b) 1.5  (c) 2.5  (d) 6

**Q8.** For those same errors, MSE is…
- (a) 1.5  (b) 2.5  (c) 10  (d) 0

**Q9.** Why does MSE punish large errors more than MAE?
- (a) It divides by n−1  (b) It squares each error before averaging  (c) It ignores small errors  (d) It uses absolute values

**Q10.** MSE on the salary model was 241,834,884. Its units are…
- (a) Rupees  (b) Rupees **squared**  (c) None  (d) Percent

**Q11.** Which metric should you quote to a non-technical stakeholder?
- (a) MSE  (b) RMSE  (c) The slope  (d) The intercept

**Q12.** MAE is 12,094 and RMSE is 15,551. The gap tells you…
- (a) A calculation error  (b) A few predictions are noticeably worse than the rest  (c) All errors are equal  (d) Nothing

**Q13.** R² of 0.0 means the model is…
- (a) Perfect  (b) No better than always predicting the average  (c) Broken  (d) Overfitted

**Q14.** Can R² be negative?
- (a) No, never  (b) Yes — it means worse than predicting the average  (c) Only for classification  (d) Only with one feature

**Q15.** R² of 1.0 on a test set most likely means…
- (a) An excellent model  (b) Data leakage or a bug  (c) Too little data  (d) The wrong metric

### Reading and improving the model

**Q16.** `Salary = 6,822 × Experience + 31,521`. The 6,822 means…
- (a) The starting salary  (b) Each extra year of experience is worth about ₹6,822  (c) The error  (d) The R² score

**Q17.** TV has the highest correlation with Sales but radio has the highest coefficient. This is…
- (a) A contradiction  (b) Two different questions: how much variation is explained, versus what one more unit buys  (c) An error  (d) Impossible

**Q18.** Dropping `Newspaper` slightly **improved** R². Why can that happen?
- (a) It cannot  (b) A feature with almost no signal still carries noise the model fits  (c) Fewer features are always better  (d) The data was sorted

**Q19.** RMSE is 503,024 and the median car price is 556,000. This means…
- (a) A good model  (b) The typical error is about 90% of a typical price — the model is not usable  (c) R² must be negative  (d) The units are wrong

**Q20.** Scaling made no difference to the car model's R². Why?
- (a) A bug  (b) Linear regression does not need scaling to fit; scaling makes coefficients comparable  (c) The data was already scaled  (d) R² ignores scale

<details><summary>Answers</summary>

**A1 — (b) A number.** Classification predicts a category.

**A2 — (b) The target.** `X` holds the features; `y` holds the answers.

**A3 — (b).** `stratify` preserves class proportions, and a continuous target has no classes.

**A4 — (b) 373 rows.** 298 + 75 = 373, and checking that they add up is a quick way to confirm the split worked.

**A5 — (b) 0.** −1 + 1 − 2 + 2 = 0.

**A6 — (b) Nothing.** **Every prediction was wrong, and the total is zero.** This is exactly why no metric simply sums the errors.

**A7 — (b) 1.5.** (1 + 1 + 2 + 2) ÷ 4.

**A8 — (b) 2.5.** (1 + 1 + 4 + 4) ÷ 4.

**A9 — (b) Squaring.** An error of 10 becomes 100; an error of 1 stays 1.

**A10 — (b) Rupees squared.** Which is why MSE is for comparing models, never for describing one to a person.

**A11 — (b) RMSE.** It is in the target's own units and it does not hide occasional large misses.

**A12 — (b).** **RMSE is always at least MAE; how much larger tells you about the shape of your errors.** A 29% gap means a handful of predictions are noticeably worse.

**A13 — (b) No better than predicting the average.**

**A14 — (b) Yes.** A model can be actively worse than the lazy guess.

**A15 — (b) Leakage or a bug.** Real data has noise; perfection almost always means the answer got into the features.

**A16 — (b).** **A model you can say in one sentence is a model you can defend in a meeting.**

**A17 — (b).** Correlation describes the past; the coefficient advises the next rupee. **Both are true.**

**A18 — (b).** The model spends a little capacity fitting that noise. **Note the honest claim is "no worse and simpler", not "better" — the gain was 0.002.**

**A19 — (b) Not usable.** "Worth ₹556,000, give or take ₹503,000" helps nobody. **R² of 0.66 hid this completely.**

**A20 — (b).** Linear regression adjusts each coefficient to suit that column's units. **Scaling changes what you can read, not how well it fits.**
</details>

---

# 🎯 Regression — Tasks

## Warm-up

**Task 1 — The metric calculator.** For the errors −3, +5, −1, +7, compute MAE, MSE and RMSE **by hand**, then check with scikit-learn. **Explain why the sum of the errors is a useless summary.**

**Task 2 — Correlation first.** For any regression dataset, print the correlation of each feature with the target **before training**. Write down which feature you expect to matter most, then check whether the model agreed.

**Task 3 — Read the equation.** Train a single-feature model on the salary data and write the learned equation as one plain-English sentence. **Then say what the x-intercept means and why it is not useful.**

**Task 4 — The duplicate decision.** Count the duplicates in the salary data. **Write a paragraph arguing both that they should and should not be removed**, then state which side you take.

**Task 5 — RMSE in context.** For each of the three use cases, express RMSE as a percentage of the target's median. **Rank the three models by how usable they are, and explain why R² gives a different ranking.**

## Applying

**Task 6 — Feature by feature.** On the advertising data, train three separate single-feature models (TV only, radio only, newspaper only). **Report R² for each and compare with the three-feature model.**

**Task 7 — Coefficients need equal footing.** Take the car dataset and print the coefficients with and without scaling. **Explain why only one of the two sets can be compared across features.**

**Task 8 — Dummy variables instead.** In use case 3, replace Label Encoding of `brand` with `pd.get_dummies`. **Report the new R² and the new column count. Was the extra complexity worth it?**

**Task 9 — Predicted versus actual.** For any of the three use cases, plot predicted against actual with a diagonal reference line. **Describe the shape and say what it tells you that R² does not.**

**Task 10 — The residual plot.** Plot the errors (`y_test - y_pred`) against the predicted values. **A good model shows a shapeless cloud. Describe what you see and what any pattern would mean.**

## Whole projects

**Task 11 — A fourth dataset.** Find a regression dataset not used in this session and run all nine steps. **Report all four metrics, and state whether the model is good enough to use, with a reason.**

**Task 12 — The reusable function.** Write `run_regression(dataset_url, target)` that performs the whole pipeline and returns the four metrics plus the coefficients. **Run it unchanged on all three datasets from this part.**

**Task 13 — Fix the car model.** Try three improvements to use case 3: log-transform the target, add dummy variables for `brand`, and remove the most extreme prices. **Report RMSE in rupees for each — converting back from log where needed — and say which helped most.**

**Task 14 — The honest report.** For one dataset, write a one-page report: what you predicted, what data you used, all four metrics, the equation in plain English, and **three situations where the model should not be trusted.**

**Task 15 — When regression is the wrong tool.** Find a question that looks like regression but is not, and explain why. *(Hint: think about targets that are bounded, or counts that cannot be negative, or categories that happen to be numbered.)*

---

# ✅ Before you move on

- [ ] I can tell regression from classification by looking at the target column
- [ ] I know why `stratify` is not used in a regression split
- [ ] I can explain why the **sum** of the errors is a useless summary
- [ ] I know MAE averages absolute errors and MSE averages squared ones
- [ ] I know MSE's units are the target **squared**, and never quote it to a person
- [ ] **I report RMSE, because it is in the target's own units**
- [ ] I know R² compares the model against always guessing the average
- [ ] I know R² can be negative, and that 1.0 on a test set is a warning
- [ ] I compare RMSE against the target's median before calling a model good
- [ ] I can read a coefficient out loud as a plain-English sentence
- [ ] I know correlation and coefficient answer different questions
- [ ] I plot predicted-vs-actual and residuals, not just the metrics
- [ ] **I know that sometimes the limit is the data, not the model**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-05-regression.ipynb) | All three use cases, runnable |
| [Session 5B — Classification](session-05b-classification.md) | The other half of supervised learning |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
