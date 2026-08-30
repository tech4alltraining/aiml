# Session 5B — Classification & Deployment

**Classification · TP, TN, FP, FN · Evaluation metrics · Six algorithms · Pipelines · Saving models · Streamlit**

| | |
|---|---|
| **Notebook** | [session-05b-classification.ipynb](../notebooks/session-05b-classification.ipynb) |
| **Previous** | [Session 5 — Regression](session-05-regression.md) |
| **Next** | [Session 6 — Augmentation & Feature Engineering](session-06-augmentation-feature-engg-red.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Session 5 predicted numbers. This one predicts categories** — and that single change means every metric has to be rebuilt from scratch.
>
> By the end you will have trained six different classifiers, understood exactly what a confusion matrix is telling you, wrapped the whole thing in a pipeline, saved it to a file, and put it behind a web app.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Run the full classification workflow on a real dataset
2. **Explain TP, TN, FP and FN, and read any confusion matrix**
3. Derive accuracy, precision, recall and F1 from those four numbers
4. **Say which metric to use, and why, for a given problem**
5. Explain how six different algorithms decide, and when each suits
6. **Build a `Pipeline`, and say why it is safer than doing the steps by hand**
7. Save a trained model and load it in a separate program
8. Put a model behind a Streamlit app

---

## How this session is organised

| | Dataset | What it teaches |
|---|---|---|
| **Use case 1** | `loan_data_10k.csv` | The full workflow, **the metrics in depth**, all six algorithms, **and pipelines** |
| **Use case 2** | `diabetes_prediction_dataset.csv` | What changes when the classes are **imbalanced** |
| **Use case 3** | `iris.csv` | What changes when there are **more than two classes** |

| # | Topic |
|---|---|
| 1 | [What classification is](#1-what-classification-is) |
| 2 | [Use case 1 — Loan approval](#2-use-case-1--loan-approval) |
| 3 | [TP, TN, FP, FN — the four outcomes](#3-tp-tn-fp-fn--the-four-outcomes) |
| 4 | [The evaluation metrics](#4-the-evaluation-metrics) |
| 5 | [The six algorithms](#5-the-six-algorithms) |
| 6 | [Pipelines](#6-pipelines) |
| 7 | [Use case 2 — Imbalanced data](#7-use-case-2--imbalanced-data) |
| 8 | [Use case 3 — More than two classes](#8-use-case-3--more-than-two-classes) |
| 9 | [Saving, loading and inference](#9-saving-loading-and-inference) |
| 10 | [Streamlit app development](#10-streamlit-app-development) |
| | [❓ 20 MCQs](#-classification--20-mcqs) · [🎯 Tasks](#-classification--tasks) |

---

# 1. What classification is

**Classification predicts a category.** Will this loan be approved? Does this patient have diabetes? Which species is this flower?

🧠 **Analogy: sorting post into pigeonholes.** Every letter goes into exactly one slot. **You cannot put a letter 60% into one slot and 40% into another** — though the model can tell you how confident it was.

| | Regression (Session 5) | Classification (here) |
|---|---|---|
| The answer is | A **number** | A **category** |
| A bad prediction is | Off by ₹4,000 | **Simply wrong** |
| Metrics | RMSE, R² | **Accuracy, precision, recall, F1** |

> **This is why the metrics had to be rebuilt.** "Off by 4,000" is a meaningful statement about a number. **There is no such thing as being "off by 40%" when the true answer is *approved* and you said *declined*.** You were just wrong.

## Binary and multi-class

| | Classes | Example |
|---|---|---|
| **Binary** | Exactly 2 | Approved / declined — **use cases 1 and 2** |
| **Multi-class** | 3 or more | Three iris species — **use case 3** |

**Almost everything in this session is explained on the binary case first**, because the four outcomes are easiest to see with two classes. **Use case 3 shows what changes when there are more.**

---

# 2. Use case 1 — Loan approval

**The question:** given an application, predict whether the loan will be approved.

---

## Step 1 — Import the libraries and load the data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"

df = pd.read_csv(dataset_url)
print(df.shape)
df.head()
```

**Output:**

```text
(10000, 14)
```

**Ten thousand applications, fourteen columns.** Thirteen of them describe the applicant; the fourteenth, `loan_status`, is what we must predict.

---

## Step 2 — Exploratory Data Analysis

```python
df.head()
df.tail()
df.info()
df.describe(include='all')
```

**The same routine as always.** `include='all'` brings the text columns into `describe()` too.

### Look at the target first

```python
df['loan_status'].unique()
```

**Output:**

```text
array([1, 0])
```

**Two values, so this is binary classification.**

```python
df['loan_status'].value_counts()
```

**Output:**

```text
1    5000
0    5000
```

> **Exactly balanced — 5,000 of each.** **This matters enormously and you check it first**, because it decides whether accuracy is a fair metric. **On this dataset it is. Use case 2 shows a dataset where it is not.**

---

## Step 3 — Preprocessing

### Missing values

```python
df.isnull().sum()
```

**Output:**

```text
person_gender          1
loan_int_rate          1
loan_percent_income    1
```

**Three missing cells in 140,000 — about 0.002%.**

**Identify what kind of column each one is, because that decides the treatment:**

| Column | Kind |
|---|---|
| `person_gender` | **Categorical** |
| `loan_int_rate` | **Numerical** |
| `loan_percent_income` | **Numerical** |

**With three gaps out of ten thousand rows, dropping costs essentially nothing.**

```python
df.dropna(inplace=True)
print(df.shape)
```

**Output:**

```text
(9997, 14)
```

> **Compare with Session 3's `pre_data.csv`**, where dropping three rows cost 25% of the data and imputation was the right answer. **Same technique, opposite decision — and the deciding factor is how much data you have.**

### Unwanted features

**Before modelling, remove columns that cannot possibly help** — names, ID numbers, phone numbers, row indexes. **They are unique per row, so a model can only memorise them.**

```python
# This dataset has none. On a real file, check for them explicitly.
```

### Duplicates

```python
df.duplicated().sum()
```

**Output:**

```text
0
```

**None found.** **Run the check anyway** — it costs one line, and discovering later that your data was duplicated is far more expensive.

### Outliers

**Check every numeric column, not just the ones you suspect.**

```python
numerical_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                      'loan_int_rate', 'loan_percent_income',
                      'cb_person_cred_hist_length', 'credit_score']

for feature in numerical_features:
    sns.boxplot(df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()
```

![Box plots of all eight numeric features, several showing outlier dots](images/s5b-boxplots-before.png)

**Almost every column has dots beyond the whiskers.** `person_income` and `loan_amnt` in particular have long right tails — a few very large values stretching the axis.

**Remove them with the IQR rule, column by column:**

```python
for feature in numerical_features:
    q1, q3 = np.percentile(df[feature], [25, 75])
    iqr = q3 - q1
    minimum = q1 - 1.5 * iqr
    maximum = q3 + 1.5 * iqr
    df = df[(df[feature] >= minimum) & (df[feature] <= maximum)]
    print(f"{feature:<28} -> {len(df)} rows remain")
```

**Output:**

```text
person_age                   -> 9501 rows remain
person_income                -> 9041 rows remain
person_emp_exp               -> 8930 rows remain
loan_amnt                    -> 8808 rows remain
loan_int_rate                -> 8808 rows remain
loan_percent_income          -> 8755 rows remain
cb_person_cred_hist_length   -> 8578 rows remain
credit_score                 -> 8494 rows remain
```

### ⚠️ Count the cost of that loop

```python
print(f"9,997 rows -> {len(df)} rows")
print(f"lost {(9997 - len(df)) / 9997:.1%} of the data")
```

**Output:**

```text
9,997 rows -> 8494 rows
lost 15.0% of the data
```

> **Fifteen percent, removed by a formula.**
>
> **Each column removes only a few percent — but the losses stack.** Eight columns, each trimming its own tail, and 1,503 rows are gone.
>
> **Session 3 warned that the IQR rule flags tails, not just errors.** Here that warning has a number attached: **most of those 1,503 applicants were probably perfectly real people who simply earned more, or borrowed more, than the middle 50%.**

**Is that acceptable? It depends on what you are building.**

| If your model is for | Then |
|---|---|
| Typical applicants | Removing the extremes is defensible — you were not going to serve them anyway |
| **All applicants** | **You have just deleted your high-value customers** |

**We continue with the removal so the walkthrough matches the trainer's notebook — but one of the tasks at the end asks you to measure what keeping them would have done.**

### Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['person_gender'] = le.fit_transform(df['person_gender'])
df['person_education'] = le.fit_transform(df['person_education'])
df['person_home_ownership'] = le.fit_transform(df['person_home_ownership'])
df['loan_intent'] = le.fit_transform(df['loan_intent'])
df['previous_loan_defaults_on_file'] = le.fit_transform(df['previous_loan_defaults_on_file'])
```

**Five text columns, all converted to integers.**

> ⚠️ **Session 3's warning applies to three of these.** `person_home_ownership` and `loan_intent` are **unordered**, so Label Encoding invents an order that does not exist. `person_education` **is** ordered, but `LabelEncoder` sorts alphabetically and gets it backwards.
>
> **For the tree-based models below this barely matters** — they can split anywhere. **For logistic regression, kNN and SVM it does.** A task at the end asks you to measure the difference.

### Scaling

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

**Every numeric column now sits between 0 and 1.**

> **This matters far more for classification than it did for regression.** **kNN and SVM measure distance**, so an income column in the tens of thousands would drown out a credit score in the hundreds. **Without scaling, those two models would be crippled.**

---

## Step 4 — Train-test split

```python
from sklearn.model_selection import train_test_split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)
```

**Output:**

```text
Shape of X_train:  (6795, 13)
Shape of X_test:  (1699, 13)
Shape of y_train:  (6795,)
Shape of y_test:  (1699,)
```

**6,795 + 1,699 = 8,494.** Nothing lost.

---

## Step 5 — Train the first model

**Logistic regression is the standard first classifier: fast, and explainable.**

> **Despite the name, logistic regression is a *classifier*, not a regression model.** It computes a weighted sum like linear regression, then squashes the result into a probability between 0 and 1, and calls anything above 0.5 a positive. **The name is a historical accident that confuses everybody once.**

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()      # create the model
lr.fit(X_train, y_train)       # train it
```

```python
y_pred = lr.predict(X_test)    # predict on data it has never seen
y_pred[:10]
```

**Output:**

```text
[1 0 1 1 0 1 0 0 1 1]
```

**Categories, not numbers.** Compare that with Session 5, where `predict` returned salaries.

---

## Step 6 — The first metric

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
```

**Output:**

```text
0.8775750441436139
```

**87.8% of predictions were correct.**

> **And now stop.** **Accuracy is one number covering four very different kinds of outcome**, and until you can see those four, you do not actually know how this model behaves. **That is the next topic, and it is the most important one in this session.**

---

# 3. TP, TN, FP, FN — the four outcomes

**Every prediction in a binary classification lands in exactly one of four buckets.**

**Two things can be true or false: what actually happened, and what you said.** Combine them and you get four outcomes.

```text
                        WHAT YOU PREDICTED
                     Negative        Positive
                  +--------------+--------------+
    WHAT       Neg |      TN      |      FP      |
   ACTUALLY        | True Negative| False Positive|
   HAPPENED        +--------------+--------------+
               Pos |      FN      |      TP      |
                   |False Negative| True Positive |
                   +--------------+--------------+
```

## Reading the names

**The names are less confusing than they look, once you know the trick:**

> **The second word is what you predicted. The first word says whether you were right.**

| Term | You predicted | Were you right? | In plain words |
|---|---|---|---|
| **TP** — True Positive | Positive | **Yes** | You said yes, and it was yes |
| **TN** — True Negative | Negative | **Yes** | You said no, and it was no |
| **FP** — False Positive | Positive | **No** | You said yes, and it was no — **a false alarm** |
| **FN** — False Negative | Negative | **No** | You said no, and it was yes — **a miss** |

**TP and TN are the correct predictions. FP and FN are the two ways of being wrong** — and they are **not** the same kind of wrong.

🧠 **Analogy: a fire alarm.**

| | |
|---|---|
| **TP** | There is a fire, and the alarm sounds |
| **TN** | There is no fire, and the alarm stays silent |
| **FP** | **The alarm sounds and there is no fire** — annoying, everyone files outside |
| **FN** | **There is a fire and the alarm stays silent** — catastrophic |

**Both are errors. Only one kills people.** **Which of your two errors is worse is the single most important question in classification**, and it determines every metric choice you make.

## The confusion matrix

**A confusion matrix is just those four numbers laid out in a square.**

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
```

**Output:**

```text
[[691 157]
 [ 51 800]]
```

**Reading it:**

```text
                  Predicted 0    Predicted 1
   Actual 0            691            157
   Actual 1             51            800
```

| Cell | Value | Meaning |
|---|---|---|
| **TN** | **691** | Correctly predicted "declined" |
| **FP** | **157** | Predicted approved, actually declined — **157 false alarms** |
| **FN** | **51** | Predicted declined, actually approved — **51 missed** |
| **TP** | **800** | Correctly predicted "approved" |

**691 + 157 + 51 + 800 = 1,699** — every test row is in exactly one cell.

### Plotting it

```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

![Confusion matrix heatmap with the four cells labelled TN, FP, FN and TP](images/s5b-confusion-matrix.png)

> **The diagonal is what you got right.** A good model has large numbers on the diagonal and small ones off it. **Anything large off the diagonal tells you which mistake this model prefers to make** — and here it is clearly FP (157) over FN (51).

## ✏️ Practice — the four outcomes

1. A spam filter marks a genuine email as spam. Which of the four is that?
2. A medical test says a healthy patient is ill. Which one?
3. A medical test says an ill patient is healthy. Which one?
4. In the loan matrix above, how many applications were wrongly predicted as approved?
5. For the fire alarm, which error would you rather have — and why?

<details><summary>Solutions</summary>

```text
1  FALSE POSITIVE. The filter said "spam" (positive) and it was wrong.
   The email is lost in the spam folder.

2  FALSE POSITIVE. The test said "ill" (positive) and was wrong.
   A false alarm - the patient is worried and takes another test.

3  FALSE NEGATIVE. The test said "healthy" (negative) and was wrong.
   A MISS - the illness goes untreated. Usually the far worse error.

4  157 - the FP cell. Predicted approved (1), actually declined (0).

5  You would rather have a FALSE POSITIVE. Filing outside for a false
   alarm is an inconvenience; a silent alarm during a real fire is
   catastrophic.
   This is the whole reason we need more than one metric.
```
</details>

---

# 4. The evaluation metrics

**All four metrics are built from TP, TN, FP and FN.** Once you can read a confusion matrix, every metric is arithmetic.

**We will use the loan model's numbers throughout:**

```text
TN = 691     FP = 157
FN =  51     TP = 800
```

---

## Accuracy — what fraction did I get right?

```text
              TP + TN            800 + 691       1491
accuracy = ---------------  =  -------------  =  ------  =  0.8776
            TP+TN+FP+FN            1699           1699
```

**Of 1,699 predictions, 1,491 were correct.**

| | |
|---|---|
| **Answers** | *Overall, how often is this model right?* |
| **Use when** | The classes are **balanced** and both errors cost about the same |
| **⚠️ Fails when** | The classes are imbalanced — see use case 2 |

> **Accuracy is the metric everyone reaches for, and it is frequently the wrong one.** **It treats a false alarm and a miss as equally bad**, which is almost never true.

---

## Precision — when I say yes, how often am I right?

```text
                  TP              800          800
precision = ---------------  =  ---------  =  ------  =  0.8359
                TP + FP          800 + 157      957
```

**The model predicted "approved" 957 times. It was right 800 of those times.**

| | |
|---|---|
| **Answers** | *Of everything I flagged, how much was genuine?* |
| **Looks at** | The **predicted positive** column |
| **Punishes** | **False positives** — false alarms |
| **Use when** | **A false alarm is expensive** |

🧠 **Precision is about trust in your alarms.** A smoke detector with high precision rarely cries wolf.

**Use precision when acting on a positive costs something:**

- **Spam filtering** — a false positive puts a job offer in the spam folder
- **Recommending a product** — a false positive annoys the customer
- **Arresting someone** — a false positive is a serious injustice

---

## Recall — of all the real cases, how many did I catch?

```text
               TP             800          800
recall = ---------------  =  --------  =  ------  =  0.9401
             TP + FN          800 + 51      851
```

**There were 851 genuinely approved applications. The model found 800 of them.**

| | |
|---|---|
| **Answers** | *Of everything that was really there, how much did I find?* |
| **Looks at** | The **actual positive** row |
| **Punishes** | **False negatives** — misses |
| **Use when** | **A miss is expensive** |

🧠 **Recall is about not missing anything.** A fishing net with high recall catches almost every fish.

**Use recall when *failing* to act costs something:**

- **Disease screening** — a false negative means an illness goes untreated
- **Fraud detection** — a false negative is money gone
- **Safety inspection** — a false negative is an accident waiting

> **Precision and recall look at the same TP from two different directions.** **Precision divides by the column; recall divides by the row.** That is the entire difference, and it is worth pausing on.

---

## The trade-off, and why you cannot have both

**You can always make recall perfect: predict positive for everything.** You will catch every real case — and your precision will be terrible.

**You can always make precision high: only predict positive when absolutely certain.** You will rarely be wrong — and you will miss most real cases.

> **Pushing one up generally pushes the other down.** **There is no setting where both are perfect**, so you must decide which one your problem needs.

**Here is that trade-off in real numbers from this dataset:**

![Confusion matrices for Gaussian Naive Bayes and Decision Tree side by side](images/s5b-precision-recall-tradeoff.png)

| Model | Recall | Precision | What it is doing |
|---|---|---|---|
| **Gaussian Naive Bayes** | **1.0000** | 0.7362 | **Says "approved" to almost everyone.** Catches every single real approval — and 305 false alarms |
| **Decision Tree** | 0.8519 | **0.8372** | Far more cautious. Trustworthy when it says yes, but **misses 126** |

> **Look at Naive Bayes: recall of exactly 1.0000, and zero false negatives.** **It did not miss a single approval.** That sounds superb until you see it also wrongly approved 305 applications.
>
> **Same data. Same split. Opposite personalities.** **This is why you never report a single number.**

---

## F1 — one number balancing both

**When you genuinely need a single figure, F1 combines precision and recall.**

```text
            2 × precision × recall       2 × 0.8359 × 0.9401
F1  =  ------------------------------ = --------------------- = 0.8850
             precision + recall          0.8359 + 0.9401
```

> **F1 is the *harmonic* mean, not the ordinary average** — and that matters. **The harmonic mean is dragged down hard by the smaller of the two.**
>
> **Precision 1.0 and recall 0.0** would give an ordinary average of 0.5, which looks respectable. **F1 gives 0.0**, which is the honest answer — a model that finds nothing is useless however precise it is.

| | |
|---|---|
| **Use when** | You need one number, and both errors matter |
| **⚠️ Careful** | It weights precision and recall **equally**, and your problem may not |

---

## The classification report

**One call gives you all of it:**

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

**Output:**

```text
              precision    recall  f1-score   support

           0       0.93      0.81      0.87       848
           1       0.84      0.94      0.88       851

    accuracy                           0.88      1699
   macro avg       0.88      0.88      0.88      1699
weighted avg       0.88      0.88      0.88      1699
```

**Reading it properly:**

| Column | Means |
|---|---|
| **precision / recall / f1-score** | Computed **once per class** |
| **support** | How many test rows genuinely belong to that class |
| **macro avg** | The plain average across classes — treats both equally |
| **weighted avg** | Averaged by support — larger classes count more |

> **Notice each class gets its own row.** **Class 0 has precision 0.93 and recall 0.81; class 1 has the reverse pattern.** The model is better at *confirming* declines and better at *finding* approvals. **A single accuracy figure of 0.88 told you none of that.**

---

## Choosing your metric — the decision that matters

> **Ask one question: which of my two mistakes is worse?**

| If the worse error is | Optimise for | Because |
|---|---|---|
| **A miss** (false negative) | **Recall** | You cannot afford to let real cases through |
| **A false alarm** (false positive) | **Precision** | Acting wrongly is expensive |
| **Neither, particularly** | **F1**, or accuracy if balanced | Both errors cost about the same |

**Worked examples:**

| Problem | Worse error | Metric |
|---|---|---|
| Cancer screening | Missing a cancer | **Recall** |
| Spam filter | Losing a real email | **Precision** |
| Fraud detection | Missing fraud | **Recall** |
| Recommending a film | Neither | **F1** |
| **Loan approval** | **Depends on the bank's appetite for risk** | Discuss it first |

> **That last row is the honest one.** **A bank that fears bad debt wants precision; a bank chasing growth wants recall.** **The metric is a business decision, not a technical one** — and it should be settled before you train anything.

## ✏️ Practice — the metrics

1. From `TN=691, FP=157, FN=51, TP=800`, compute accuracy, precision, recall and F1 by hand. Check against scikit-learn.
2. A model has precision 1.0 and recall 0.0. Compute the ordinary average and F1. **Which is the honest summary?**
3. For a cancer screening test, which metric matters most, and why?
4. For an email spam filter, which metric matters most, and why?
5. Print the `classification_report` and explain what `support` and `macro avg` mean.

<details><summary>Solutions</summary>

```python
TN, FP, FN, TP = 691, 157, 51, 800

accuracy  = (TP + TN) / (TP + TN + FP + FN)                            # 1
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
print(f"accuracy  {accuracy:.4f}")     # 0.8776
print(f"precision {precision:.4f}")    # 0.8359
print(f"recall    {recall:.4f}")       # 0.9401
print(f"f1        {f1:.4f}")           # 0.8850

p, r = 1.0, 0.0                                                        # 2
print("ordinary average:", (p + r) / 2)          # 0.5 - looks respectable
print("F1              :", 0.0)                  # the honest answer
# F1 is the HARMONIC mean, so it is dragged down by the smaller value.
# A model that finds nothing is useless however precise it is.

# 3 - RECALL. A false negative means a cancer goes undetected, possibly
#     for years. A false positive means one more test. The two errors
#     are nowhere near equally bad.

# 4 - PRECISION. A false positive puts a genuine email - possibly a job
#     offer - in the spam folder where nobody looks. A false negative is
#     one piece of junk in the inbox, which is merely annoying.

# 5 - support   : how many test rows genuinely belong to that class
#     macro avg : the plain average across classes, treating both equally
#     weighted  : averaged by support, so larger classes count more
```
</details>

---

# 5. The six algorithms

**Every one of these uses the same three lines.** What changes is *how* they decide.

```python
# illustrative: a syntax reference, not runnable as written.
model = SomeClassifier()       # create
model.fit(X_train, y_train)    # train
y_pred = model.predict(X_test) # predict
```

---

## Logistic Regression

**How it decides:** computes a weighted sum of the features, squashes it into a probability between 0 and 1, and calls anything above 0.5 positive.

🧠 **Analogy: a points system for a visa application.** Each factor adds or subtracts points. Add them up, and if the total clears the threshold you are approved. **The weights are exactly what the model learns.**

| | |
|---|---|
| **Good for** | A fast, explainable baseline. **Always try this first** |
| **Needs scaling** | Yes |
| **Gives you** | A coefficient per feature — you can see what mattered |

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

**Output:** `0.8776`

---

## k-Nearest Neighbours (kNN)

**How it decides:** finds the *k* most similar rows in the training data and lets them vote.

🧠 **Analogy: asking your five nearest neighbours.** *"Did people like this get approved?"* If four of the five did, you predict approved. **Move house and the answer changes** — which is exactly why the scale of your columns matters so much.

| | |
|---|---|
| **Good for** | Simple problems; no real training step |
| **Needs scaling** | **Essential** — it measures distance |
| **Watch out** | Slow at prediction time on large data; **k must be chosen** |

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

**Output:** `0.8487`

> **`n_neighbors=5` is a choice, not a law.** **Small k** follows the data closely and is jumpy; **large k** smooths everything into mush. **Session 8 shows how to choose it with evidence rather than guessing.**

---

## Decision Tree

**How it decides:** asks a series of yes/no questions, like a flowchart, until it reaches an answer.

🧠 **Analogy: a doctor's triage flowchart.** *Is the credit score above 600? If yes, is the loan under 20% of income? If yes, approve.* **Each question splits the applicants into two groups, and the tree learns which questions to ask.**

| | |
|---|---|
| **Good for** | **Explaining a decision to a human** — you can print the flowchart |
| **Needs scaling** | **No** — it splits one column at a time |
| **Watch out** | **Overfits badly** if you let it grow without limit |

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

**Output:** `0.8428`

> **The lowest score of the six** — and that is characteristic. **A single unrestricted tree grows until it has memorised the training data**, which is exactly what Session 8 calls overfitting. **The fix is to grow many trees instead, which is the next-but-two algorithm.**

---

## Support Vector Machine (SVM)

**How it decides:** finds the boundary that leaves the widest possible gap between the two classes.

🧠 **Analogy: drawing the widest possible road between two neighbourhoods.** Not just *a* line — the line with the most clearance on both sides. **The houses right on the kerb are the "support vectors"**, and they alone determine where the road goes.

| | |
|---|---|
| **Good for** | Clean, well-separated classes; medium-sized data |
| **Needs scaling** | **Essential** — it measures distance |
| **Watch out** | **Slow on large datasets** |

```python
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

**Output:** `0.8805`

> **`kernel='linear'` draws a straight boundary.** Other kernels bend it, which handles classes that a straight line cannot separate. **Note this took noticeably longer to train than logistic regression for a very similar score.**

---

## Gaussian Naive Bayes

**How it decides:** uses probability, assuming every feature is independent of the others.

🧠 **Analogy: judging a book by several unrelated clues.** *The cover is blue, the title has three words, it is 300 pages.* **Naive Bayes treats each clue as if it tells you nothing about the others** — which is usually false, and it works surprisingly well anyway.

| | |
|---|---|
| **Good for** | **Very fast**; text classification; small datasets |
| **Needs scaling** | Not really |
| **Watch out** | **The independence assumption is almost always wrong** |

```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

**Output:** `0.8205`

> **The lowest accuracy of the six — and the highest recall, at exactly 1.0000.** **It missed nothing, because it says "approved" to almost everyone.** **If a miss were catastrophic in your problem, this "worst" model might be the right one.**

---

## Random Forest

**How it decides:** builds hundreds of decision trees, each on a different random slice of the data, and lets them vote.

🧠 **Analogy: a panel of 200 doctors.** Each sees a slightly different subset of the notes and gives an opinion. **One doctor's odd view gets outvoted; the panel's consensus is far more reliable than any individual.**

| | |
|---|---|
| **Good for** | **The default choice on tabular data.** Usually the best score |
| **Needs scaling** | **No** |
| **Gives you** | Feature importances — which columns mattered |
| **Watch out** | Slower; harder to explain than a single tree |

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

**Output:** `0.8929`

> **The best accuracy of the six, and it fixes exactly the weakness of the single Decision Tree.** **Each tree overfits its own random slice, and averaging cancels their individual mistakes.**

---

## All six compared

![Bar chart comparing accuracy, precision, recall and F1 for six classifiers](images/s5b-model-comparison.png)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8776 | 0.8359 | 0.9401 | 0.8850 |
| kNN (k=5) | 0.8487 | 0.8200 | 0.8942 | 0.8555 |
| Decision Tree | 0.8428 | 0.8372 | 0.8519 | 0.8445 |
| SVM (linear) | 0.8805 | 0.8333 | **0.9518** | 0.8886 |
| Gaussian NB | 0.8205 | **0.7362** | **1.0000** | 0.8480 |
| **Random Forest** | **0.8929** | **0.8737** | 0.9189 | **0.8958** |

**Three things worth noticing:**

1. **Random Forest wins on accuracy, precision and F1** — which is why it is the usual default for tabular data.
2. **Gaussian NB wins on recall with a perfect 1.0000** — and has the worst precision by a wide margin. **The "worst" model is the best model if a miss is what you fear.**
3. **The spread is only 7 percentage points**, from 0.82 to 0.89. **Choosing the algorithm matters less than beginners expect** — and far less than choosing the right metric.

> **Session 8 will show you that some of these gaps are inside the noise of a single split.** **Do not rank models on one run.**

## ✏️ Practice — the algorithms

1. Train all six models and print accuracy for each. Which wins?
2. Print the confusion matrix for Gaussian NB. **Explain its recall of 1.0000 from the matrix alone.**
3. Compare the Decision Tree's training accuracy with its test accuracy. What does the gap tell you?
4. Run kNN without scaling. How much accuracy is lost, and why?
5. Which two models would you not bother scaling, and why?

<details><summary>Solutions</summary>

```python
# Rebuild the use case 1 state so this block stands alone.
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"
NUMERICAL = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
             'loan_int_rate', 'loan_percent_income',
             'cb_person_cred_hist_length', 'credit_score']
df = pd.read_csv(dataset_url).dropna()
for f in NUMERICAL:
    q1, q3 = np.percentile(df[f], [25, 75]); iqr = q3 - q1
    df = df[(df[f] >= q1 - 1.5 * iqr) & (df[f] <= q3 + 1.5 * iqr)]
for c in ['person_gender', 'person_education', 'person_home_ownership',
          'loan_intent', 'previous_loan_defaults_on_file']:
    df[c] = LabelEncoder().fit_transform(df[c])
df[NUMERICAL] = MinMaxScaler().fit_transform(df[NUMERICAL])
X, y = df.drop('loan_status', axis=1), df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

models = [("LogisticRegression", LogisticRegression()),                # 1
          ("kNN", KNeighborsClassifier(n_neighbors=5)),
          ("DecisionTree", DecisionTreeClassifier(random_state=42)),
          ("SVM", SVC(kernel='linear')),
          ("GaussianNB", GaussianNB()),
          ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42))]
for name, m in models:
    m.fit(X_train, y_train)
    print(f"{name:<20}{accuracy_score(y_test, m.predict(X_test)):.4f}")
# Random Forest wins at 0.8929.

nb = GaussianNB().fit(X_train, y_train)                                # 2
print(confusion_matrix(y_test, nb.predict(X_test)))
# The bottom-left cell (FN) is ZERO - it missed nothing. But the
# top-right cell (FP) is large: it approves masses of bad applications
# too. Recall 1.0 with poor precision means "says yes to almost
# everyone".

dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)     # 3
print("train:", accuracy_score(y_train, dt.predict(X_train)))   # ~1.0
print("test :", accuracy_score(y_test, dt.predict(X_test)))     # ~0.84
# A large gap means it MEMORISED the training data. That is overfitting,
# and Session 8 covers it properly.

# 4 - kNN measures DISTANCE. Unscaled, person_income runs to tens of
#     thousands while credit_score runs to hundreds, so income dominates
#     every distance calculation and the other twelve columns barely
#     count. Accuracy drops noticeably.

# 5 - DECISION TREE and RANDOM FOREST. They split one column at a time,
#     so the relative scale of the columns is irrelevant to them.
```
</details>

---

# 6. Pipelines

**Everything so far was done step by step: scale, then split, then train.** That works, and it is fragile.

🧠 **Analogy: a recipe card versus a ready-meal.** Following steps by hand works — until you forget one, or do them in the wrong order, or hand the recipe to someone who skips a step. **A pipeline packs the steps into one object that always does them in the right order.**

## The problem a pipeline solves

**Look at what the manual version requires you to remember:**

```python
# illustrative: a syntax reference, not runnable as written.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform on train
X_test_scaled = scaler.transform(X_test)         # ONLY transform on test
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)            # remember to scale here too
```

**Four places to go wrong:**

| Mistake | Consequence |
|---|---|
| `fit_transform` on the test set | **Data leakage** — your score is inflated |
| Forgetting to scale `X_test` | Predictions are garbage, often silently |
| Scaling before splitting | **Leakage again** |
| Forgetting to scale new data at prediction time | **The number one deployment bug** |

> **All four are silent.** **None of them raises an error** — you just get a wrong answer that looks fine.

## The pipeline version

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', LogisticRegression()),
])

pipe.fit(X_train, y_train)          # scales AND trains, in the right order
y_pred = pipe.predict(X_test)       # scales AND predicts, automatically

print(accuracy_score(y_test, y_pred))
```

**Output:**

```text
0.877
```

**Identical to the manual version — which is the point.** **The pipeline is not more accurate; it is harder to get wrong.**

## What `fit` and `predict` do inside a pipeline

```text
pipe.fit(X_train, y_train)
    step 1  scaler.fit_transform(X_train)     <- fit ONLY on training data
    step 2  model.fit(scaled_X_train, y_train)

pipe.predict(X_test)
    step 1  scaler.transform(X_test)          <- transform ONLY, never fit
    step 2  model.predict(scaled_X_test)
```

> **The pipeline knows that `fit` means "learn and apply" while `predict` means "apply only".** **You cannot accidentally fit a scaler on your test data, because the pipeline never offers you the option.**
>
> **Structure beats discipline.** A rule you have to remember is a rule you will eventually forget.

## The complete pipelined version of use case 1

**The whole walkthrough, rewritten:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"

NUMERICAL = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
             'loan_int_rate', 'loan_percent_income',
             'cb_person_cred_hist_length', 'credit_score']
CATEGORICAL = ['person_gender', 'person_education', 'person_home_ownership',
               'loan_intent', 'previous_loan_defaults_on_file']

# ---------- 1. LOAD AND CLEAN (no statistics, so safe before the split)
df = pd.read_csv(dataset_url).dropna()

for feature in NUMERICAL:                       # outlier removal
    q1, q3 = np.percentile(df[feature], [25, 75])
    iqr = q3 - q1
    df = df[(df[feature] >= q1 - 1.5 * iqr) & (df[feature] <= q3 + 1.5 * iqr)]

for col in CATEGORICAL:                         # encoding
    df[col] = LabelEncoder().fit_transform(df[col])

# ---------- 2. SPLIT
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# ---------- 3. PIPELINE: scaling and the model, as one object
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', LogisticRegression()),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ---------- 4. EVALUATE
print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Output:**

```text
accuracy: 0.8770
```

> **Notice what moved and what did not.** **Cleaning and encoding stay outside the pipeline** — they use no statistics from the data, so running them first is safe. **Scaling goes inside**, because it computes a minimum and maximum, and those must come from the training data only.

## Swapping the model is now one line

```python
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', RandomForestClassifier(n_estimators=200)),   # <- the only change
])
pipe.fit(X_train, y_train)
```

**Trying all six algorithms becomes a loop:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = {
    'Logistic Regression': LogisticRegression(),
    'kNN':                 KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'SVM':                 SVC(kernel='linear'),
    'Gaussian NB':         GaussianNB(),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, clf in models.items():
    pipe = Pipeline([('scaler', MinMaxScaler()), ('model', clf)])
    pipe.fit(X_train, y_train)
    score = accuracy_score(y_test, pipe.predict(X_test))
    print(f"{name:<22}{score:.4f}")
```

**Six models, one loop, and no chance of scaling one of them wrongly.**

## Why this matters most at deployment

```python
# illustrative: a syntax reference, not runnable as written.
import joblib

joblib.dump(pipe, 'loan_model.joblib')     # saves the SCALER and the MODEL
```

> **Save the pipeline, not the bare model.** **The saved file now contains the scaler's learned minimum and maximum**, so when your app loads it six months later and feeds in raw values, they get scaled exactly as they were during training. **Save only the model and the app silently produces nonsense.** Topic 9 returns to this.

## ✏️ Practice — pipelines

1. Build the two-step pipeline and confirm it gives the same accuracy as the manual version.
2. List four mistakes the pipeline makes impossible.
3. Swap `LogisticRegression` for `RandomForestClassifier` by changing one line.
4. Write the loop that runs all six models through a pipeline and prints accuracy.
5. Explain why encoding stays outside the pipeline but scaling goes inside.

<details><summary>Solutions</summary>

```python
# Rebuild the use case 1 state so this block stands alone.
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv"
NUMERICAL = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
             'loan_int_rate', 'loan_percent_income',
             'cb_person_cred_hist_length', 'credit_score']
df = pd.read_csv(dataset_url).dropna()
for f in NUMERICAL:
    q1, q3 = np.percentile(df[f], [25, 75]); iqr = q3 - q1
    df = df[(df[f] >= q1 - 1.5 * iqr) & (df[f] <= q3 + 1.5 * iqr)]
for c in ['person_gender', 'person_education', 'person_home_ownership',
          'loan_intent', 'previous_loan_defaults_on_file']:
    df[c] = LabelEncoder().fit_transform(df[c])
df[NUMERICAL] = MinMaxScaler().fit_transform(df[NUMERICAL])
X, y = df.drop('loan_status', axis=1), df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pipe = Pipeline([('scaler', MinMaxScaler()),                           # 1
                 ('model', LogisticRegression())]).fit(X_train, y_train)
print("pipeline:", accuracy_score(y_test, pipe.predict(X_test)))

sc = MinMaxScaler().fit(X_train)
manual = LogisticRegression().fit(sc.transform(X_train), y_train)
print("manual  :", accuracy_score(y_test, manual.predict(sc.transform(X_test))))
# Identical. The pipeline is not more accurate - it is harder to get wrong.

# 2 - fit_transform on the test set (leakage); forgetting to scale X_test;
#     scaling before splitting (leakage); forgetting to scale new data at
#     prediction time. ALL FOUR ARE SILENT - none raises an error.

from sklearn.ensemble import RandomForestClassifier                    # 3
pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('model', RandomForestClassifier(n_estimators=200,
                                                  random_state=42))])
pipe.fit(X_train, y_train)
print(accuracy_score(y_test, pipe.predict(X_test)))

from sklearn.neighbors import KNeighborsClassifier                     # 4
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
models = {'LogReg': LogisticRegression(), 'kNN': KNeighborsClassifier(5),
          'Tree': DecisionTreeClassifier(random_state=42), 'SVM': SVC(kernel='linear'),
          'GNB': GaussianNB(),
          'Forest': RandomForestClassifier(n_estimators=200, random_state=42)}
for name, clf in models.items():
    p = Pipeline([('scaler', MinMaxScaler()), ('model', clf)]).fit(X_train, y_train)
    print(f"{name:<8}{accuracy_score(y_test, p.predict(X_test)):.4f}")

# 5 - Encoding uses a FIXED MAPPING (text -> integer) that does not depend
#     on the distribution of the data, so running it first is safe.
#     Scaling computes a MINIMUM AND MAXIMUM from the data. Computing
#     those from the test set too is leakage - so it must happen inside
#     the pipeline, where it only ever sees the training fold.
```
</details>

---

# 7. Use case 2 — Imbalanced data

**Use case 1 had exactly 5,000 of each class. Real data is rarely so obliging.**

**The question:** given routine health measurements, predict whether a patient likely has diabetes.

---

## Step 1 — Load, and check the balance immediately

```python
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/diabetes_prediction_dataset.csv"

df = pd.read_csv(dataset_url).dropna()
print(df.shape)
print(df['diabetes'].value_counts())
print(f"positive rate: {df['diabetes'].mean():.1%}")
```

**Output:**

```text
(100000, 9)

0    91500
1     8500

positive rate: 8.5%
```

> **Only 8.5% of patients have diabetes.** **This one number changes everything that follows.**

---

## Step 2 — The baseline that exposes the problem

**Before training anything, ask: what does a model that does nothing score?**

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

for col in df.select_dtypes('object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)

print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"recall  : {recall_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
```

**Output:**

```text
accuracy: 0.9150
recall  : 0.0000
[[18300     0]
 [ 1700     0]]
```

### ⚠️ Read that confusion matrix

```text
                 Predicted 0    Predicted 1
   Actual 0         18300              0
   Actual 1          1700              0     <- 1,700 diabetics, ZERO found
```

> **91.5% accuracy. Zero patients found.**
>
> **The model predicts "no diabetes" for every single person.** It never predicts positive at all, which is why the entire right-hand column is zero.
>
> **This is the most important demonstration in the session.** A number that looks like success on a dashboard, from a model that is completely useless.

**`stratify=y` is essential here.** Without it, a random split could leave the test set with a materially different positive rate, and every metric would be measured against the wrong baseline.

---

## Step 3 — A real model, and the metric that matters

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=2000)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"precision: {precision_score(y_test, y_pred):.4f}")
print(f"recall   : {recall_score(y_test, y_pred):.4f}")
print(f"f1       : {f1_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
```

**Output:**

```text
accuracy : 0.9603
precision: 0.8588
recall   : 0.6371
f1       : 0.7315
[[18122   178]
 [  617  1083]]
```

**96% accuracy — better than the useless baseline's 91.5%, but only by 4.5 points.**

> **Look at recall: 0.6371.** **The model found 1,083 of the 1,700 diabetic patients and missed 617 of them.**
>
> **For a screening tool, missing 36% of cases is the number that matters** — and accuracy of 96% hid it completely.

---

## Step 4 — Trading precision for recall

**If a miss is what you fear, tell the model that.** `class_weight='balanced'` makes mistakes on the rare class cost more.

```python
# class_weight tells the model that mistakes on the rare class cost more
pipe_balanced = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=2000, class_weight='balanced')),
])
pipe_balanced.fit(X_train, y_train)
y_pred_bal = pipe_balanced.predict(X_test)

print(f"accuracy : {accuracy_score(y_test, y_pred_bal):.4f}")
print(f"precision: {precision_score(y_test, y_pred_bal):.4f}")
print(f"recall   : {recall_score(y_test, y_pred_bal):.4f}")
print(confusion_matrix(y_test, y_pred_bal))
```

**Output:**

```text
accuracy : 0.8875
precision: 0.4230
recall   : 0.8882
[[16240  2060]
 [  190  1510]]
```

### The trade, in one table

| | Accuracy | Precision | Recall | Missed patients |
|---|---|---|---|---|
| Baseline (predict "no") | **0.9150** | 0.0000 | 0.0000 | **1,700** |
| Logistic Regression | **0.9603** | 0.8588 | 0.6371 | 617 |
| **Balanced weights** | 0.8875 | 0.4230 | **0.8882** | **190** |

> **The balanced model has *worse* accuracy — 0.8875 against 0.9603 — and it is almost certainly the one you want.**
>
> **It finds 1,510 of the 1,700 patients instead of 1,083.** **It misses 190 instead of 617.**
>
> **The price is precision falling to 0.42:** it now flags 2,060 healthy people who then need a confirmatory blood test.

**Is that a good trade? For a screening tool, clearly yes.** A blood test costs a little money and a little worry. **A missed diabetes diagnosis can cost years of untreated illness.**

> **This is what "choose the metric before you train" means in practice.** Rank these three models by accuracy and you pick the wrong one.

## ✏️ Practice — imbalanced data

1. Compute the positive rate. What accuracy does "always predict no" achieve?
2. Print that baseline's confusion matrix and explain why the right column is all zeros.
3. Train a logistic regression and compare its recall with its accuracy.
4. Add `class_weight='balanced'` and report how many more patients it finds.
5. **Which of the three models would you deploy for screening, and why?**

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix)

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/diabetes_prediction_dataset.csv"
df = pd.read_csv(dataset_url).dropna()
print(f"positive rate {df['diabetes'].mean():.1%}")                    # 1
print(f"'always no' would score {1 - df['diabetes'].mean():.1%}")

for c in df.select_dtypes('object').columns:
    df[c] = LabelEncoder().fit_transform(df[c])
X, y = df.drop('diabetes', axis=1), df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)

d = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)    # 2
print(confusion_matrix(y_test, d.predict(X_test)))
# The right column is entirely zero because the model NEVER predicts
# positive. It has no true positives and no false positives - it simply
# says "no" to everyone.

p1 = Pipeline([('s', StandardScaler()),                                # 3
               ('m', LogisticRegression(max_iter=2000))]).fit(X_train, y_train)
pred1 = p1.predict(X_test)
print(f"accuracy {accuracy_score(y_test, pred1):.4f}  "
      f"recall {recall_score(y_test, pred1):.4f}")
# 96% accuracy but only 64% recall - it misses over a third of patients.

p2 = Pipeline([('s', StandardScaler()),                                # 4
               ('m', LogisticRegression(max_iter=2000,
                                        class_weight='balanced'))]).fit(X_train, y_train)
pred2 = p2.predict(X_test)
print(confusion_matrix(y_test, pred2))
print(f"missed: {confusion_matrix(y_test, pred1)[1,0]} -> "
      f"{confusion_matrix(y_test, pred2)[1,0]}")
# 617 missed -> 190 missed. It finds 427 more patients.

# 5 - The BALANCED model, despite its lower accuracy. For screening, a
#     missed diagnosis is far worse than a false alarm: the false alarm
#     costs one blood test, the miss can cost years of untreated illness.
#     Ranking by accuracy would have chosen the wrong model.
```
</details>

---

# 8. Use case 3 — More than two classes

**Both previous use cases had two classes. Many real problems have more.**

**The question:** given four measurements of an iris flower, predict which of three species it is.

---

## The whole thing, in one block

**The code barely changes — which is the point.**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/iris.csv"

df = pd.read_csv(dataset_url)
print(df.shape)
print(df['species'].value_counts())

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=200)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**Output:**

```text
(150, 5)

setosa        50
versicolor    50
virginica     50

accuracy: 0.9333
```

**Three balanced classes, 50 each.** **Not one line of the modelling code had to change** — scikit-learn handles multi-class automatically.

---

## What *does* change: the confusion matrix

```python
print(confusion_matrix(y_test, y_pred))
```

**Output:**

```text
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
```

**It is now 3×3 instead of 2×2** — one row and one column per class, in alphabetical order: setosa, versicolor, virginica.

```text
                 Pred setosa   Pred versicolor   Pred virginica
   setosa             10              0                0
   versicolor          0              9                1
   virginica           0              1                9
```

> **Read the off-diagonal cells — they tell you exactly which classes the model confuses.**
>
> **Setosa: 10 out of 10, perfectly separated.** Nothing was ever mistaken for it, and it was never mistaken for anything.
>
> **Versicolor and virginica each got one wrong — and they were confused with *each other*.** The model never mixed either of them up with setosa.

**That is a real biological fact showing up in a matrix:** setosa is visibly distinct, while versicolor and virginica overlap. **A single accuracy of 0.9333 would never have told you which two species the model struggles to tell apart.**

> ⚠️ **TP, TN, FP and FN do not have a single meaning with three classes.** They are defined *per class*: for setosa, a "positive" is setosa and everything else is negative. **That is exactly what the classification report computes.**

---

## The per-class report

```python
print(classification_report(y_test, y_pred))
```

**Output:**

```text
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```

**Setosa is perfect. The other two sit at 0.90 on everything.**

| Row | Means |
|---|---|
| **macro avg** | The plain average across classes — **every class counts equally** |
| **weighted avg** | Averaged by support — **larger classes count more** |

> **With balanced classes these two are identical, as here.** **On imbalanced data they differ sharply**, and which one you quote is a real choice: macro treats a rare class as equally important, weighted lets the common class dominate.

## ✏️ Practice — multi-class

1. Load iris, check the class balance, and train the pipeline. What accuracy do you get?
2. Print the 3×3 confusion matrix. **Which two species does the model confuse?**
3. Print the classification report. Which class is easiest, and how can you tell?
4. Explain why TP/TN/FP/FN must be defined per class when there are three.
5. Explain the difference between `macro avg` and `weighted avg`, and when they differ.

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/iris.csv"
df = pd.read_csv(dataset_url)
print(df['species'].value_counts())                                    # 1
X, y = df.drop('species', axis=1), df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)
pipe = Pipeline([('s', StandardScaler()),
                 ('m', LogisticRegression(max_iter=200))]).fit(X_train, y_train)
pred = pipe.predict(X_test)
print(accuracy_score(y_test, pred))            # 0.9333

print(confusion_matrix(y_test, pred))                                  # 2
# VERSICOLOR and VIRGINICA - one mistaken for the other in each
# direction. Setosa is never confused with anything.

print(classification_report(y_test, pred))                             # 3
# SETOSA is easiest: precision, recall and f1 are all 1.00. It is
# visibly distinct from the other two, and the matrix shows it was
# never mixed up with either.

# 4 - "Positive" has no single meaning with three classes. The metrics
#     are computed ONE CLASS AT A TIME: for setosa, a positive is setosa
#     and the other two species are both negative. Repeat for each class.

# 5 - macro avg treats every class equally, whatever its size.
#     weighted avg weights by support, so common classes dominate.
#     With balanced classes they are identical, as here. On imbalanced
#     data they diverge sharply, and quoting one rather than the other
#     is a real choice.
```
</details>

---

---

# ❓ Classification — 20 MCQs

**Answer from memory first, then check.**

### The four outcomes

**Q1.** A spam filter marks a genuine email as spam. That is a…
- (a) True positive  (b) False positive  (c) False negative  (d) True negative

**Q2.** A medical test tells an ill patient they are healthy. That is a…
- (a) False positive  (b) False negative  (c) True negative  (d) True positive

**Q3.** In the name "false negative", what does the second word tell you?
- (a) Whether you were right  (b) **What you predicted**  (c) The class balance  (d) Nothing

**Q4.** In the confusion matrix `[[691, 157], [51, 800]]`, how many false positives are there?
- (a) 691  (b) 157  (c) 51  (d) 800

**Q5.** For a fire alarm, which error is worse?
- (a) False positive — the alarm sounds with no fire  (b) **False negative — silence during a real fire**  (c) They are equally bad  (d) Neither matters

### The metrics

**Q6.** Precision is…
- (a) TP / (TP + FN)  (b) TP / (TP + FP)  (c) (TP + TN) / all  (d) TP / all

**Q7.** Recall is…
- (a) TP / (TP + FN)  (b) TP / (TP + FP)  (c) (TP + TN) / all  (d) TN / (TN + FP)

**Q8.** Precision answers…
- (a) Of all the real cases, how many did I find?  (b) When I say yes, how often am I right?  (c) How many did I get right overall?  (d) How balanced are the classes?

**Q9.** For cancer screening you should optimise for…
- (a) Precision  (b) Recall  (c) Accuracy  (d) Training speed

**Q10.** For a spam filter you should optimise for…
- (a) Precision  (b) Recall  (c) Accuracy  (d) Speed

**Q11.** A model has precision 1.0 and recall 0.0. Its F1 is…
- (a) 0.5  (b) 1.0  (c) **0.0**  (d) Undefined

**Q12.** Why is F1 the harmonic mean rather than the ordinary average?
- (a) It is faster  (b) It is dragged down by the smaller of the two, which is more honest  (c) Tradition  (d) It is not

**Q13.** You can always make recall 1.0 by…
- (a) Training longer  (b) Predicting positive for everything  (c) Adding features  (d) Scaling

**Q14.** In a classification report, `support` means…
- (a) The model's confidence  (b) How many test rows genuinely belong to that class  (c) The F1 score  (d) The number of features

### The algorithms and the workflow

**Q15.** Which two algorithms absolutely require scaled features?
- (a) Decision Tree and Random Forest  (b) **kNN and SVM**  (c) Naive Bayes and Logistic Regression  (d) None

**Q16.** A single Decision Tree scored lowest of the six. The usual reason is…
- (a) Too little data  (b) It overfits — it memorises the training set  (c) It needs scaling  (d) A bug

**Q17.** Gaussian Naive Bayes achieved recall of exactly 1.0000 and the worst precision. This means it…
- (a) Is the best model  (b) **Says "approved" to almost everyone**  (c) Says "declined" to almost everyone  (d) Is broken

**Q18.** What does a `Pipeline` prevent?
- (a) Overfitting  (b) Silently scaling the test set wrongly, or forgetting to scale it at all  (c) Missing values  (d) Class imbalance

**Q19.** On the diabetes data, "always predict no" scored 91.5% accuracy. This shows…
- (a) An excellent model  (b) **Accuracy is meaningless when classes are imbalanced**  (c) The data is broken  (d) Recall is unnecessary

**Q20.** The iris confusion matrix showed setosa perfect and one error each between versicolor and virginica. This tells you…
- (a) Nothing useful  (b) **Which two classes the model confuses — setosa is separable, the other two overlap**  (c) The model is broken  (d) Accuracy is wrong

<details><summary>Answers</summary>

**A1 — (b) False positive.** It said "spam" (positive) and was wrong. **The email is lost.**

**A2 — (b) False negative.** It said "healthy" (negative) and was wrong. **A miss — usually the far worse error in medicine.**

**A3 — (b) What you predicted.** **The first word says whether you were right.** "False negative" = you predicted negative, and you were wrong.

**A4 — (b) 157.** Top-right: actual 0, predicted 1.

**A5 — (b) The false negative.** Filing outside for a false alarm is an inconvenience; **a silent alarm during a real fire is catastrophic.**

**A6 — (b) TP / (TP + FP).** It divides by the **predicted-positive column**.

**A7 — (a) TP / (TP + FN).** It divides by the **actual-positive row**. **That is the whole difference between the two.**

**A8 — (b).** Recall answers the other one.

**A9 — (b) Recall.** A missed cancer can cost years; a false alarm costs one more test.

**A10 — (a) Precision.** A false positive puts a job offer in the spam folder.

**A11 — (c) 0.0.** The ordinary average would say 0.5, which flatters a model that finds nothing.

**A12 — (b).** **A model that finds nothing is useless however precise it is**, and F1 says so.

**A13 — (b) Predict positive for everything.** Your precision will collapse — which is exactly the trade-off.

**A14 — (b).** It tells you how much weight that class's scores deserve.

**A15 — (b) kNN and SVM.** Both measure distance, so a large-range column drowns out the rest. **Trees do not care.**

**A16 — (b) Overfitting.** An unrestricted tree grows until it memorises. **The Random Forest fixes exactly this by averaging many trees.**

**A17 — (b).** Approving nearly everyone catches every real approval (recall 1.0) while being wrong often (precision 0.74). **If a miss were catastrophic, this "worst" model would be the right one.**

**A18 — (b).** All four of those mistakes are **silent** — none raises an error. **Structure beats discipline.**

**A19 — (b).** **It found zero of the 1,700 diabetic patients.** Check the class balance before choosing a metric.

**A20 — (b).** **A single accuracy figure of 0.9333 would never have told you which two species overlap.**
</details>

---

# 🎯 Classification — Tasks

## Warm-up

**Task 1 — Read a matrix.** Given `[[450, 50], [30, 470]]`, identify TN, FP, FN and TP, then compute accuracy, precision, recall and F1 **by hand**. Check with scikit-learn.

**Task 2 — Name the error.** For six situations of your own (a security scanner, a plagiarism checker, a job-application filter, and three more), say which of FP or FN is worse and why. **Then name the metric each implies.**

**Task 3 — The balance check.** For three classification datasets, print the class balance and state whether accuracy would be a fair metric. **For any that fail, say which metric you would use instead.**

**Task 4 — Baseline first.** For any classification dataset, compute the `DummyClassifier` score **before** training anything. **Report how much your real model beats it by.**

**Task 5 — The report, explained.** Print a `classification_report` and write one sentence per column explaining what it tells you.

## Applying

**Task 6 — The six-model bake-off.** Run all six algorithms through a pipeline on the loan data. Produce a table of accuracy, precision, recall and F1. **Recommend one, and name the metric that drove your choice.**

**Task 7 — Measure the outlier cost.** Run use case 1 twice — once with the outlier-removal loop and once without. **Report the row count and all four metrics for each. Was removing 15% of the data worth it?**

**Task 8 — Encoding properly.** Replace Label Encoding of `loan_intent` and `person_home_ownership` with `pd.get_dummies`. **Report the change in accuracy for logistic regression and for Random Forest separately, and explain why they differ.**

**Task 9 — Choosing k.** Run kNN with k = 1, 3, 5, 11, 25 and 51. Plot accuracy against k. **Where is the sweet spot, and what happens at each extreme?**

**Task 10 — The tree you can read.** Train a `DecisionTreeClassifier(max_depth=3)` and print it with `export_text`. **Hand the output to someone non-technical and see whether they can follow it.**

## Pipelines and imbalance

**Task 11 — Build a pipeline.** Write the two-step pipeline for use case 1 and confirm it matches the manual version exactly. **Then break the manual version deliberately** by fitting the scaler on all the data, and report the inflated score.

**Task 12 — Swap models in a loop.** Write one loop that runs all six models through a pipeline. **Confirm no model can be accidentally left unscaled.**

**Task 13 — Rescue the recall.** On the diabetes data, try three ways to improve recall: `class_weight='balanced'`, a Random Forest, and lowering the decision threshold using `predict_proba`. **Report recall and precision for each, and recommend one.**

**Task 14 — The threshold dial.** Using `predict_proba`, compute precision and recall at thresholds 0.2, 0.3, 0.5, 0.7 and 0.9. **Plot both against the threshold and mark where you would set it for a screening tool.**

**Task 15 — Multi-class confusion.** On the iris data, print the 3×3 confusion matrix as a heatmap. **Write two sentences on what the off-diagonal cells tell you that accuracy cannot.**

## Whole projects

**Task 16 — A fourth dataset.** Take a classification dataset not used here and run the whole workflow with a pipeline. **Report the class balance, your chosen metric with a justification, the confusion matrix, and all four scores.**

**Task 17 — The reusable function.** Write `classify(dataset_url, target)` that runs the pipeline and returns the four metrics plus the confusion matrix. **Run it unchanged on all three datasets from this session.**

**Task 18 — Save and reload.** Save your best pipeline with `joblib`, then in a **separate script** load it and predict on three new rows. **The second script must not import anything from the first.**

**Task 19 — Ship it.** Build a Streamlit app around your saved pipeline: one input per feature, a Predict button, the probability shown with `st.metric`, and an honest disclaimer. **State on screen what the model should not be used for.**

**Task 20 — The honest write-up.** For one model, write a one-page report: the class balance, the metric you chose and why, the confusion matrix in words, and **three situations where this model should not be trusted.**

---

# ✅ Before you move on

**The four outcomes and the metrics**

- [ ] I can name TP, TN, FP and FN, and read any confusion matrix
- [ ] I know the second word is what you predicted and the first says whether you were right
- [ ] I can compute accuracy, precision, recall and F1 from four numbers
- [ ] I know precision divides by the column and recall divides by the row
- [ ] **I can say which metric a problem needs, and defend it**
- [ ] I know F1 is the harmonic mean, and why that matters
- [ ] I read the per-class rows of a classification report, not just accuracy

**The algorithms**

- [ ] I can explain how each of the six decides
- [ ] I know kNN and SVM need scaling and trees do not
- [ ] I know why a single Decision Tree underperforms a Random Forest
- [ ] I know a model with perfect recall may simply be saying yes to everything

**Building it properly**

- [ ] **I use a `Pipeline`, and I can name four mistakes it prevents**
- [ ] I know cleaning and encoding go outside it and scaling goes inside
- [ ] I check the class balance before choosing a metric
- [ ] I compute a baseline before celebrating any score
- [ ] **I know 91.5% accuracy can mean finding nobody at all**
- [ ] I save the **pipeline**, not the bare model
- [ ] I have put a model behind a Streamlit app

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-05b-classification.ipynb) | Everything above, runnable |
| [Session 5 — Regression](session-05-regression.md) | The other half of supervised learning |
| [Streamlit tutorials](../tutorials/apps/streamlit-apps-collection.md) | More app examples |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
