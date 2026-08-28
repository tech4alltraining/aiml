# Session 8 — Model Evaluation & Improvement

**Holdout Validation · Cross-Validation · Bootstrapping · K-Fold & LOOCV · Overfitting & Underfitting · Hyperparameter Tuning: Grid Search, Random Search, Bayesian Optimization**

| | |
|---|---|
| **Notebook** | [session-08-evaluation-tuning.ipynb](../notebooks/session-08-evaluation-tuning.ipynb) |
| **Previous** | [Session 7 — Unsupervised Learning](session-07-unsupervised.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Until now you have trusted a single number from a single split.** This session shows you that number moves by more than a percentage point for no reason at all — and teaches you how to report a score you can actually defend.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Show that a single train/test split is unreliable, with measured evidence
2. Run K-Fold and Stratified K-Fold cross-validation, and report mean ± std
3. Explain when LOOCV is worth it and when it is a waste
4. Use bootstrapping to put a **confidence interval** on a score
5. Diagnose overfitting and underfitting from a train-versus-test curve
6. Run Grid Search, Random Search and Bayesian optimisation
7. Explain why Random Search usually beats Grid Search
8. **Recognise when tuning is not worth doing at all**

---

## The five topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Why one split is not enough](#1-holdout-validation-and-why-one-split-is-not-enough) | The seed alone moves your score by 1.4pp |
| 2 | [Cross-validation](#2-cross-validation-k-fold-stratified-and-loocv) | Report **mean ± std**, never a lone number |
| 3 | [Bootstrapping](#3-bootstrapping-and-confidence-intervals) | "0.89" is really "0.88 to 0.91" |
| 4 | [Overfitting & underfitting](#4-overfitting-and-underfitting) | Watch the **gap**, not the training score |
| 5 | [Hyperparameter tuning](#5-hyperparameter-tuning) | Random beats Grid. And often nothing beats both |

---

# 1. Holdout validation, and why one split is not enough

**Holdout** is what you have been doing since Session 3: one `train_test_split`, train on 80%, score on 20%.

🧠 **Analogy: judging a restaurant from one meal.** It could have been an off night, or the chef's best dish. **One visit is evidence; it is not a verdict.**

## 📘 Examples

**Example 1 — the same model, ten different seeds**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

scores = []
for seed in range(10):
    a, b, c, d = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    scores.append(RandomForestClassifier(n_estimators=100, random_state=42).fit(a, c).score(b, d))
```

Measured:

| | |
|---|---|
| Mean | 0.8945 |
| Std | 0.0038 |
| **Lowest** | **0.8880** |
| **Highest** | **0.9020** |
| **Spread** | **0.0140** |

**Nothing changed except the random seed, and the score moved by 1.4 percentage points.**

If you had run seed 0 you would report 0.89. Run another and you report 0.90. **Both are the same model on the same data.** Every time you see two models "compared" on a single split with a gap smaller than this, the comparison is meaningless.

**Example 2 — what this means for your reports**

```python
# ❌ Not defensible
print(f"Accuracy: {model.score(X_test, y_test):.4f}")

# ✅ Defensible
from sklearn.model_selection import cross_val_score
cv = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")
```

**Example 3 — when holdout is still the right choice**

Holdout is not wrong; it is *cheap*. Use it when:

- The dataset is large (millions of rows — the split variance shrinks)
- Training takes hours, and 5 folds means 5× the hours
- You are iterating quickly and only need a rough signal

> **Use holdout to explore. Use cross-validation to report.**

## ✏️ Practice

1. Run the ten-seed experiment. What spread do you get?
2. Repeat with `test_size=0.1`. Is the spread larger or smaller? Why?
3. Repeat with a `DecisionTreeClassifier`. Is it more or less stable than a forest?
4. Take a 500-row sample and repeat. What happens to the spread?
5. Two models score 0.891 and 0.894 on one split. What can you conclude?

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])
X, y = L.drop(columns=["loan_status"]), L["loan_status"]

def spread(model_fn, Xd=X, yd=y, test_size=.2, seeds=10):
    s = []
    for seed in range(seeds):
        a, b, c, d = train_test_split(Xd, yd, test_size=test_size,
                                      random_state=seed, stratify=yd)
        s.append(model_fn().fit(a, c).score(b, d))
    return np.mean(s), np.std(s), max(s) - min(s)

rf = lambda: RandomForestClassifier(n_estimators=100, random_state=42)
dt = lambda: DecisionTreeClassifier(random_state=42)

print("forest, 20%% test : mean %.4f std %.4f spread %.4f" % spread(rf))          # 1
print("forest, 10%% test : mean %.4f std %.4f spread %.4f" % spread(rf, test_size=.1))  # 2
# A SMALLER test set gives a LARGER spread -- fewer rows to average over.
print("tree,   20%% test : mean %.4f std %.4f spread %.4f" % spread(dt))          # 3
# The single tree is less stable: a forest already averages 100 trees.

small = L.sample(500, random_state=0)                                            # 4
print("forest, 500 rows : mean %.4f std %.4f spread %.4f" %
      spread(rf, small.drop(columns=["loan_status"]), small["loan_status"]))
# Much larger spread. Small data means unreliable estimates -- and this is
# exactly when people report a single number most confidently.

# 5 - NOTHING. A 0.003 gap is far inside the 0.014 the seed alone produces.
#     To compare them you need cross-validation on both, and even then you
#     compare mean +/- std, not the means alone.
```
</details>

## ❓ MCQs

**Q1.** Changing only `random_state` moved accuracy from 0.888 to 0.902. This shows…
- (a) The model is broken  (b) A single split is an unreliable estimate  (c) The data is corrupt  (d) You need more trees

**Q2.** Two models score 0.891 and 0.894 on one split. You can conclude…
- (a) The second is better  (b) Nothing — the gap is smaller than split-to-split noise  (c) The first is better  (d) Both are overfitted

**Q3.** A smaller test set produces…
- (a) A more reliable estimate  (b) A less reliable estimate with more variance  (c) The same variance  (d) No estimate

**Q4.** When is plain holdout a reasonable choice?
- (a) Never  (b) On very large data, or when training is slow and you need a rough signal  (c) Always  (d) Only for regression

**Q5.** What should a defensible report show?
- (a) The best score you saw  (b) Mean ± standard deviation across folds  (c) The training score  (d) A single test score

<details><summary>Answers</summary>

**A1 — (b).** Same model, same data, same everything except which rows landed where.

**A2 — (b).** **This is the most common mistake in student projects.** The gap is inside the noise.

**A3 — (b).** Fewer rows to average over means a noisier estimate.

**A4 — (b).** **Use holdout to explore, cross-validation to report.**

**A5 — (b).** A number without a spread is not a result.
</details>

## 🎯 Tasks

**Task 1 — Your own variance study.** Take a dataset and model of your choice and run 20 seeds. Report mean, std and spread, and **draw a histogram of the scores.** Then state the smallest difference between two models you would be willing to call real on this data.

**Task 2 — Re-audit your past work.** Go back to your Session 5 six-model comparison. **Were any of the gaps you reported smaller than the split noise?** Rewrite the conclusion honestly if so. This is the single most valuable exercise in the session.

---

# 2. Cross-validation: K-Fold, Stratified and LOOCV

**Do not split once. Split k times, so every row is tested exactly once.**

🧠 **Analogy: five judges instead of one.** Each judge sits out one round and scores the round they did not compete in. Everyone gets judged, everyone gets to judge, and you report the average. **One judge might be harsh; five together are hard to fool.**

```text
5-Fold Cross-Validation

fold 1  [TEST ][train][train][train][train]
fold 2  [train][TEST ][train][train][train]
fold 3  [train][train][TEST ][train][train]
fold 4  [train][train][train][TEST ][train]
fold 5  [train][train][train][train][TEST ]
                                              -> mean ± std of 5 scores
```

| Method | Folds | Use when |
|---|---|---|
| **K-Fold** | k (usually 5 or 10) | The default |
| **Stratified K-Fold** | k | **Classification — keeps class balance in every fold** |
| **LOOCV** | n (one per row) | Very small datasets only |

> `cross_val_score` uses **Stratified** K-Fold automatically for classifiers. You get the right behaviour by default.

## 📘 Examples

**Example 1 — 5-fold on the loan data**

```python
from sklearn.model_selection import cross_val_score

cv = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5)
print(cv.round(4))
print(f"{cv.mean():.4f} ± {cv.std():.4f}")
```

Measured: `[0.8915 0.8945 0.8804 0.9075 0.8999]` → **0.8948 ± 0.0090**

**Look at the individual folds: 0.8804 to 0.9075.** That range is the honest picture of how much your score depends on which rows you happened to test on.

**Example 2 — is 10-fold better than 5-fold?**

| | Mean | Std | Time |
|---|---|---|---|
| 5-fold | 0.8948 | 0.0090 | 2.7 s |
| 10-fold | 0.8946 | 0.0103 | 6.1 s |

**The same answer for 2.3× the time.** More folds means each model trains on more data, but you pay linearly for it and the estimate barely improves. **5-fold is the sensible default; 10-fold when data is scarce.**

**Example 3 — LOOCV, and its real cost**

LOOCV trains **n** models, each leaving out one row. On 200 rows:

| Method | Mean | Fits |
|---|---|---|
| LOOCV | 0.8500 | **200** |
| 5-fold | 0.7900 | **5** |

**LOOCV scores higher because each of its models trains on 199 rows, while each 5-fold model trains on only 160.** With 200 rows that difference is substantial — which is exactly why LOOCV exists.

But scale it up: LOOCV on the full 9,997-row dataset would mean **9,997 model fits**. At roughly half a second each, that is well over an hour for one number.

> **LOOCV is for when data is so scarce that you cannot spare 20% for a test set.** Below a few hundred rows, consider it. Above that, use 5-fold.

## ✏️ Practice

1. Run 5-fold on the loan data. Report mean ± std and the individual folds.
2. Compare 3-fold, 5-fold and 10-fold on mean, std and runtime.
3. Take a 200-row sample and compare LOOCV with 5-fold. Explain the gap.
4. Why does `cross_val_score` use *stratified* folds for classifiers?
5. Use `cross_val_score(..., scoring="f1")`. Does the ranking of two models change?

<details><summary>Solutions</summary>

```python
import time, numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])
X, y = L.drop(columns=["loan_status"]), L["loan_status"]
rf = lambda: RandomForestClassifier(n_estimators=100, random_state=42)

cv = cross_val_score(rf(), X, y, cv=5)                                 # 1
print(cv.round(4), f"-> {cv.mean():.4f} +/- {cv.std():.4f}")

for k in [3, 5, 10]:                                                   # 2
    t0 = time.time(); s = cross_val_score(rf(), X, y, cv=k)
    print(f"{k:>2}-fold  mean {s.mean():.4f}  std {s.std():.4f}  {time.time()-t0:.1f}s")
# More folds costs linearly and the estimate barely improves.

sub = L.sample(200, random_state=42)                                   # 3
Xs, ys = sub.drop(columns=["loan_status"]), sub["loan_status"]
dt = lambda: DecisionTreeClassifier(max_depth=4, random_state=42)
print("LOOCV :", round(cross_val_score(dt(), Xs, ys, cv=LeaveOneOut()).mean(), 4), "(200 fits)")
print("5-fold:", round(cross_val_score(dt(), Xs, ys, cv=5).mean(), 4), "(5 fits)")
# LOOCV is higher because each of its models trains on 199 rows,
# while each 5-fold model trains on only 160.

# 4 - Without stratification a fold could get very few (or none) of a rare
#     class, making that fold's score meaningless. Stratifying keeps the
#     class balance of the full dataset inside every fold.

for name, m in [("forest", rf()), ("tree", dt())]:                     # 5
    acc = cross_val_score(m, X, y, cv=5).mean()
    f1 = cross_val_score(m, X, y, cv=5, scoring="f1").mean()
    print(f"{name:<8} acc {acc:.4f}   f1 {f1:.4f}")
# Ranking usually holds here because the classes are balanced. On the
# IMBALANCED diabetes data from Session 6 it can flip -- which is why you
# choose the metric BEFORE you compare.
```
</details>

## ❓ MCQs

**Q1.** In 5-fold cross-validation, each row is used for testing…
- (a) Five times  (b) Exactly once  (c) Never  (d) It depends on the seed

**Q2.** Why does `cross_val_score` stratify by default for classifiers?
- (a) It is faster  (b) So every fold keeps the class balance of the full data  (c) It reduces overfitting  (d) It does not

**Q3.** 10-fold gave the same mean as 5-fold for 2.3× the time. This suggests…
- (a) 10-fold is broken  (b) 5-fold is a sensible default on data this size  (c) Always use 10-fold  (d) Always use 3-fold

**Q4.** LOOCV on 10,000 rows requires how many model fits?
- (a) 10  (b) 100  (c) 10,000  (d) 1

**Q5.** LOOCV scored higher than 5-fold on the same 200 rows because…
- (a) It is more accurate  (b) Each of its models trains on more data (199 vs 160 rows)  (c) It uses a different metric  (d) Random chance

**Q6.** When is LOOCV genuinely worth it?
- (a) Always  (b) When data is so scarce you cannot spare a test set  (c) On large datasets  (d) For deep learning

<details><summary>Answers</summary>

**A1 — (b) Exactly once.** That is the whole point: every row contributes to the estimate.

**A2 — (b).** Otherwise a fold might contain almost none of a rare class.

**A3 — (b).** More folds is not automatically better — you pay linearly for a marginal gain.

**A4 — (c) 10,000.** Over an hour for one number.

**A5 — (b).** With only 200 rows, 199-vs-160 training rows is a real difference.

**A6 — (b).** Below a few hundred rows. Above that, use 5-fold.
</details>

## 🎯 Tasks

**Task 1 — The honest comparison.** Take three models and compare them with 5-fold CV, reporting mean ± std for each. **Then state which differences you believe are real** — a difference smaller than the standard deviations overlapping is not a finding.

**Task 2 — The fold-count curve.** Plot CV mean and runtime against k for k = 2, 3, 5, 10, 20. **Mark the k you would use and justify it with both axes.**

**Task 3 — Small-data study.** Sample 100, 200, 500 and 1,000 rows. For each, compare LOOCV with 5-fold on score and on runtime. **At what dataset size does LOOCV stop being worth it?**

---

# 3. Bootstrapping and confidence intervals

**Bootstrapping answers a question none of the above do: how uncertain is my number?**

🧠 **Analogy: an exit poll.** You cannot ask every voter. So you sample, and sample again, and again — and the spread across your samples tells you the margin of error. **"52%, ±3 points" is a far more honest statement than "52%".**

The trick is **resampling with replacement**: draw a new test set the same size as the original, allowing duplicates. Score. Repeat 200 times. The middle 95% of those scores is your confidence interval.

```python
import numpy as np

boots = []
rng = np.random.default_rng(0)
for _ in range(200):
    idx = rng.integers(0, len(X_test), len(X_test))       # with replacement
    boots.append(model.score(X_test.iloc[idx], y_test.iloc[idx]))

lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"{np.mean(boots):.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
```

## 📘 Examples

**Example 1 — a real confidence interval**

Measured on the loan data:

| | |
|---|---|
| Point estimate | 0.8949 |
| **95% CI** | **[0.8805, 0.9075]** |
| Width | 0.0270 |

**Your "89.5% accurate" model is really "somewhere between 88.1% and 90.8%".**

Now go back to Session 5's six-model table, where Random Forest scored 0.8910 and SVM 0.8780. **That 1.3-point gap sits comfortably inside a 2.7-point confidence interval.** The forest is probably better — but "probably" is the honest word, and a single split could never have told you that.

**Example 2 — what the interval is telling you**

```python
import matplotlib.pyplot as plt
plt.hist(boots, bins=30)
plt.axvline(lo, color="red", ls="--")
plt.axvline(hi, color="red", ls="--")
```

The histogram is the distribution of scores you might have reported **if the world had handed you a slightly different test set.**

**Example 3 — the three techniques compared**

| Technique | Answers |
|---|---|
| **Holdout** | *What did this model score once?* |
| **Cross-validation** | *What does it score on average, and how much does that vary?* |
| **Bootstrapping** | *What is the range my true score plausibly lies in?* |

> They are not competitors. **Cross-validate to choose your model; bootstrap to report its uncertainty.**

## ✏️ Practice

1. Compute a 95% bootstrap CI for your model. How wide is it?
2. Try 50, 200 and 1,000 resamples. Does the interval stabilise?
3. Compute a 99% interval. Is it wider or narrower? Why?
4. Bootstrap a 500-row test set instead. What happens to the width?
5. Two models differ by 0.008 and both CIs are 0.027 wide. What do you report?

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])
X, y = L.drop(columns=["loan_status"]), L["loan_status"]
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
m = RandomForestClassifier(n_estimators=100, random_state=42).fit(a, c)

def boot(Xt, yt, n=200, pct=(2.5, 97.5), seed=0):
    rng = np.random.default_rng(seed)
    s = [m.score(Xt.iloc[i], yt.iloc[i])
         for i in (rng.integers(0, len(Xt), len(Xt)) for _ in range(n))]
    lo, hi = np.percentile(s, pct)
    return np.mean(s), lo, hi

for n in [50, 200, 1000]:                                              # 1, 2
    mu, lo, hi = boot(b, d, n=n)
    print(f"n={n:<5} {mu:.4f}  CI [{lo:.4f}, {hi:.4f}]  width {hi-lo:.4f}")
# The interval settles down by about 200 resamples.

mu, lo, hi = boot(b, d, pct=(0.5, 99.5))                               # 3
print(f"99% CI [{lo:.4f}, {hi:.4f}] width {hi-lo:.4f}")
# WIDER. More confidence requires a bigger net.

small_b, small_d = b.iloc[:500], d.iloc[:500]                          # 4
mu, lo, hi = boot(small_b, small_d)
print(f"500-row test CI width {hi-lo:.4f}")
# Much wider -- a smaller test set means a less certain estimate.

# 5 - Report BOTH with their intervals and say the difference is not
#     established. A 0.008 gap inside 0.027-wide intervals is not evidence.
#     If the decision matters, get more test data or use paired CV folds.
```
</details>

## ❓ MCQs

**Q1.** Bootstrapping resamples…
- (a) Without replacement  (b) With replacement, so rows can repeat  (c) Only the training set  (d) The features

**Q2.** A 95% CI of [0.8805, 0.9075] means…
- (a) The model is 95% accurate  (b) The true score plausibly lies in that range  (c) 95% of predictions are right  (d) An error

**Q3.** A 99% interval compared with a 95% interval is…
- (a) Narrower  (b) Wider  (c) The same  (d) Undefined

**Q4.** Two models differ by 0.008 with overlapping 0.027-wide intervals. You should report…
- (a) The better model wins  (b) That the difference is not established  (c) Only the higher score  (d) Neither model

**Q5.** Which technique tells you the *uncertainty* of a score?
- (a) Holdout  (b) Grid search  (c) Bootstrapping  (d) Stratification

**Q6.** A smaller test set gives a bootstrap interval that is…
- (a) Narrower  (b) Wider  (c) Unchanged  (d) Zero

<details><summary>Answers</summary>

**A1 — (b) With replacement.** That is what simulates "a slightly different test set".

**A2 — (b).** It is a statement about the *estimate*, not about individual predictions.

**A3 — (b) Wider.** More confidence needs a bigger net.

**A4 — (b).** **This is intellectual honesty, and it is what separates a real report from a student project.**

**A5 — (c) Bootstrapping.**

**A6 — (b).** Fewer rows means more uncertainty.
</details>

## 🎯 Tasks

**Task 1 — Put an interval on your model.** Compute a 95% bootstrap CI and **draw the histogram with the interval marked.** Write the one-sentence version you would say in a presentation.

**Task 2 — Revisit Session 5 honestly.** Bootstrap the top three models from your Session 5 comparison. **Do their intervals overlap?** Rewrite your recommendation in light of the answer.

---

# 4. Overfitting and underfitting

🧠 **Analogy: two students preparing for an exam.**

- **The memoriser** learns every past paper by heart, including the typos. Perfect on those papers, lost on a new question. **That is overfitting.**
- **The skimmer** reads the chapter titles only. Bad on the past papers *and* the new ones. **That is underfitting.**
- **The one you want** understood the method: good on both.

| | Training score | Test score | Gap |
|---|---|---|---|
| **Underfitting** | Low | Low | Small |
| **Good fit** | High | High | Small |
| **Overfitting** | **Very high** | **Lower** | **Large** |

> **Watch the gap, not the training score.** A training accuracy of 1.0000 is not an achievement — it is a warning.

## 📘 Examples

**Example 1 — a decision tree overfitting in real time**

Measured on the loan data:

| max_depth | Train | Test | Gap |
|---|---|---|---|
| 1 | 0.8286 | 0.8240 | +0.005 |
| 3 | 0.8327 | 0.8070 | +0.026 |
| 5 | 0.8803 | 0.8515 | +0.029 |
| **8** | 0.9007 | **0.8665** ← best | +0.034 |
| 12 | 0.9416 | 0.8600 | +0.082 |
| 20 | 0.9961 | 0.8480 | +0.148 |
| **None** | **1.0000** | **0.8435** | **+0.157** |

**Read the last row.** The tree gets **every single training row right** — and is the *worst* model in the table on unseen data. It memorised.

**The test score peaks at depth 8 and falls after.** That peak is the model you ship. Everything to the left is underfitting; everything to the right is overfitting.

**Example 2 — the fixes**

| Problem | Fixes |
|---|---|
| **Underfitting** | A more capable model, more features, less regularisation, train longer |
| **Overfitting** | **More data** (best), simpler model, regularisation, `max_depth`, early stopping, ensembles |

**Example 3 — how a Random Forest fights overfitting**

```python
DecisionTreeClassifier()                          # train 1.0000, test 0.8435
RandomForestClassifier(n_estimators=200)          # train ~1.0000, test ~0.8910
```

**Both memorise the training set. Only one generalises.** Each tree in the forest overfits a *different* random slice, and averaging cancels their individual mistakes — the panel-of-doctors analogy from Session 5, now with numbers attached.

## ✏️ Practice

1. Build the depth table above. Where does test accuracy peak?
2. Plot train and test accuracy against depth on one chart. Where do the lines diverge?
3. What is the gap at unlimited depth? What does a training score of 1.0 tell you?
4. Compare an unlimited-depth single tree with a 200-tree forest. Explain the difference.
5. Take 500 rows and rebuild the curve. Does overfitting start earlier or later?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])
X, y = L.drop(columns=["loan_status"]), L["loan_status"]
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

for dep in [1, 2, 3, 5, 8, 12, 20, None]:                              # 1, 2, 3
    t = DecisionTreeClassifier(max_depth=dep, random_state=42).fit(a, c)
    tr, te = t.score(a, c), t.score(b, d)
    print(f"depth {str(dep):<5} train {tr:.4f}  test {te:.4f}  gap {tr-te:+.4f}")
# Test peaks at depth 8. The lines diverge sharply after depth 8.
# At unlimited depth train = 1.0000 -- pure memorisation, and the WORST
# test score in the table.

tree = DecisionTreeClassifier(random_state=42).fit(a, c)               # 4
forest = RandomForestClassifier(n_estimators=200, random_state=42).fit(a, c)
print(f"\\nsingle tree  train {tree.score(a,c):.4f}  test {tree.score(b,d):.4f}")
print(f"200 trees    train {forest.score(a,c):.4f}  test {forest.score(b,d):.4f}")
# Both memorise. Only the forest generalises: each tree overfits a
# DIFFERENT random slice, and averaging cancels their mistakes.

small = L.sample(500, random_state=0)                                  # 5
Xs, ys = small.drop(columns=["loan_status"]), small["loan_status"]
a2, b2, c2, d2 = train_test_split(Xs, ys, test_size=.2, random_state=42, stratify=ys)
print()
for dep in [1, 3, 5, 8, 12, None]:
    t = DecisionTreeClassifier(max_depth=dep, random_state=42).fit(a2, c2)
    print(f"500 rows, depth {str(dep):<5} train {t.score(a2,c2):.4f}  test {t.score(b2,d2):.4f}")
# Overfitting starts EARLIER with less data: there is less to learn before
# the model runs out of real pattern and starts memorising noise.
```
</details>

## ❓ MCQs

**Q1.** Training accuracy 1.0000 and test accuracy 0.8435 indicates…
- (a) An excellent model  (b) Overfitting  (c) Underfitting  (d) A bug

**Q2.** Both training and test accuracy are low. This is…
- (a) Overfitting  (b) Underfitting  (c) A good fit  (d) Data leakage

**Q3.** Which number should you watch most closely?
- (a) The training score  (b) The gap between training and test  (c) The number of features  (d) Training time

**Q4.** The best fix for overfitting, when available, is…
- (a) A more complex model  (b) More training data  (c) More features  (d) Training longer

**Q5.** Test accuracy peaks at depth 8 and falls after. You should ship…
- (a) Unlimited depth  (b) Depth 8  (c) Depth 1  (d) Depth 20

**Q6.** With only 500 rows instead of 10,000, overfitting begins…
- (a) Later  (b) Earlier  (c) At the same depth  (d) Never

**Q7.** A Random Forest also reaches ~1.0 training accuracy, yet generalises far better. Why?
- (a) It is simpler  (b) Each tree overfits a different slice, and averaging cancels the mistakes  (c) It uses fewer features  (d) It does not really overfit

<details><summary>Answers</summary>

**A1 — (b) Overfitting.** It memorised, including the noise.

**A2 — (b) Underfitting.** The model is too simple to capture the pattern.

**A3 — (b) The gap.** A high training score on its own tells you nothing good.

**A4 — (b) More data.** Every other fix is a compromise; more data is a genuine solution.

**A5 — (b) Depth 8.** The peak of the *test* curve.

**A6 — (b) Earlier.** Less real pattern to learn before it starts memorising noise.

**A7 — (b).** The panel-of-doctors analogy from Session 5, now measured.
</details>

## 🎯 Tasks

**Task 1 — The overfitting curve.** Plot training and test accuracy against model complexity for two different model families. **Mark the peak on each and state the depth you would ship.**

**Task 2 — The learning curve.** Plot test accuracy against *training set size* (100, 500, 1,000, 5,000, all). **If the curve is still rising at the right-hand edge, more data would still help you** — say so explicitly, because it changes what you should do next.

**Task 3 — Diagnose three models.** Deliberately build one underfitted, one well-fitted and one overfitted model on the same data. **Present all three with train, test and gap, and write the one-line diagnosis for each.**

---

# 5. Hyperparameter Tuning

**Parameters are learned from data. Hyperparameters are set by you before training.**

| | Examples |
|---|---|
| **Parameters** (learned) | Regression coefficients, tree split points |
| **Hyperparameters** (yours) | `n_estimators`, `max_depth`, `k`, `C`, learning rate |

🧠 **Analogy: tuning a radio.** The station is fixed; the dial is yours. Grid search turns the dial through every marked position. Random search jumps around. Bayesian optimisation **listens to where the signal was getting stronger and tries there next.**

## The three strategies

| Strategy | How it searches | Cost |
|---|---|---|
| **Grid Search** | Every combination in your grid | Explodes: 3 × 4 × 3 = 36 combos × 3 folds = 108 fits |
| **Random Search** | n random combinations | **You choose the budget** |
| **Bayesian** | Builds a model of the score surface, tries where it is most promising | Fewest evaluations |

## 📘 Examples

**Example 1 — Grid versus Random, measured**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=3, n_jobs=-1).fit(a, c)

rs = RandomizedSearchCV(RandomForestClassifier(random_state=42),
        {"n_estimators": randint(50, 250), "max_depth": [5, 10, 20, None],
         "min_samples_split": randint(2, 15)},
        n_iter=12, cv=3, random_state=42, n_jobs=-1).fit(a, c)
```

Measured:

| | Fits | Time | Best CV | Test |
|---|---|---|---|---|
| Grid Search | 108 | 12.2 s | 0.8936 | 0.8860 |
| **Random Search** | **36** | **4.9 s** | **0.8937** | **0.8895** |

**Random Search used one third of the fits, ran 2.5× faster, and came out slightly ahead on both scores.**

**Why does this happen?** Grid search wastes most of its budget on hyperparameters that barely matter. If `max_depth` matters a lot and `min_samples_split` barely at all, a grid still dutifully tries every `min_samples_split` value at every depth. Random search samples *different* depths every single time. **With the same budget, it explores the dimension that matters far more finely.**

**Example 2 — the finding nobody puts in a tutorial**

Compare the tuned results with the plain untuned default from Topic 2:

| | Score |
|---|---|
| **Untuned default, 5-fold CV** | **0.8948** |
| Grid Search best, 3-fold CV | 0.8936 |
| Random Search best, 3-fold CV | 0.8937 |

**Tuning did not help.** All three are inside the ±0.009 standard deviation you measured in Topic 2, and inside the 0.027 bootstrap interval from Topic 3.

> **Sometimes the honest answer is "the defaults were fine".** scikit-learn's defaults are well chosen. Tune when you have a reason — a specific hyperparameter you suspect matters, or a metric you need to push. **Do not tune out of habit, and never report a tuning gain smaller than your noise.**

**Example 3 — Bayesian optimisation, built from scratch**

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Fit a model of "score as a function of depth" to what we have tried so far
gp = GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=5.0, nu=2.5),
                              alpha=1e-4, normalize_y=True).fit(tried, scores)

mu, sd = gp.predict(all_depths, return_std=True)
next_depth = all_depths[np.argmax(mu + 1.5 * sd)]     # promising OR unexplored
```

That `mu + 1.5 * sd` is the whole idea. It scores each untried value by **how good we predict it is (`mu`) plus how uncertain we are about it (`sd`)** — so it tries things that look good *and* things nobody has checked. That balance has a name: **exploitation versus exploration.**

Measured, searching `max_depth` from 2 to 25:

| Method | Fits | Best depth found | CV score |
|---|---|---|---|
| Exhaustive | **24** | 8 | 0.8877 |
| **Bayesian** | **10** | **8** | **0.8877** |

**The same optimum, for under half the work.** With a 12-hour training run, that difference is days.

> **In production:** `pip install optuna` and let it handle the search. But the idea is exactly what you just read — model the surface, then try where the model is most hopeful.

## ✏️ Practice

1. Run Grid Search on the loan data. How many fits, and how long?
2. Run Random Search with `n_iter=12`. Compare fits, time and best score.
3. Compare both against the **untuned default**. Did tuning actually help?
4. Explain why Random Search often beats Grid Search at equal budget.
5. Explain `mu + 1.5 * sd` in one sentence.

<details><summary>Solutions</summary>

```python
import time, pandas as pd
from scipy.stats import randint
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])
X, y = L.drop(columns=["loan_status"]), L["loan_status"]
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

t0 = time.time()                                                       # 1
gs = GridSearchCV(RandomForestClassifier(random_state=42),
                  {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20, None],
                   "min_samples_split": [2, 5, 10]}, cv=3, n_jobs=-1).fit(a, c)
print(f"Grid   {len(gs.cv_results_['params'])*3:>4} fits {time.time()-t0:>6.1f}s "
      f"cv {gs.best_score_:.4f} test {gs.score(b,d):.4f}")

t0 = time.time()                                                       # 2
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42),
        {"n_estimators": randint(50, 250), "max_depth": [5, 10, 20, None],
         "min_samples_split": randint(2, 15)},
        n_iter=12, cv=3, random_state=42, n_jobs=-1).fit(a, c)
print(f"Random {12*3:>4} fits {time.time()-t0:>6.1f}s "
      f"cv {rs.best_score_:.4f} test {rs.score(b,d):.4f}")

untuned = cross_val_score(RandomForestClassifier(n_estimators=100,             # 3
                          random_state=42), X, y, cv=5)
print(f"\\nuntuned default 5-fold: {untuned.mean():.4f} +/- {untuned.std():.4f}")
# Tuning did NOT help: every result is inside the untuned std.
# Sometimes the honest answer is "the defaults were fine".

# 4 - Grid search wastes most of its budget on hyperparameters that barely
#     matter: it dutifully tries every min_samples_split at every depth.
#     Random search draws a DIFFERENT depth every single time, so at equal
#     budget it explores the dimension that matters far more finely.

# 5 - Try the value with the best PREDICTED score plus a bonus for being
#     UNCERTAIN -- balancing exploiting what looks good against exploring
#     what nobody has checked.
```
</details>

## ❓ MCQs

**Q1.** Which is a hyperparameter?
- (a) A regression coefficient  (b) `max_depth`  (c) A tree's split point  (d) The prediction

**Q2.** A grid of 3 × 4 × 3 with `cv=3` runs how many fits?
- (a) 10  (b) 36  (c) 108  (d) 12

**Q3.** Why does Random Search often beat Grid Search at equal budget?
- (a) It is luckier  (b) It samples a different value of each hyperparameter every time, exploring the ones that matter more finely  (c) It uses fewer folds  (d) It skips bad models

**Q4.** Tuning produced 0.8937 against an untuned 0.8948 ± 0.0090. You should conclude…
- (a) Tuning hurt the model  (b) Tuning made no measurable difference here  (c) Tuning helped  (d) The grid was wrong

**Q5.** In Bayesian optimisation, `mu + 1.5 * sd` balances…
- (a) Speed and accuracy  (b) Exploitation and exploration  (c) Train and test  (d) Precision and recall

**Q6.** Bayesian optimisation found the same optimum as exhaustive search in 10 fits instead of 24. Its advantage is greatest when…
- (a) Data is small  (b) Each evaluation is expensive  (c) There is one hyperparameter  (d) Never

**Q7.** Your search's best score beats the default by 0.002, with CV std 0.009. Report it as…
- (a) A 0.2% improvement  (b) No measurable improvement  (c) The new best model  (d) A tuning success

<details><summary>Answers</summary>

**A1 — (b) `max_depth`.** You set it; the model does not learn it.

**A2 — (c) 108.** 36 combinations × 3 folds. Grids explode fast.

**A3 — (b).** Grid search re-tries the same depth values over and over; random search does not.

**A4 — (b).** The difference is well inside the noise. **Do not report noise as a gain.**

**A5 — (b).** Try what looks good, and what nobody has checked yet.

**A6 — (b).** With a 12-hour training run, halving the evaluations saves days.

**A7 — (b).** **A gain smaller than your standard deviation is not a gain.**
</details>

## 🎯 Tasks

**Task 1 — The three-way search.** On one dataset run Grid, Random and (optionally) a Bayesian search with a matched budget. Report fits, runtime and best score for each. **Then compare all three with the untuned default and state honestly whether tuning was worth doing.**

**Task 2 — The budget curve.** Run Random Search with `n_iter` = 5, 10, 25, 50, 100 and plot best-score-found against budget. **Where does the curve flatten?** That is where you should have stopped.

**Task 3 — The full evaluation report.** For one model on one dataset, produce a single page containing: 5-fold CV mean ± std, a bootstrap 95% interval, a train-vs-test overfitting curve, and the tuning result. **Finish with a one-paragraph recommendation that states what you are confident about and what you are not.** This is the evaluation section of your capstone.

---

# ✅ Before you move on

- [ ] I can show that a single split moves by over a percentage point on seed alone
- [ ] I report **mean ± std**, never a lone number
- [ ] I know why `cross_val_score` stratifies for classifiers
- [ ] I know when LOOCV is worth it and when it is a waste
- [ ] I can put a bootstrap confidence interval on a score
- [ ] I diagnose overfitting from the **gap**, not the training score
- [ ] I know a training accuracy of 1.0 is a warning, not an achievement
- [ ] I can run Grid, Random and Bayesian search, and explain why Random usually wins
- [ ] **I never report a gain smaller than my noise**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-08-evaluation-tuning.ipynb) | Every example above, runnable |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
