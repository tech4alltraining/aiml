# Session 6 — Data Augmentation, Feature Engineering & Feature Reduction

**What to do when you do not have enough data, when your columns are not informative enough, and when you have too many of them**

| | |
|---|---|
| **Notebook** | [session-06-augmentation-feature-engg-red.ipynb](../notebooks/session-06-augmentation-feature-engg-red.ipynb) |
| **Previous** | [Session 5C — Model Deployment](session-05c-deployment.md) |
| **Next** | [Session 7 — Unsupervised Learning](session-07-unsupervised.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Sessions 5 and 5B were about choosing and training models. This session is about the data you feed them** — and on real projects, this is where most of the gains actually come from.
>
> **Every technique here is a trade, not a free win.** You will measure the cost of each one.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Say what data augmentation is, when it is used, and why
2. Balance an imbalanced dataset with over-sampling and SMOTE — **and explain how they differ**
3. Name augmentation techniques for tabular, image and text data
4. Create new features from existing ones, and **measure whether they helped**
5. Explain why feature reduction is needed
6. Use filter, wrapper and embedded selection methods
7. Apply PCA, LDA and t-SNE, and **say which is right for which job**

---

## The three parts

| Part | Question it answers |
|---|---|
| **A — [Data Augmentation](#part-a--data-augmentation)** | *I do not have enough data. What now?* |
| **B — [Feature Engineering](#part-b--feature-engineering)** | *My columns are not informative enough. Can I build better ones?* |
| **C — [Feature Reduction](#part-c--feature-reduction)** | *I have too many columns. Which can I drop?* |

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [What is augmentation](#1-what-is-data-augmentation) | | 8 | [What is feature engineering](#8-what-is-feature-engineering) |
| 2 | [When it is used](#2-when-augmentation-is-used) | | 9 | [The benefits](#9-the-benefits-of-feature-engineering) |
| 3 | [Why use it](#3-why-use-augmentation) | | 10 | [Example — car prices](#10-example-1--car-prices) |
| 4 | [Types of augmentation](#4-types-of-augmentation) | | 11 | [Example — heart failure](#11-example-2--heart-failure) |
| 5 | [Tabular techniques](#5-common-tabular-augmentation-techniques) | | 12 | [Why reduction is needed](#12-why-feature-reduction-is-needed) |
| 6 | [Image techniques](#6-image-augmentation-techniques) | | 13 | [Filter, wrapper, embedded](#13-types-of-feature-reduction) |
| 7 | [Text techniques](#7-text-augmentation-techniques) | | 14 | [Projection methods](#14-projection-methods) |
| | | | 15 | [What reduction costs](#15-what-reduction-actually-costs) |

**Practices sit between the topics.** The [20 MCQs](#-session-6--20-mcqs) and [tasks](#-session-6--tasks) are at the end.

---

# Part A — Data Augmentation

# 1. What is data augmentation

> **Data augmentation means creating additional training examples from the data you already have.**

**You do not go and collect more.** You take what you have and produce more from it — by copying, by transforming, or by generating new examples that resemble the real ones.

🧠 **Analogy: revising from one photograph of a friend.** If you have only ever seen one photo, you might not recognise them in a hat, in dim light, or from the side. **Show yourself that same photo flipped, darkened and rotated, and you learn the friend rather than the photograph.**

**Augmentation teaches the model the thing, not the example.**

## The one-line version

```text
BEFORE   100 training examples
AFTER    500 training examples, made from those 100

The MODEL sees more variety.
The WORLD has not given you any more information.
```

> **That second line matters.** **Augmentation does not create new information** — it re-presents what you have in more forms. **It helps a model generalise; it cannot tell the model something the original data never contained.**

---

# 2. When augmentation is used

**Three situations, and you will meet all three.**

| Situation | What it looks like | Example |
|---|---|---|
| **1. Too little data overall** | A few hundred examples, or fewer | 60 photographs for a helmet detector |
| **2. Imbalanced classes** | One class far rarer than another | 8.5% of patients have diabetes |
| **3. The model overfits** | High training score, poor test score | Session 8's central problem |

## Situation 1 — too little data

**Deep learning in particular is hungry.** A network trained on 60 images will memorise all 60 and fail on the 61st. **Augmenting to 600 gives it enough variety to learn a pattern rather than a list.**

## Situation 2 — imbalanced classes

**This is the commonest reason on tabular data.**

**A model trained on 91.5% negatives learns that predicting "negative" is usually right** — and Session 5B showed exactly that: 91.5% accuracy, zero patients found. **Balancing the classes removes that easy escape.**

## Situation 3 — the model overfits

**More variety in the training data makes memorisation harder.** If the model sees the same image flipped, rotated and brightened, it cannot simply memorise pixel positions.

> ⚠️ **Augmentation is not always the answer.** **If you have 100,000 balanced rows and your model is underfitting, augmentation will not help.** The problem is elsewhere.

---

# 3. Why use augmentation

**Four reasons, in the order they usually matter.**

| Reason | What it buys you |
|---|---|
| **Better generalisation** | The model learns the pattern, not the specific examples |
| **Less overfitting** | Harder to memorise a varied training set |
| **Fairer treatment of rare classes** | The model stops ignoring the minority |
| **Cheaper than collecting data** | Labelling 500 new images costs real money and time |

## And the honest cost

> ⚠️ **Augmentation is a trade, and Session 6's central lesson is that the trade is measurable.**

**Balancing a dataset typically raises recall and lowers precision.** The model now shouts the rare class more often — and most of those extra shouts are wrong. **Whether that is a good trade depends entirely on which error costs you more**, which is exactly the question Session 5B asked.

## ⚠️ The rule you must not break

```text
SPLIT FIRST.  Then augment the TRAINING half only.
```

> **Augmenting before splitting puts near-copies of the same row on both sides of the split.** Your model then gets tested on data it has effectively already seen, and **your score becomes fiction.**
>
> **This is Session 3's leakage lesson wearing a different hat.**

---

# 4. Types of augmentation

**The techniques differ completely by data type, because what counts as a "valid variation" differs.**

| Data type | Techniques | The question each asks |
|---|---|---|
| **Tabular** | Over-sampling, under-sampling, SMOTE, noise injection | *What is another plausible row?* |
| **Image** | Flip, rotate, crop, brightness, noise, zoom | *What is another plausible photograph of this?* |
| **Text** | Synonym swap, back-translation, random deletion | *What is another way of saying this?* |
| **Audio** | Pitch shift, time stretch, background noise | *What is another way this could sound?* |

## The rule that governs all of them

> **The augmentation must preserve the label.**

**A cat photographed upside down is still a cat.** **A handwritten `6` rotated 180° is a `9`** — so rotation is valid augmentation for cats and destroys the label for digits.

**This is a judgement about your data, not a setting in a library.** **You have to make it every time.**

---

# 5. Common tabular augmentation techniques

**Tabular augmentation is almost always about class imbalance.**

## Setting up the problem

**We will make the iris dataset deliberately imbalanced** — 50 of class 0 and only 25 of class 1.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# Take 50 samples from class 0 and only 25 from class 1
X_imbalanced = np.vstack([X[y == 0][:50], X[y == 1][:25]])
y_imbalanced = np.hstack([y[y == 0][:50], y[y == 1][:25]])

print("Class Distribution Before Sampling:")
print(pd.Series(y_imbalanced).value_counts())
```

**Output:**

```text
Class Distribution Before Sampling:
0    50
1    25
```

**Two classes, one twice the size of the other.**

---

## Technique 1 — Random over-sampling

> **Duplicate rows from the minority class until the classes are balanced.**

**That is the whole idea.** No new information is created — existing rows are simply repeated.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_imbalanced, y_imbalanced)

print("Class Distribution After Over-sampling:")
print(pd.Series(y_resampled).value_counts())
```

**Output:**

```text
Class Distribution After Over-sampling:
0    50
1    50
```

**Twenty-five rows from class 1 were duplicated to make up the difference.**

> **`imblearn` is a separate package: `pip install imbalanced-learn`.** It is the standard tool for this, and it is now in the course requirements.

| | |
|---|---|
| ✅ **Simple, and fast** | |
| ✅ **Never invents impossible values** | Every row is a real observation |
| ⚠️ **Encourages overfitting** | The model sees the same rows repeatedly and can memorise them |

---

## Technique 2 — SMOTE

> **SMOTE — Synthetic Minority Over-sampling Technique — creates *new* rows between existing minority rows.**

**Rather than copying a row, it picks a real minority row, picks one of its near neighbours, and places a new point somewhere on the line between them.**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imbalanced, y_imbalanced)

print("Class Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())
```

**Output:**

```text
Class Distribution After SMOTE:
0    50
1    50
```

**The same counts — and completely different rows.**

## Seeing the difference

![Class counts before, after over-sampling, and after SMOTE](images/s6-class-balance.png)

**The bar charts look identical after either method.** **The counts tell you nothing about which one you used** — so look at where the new points actually land:

![Scatter plots showing over-sampled copies on top of existing points, and SMOTE points between them](images/s6-oversample-vs-smote.png)

> **On the left, every green cross sits exactly on top of a blue or red circle.** They are duplicates — the same point, counted twice.
>
> **On the right, the green crosses sit in the gaps between real points.** They are new locations that no observation actually occupied.

**That picture is the whole difference between the two techniques.**

| | Random over-sampling | SMOTE |
|---|---|---|
| Creates | **Copies** of real rows | **New** rows between real ones |
| Risk | Overfitting to repeated rows | **Can invent impossible combinations** |
| Works on | Anything | **Numeric features only** |

> ⚠️ **SMOTE's danger is real.** Interpolating between a row with `pregnancies=0` and one with `pregnancies=8` gives `pregnancies=3.7`. **On a categorical or count column, SMOTE can produce rows that could not exist.**

---

## Technique 3 — Under-sampling

> **Instead of adding to the minority, remove from the majority.**

```python
# illustrative: a syntax reference, not runnable as written.
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_imbalanced, y_imbalanced)
# 25 and 25 - the majority class is cut down
```

| | |
|---|---|
| ✅ **Fast, and no synthetic data** | |
| ⚠️ **You throw away real data** | On a small dataset this is expensive |

**Use it when the majority class is enormous** — discarding 90% of a million rows still leaves plenty.

---

## Technique 4 — Class weights, which are not augmentation at all

> **Often the best answer is to not augment.**

```python
# illustrative: a syntax reference, not runnable as written.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
```

**This tells the model that mistakes on the rare class cost more**, without touching the data at all.

**Session 5B measured this on the diabetes data:** recall went from 0.6371 to 0.8882, and missed patients from 617 to 190. **One keyword, no synthetic rows, no leakage risk.**

> **Try `class_weight='balanced'` before you reach for SMOTE.** It is simpler, it cannot invent impossible rows, and it very often works just as well.

## ✏️ Practice — tabular augmentation

1. Build the imbalanced iris subset and print the class distribution.
2. Apply `RandomOverSampler` and print the distribution after.
3. Apply `SMOTE` and print the distribution after. **How do the counts compare with over-sampling?**
4. Plot the petal length against petal width for both methods, marking which points are synthetic. **Describe the difference.**
5. Name one risk of over-sampling and one risk of SMOTE.

<details><summary>Solutions</summary>

```python
# needs-install: pip install imbalanced-learn
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from imblearn.over_sampling import RandomOverSampler, SMOTE

iris = load_iris(); X, y = iris.data, iris.target
X_imb = np.vstack([X[y == 0][:50], X[y == 1][:25]])                    # 1
y_imb = np.hstack([y[y == 0][:50], y[y == 1][:25]])
print(pd.Series(y_imb).value_counts())          # 50 and 25

Xo, yo = RandomOverSampler(random_state=42).fit_resample(X_imb, y_imb)  # 2
print(pd.Series(yo).value_counts())             # 50 and 50

Xs, ys = SMOTE(random_state=42).fit_resample(X_imb, y_imb)             # 3
print(pd.Series(ys).value_counts())             # 50 and 50
# IDENTICAL counts. The counts cannot tell you which method was used -
# only looking at WHERE the new points sit can.

# 4 - Over-sampling puts every new point exactly on top of an existing
#     one, because they are duplicates. SMOTE puts them in the GAPS
#     between real points, because they are interpolations.

# 5 - Over-sampling: the model sees the same rows repeatedly and can
#       memorise them, which encourages overfitting.
#     SMOTE: interpolating between rows can invent impossible values -
#       3.7 pregnancies, or a category halfway between two categories.
```
</details>

---

# 6. Image augmentation techniques

**Images are the classic case for augmentation**, because a photograph has so many valid variations.

| Technique | What it does | Valid when |
|---|---|---|
| **Horizontal flip** | Mirror left-right | The object is symmetric in meaning |
| **Rotation** | Turn by a few degrees | Orientation does not carry meaning |
| **Crop and resize** | Take a portion, scale back up | The object may not be centred |
| **Brightness / contrast** | Lighten or darken | Lighting varies in the real world |
| **Zoom** | Scale in or out | Distance to the subject varies |
| **Noise** | Add random pixel variation | Cameras are imperfect |
| **Shift / translate** | Move the image within the frame | Position varies |

## One image becomes six

![Six versions of the same shape: original, flipped, rotated, cropped, brightened and noisy](images/s6-image-augmentation.png)

```python
from PIL import Image, ImageEnhance
import numpy as np

# A simple shape standing in for a photograph
img = np.zeros((72, 72), dtype=np.uint8)
img[14:58, 20:29] = 255
img[49:58, 20:52] = 255
img[31:40, 20:52] = 255
img[31:58, 43:52] = 255
base = Image.fromarray(img)

rng = np.random.default_rng(0)
views = {
    "original":      base,
    "flip":          base.transpose(Image.FLIP_LEFT_RIGHT),
    "rotate 20":     base.rotate(20),
    "crop + resize": base.crop((8, 8, 64, 64)).resize((72, 72)),
    "brighter":      ImageEnhance.Brightness(base).enhance(1.7),
    "noise":         Image.fromarray(
        np.clip(img + rng.normal(0, 45, img.shape), 0, 255).astype(np.uint8)),
}
for name, im in views.items():
    print(f"{name:<16}{im.size}")
```

**Sixty photographs become 360 training examples** — which is the difference between a model that memorises and one that learns.

## ⚠️ The label-preservation question

**Look at that figure again and ask, for each variation: is it still the same thing?**

| For **cat vs dog** | For **handwritten digits** |
|---|---|
| Flip ✅ — a mirrored cat is a cat | **Flip ❌ — a mirrored 2 is not a 2** |
| Rotate ✅ | **Rotate 180° ❌ — a 6 becomes a 9** |
| Brightness ✅ | Brightness ✅ |
| Crop ✅ | Crop ⚠️ — may cut off part of the digit |

> **The same technique is correct for one dataset and destroys another.** **Augmentation is a judgement about your data, and no library can make it for you.**

**In practice you would use a library** — `torchvision.transforms`, `albumentations`, or Keras's `ImageDataGenerator` — but **the decision about which transforms are valid is always yours.**

---

# 7. Text augmentation techniques

**Text is the hardest of the three, because meaning is fragile.**

| Technique | What it does | Example |
|---|---|---|
| **Synonym replacement** | Swap a word for a similar one | *"The film was **great**"* → *"The film was **excellent**"* |
| **Random deletion** | Remove a word at random | *"The film was really great"* → *"The film was great"* |
| **Random swap** | Exchange two words' positions | *"really great film"* → *"great really film"* |
| **Random insertion** | Add a related word | *"The film was great"* → *"The film was truly great"* |
| **Back-translation** | Translate out and back | English → French → English |
| **LLM paraphrasing** | Ask a model to rewrite it | Session 10's territory |

## Back-translation, the most reliable of them

```text
ORIGINAL   "The delivery was late but the product is excellent."
              ↓  translate to French
FRENCH     "La livraison était en retard mais le produit est excellent."
              ↓  translate back to English
RESULT     "Delivery was delayed but the product is excellent."
```

**Different words, same meaning, same label.** **That is exactly what good augmentation produces.**

## ⚠️ Why text augmentation is risky

```text
ORIGINAL   "The service was not good."          -> NEGATIVE

Random deletion removes "not":
RESULT     "The service was good."              -> POSITIVE

The label was flipped by deleting one word.
```

> **A single word can invert the meaning of a sentence.** **Negations, sarcasm and qualifiers are all fragile** — which is why random deletion and swapping are used cautiously, and why back-translation and paraphrasing are safer.

## A simple synonym-swap example

```python
import random

SYNONYMS = {
    "great": ["excellent", "superb", "fantastic"],
    "bad":   ["poor", "terrible", "awful"],
    "quick": ["fast", "rapid", "speedy"],
}

def synonym_swap(sentence, seed=0):
    random.seed(seed)
    words = sentence.split()
    out = [random.choice(SYNONYMS[w]) if w in SYNONYMS else w for w in words]
    return " ".join(out)

original = "the delivery was quick and the product was great"
for s in range(3):
    print(synonym_swap(original, seed=s))
```

**Three variations from one sentence, all carrying the same sentiment.**

## ✏️ Practice — image and text augmentation

1. Generate six augmented versions of an image and display them.
2. For each of your six, say whether it would preserve the label for (a) cat-vs-dog and (b) handwritten digits.
3. Write a synonym-swap function and produce three variants of a sentence.
4. Write a sentence where deleting one word **reverses** the label.
5. Explain back-translation in one sentence, and say why it is safer than random deletion.

<details><summary>Solutions</summary>

```python
import numpy as np, random
from PIL import Image, ImageEnhance

img = np.zeros((72, 72), dtype=np.uint8)                               # 1
img[14:58, 20:29] = 255; img[49:58, 20:52] = 255
img[31:40, 20:52] = 255; img[31:58, 43:52] = 255
base = Image.fromarray(img)
rng = np.random.default_rng(0)
views = {"original": base,
         "flip": base.transpose(Image.FLIP_LEFT_RIGHT),
         "rotate": base.rotate(20),
         "crop": base.crop((8, 8, 64, 64)).resize((72, 72)),
         "bright": ImageEnhance.Brightness(base).enhance(1.7),
         "noise": Image.fromarray(np.clip(img + rng.normal(0, 45, img.shape),
                                          0, 255).astype(np.uint8))}
print(list(views))

# 2 - CAT vs DOG: all six preserve the label. A mirrored, rotated,
#       cropped, brightened or noisy cat is still a cat.
#     DIGITS: flip DESTROYS it (a mirrored 2 is not a 2) and a 180
#       rotation turns 6 into 9. Brightness and noise are fine; cropping
#       risks cutting off part of the digit.

SYNONYMS = {"great": ["excellent", "superb"], "quick": ["fast", "rapid"]}   # 3
def swap(s, seed=0):
    random.seed(seed)
    return " ".join(random.choice(SYNONYMS[w]) if w in SYNONYMS else w
                    for w in s.split())
for s in range(3):
    print(swap("the delivery was quick and the product was great", s))

# 4 - "The service was not good."  -> NEGATIVE
#     Delete "not" and it becomes "The service was good." -> POSITIVE.
#     One word inverted the label.

# 5 - Back-translation means translating a sentence into another language
#     and then back again, producing different wording with the same
#     meaning. It is safer than random deletion because a translator
#     preserves meaning by design, whereas deletion can remove the one
#     word - a negation - that the label depended on.
```
</details>

---

# Part B — Feature Engineering

# 8. What is feature engineering

> **Feature engineering means creating new columns from the ones you already have, so the pattern is easier for a model to see.**

🧠 **Analogy: judging whether a loan is affordable.** You are told the loan is ₹800,000 and the income is ₹400,000. **You could look at both — but the number you actually care about is the ratio: 2.0.** Nobody handed you that column. You built it.

## Why a model cannot do this for you

**A model can only find patterns among the columns it is given.**

```text
Given  loan_amount  and  income
A linear model can compute:   a × loan_amount + b × income
It CANNOT compute:            loan_amount ÷ income
```

> **A linear model has no way to represent division.** **A tree can approximate a ratio by splitting on both columns repeatedly**, but only clumsily. **If the useful quantity is a relationship between columns, somebody has to construct it — and that somebody is you.**

## The main techniques

| Technique | What it does | Example |
|---|---|---|
| **Ratios** | Divide one column by another | `loan ÷ income`, `power ÷ engine` |
| **Differences** | Subtract one from another | `end_date − start_date` |
| **Rates** | Amount per unit of something | `km ÷ years` |
| **Binning** | Group a number into bands | age → young / middle / senior |
| **Flags** | A yes/no derived from a threshold | `ejection_fraction < 40` |
| **Interactions** | Combine two columns | `age × ejection_fraction` |
| **Aggregations** | Summarise related rows | a customer's average past order |
| **Transformations** | Reshape a skewed column | `log(price)` |
| **Concatenation** | Join two categories | `brand + model` |

---

# 9. The benefits of feature engineering

| Benefit | Why it happens |
|---|---|
| **Better accuracy** | The pattern becomes visible to the model |
| **Simpler models work** | A linear model given a ratio can match a tree without one |
| **More explainable** | *"loan-to-income above 2"* is a sentence anyone understands |
| **Encodes domain knowledge** | Decades of expertise, put into a column |
| **Reduces data needed** | The model has less to discover for itself |

## ⚠️ And the honest caveat, measured

**Feature engineering is often described as the highest-value activity in machine learning. It is also often oversold.**

**Here is what the car features below actually did** — the same model, with and without five engineered columns:

| Model | Base features | With 5 engineered features | Change |
|---|---|---|---|
| Linear Regression | 0.6604 | 0.6605 | **+0.0001** |
| Random Forest | 0.9236 | 0.9201 | **−0.0035** |

> **Essentially nothing for the linear model, and very slightly *worse* for the forest.**
>
> **Why so little?** A Random Forest can already approximate a ratio by splitting on both columns. **The features you built are things it had largely found on its own.**

**So does that make feature engineering pointless? No — but it does mean you should measure rather than assume.** Features earn their place most when:

- The model is **linear** and cannot represent the relationship at all
- The feature encodes **domain knowledge the data does not contain** — a clinical threshold, a regulatory limit
- You need the model to be **explainable**, and a named ratio is easier to defend than a split

---

# 10. Example 1 — Car prices

**The cardekho dataset has raw measurements. We will build features a car dealer would think in.**

## The features, and why each one

| New feature | Meaning | Why it should help |
|---|---|---|
| `km_per_year` | Average kilometres driven per year | **Usage intensity** — 100,000 km over 10 years is different from over 2 |
| `age_category` | new / moderate / old / very old | Price falls in bands, not smoothly |
| `high_usage` | Driven more than the median rate | Heavy use lowers value |
| `power_to_engine_ratio` | Power relative to engine size | **Performance efficiency** |
| `mileage_to_engine_ratio` | Mileage relative to engine size | **Fuel efficiency** |
| `power_per_seat` | Power distributed per seat | Compares compact and family cars fairly |
| `brand_model` | Brand and model joined | Captures the specific car's identity |
| `fuel_transmission` | Fuel type and transmission joined | Captures combined market preference |

## The code

```python
import pandas as pd
import numpy as np

dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/cardekho_dataset.csv"
cars_data = pd.read_csv(dataset_url)

# 1. Usage-based feature
cars_data["km_per_year"] = cars_data["km_driven"] / (cars_data["vehicle_age"] + 1)

# 2. Age category
cars_data["age_category"] = pd.cut(
    cars_data["vehicle_age"],
    bins=[0, 3, 7, 12, 30],
    labels=["new", "moderate", "old", "very_old"]
)

# 3. High usage indicator
cars_data["high_usage"] = (
    cars_data["km_per_year"] > cars_data["km_per_year"].median()
).astype(int)

# 4. Performance-related features
cars_data["power_to_engine_ratio"] = cars_data["max_power"] / cars_data["engine"]
cars_data["mileage_to_engine_ratio"] = cars_data["mileage"] / cars_data["engine"]
cars_data["power_per_seat"] = cars_data["max_power"] / cars_data["seats"]

# 5. Interaction categorical features
cars_data["brand_model"] = (
    cars_data["brand"].astype(str) + "_" + cars_data["model"].astype(str)
)
cars_data["fuel_transmission"] = (
    cars_data["fuel_type"].astype(str) + "_" + cars_data["transmission_type"].astype(str)
)

cars_data.head()
```

## ⚠️ Two bugs hiding in that code — and both are common

**Run it and then check your work. Always check your work.**

```python
print("infinities in power_per_seat :", np.isinf(cars_data["power_per_seat"]).sum())
print("missing in age_category      :", cars_data["age_category"].isna().sum())
```

**Output:**

```text
infinities in power_per_seat : 2
missing in age_category      : 5
```

### Bug 1 — division by zero

```python
print(cars_data["seats"].value_counts().sort_index().to_dict())
```

**Output:**

```text
{0: 2, 2: 7, 4: 77, 5: 12910, 6: 127, 7: 1922, 8: 311, 9: 55}
```

> **Two cars have `seats = 0`.** Dividing by zero gives infinity, and **scikit-learn refuses to train on infinite values** — you get `ValueError: Input X contains infinity`.
>
> **This is exactly why `km_per_year` uses `vehicle_age + 1`.** The `+ 1` protects against a brand-new car with age 0. **The same protection was needed for `seats` and was not applied.**

**The fix:**

```python
cars_data["power_per_seat"] = (
    cars_data["max_power"] / cars_data["seats"].replace(0, np.nan)
)
cars_data["power_per_seat"] = cars_data["power_per_seat"].fillna(
    cars_data["power_per_seat"].median()
)
print("infinities now:", np.isinf(cars_data["power_per_seat"]).sum())
```

**Output:** `infinities now: 0`

> **Every ratio you build needs this check.** **Ask: can the denominator be zero?**

### Bug 2 — `pd.cut` excludes the left edge

```python
print("cars with vehicle_age == 0:", (cars_data["vehicle_age"] == 0).sum())
```

**Output:**

```text
cars with vehicle_age == 0: 5
```

> **`pd.cut(bins=[0, 3, 7, 12, 30])` creates intervals that are *open on the left*: `(0, 3]`, `(3, 7]`, and so on.** **A vehicle age of exactly 0 falls into none of them** and becomes `NaN`.

**Two fixes, both valid:**

```python
# Option A - start the bins below the minimum
cars_data["age_category"] = pd.cut(
    cars_data["vehicle_age"], bins=[-1, 3, 7, 12, 30],
    labels=["new", "moderate", "old", "very_old"])

# Option B - include the left edge explicitly
cars_data["age_category"] = pd.cut(
    cars_data["vehicle_age"], bins=[0, 3, 7, 12, 30],
    labels=["new", "moderate", "old", "very_old"], include_lowest=True)

print("missing now:", cars_data["age_category"].isna().sum())
```

**Output:** `missing now: 0`

> **Five rows out of 15,411 is 0.03% — easy to miss, and it would silently become five missing values in your feature.** **Check every engineered column for `NaN` and `inf` the moment you create it.**

## Did the features help?

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

BASE = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
NEW = ["km_per_year", "high_usage", "power_to_engine_ratio",
       "mileage_to_engine_ratio", "power_per_seat"]

def evaluate(cols):
    X, y = cars_data[cols], cars_data["selling_price"]
    a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=42)
    lin = r2_score(d, LinearRegression().fit(a, c).predict(b))
    rf = r2_score(d, RandomForestRegressor(n_estimators=100, random_state=42,
                                           n_jobs=-1).fit(a, c).predict(b))
    return lin, rf

base_lin, base_rf = evaluate(BASE)
eng_lin, eng_rf = evaluate(BASE + NEW)

print(f"linear  R2  {base_lin:.4f} -> {eng_lin:.4f}  ({eng_lin - base_lin:+.4f})")
print(f"forest  R2  {base_rf:.4f} -> {eng_rf:.4f}  ({eng_rf - base_rf:+.4f})")
```

**Output:**

```text
linear  R2  0.6604 -> 0.6605  (+0.0001)
forest  R2  0.9236 -> 0.9201  (-0.0035)
```

> **Almost nothing, and slightly negative for the forest.**
>
> **Report this honestly.** It is a real result, and pretending otherwise would be worse than useless. **The forest had already found these relationships by splitting on the raw columns twice.**

**What this does *not* mean:** that the features are worthless. **They are far more explainable** — *"this car is driven 15,000 km a year"* is a sentence a dealer understands, and `km_driven=150000, vehicle_age=10` is not. **Explainability is a real benefit even when accuracy does not move.**

## ✏️ Practice — car features

1. Build all eight features and print `head()`.
2. Check every new numeric column for `NaN` and `inf`. **Which two have problems?**
3. Fix both bugs and confirm they are gone.
4. Measure R² with and without the engineered features, for a linear model and a forest.
5. **Which single feature would you keep if you could keep only one, and why?**

<details><summary>Solutions</summary>

```python
import pandas as pd, numpy as np
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/cardekho_dataset.csv"
cars = pd.read_csv(dataset_url)

cars["km_per_year"] = cars["km_driven"] / (cars["vehicle_age"] + 1)     # 1
cars["age_category"] = pd.cut(cars["vehicle_age"], bins=[0, 3, 7, 12, 30],
                              labels=["new", "moderate", "old", "very_old"])
cars["high_usage"] = (cars["km_per_year"] > cars["km_per_year"].median()).astype(int)
cars["power_to_engine_ratio"] = cars["max_power"] / cars["engine"]
cars["mileage_to_engine_ratio"] = cars["mileage"] / cars["engine"]
cars["power_per_seat"] = cars["max_power"] / cars["seats"]
cars["brand_model"] = cars["brand"].astype(str) + "_" + cars["model"].astype(str)
cars["fuel_transmission"] = (cars["fuel_type"].astype(str) + "_"
                             + cars["transmission_type"].astype(str))
print(cars.head())

for c in ["km_per_year", "power_to_engine_ratio", "mileage_to_engine_ratio",  # 2
          "power_per_seat"]:
    print(f"{c:<26} nan {cars[c].isna().sum():>3}  inf {np.isinf(cars[c]).sum():>3}")
print("age_category nan:", cars["age_category"].isna().sum())
# power_per_seat has 2 infinities (seats == 0 for two cars)
# age_category has 5 NaN (pd.cut excludes the left edge, and 5 cars are age 0)

cars["power_per_seat"] = cars["max_power"] / cars["seats"].replace(0, np.nan)  # 3
cars["power_per_seat"] = cars["power_per_seat"].fillna(cars["power_per_seat"].median())
cars["age_category"] = pd.cut(cars["vehicle_age"], bins=[-1, 3, 7, 12, 30],
                              labels=["new", "moderate", "old", "very_old"])
print("inf:", np.isinf(cars["power_per_seat"]).sum(),
      " nan:", cars["age_category"].isna().sum())      # 0 and 0

from sklearn.model_selection import train_test_split                    # 4
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
BASE = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
NEW = ["km_per_year", "high_usage", "power_to_engine_ratio",
       "mileage_to_engine_ratio", "power_per_seat"]
def ev(cols):
    X, y = cars[cols], cars["selling_price"]
    a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42)
    return (r2_score(d, LinearRegression().fit(a, c).predict(b)),
            r2_score(d, RandomForestRegressor(n_estimators=100, random_state=42,
                                              n_jobs=-1).fit(a, c).predict(b)))
print("base      ", [round(v, 4) for v in ev(BASE)])
print("engineered", [round(v, 4) for v in ev(BASE + NEW)])
# Barely any change, and slightly WORSE for the forest. Report it honestly.

# 5 - km_per_year. It is the one a dealer would actually name, it turns
#     two raw columns into a single interpretable rate, and it is the
#     feature a LINEAR model most needs, since it cannot divide.
```
</details>

---

# 11. Example 2 — Heart failure

**Medical data is where feature engineering pays best**, because clinical thresholds are real knowledge that the raw numbers do not contain.

## ⚠️ First: this dataset needs preprocessing

**The trainer's notebook has a comment saying "Do preprocessing first". Here is why that matters.**

```python
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/heart_failure_raw.csv"
heart = pd.read_csv(dataset_url)

print(heart.shape)
print(heart.dtypes)
print("\nmissing:")
print(heart.isnull().sum()[heart.isnull().sum() > 0])
```

**Output:**

```text
(299, 14)

anaemia                 object
diabetes                object
high_blood_pressure     object
smoking                 object
sex                     object
DEATH_EVENT             object
...

missing:
age                  15
ejection_fraction    15
serum_creatinine     15
```

**Two problems:**

1. **The binary columns are text**, holding `"Yes"` and `"No"` — not 1 and 0
2. **45 missing values** across three numeric columns

> **The feature engineering below uses columns called `anaemia_bin`, `diabetes_bin` and so on.** **Those do not exist in the raw file — you have to create them.** That is precisely what "do preprocessing first" means.

```python
# Convert the Yes/No columns into 1/0
BINARY = ["anaemia", "diabetes", "high_blood_pressure", "smoking"]
for col in BINARY:
    heart[col + "_bin"] = heart[col].map({"Yes": 1, "No": 0})

# Fill the numeric gaps with the median (Session 3's rule for skewed columns)
for col in ["age", "ejection_fraction", "serum_creatinine"]:
    heart[col] = heart[col].fillna(heart[col].median())

print("missing now:", heart.isnull().sum().sum())
print("created:", [c + "_bin" for c in BINARY])
```

**Output:**

```text
missing now: 0
created: ['anaemia_bin', 'diabetes_bin', 'high_blood_pressure_bin', 'smoking_bin']
```

## Now the feature engineering

**Notice that almost every feature below encodes a *clinical threshold* — a number that means something to a cardiologist and nothing to a model.**

```python
# 1. Age group
heart["age_group"] = pd.cut(
    heart["age"], bins=[0, 40, 60, 80, 120],
    labels=["young", "middle_age", "senior", "elderly"])

# 2. Ejection fraction risk features
heart["low_ejection_fraction"] = (heart["ejection_fraction"] < 40).astype(int)
heart["very_low_ejection_fraction"] = (heart["ejection_fraction"] < 30).astype(int)

# 3. Serum creatinine risk feature
heart["high_serum_creatinine"] = (heart["serum_creatinine"] > 1.5).astype(int)

# 4. Serum sodium risk feature
heart["low_serum_sodium"] = (heart["serum_sodium"] < 135).astype(int)

# 5. Creatinine phosphokinase risk feature
heart["high_cpk"] = (heart["creatinine_phosphokinase"] > 500).astype(int)

# 6. Platelet-related features
heart["low_platelets"] = (heart["platelets"] < 150000).astype(int)
heart["high_platelets"] = (heart["platelets"] > 450000).astype(int)
heart["platelets_lakh"] = heart["platelets"] / 100000

# 7. Ratio feature
heart["creatinine_sodium_ratio"] = heart["serum_creatinine"] / heart["serum_sodium"]

# 8. Log transformation of a skewed column
heart["cpk_log"] = np.log1p(heart["creatinine_phosphokinase"])

# 9. Comorbidity count
heart["comorbidity_count"] = heart[
    ["anaemia_bin", "diabetes_bin", "high_blood_pressure_bin", "smoking_bin"]
].sum(axis=1)

# 10. Risk score features
heart["renal_risk_score"] = heart["high_serum_creatinine"] + heart["low_serum_sodium"]
heart["cardiac_risk_score"] = heart["low_ejection_fraction"] + heart["high_cpk"]

# 11. Interaction features
heart["age_ef_interaction"] = heart["age"] * heart["ejection_fraction"]
heart["creatinine_ef_interaction"] = heart["serum_creatinine"] * heart["ejection_fraction"]

# 12. Treatment-age interaction feature
heart["treatment_risk_group"] = (
    heart["treatment_type"].astype(str) + "_" + heart["age_group"].astype(str))

print(heart.shape)
heart.head()
```

**Output:**

```text
(299, 34)
```

**Fourteen columns became thirty-four.**

## The techniques on display

| # | Technique | Feature | The knowledge it encodes |
|---|---|---|---|
| 1 | **Binning** | `age_group` | Risk changes by life stage, not smoothly |
| 2 | **Threshold flag** | `low_ejection_fraction` | **Below 40% is clinically abnormal** |
| 3–5 | **Threshold flags** | creatinine, sodium, CPK | Published clinical reference ranges |
| 6 | **Unit change** | `platelets_lakh` | Readable numbers for a human |
| 7 | **Ratio** | `creatinine_sodium_ratio` | Kidney function relative to electrolytes |
| 8 | **Log transform** | `cpk_log` | CPK is heavily skewed |
| 9 | **Count** | `comorbidity_count` | **How many conditions, in one number** |
| 10 | **Composite score** | `renal_risk_score` | Combines related flags |
| 11 | **Interaction** | `age_ef_interaction` | Poor heart function matters more when older |

> **Look at feature 2.** `ejection_fraction < 40` is not a number a model would discover — **it is a threshold agreed by cardiologists.** **The model cannot know that 39 and 41 are meaningfully different; a doctor can tell it.**
>
> **That is what "encoding domain knowledge" means, and it is the strongest argument for feature engineering.**

## The comorbidity count, which is worth pausing on

```python
print(heart["comorbidity_count"].value_counts().sort_index())
```

**Output:**

```text
0     43
1    120
2     94
3     37
4      5
```

> **Four separate yes/no columns became one number from 0 to 4.** **A model now has a single "how ill is this person overall" signal**, rather than having to discover that those four columns should be added together.

## ✏️ Practice — heart failure features

1. Load the dataset and print its shape, dtypes and missing values. **What two problems must be fixed before the feature code will run?**
2. Create the four `*_bin` columns and fill the numeric gaps.
3. Build all the engineered features and print the new shape.
4. Print the distribution of `comorbidity_count`. **What does a value of 4 mean?**
5. **Pick three features and say what clinical knowledge each one encodes.**

<details><summary>Solutions</summary>

```python
import pandas as pd, numpy as np
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/classification/heart_failure_raw.csv"
heart = pd.read_csv(dataset_url)

print(heart.shape)                                                     # 1
print(heart.dtypes.to_string())
print(heart.isnull().sum()[heart.isnull().sum() > 0])
# PROBLEM 1: anaemia, diabetes, high_blood_pressure and smoking are TEXT
#   ("Yes"/"No"), but the feature code expects *_bin columns of 1/0.
# PROBLEM 2: 45 missing values across age, ejection_fraction and
#   serum_creatinine.

BINARY = ["anaemia", "diabetes", "high_blood_pressure", "smoking"]      # 2
for c in BINARY:
    heart[c + "_bin"] = heart[c].map({"Yes": 1, "No": 0})
for c in ["age", "ejection_fraction", "serum_creatinine"]:
    heart[c] = heart[c].fillna(heart[c].median())
print("missing now:", heart.isnull().sum().sum())

heart["low_ejection_fraction"] = (heart["ejection_fraction"] < 40).astype(int)  # 3
heart["high_serum_creatinine"] = (heart["serum_creatinine"] > 1.5).astype(int)
heart["low_serum_sodium"] = (heart["serum_sodium"] < 135).astype(int)
heart["cpk_log"] = np.log1p(heart["creatinine_phosphokinase"])
heart["comorbidity_count"] = heart[[c + "_bin" for c in BINARY]].sum(axis=1)
heart["renal_risk_score"] = (heart["high_serum_creatinine"]
                             + heart["low_serum_sodium"])
print(heart.shape)

print(heart["comorbidity_count"].value_counts().sort_index())          # 4
# A value of 4 means the patient has ALL FOUR conditions - anaemia,
# diabetes, high blood pressure and smoking. Only 5 patients do.

# 5 - low_ejection_fraction: below 40% is a clinically recognised
#       threshold for impaired heart function. A model cannot know that
#       39 and 41 differ meaningfully; a cardiologist can tell it.
#     high_serum_creatinine: above 1.5 indicates impaired kidney
#       function, a published reference range.
#     comorbidity_count: clinicians reason about how MANY conditions a
#       patient has, not just which. One number carries that.
```
</details>

---

# Part C — Feature Reduction

**Part B added columns. Part C takes them away.**

**That is not a contradiction — it is the two halves of the same job.** **You create every feature that might carry signal, then you keep only the ones that do.**

---

# 12. Why feature reduction is needed

> **More columns is not the same as more information.**

## Reason 1 — the curse of dimensionality

**As you add columns, the space the data lives in grows exponentially, and your rows spread thinner and thinner across it.**

```text
1 feature,  10 bins            ->     10 cells,  100 rows = 10 rows per cell
2 features, 10 bins each       ->    100 cells,  100 rows =  1 row  per cell
3 features, 10 bins each       ->  1,000 cells,  100 rows =  0.1 rows per cell
```

> **With three features, 90% of the space is empty.** **A model asked "what happens near this point?" has no neighbours to answer with.**

## Reason 2 — overfitting

**Every extra column is another chance for the model to memorise noise.** A column that is pure random numbers will, by chance, correlate slightly with the target in your training set — and the model will use it.

## Reason 3 — speed and cost

**Training time and memory grow with the number of columns.** For a small dataset this is invisible; for a million rows and a thousand columns it is the difference between minutes and hours.

## Reason 4 — multicollinearity

**Two columns that say the same thing confuse a linear model.** If `height_cm` and `height_inches` are both present, the model cannot decide which one gets the credit, and the coefficients become unstable and unreadable.

## Reason 5 — explainability

**A model using 6 columns can be explained to a customer. A model using 600 cannot.**

## Reason 6 — visualisation

**You cannot plot 30 dimensions.** Reducing to 2 lets you *look* at the data — often the fastest way to notice something is wrong.

| Reason | Symptom you would see |
|---|---|
| Curse of dimensionality | Test accuracy far below train accuracy |
| Overfitting | The same |
| Speed | Training takes too long to iterate |
| Multicollinearity | Coefficients flip sign when you retrain |
| Explainability | You cannot answer "why was I rejected?" |
| Visualisation | You cannot see the data at all |

---

# 13. Types of feature reduction

**There are two fundamentally different things people call "feature reduction", and confusing them causes real problems.**

| | **Selection** | **Projection** |
|---|---|---|
| What it does | **Keeps some original columns, drops the rest** | **Builds new columns from combinations of all of them** |
| Output columns | `petal_length`, `petal_width` | `PC1`, `PC2` |
| Explainable? | **Yes — they are your real columns** | **No — what is `PC1` in ₹?** |
| Methods | Filter, Wrapper, Embedded | PCA, LDA, t-SNE |

> **Selection is subtraction. Projection is rotation.** **This section covers selection; §14 covers projection.**

**Selection itself comes in three families:**

| Family | How it decides | Model involved? | Cost |
|---|---|---|---|
| **Filter** | Statistics on each column vs the target | **No** | **Very cheap** |
| **Wrapper** | Trains a model on subsets and compares | **Yes, many times** | **Expensive** |
| **Embedded** | The model selects while it trains | **Yes, once** | **Cheap** |

---

## 13.1 Filter methods

> **Score every column on its own, keep the best ones. No model is trained.**

**Because no model is involved, filter methods are fast and model-agnostic — but they judge each column in isolation, so they cannot see that two columns are useful only *together*.**

### The four filters you will use

| Filter | Use when | What it measures |
|---|---|---|
| `VarianceThreshold` | Always, first | **Does the column vary at all?** |
| `f_classif` (ANOVA F) | Numeric X, categorical y | **Do the class means differ?** |
| `chi2` | **Non-negative** X, categorical y | Dependence between column and class |
| `mutual_info_classif` | Anything | **Any** relationship, including non-linear |

### Step 1 — drop columns that barely vary

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print(X.var().round(4).to_dict())

vt = VarianceThreshold(threshold=0.3).fit(X)
print("kept:", list(X.columns[vt.get_support()]))
```

**Output:**

```text
{'sepal length (cm)': 0.6857, 'sepal width (cm)': 0.19,
 'petal length (cm)': 3.1163, 'petal width (cm)': 0.581}

kept: ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
```

> **`sepal width` is dropped: its values barely move, so it cannot explain a target that does.**
>
> ⚠️ **Variance depends on units.** A column measured in metres has 10,000× less variance than the same column in centimetres. **Either scale first, or set the threshold per column with your eyes open.** **A pure zero threshold — `VarianceThreshold(0)`, which drops only constant columns — is always safe.**

### Step 2 — ANOVA F: do the classes have different means?

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=2).fit(X, y)
for name, score, p in zip(X.columns, selector.scores_, selector.pvalues_):
    print(f"{name:<20} F = {score:>8.2f}   p = {p:.2e}")

print("\nselected:", list(X.columns[selector.get_support()]))
```

**Output:**

```text
sepal length (cm)    F =   119.26   p = 1.67e-31
sepal width (cm)     F =    49.16   p = 4.49e-17
petal length (cm)    F =  1180.16   p = 2.86e-91
petal width (cm)     F =   960.01   p = 4.17e-85

selected: ['petal length (cm)', 'petal width (cm)']
```

> **How to read this: a high F means the three species have very different average petal lengths, and very little spread within each species.** **That is exactly what makes a column useful for telling classes apart.**
>
> **`petal length` scores 1180 against `sepal width`'s 49 — a 24× difference.**

### Step 3 — chi², for counts and non-negative values

```python
from sklearn.feature_selection import chi2

chi_sel = SelectKBest(score_func=chi2, k=2).fit(X, y)
for name, score in zip(X.columns, chi_sel.scores_):
    print(f"{name:<20} chi2 = {score:>8.2f}")
```

**Output:**

```text
sepal length (cm)    chi2 =    10.82
sepal width (cm)     chi2 =     3.71
petal length (cm)    chi2 =   116.31
petal width (cm)     chi2 =    67.05
```

> ⚠️ **`chi2` requires non-negative values.** **If you have standard-scaled your data, half your values are negative and `chi2` will raise an error.** Use `MinMaxScaler`, or use `f_classif` instead.

### Step 4 — mutual information, which catches non-linear relationships

```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X, y, random_state=42)
for name, score in zip(X.columns, mi):
    print(f"{name:<20} MI = {score:.4f}")
```

**Output:**

```text
sepal length (cm)    MI = 0.5114
sepal width (cm)     MI = 0.2994
petal length (cm)    MI = 0.9926
petal width (cm)     MI = 0.9856
```

> **Mutual information asks: "how much does knowing this column reduce my uncertainty about the class?"** **0 means nothing at all. It does not assume the relationship is linear, which is why it is the safest general-purpose filter.**

### They agree — and that is the point

![Four filter methods, one answer](images/s6-filter-methods-agree.png)

> **Four different statistics, computed four different ways, all rank the two petal columns far above the two sepal columns.**
>
> **When independent methods agree, you can act with confidence. When they disagree, that disagreement is telling you something** — usually that a column has a non-linear relationship (mutual information sees it, ANOVA does not) or that the scales are misleading you.

---

## 13.2 Wrapper methods

> **Actually train the model on different subsets of columns, and keep the subset that scores best.**

**Filter methods guess. Wrapper methods measure.** The price is that you train the model many times.

### RFE — recursive feature elimination

**The algorithm:**

```text
1. Train the model on ALL features
2. Ask the model which feature it relied on least
3. Delete that feature
4. Repeat from step 1 until only k features remain
```

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=2).fit(X, y)

for name, rank in zip(X.columns, rfe.ranking_):
    print(f"{name:<20} rank {rank}")

print("\nselected:", list(X.columns[rfe.support_]))
```

**Output:**

```text
sepal length (cm)    rank 3
sepal width (cm)     rank 2
petal length (cm)    rank 1
petal width (cm)     rank 1

selected: ['petal length (cm)', 'petal width (cm)']
```

> **Rank 1 means "kept". Higher ranks are the order in which columns were eliminated** — `sepal length` was dropped first, `sepal width` second.
>
> **Note that RFE reached the same answer as every filter method — but it had to train logistic regression three times to get there.**

### Forward and backward selection

| Method | Starts with | Each step |
|---|---|---|
| **Forward selection** | **No features** | **Adds** the one that helps most |
| **Backward elimination** | **All features** | **Removes** the one that hurts least |
| **RFE** | All features | Removes the lowest-ranked, using the model's own weights |

```python
# illustrative: a syntax reference, not runnable as written.
from sklearn.feature_selection import SequentialFeatureSelector

forward = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000), n_features_to_select=2, direction="forward")

backward = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000), n_features_to_select=2, direction="backward")
```

> ⚠️ **The cost is real.** **With 100 columns, forward selection to 10 features trains roughly 955 models.** **On a large dataset that is hours.** **This is why wrapper methods are used on tens of columns, not thousands.**

---

## 13.3 Embedded methods

> **The model selects features as a side effect of training. One training run, no extra cost.**

### Lasso — the coefficients that go to exactly zero

**Lasso is linear regression with a penalty that pushes small coefficients all the way to zero.** A coefficient of exactly zero means the column is not used at all.

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
lasso = Lasso(alpha=0.1).fit(X_scaled, y)

for name, coef in zip(X.columns, lasso.coef_):
    kept = "kept   " if coef != 0 else "DROPPED"
    print(f"{name:<20} coef = {coef:>8.4f}   {kept}")
```

**Output:**

```text
sepal length (cm)    coef =   0.0000   DROPPED
sepal width (cm)     coef =  -0.0000   DROPPED
petal length (cm)    coef =   0.2633   kept
petal width (cm)     coef =   0.4275   kept
```

> **Lasso zeroed two of the four columns by itself, without being told to keep any particular number.**
>
> ⚠️ **Always scale before Lasso.** **The penalty is applied to the raw coefficient size, so an unscaled column with large values gets unfairly punished.**

### Tree importance

**Every tree-based model records how much each feature reduced impurity across all its splits.**

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
importances = pd.Series(forest.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).round(4))
```

**Output:**

```text
petal length (cm)    0.4593
petal width (cm)     0.4179
sepal length (cm)    0.1020
sepal width (cm)     0.0209
```

> **The two petal columns account for 87.7% of the total importance.**
>
> ⚠️ **Two traps with tree importance.** **First, it is biased towards high-cardinality columns** — an ID column can look important because it can split anything. **Second, when two columns are correlated, the forest splits the credit between them arbitrarily**, so a genuinely important feature can look weak simply because a twin is standing beside it.

### Choosing a family

| Situation | Use |
|---|---|
| Thousands of columns, need a fast first cut | **Filter** |
| Tens of columns, accuracy matters most | **Wrapper (RFE)** |
| You are already training a linear or tree model | **Embedded** |
| **In practice** | **Filter to cut it down, then embedded or wrapper on what survives** |

## ✏️ Practice — feature selection

1. Load iris. Compute the variance of each column and apply `VarianceThreshold(0.3)`. **Which column is dropped, and does its low variance actually mean it is useless?**
2. Run `SelectKBest(f_classif, k=2)` and `SelectKBest(chi2, k=2)`. Do they choose the same two columns?
3. Run `RFE` with logistic regression to 2 features. Compare its answer with the filters'.
4. Fit `Lasso(alpha=0.1)` on scaled data. **How many coefficients are exactly zero?** Now try `alpha=0.5` — what changes?
5. **Rank the four columns by forest importance, and explain in one sentence why all four methods agree.**

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       f_classif, chi2, RFE)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print(X.var().round(4).to_dict())                                       # 1
vt = VarianceThreshold(threshold=0.3).fit(X)
print("kept:", list(X.columns[vt.get_support()]))
# sepal width is dropped (variance 0.19). Low variance is a HINT, not
# proof - a column could vary little and still separate the classes
# perfectly. Here the other methods happen to agree, so it is safe.

f = SelectKBest(f_classif, k=2).fit(X, y)                               # 2
c = SelectKBest(chi2, k=2).fit(X, y)
print("f_classif:", list(X.columns[f.get_support()]))
print("chi2     :", list(X.columns[c.get_support()]))
# Both pick petal length and petal width.

r = RFE(LogisticRegression(max_iter=1000), n_features_to_select=2).fit(X, y)  # 3
print("RFE:", list(X.columns[r.support_]))
# Same two columns - but RFE trained a model three times to get there.

Xs = StandardScaler().fit_transform(X)                                  # 4
for a in [0.1, 0.5]:
    la = Lasso(alpha=a).fit(Xs, y)
    print(f"alpha={a}", la.coef_.round(4), " zeros:", (la.coef_ == 0).sum())
# alpha=0.1 -> 2 zeros. alpha=0.5 -> MORE zeros (a stronger penalty
# drops more columns). Turn alpha up far enough and everything goes to
# zero, which is a useless model - alpha is a dial, not a switch.

rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)  # 5
print(pd.Series(rf.feature_importances_,
                index=X.columns).sort_values(ascending=False).round(4))
# petal length > petal width >> sepal length >> sepal width.
# All four methods agree because the signal in iris is genuinely
# concentrated in the petals: the three species overlap heavily in
# sepal measurements and separate cleanly on petal measurements.
```
</details>

---

# 14. Projection methods

> **Instead of choosing among your columns, build new ones — fewer, each a combination of all the originals.**

🧠 **Analogy: photographing a chair.** A chair is a 3-D object. **A photograph is 2-D — you have lost a dimension — but from the right angle you can still tell it is a chair.** **Projection is choosing that angle.**

**The three methods differ in what "right angle" means.**

| Method | Optimises for | Uses labels? | Use it for |
|---|---|---|---|
| **PCA** | **Maximum spread** | **No** | Compression, de-correlation, preprocessing |
| **LDA** | **Maximum class separation** | **Yes** | Classification preprocessing |
| **t-SNE** | **Keeping neighbours together** | No | **Visualisation only** |

---

## 14.1 PCA — Principal Component Analysis

**PCA finds the direction in which the data spreads out most, calls that PC1, then finds the direction of most remaining spread perpendicular to it, calls that PC2, and so on.**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X_scaled = StandardScaler().fit_transform(X)

pca_full = PCA().fit(X_scaled)
print("each component :", (pca_full.explained_variance_ratio_ * 100).round(2))
print("cumulative     :", (np.cumsum(pca_full.explained_variance_ratio_) * 100).round(2))
```

**Output:**

```text
each component : [72.96 22.85  3.67  0.52]
cumulative     : [ 72.96  95.81  99.48 100.  ]
```

> **Read the cumulative row: two components carry 95.81% of all the variation in four columns.** **The last two components together hold 4.2% — mostly noise.**

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("before:", X_scaled.shape, " after:", X_pca.shape)
```

**Output:** `before: (150, 4)  after: (150, 2)`

### ⚠️ Three rules for PCA

1. **Always scale first.** **PCA maximises variance, so an unscaled column measured in large units will dominate PC1 purely because of its units.**
2. **Fit on train only.** `pca.fit(X_train)` then `pca.transform(X_test)`. **Fitting on everything leaks test information into training.**
3. **You lose interpretability.** **`PC1 = 0.52×sepal_length − 0.27×sepal_width + 0.58×petal_length + 0.56×petal_width` is not a sentence you can say to a customer.**

### How many components to keep

```python
# illustrative: a syntax reference, not runnable as written.
PCA(n_components=2)      # a fixed number
PCA(n_components=0.95)   # enough components to keep 95% of the variance
```

> **`n_components=0.95` is usually the better choice** — you state how much information you are willing to lose, and let PCA work out the number.

---

## 14.2 LDA — Linear Discriminant Analysis

**PCA does not know the labels. LDA does — and that changes everything.**

**LDA looks for the direction that pushes the class means as far apart as possible while keeping each class tightly packed.**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

print("separation kept:", (lda.explained_variance_ratio_ * 100).round(2))
```

**Output:**

```text
separation kept: [99.12  0.88]
```

> **The first LDA component alone captures 99.12% of the separation between the three species.** **One number per flower is nearly enough to classify it.**

⚠️ **LDA has a hard limit:** **at most `n_classes − 1` components.** With 3 species you can have at most 2. **With a binary target you get exactly one** — no matter how many columns you started with.

| | PCA | LDA |
|---|---|---|
| Uses `y` | No | **Yes** |
| Max components | Number of features | **`n_classes − 1`** |
| Best for | General compression | **Classification** |
| Works when there is no target | **Yes** | No |

---

## 14.3 t-SNE — for looking, not for modelling

**t-SNE tries to keep points that were close together in the original space close together in 2-D. It does not care about the global layout at all.**

```python
# needs-download: t-SNE on a large dataset is slow; iris runs in seconds.
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)
```

**Output:** `(150, 2)`

### ⚠️ Four things about t-SNE you must know

1. **There is no `transform()`.** **t-SNE cannot map new data into the same space.** You cannot fit it on train and apply it to test — which means **it can never be a preprocessing step in a model pipeline.**
2. **The distances between clusters are meaningless.** Two clusters drawn far apart are not more different than two drawn close.
3. **Cluster sizes are meaningless.** A big blob is not a more spread-out group.
4. **Change `perplexity` and the picture changes.** **Always look at two or three settings before believing a shape.**

> **Use t-SNE to answer "are there groups in here at all?" — and then use something else to model them.** **UMAP is a modern alternative that is faster and does support `transform()`.**

---

## The three, side by side

![PCA, LDA and t-SNE compared](images/s6-projection-methods.png)

> **Look at what each optimised for.** **PCA spreads the data out but lets two species overlap — it was never told there were species.** **LDA, which was told, pushes the three groups into clean separated bands.** **t-SNE produces three tight islands that look wonderfully convincing — and the distance between those islands means nothing.**

---

# 15. What reduction actually costs

**Reduction is a trade. Here is the trade, measured.**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

for k in [1, 2, 3, 4]:
    kept = SelectKBest(f_classif, k=k).fit_transform(X, y)
    comp = PCA(n_components=k).fit_transform(X_scaled)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    s = cross_val_score(model, kept, y, cv=5).mean()
    p = cross_val_score(model, comp, y, cv=5).mean()
    print(f"k={k}   SelectKBest {s:.4f}    PCA {p:.4f}")
```

**Output:**

```text
k=1   SelectKBest 0.9200    PCA 0.9067
k=2   SelectKBest 0.9667    PCA 0.9067
k=3   SelectKBest 0.9600    PCA 0.9267
k=4   SelectKBest 0.9667    PCA 0.9400
```

![What reduction costs](images/s6-reduction-curve.png)

**Three findings, and none of them is the one people expect:**

> **1. Halving the features cost nothing.** **Two selected columns score 0.9667 — identical to all four.** **Half the data, same accuracy.**
>
> **2. PCA was worse at every single k — including k=4, where nothing was dropped at all.** **At k=4 PCA is a pure rotation, no information lost, and accuracy still fell from 0.9667 to 0.9400.** **Rotating the axes destroyed the axis-aligned splits a tree relies on.**
>
> **3. Three features scored *lower* than two.** **0.9600 against 0.9667.** **Adding `sepal length` back in actively hurt.**

## The lesson

| If you need… | Use |
|---|---|
| **Explainability** | **Selection** — you keep real, nameable columns |
| A tree-based model | **Selection** — PCA fights against how trees split |
| To compress hundreds of correlated columns | **PCA** |
| To feed a distance-based model (KNN, SVM) | **PCA** is often genuinely better |
| To *look* at your data | **t-SNE** or **UMAP**, and nothing else |

> **Do not reach for PCA by reflex.** **On this dataset, the simplest possible method — score each column, keep the top two — beat it every time.**

## ✏️ Practice — projection

1. Scale iris and fit `PCA()`. **How many components do you need for 95% of the variance?**
2. Fit `PCA(n_components=2)` on **unscaled** data and compare the explained variance with the scaled version. **Why do they differ?**
3. Fit LDA with `n_components=2`. **Then try `n_components=3` — what error do you get, and why?**
4. Run t-SNE with `perplexity=5`, `30` and `50`. Plot all three. **Do the clusters stay in the same places?**
5. **Compare 5-fold CV accuracy for: all 4 columns, the 2 best-selected columns, 2 PCA components, and 2 LDA components. Which wins, and does that surprise you?**

<details><summary>Solutions</summary>

```python
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
Xs = StandardScaler().fit_transform(X)

p = PCA().fit(Xs)                                                       # 1
cum = np.cumsum(p.explained_variance_ratio_)
print((cum * 100).round(2))
print("components for 95%:", int(np.argmax(cum >= 0.95)) + 1)     # 2

scaled = PCA(n_components=2).fit(Xs).explained_variance_ratio_          # 2
raw = PCA(n_components=2).fit(X).explained_variance_ratio_
print("scaled  :", (scaled * 100).round(2), " total", round(scaled.sum()*100, 2))
print("unscaled:", (raw * 100).round(2), " total", round(raw.sum()*100, 2))
# Unscaled PCA looks BETTER (higher % in 2 components) but it is
# misleading: petal length has the largest raw variance (3.12 vs 0.19),
# so PC1 is mostly just "petal length in disguise". PCA maximises
# variance, and variance depends on units - so scale first.

l2 = LDA(n_components=2).fit(X, y)                                      # 3
print("LDA 2 ok:", (l2.explained_variance_ratio_ * 100).round(2))
try:
    LDA(n_components=3).fit(X, y)
except ValueError as e:
    print("LDA 3 fails:", e)
# n_components cannot exceed n_classes - 1. With 3 species the maximum
# is 2. This is a mathematical limit, not a tuning choice.

from sklearn.manifold import TSNE                                       # 4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, per in zip(axes, [5, 30, 50]):
    Z = TSNE(n_components=2, random_state=42, perplexity=per).fit_transform(X)
    for c in range(3):
        ax.scatter(Z[y == c, 0], Z[y == c, 1], s=25)
    ax.set_title(f"perplexity = {per}")
plt.close(fig)
# The three species stay separated, but the POSITIONS, ORIENTATION and
# spacing of the clusters change completely with perplexity. Never
# read meaning into where a t-SNE cluster sits.

def cv(Z):                                                              # 5
    return cross_val_score(LogisticRegression(max_iter=1000), Z, y, cv=5).mean()
sel = SelectKBest(f_classif, k=2).fit_transform(X, y)
print("all 4      ", round(cv(Xs), 4))
print("best 2     ", round(cv(StandardScaler().fit_transform(sel)), 4))
print("PCA 2      ", round(cv(PCA(n_components=2).fit_transform(Xs)), 4))
print("LDA 2      ", round(cv(LDA(n_components=2).fit_transform(Xs, y)), 4))
# LDA 2 wins (0.98) - HIGHER than using all four raw columns. That is
# the surprise: LDA saw the labels, so its two components are built
# specifically to separate the classes. PCA 2 is the worst (0.9133),
# because it optimised for spread, which is not the same thing.
```
</details>

---

# ❓ Session 6 — 20 MCQs

**Answer from memory first, then check.**

### Data augmentation

**Q1.** Data augmentation should be applied to…
- (a) The whole dataset before splitting  (b) **The training set only, after splitting**  (c) The test set only  (d) Both sets equally

**Q2.** Random over-sampling creates new minority rows by…
- (a) Interpolating between existing rows  (b) **Copying existing rows**  (c) Generating random numbers  (d) Deleting majority rows

**Q3.** SMOTE creates new minority rows by…
- (a) Copying existing rows  (b) **Interpolating between a row and its nearest neighbours**  (c) Adding Gaussian noise  (d) Rotating the data

**Q4.** After both over-sampling and SMOTE the class counts are identical. So how do you tell them apart?
- (a) You cannot  (b) **By looking at *where* the new points sit — copies land on top of originals, SMOTE points land in the gaps**  (c) By the accuracy  (d) By the row order

**Q5.** Horizontally flipping an image of a handwritten `2` is a bad augmentation because…
- (a) It is too slow  (b) **It destroys the label — a mirrored 2 is not a 2**  (c) It changes the file size  (d) Flipping is never allowed

**Q6.** The safest text augmentation technique is generally…
- (a) Random word deletion  (b) **Back-translation**  (c) Random character swaps  (d) Reversing the sentence

**Q7.** Using class weights instead of SMOTE means…
- (a) You create more rows  (b) **You create no new rows — you tell the model to penalise minority mistakes more**  (c) You delete majority rows  (d) You change the metric

### Feature engineering

**Q8.** A linear model cannot represent `loan ÷ income` because…
- (a) Division is too slow  (b) **A linear model can only add weighted columns, never divide one by another**  (c) The values are too large  (d) It can, actually

**Q9.** `cars["power_per_seat"] = cars["max_power"] / cars["seats"]` produced two infinities because…
- (a) `max_power` was missing  (b) **Two cars have `seats == 0`**  (c) The column was text  (d) The data was unscaled

**Q10.** `pd.cut(vehicle_age, bins=[0, 3, 7, 12, 30])` left five NaNs because…
- (a) Five ages were missing  (b) **The intervals are open on the left, so `vehicle_age == 0` fits into none of them**  (c) The bins overlap  (d) 30 is too small

**Q11.** Adding five engineered features changed the forest's R² from 0.9236 to 0.9201. The right conclusion is…
- (a) The code is broken  (b) **The forest had already found those relationships by splitting twice — but the named features are still more explainable**  (c) Feature engineering never works  (d) Use more features

**Q12.** `ejection_fraction < 40` is a valuable feature mainly because…
- (a) 40 is a round number  (b) **It encodes a clinical threshold the model has no way to discover**  (c) It reduces the column count  (d) It scales the data

**Q13.** `comorbidity_count` turns four yes/no columns into one number from 0 to 4. Its benefit is…
- (a) It saves memory  (b) **It gives the model a single "how ill overall" signal instead of making it discover that those four should be added**  (c) It removes missing values  (d) It is required by scikit-learn

**Q14.** The heart-failure feature code fails on the raw file because…
- (a) The file is corrupt  (b) **The `*_bin` columns do not exist — the binary columns are `"Yes"`/`"No"` text and must be converted first**  (c) There are too many rows  (d) The target is missing

### Feature reduction

**Q15.** `VarianceThreshold` should be used with care because…
- (a) It is slow  (b) **Variance depends on units, so a column measured in metres looks less variable than the same column in centimetres**  (c) It needs the target  (d) It only works on text

**Q16.** `chi2` raises an error on standard-scaled data because…
- (a) The data is too large  (b) **`chi2` requires non-negative values, and standard scaling produces negatives**  (c) It needs integers  (d) Scaling removes the target

**Q17.** The key difference between filter and wrapper methods is…
- (a) Filters are more accurate  (b) **Filters score columns statistically without training a model; wrappers train the model repeatedly on different subsets**  (c) Wrappers only work on text  (d) There is none

**Q18.** Lasso performs feature selection because…
- (a) It ranks columns  (b) **Its penalty drives small coefficients to exactly zero, and a zero coefficient means the column is unused**  (c) It deletes rows  (d) It uses the labels

**Q19.** LDA on a **binary** target can produce at most…
- (a) 2 components  (b) **1 component**  (c) As many as there are features  (d) 3 components

**Q20.** t-SNE can never be used as a preprocessing step in a model pipeline because…
- (a) It is too slow  (b) **It has no `transform()` — it cannot map new data into the same space**  (c) It needs the labels  (d) It only handles 2 columns

<details><summary>Answers</summary>

**A1 — (b) The training set only, after splitting.** **Augmenting before the split copies rows into both sides**, so the model is tested on rows it has already seen. Your score will look wonderful and mean nothing.

**A2 — (b) Copying existing rows.** No new information is created — the same rows just count more.

**A3 — (b) Interpolating between a row and its nearest neighbours.** A new point is placed somewhere along the line between two real minority points.

**A4 — (b) By looking at *where* the new points sit.** **The counts are identical, so counts cannot tell you.** The scatter plot can: copies stack exactly on originals, SMOTE points fill the gaps.

**A5 — (b) It destroys the label.** **This is the one rule augmentation must never break: the transformation has to preserve the answer.** A flipped cat is a cat; a flipped 2 is not a 2.

**A6 — (b) Back-translation.** Translating to another language and back rephrases the sentence while keeping the meaning — and therefore the label. **Random deletion can delete the word "not", which reverses the sentiment.**

**A7 — (b) You create no new rows.** `class_weight="balanced"` changes the *loss function*, not the data. **It is often the better first choice: nothing is invented, and nothing is duplicated.**

**A8 — (b) A linear model can only add weighted columns.** `a×loan + b×income` cannot become `loan/income` for any `a` and `b`. **If you need the ratio, you must build it.**

**A9 — (b) Two cars have `seats == 0`.** And scikit-learn refuses to train on infinity. **Note the same code protects `km_per_year` with `+ 1` — the same protection was simply forgotten for seats.**

**A10 — (b) The intervals are open on the left.** `(0, 3]` excludes 0. **Use `bins=[-1, 3, ...]` or `include_lowest=True`.** Five rows out of 15,411 — easy to miss, which is why you check every new column for NaN.

**A11 — (b).** **Report results like this honestly.** The features did not raise accuracy, but *"driven 15,000 km a year"* is something a dealer understands and `km_driven=150000, vehicle_age=10` is not.

**A12 — (b) It encodes a clinical threshold.** **The model cannot know that 39 and 41 differ meaningfully. A cardiologist can tell it.** This is the strongest case for feature engineering.

**A13 — (b) A single "how ill overall" signal.** Clinicians reason about *how many* conditions a patient has. One column now carries that.

**A14 — (b) The `*_bin` columns do not exist.** This is what the notebook's "do preprocessing first" comment means — convert the Yes/No text to 1/0 and fill the 45 missing values before any feature code runs.

**A15 — (b) Variance depends on units.** **A zero threshold, which drops only constant columns, is always safe. Anything above zero needs scaled data or a deliberate per-column decision.**

**A16 — (b) `chi2` requires non-negative values.** Use `MinMaxScaler`, or switch to `f_classif`.

**A17 — (b).** Filters are cheap and model-agnostic but judge each column alone. **Wrappers measure what actually helps *your* model — at the cost of training it many times.**

**A18 — (b) Its penalty drives coefficients to exactly zero.** On iris it dropped two of four columns without being told how many to keep. **Always scale first**, or the penalty punishes large-valued columns unfairly.

**A19 — (b) 1 component.** The limit is `n_classes − 1`, regardless of how many features you started with.

**A20 — (b) It has no `transform()`.** **You cannot fit it on train and apply it to test.** t-SNE is an exploration tool: use it to see whether groups exist, then model them with something else.
</details>

---

# 🎯 Session 6 — Tasks

## Augmentation

**Task 1 — Build the imbalance yourself.** Take any classification dataset and deliberately cut one class down to 20% of its original size. **Print the class balance before and after.** Then train a model and record the recall on the minority class.

**Task 2 — Three fixes, one problem.** On that same imbalanced data, apply (a) random over-sampling, (b) SMOTE, and (c) `class_weight="balanced"`. **Report accuracy, precision, recall and F1 for all three plus the untouched baseline in a single table. Which would you ship, and why?**

**Task 3 — Prove the leak.** Augment *before* splitting and record the test accuracy. Then augment *after* splitting and record it again. **Explain the gap in two sentences.**

**Task 4 — Where do the points go?** Scatter-plot the minority class after over-sampling and after SMOTE, on two panels. **Write one sentence describing what is visibly different.**

**Task 5 — Label preservation.** List six image augmentations and six text augmentations. **For each, name one dataset where it preserves the label and one where it destroys it.**

## Feature engineering

**Task 6 — Build and check.** Create all eight cardekho features. **Then write a loop that checks every new numeric column for NaN and infinity, and prints a clean report.** Fix anything it finds.

**Task 7 — Measure, do not assume.** Compare R² with and without the engineered features, for a linear model and a forest. **Report the actual numbers, even if they are disappointing.**

**Task 8 — One good feature.** Invent **one** new cardekho feature that is not in the list, justify it in a sentence a car dealer would accept, and measure whether it helps.

**Task 9 — Preprocess first.** Load `heart_failure_raw.csv` and write the preprocessing that must happen before any feature code: convert the four Yes/No columns, fill the 45 missing values. **Print proof that both problems are gone.**

**Task 10 — Clinical thresholds.** Build five threshold flags for the heart data. **For each, find and cite the clinical reference range you used.** *(A model cannot look these up. You can.)*

**Task 11 — Count features.** Build `comorbidity_count` and plot the death rate for each value 0–4. **Does risk rise with the count?**

## Feature reduction

**Task 12 — Four filters, one table.** On iris, run `VarianceThreshold`, `f_classif`, `chi2` and `mutual_info_classif`. **Put all four rankings in one table. Where do they disagree, and why?**

**Task 13 — The price of a wrapper.** Time `SelectKBest` and `RFE` on the same data. **Report both times and the ratio.** Then estimate how long RFE would take on 100 columns.

**Task 14 — Lasso's dial.** Fit Lasso at `alpha` = 0.01, 0.1, 0.5 and 1.0. **Plot the number of surviving features against alpha.** What happens at the top end?

**Task 15 — The correlated-twin trap.** Duplicate one iris column under a new name, then refit the forest. **What happens to the importance of the original, and what does that teach you about trusting tree importance?**

**Task 16 — Scaled and unscaled PCA.** Fit PCA on raw and on scaled iris. **Report the explained variance for two components in both cases, and explain why the unscaled version looks better but is worse.**

**Task 17 — How many components?** Plot the cumulative explained variance curve and mark 90%, 95% and 99%. **State how many components each threshold needs.**

**Task 18 — PCA against LDA.** Project iris to 2-D with both, plot them side by side, and compare 5-fold CV accuracy. **Explain the difference by what each method optimises for.**

**Task 19 — t-SNE is not stable.** Run t-SNE at three perplexities and plot all three. **Write two sentences on what stayed the same and what did not** — and say which of the two you are allowed to draw conclusions from.

**Task 20 — The full pipeline.** On any dataset of your choice: engineer at least three features, select the best `k` with a filter, train a model, and compare against the raw baseline. **Report the honest result — including if reduction cost you accuracy.**

---

## ✅ Session 6 checklist

- [ ] I can explain the difference between over-sampling and SMOTE **by where the points land**
- [ ] I always augment **after** the split, never before
- [ ] I check every engineered column for `NaN` and `inf` the moment I create it
- [ ] I **measure** whether a feature helped rather than assuming it did
- [ ] I can name a filter, a wrapper and an embedded method, and say when each is right
- [ ] I know PCA needs scaling, LDA caps at `n_classes − 1`, and t-SNE has no `transform()`

---

| | |
|---|---|
| **Previous** | [Session 5C — Model Deployment](session-05c-deployment.md) |
| **Next** | [Session 7 — Unsupervised Learning](session-07-unsupervised.md) |
| **Notebook** | [session-06-augmentation-feature-engg-red.ipynb](../notebooks/session-06-augmentation-feature-engg-red.ipynb) |
