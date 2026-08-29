# Session 4 — Introduction to Machine Learning & AI

**What ML and AI are · Real-world applications · Types of Machine Learning · The ML Workflow · ML & AI APIs: Scikit-learn, TensorFlow, PyTorch, Keras**

| | |
|---|---|
| **Notebook** | [session-04-intro-ml-ai.ipynb](../notebooks/session-04-intro-ml-ai.ipynb) |
| **Previous** | [Session 3 — Visualisation & Preprocessing](session-03-eda-preprocessing.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Sessions 1–3 taught you to handle data. This session is where the Machine Learning starts.** Nothing here needs maths beyond arithmetic.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Explain how Machine Learning differs from traditional programming, in one sentence
2. Place AI, ML, Deep Learning and Generative AI correctly inside one another
3. Give a real application for each type of ML
4. Decide whether a problem is regression, classification, clustering or generation
5. Name the ten steps of the ML workflow, in order
6. Say what each of scikit-learn, TensorFlow, PyTorch and Keras is *for*
7. Choose the right framework for a given problem — and justify it

---

## The five topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [What ML actually is](#1-what-machine-learning-actually-is) | Data + Answers → Rules |
| 2 | [AI, ML, DL, GenAI](#2-ai-ml-deep-learning-and-genai) | They nest inside one another |
| 3 | [Types of Machine Learning](#3-types-of-machine-learning) | Number or category is the first decision |
| 4 | [The ML workflow](#4-the-machine-learning-workflow) | Ten steps, and most of the time is step 4 |
| 5 | [ML & AI APIs](#5-ml--ai-apis) | Tabular → scikit-learn. Images/text → PyTorch |

---

# 1. What Machine Learning actually is

🧠 **Analogy: the recipe and the chef.**

**The recipe.** Someone hands you exact instructions: 200g onion, fry 8 minutes, 2 teaspoons chilli. You follow them. If the onions are unusually sweet today, the recipe cannot adapt. **That is traditional programming.**

**The chef.** Someone cooks a thousand curries, tastes each one, and works out for themselves what makes a good curry. Nobody wrote the rules down. **That is Machine Learning.**

```text
Traditional programming:   Data + RULES    ->  Answers
Machine Learning:          Data + ANSWERS  ->  Rules
```

The rules that come out are called a **model**.

## 📘 Examples

**Example 1 — the rule you cannot write**

```python
def is_spam(text):
    if "lottery" in text.lower():
        return True
    if "free money" in text.lower():
        return True
    return False

print(is_spam("You have WON a lottery prize!"))   # True
print(is_spam("Cheap m3dicine, no prescription")) # False - and it IS spam
```

The spammer wrote `m3dicine` to dodge the keyword. **You cannot write a rule for every spelling.** A model trained on examples catches it, because the *shape* of the message is still spam-like.

**Example 2 — the same problem, learned from examples**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = ["win a free prize now", "claim your lottery money",
          "meeting moved to 3pm", "notes from the lecture attached",
          "cheap m3dicine no prescription", "fee payment reminder"]
labels = [1, 1, 0, 0, 1, 0]        # 1 = spam

vec = CountVectorizer()
model = MultinomialNB().fit(vec.fit_transform(emails), labels)

tests = ["free money claim now", "lecture notes for tomorrow"]
print(model.predict(vec.transform(tests)))   # [1 0]
```

**Nobody wrote a rule.** Six examples were enough to separate them.

**Example 3 — what "learning" looks like as a number**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Hours studied -> exam mark. The relationship is roughly mark = 8*hours + 30
hours = np.array([[1], [2], [3], [4], [5], [6]])
marks = np.array([38, 46, 54, 62, 70, 78])

model = LinearRegression().fit(hours, marks)
print(f"learned: mark = {model.coef_[0]:.1f} * hours + {model.intercept_:.1f}")
```

The model was never told the formula. **It recovered it from six examples.**

## ✏️ Practice

1. Write a rule-based function that classifies a number as "small" (<10), "medium" (<100) or "large". Why is this *not* Machine Learning?
2. Train the spam example above on your own six messages. Does it work on a seventh?
3. In the linear example, change `marks` to `[38, 46, 54, 62, 70, 200]`. What happens to the learned slope, and why?
4. Name three tasks a rule could solve better than a model, and say why.
5. Name three tasks where rules fail and a model is needed.

<details><summary>Solutions</summary>

```python
def size(n):                                                  # 1
    return "small" if n < 10 else "medium" if n < 100 else "large"
# YOU wrote the rules. Nothing was learned from data.

# 2 - swap in your own messages and labels, then predict on a new one.

# 3 - the outlier 200 drags the fitted line upward; the slope rises well
#     above 8. A single extreme value moves a linear model a long way.

# 4 - Rules win when the logic is KNOWN and EXACT:
#     tax bands, chess legal-move checking, VAT calculation.
# 5 - Models win when the pattern is real but unwriteable:
#     spam, face recognition, speech-to-text, predicting house prices.
```
</details>

## ❓ MCQs

**Q1.** In one sentence, how does ML differ from traditional programming?
- (a) ML is faster
- (b) Traditional takes data + rules → answers; ML takes data + answers → rules
- (c) ML does not need data
- (d) ML always uses neural networks

**Q2.** What is a "model"?
- (a) The dataset
- (b) The rules the algorithm worked out from the examples
- (c) The programming language
- (d) The accuracy score

**Q3.** A rule-based spam filter misses `m3dicine`. Why does a trained model catch it?
- (a) It has a larger dictionary
- (b) It learned the overall *shape* of spam, not one exact string
- (c) It checks spelling
- (d) It does not — models miss it too

**Q4.** Which of these is best solved with **rules**, not ML?
- (a) Recognising handwriting
- (b) Calculating income tax from published bands
- (c) Predicting house prices
- (d) Detecting fraud

**Q5.** You have data but no labelled answers. What does that rule out?
- (a) Supervised learning
- (b) Unsupervised learning
- (c) All Machine Learning
- (d) Nothing

<details><summary>Answers</summary>

**A1 — (b).** That single swap is the whole idea.

**A2 — (b).** It is the thing you save to a file and reuse to make predictions.

**A3 — (b).** Rules match exact strings; models learn patterns that survive small changes.

**A4 — (b).** Tax bands are **known and exact**. Using ML there would be slower, less accurate and impossible to audit. **If you can write the rule correctly, write the rule.**

**A5 — (a).** Supervised learning needs labelled answers. Unsupervised learning works without them.
</details>

## 🎯 Tasks

**Task 1 — Rules or model?** For ten real systems you use (bank OTP check, Netflix recommendations, spell-check, traffic lights, face unlock, tax filing, spam filter, chess engine, Google Translate, ATM PIN), decide **rules or ML** and give one sentence of justification each. Some are genuinely arguable — say which and why.

**Task 2 — The rule that grew too big.** Write a rule-based classifier for "is this email spam" and keep adding rules until it handles ten test messages. Record how many rules you needed. **Then write one paragraph on what happens when the spammer adapts.**

---

# 2. AI, ML, Deep Learning and GenAI

🧠 **Analogy: Russian dolls.** Each fits inside the one before it.

```text
┌─────────────────────────────────────────────────┐
│ ARTIFICIAL INTELLIGENCE                         │
│ Any machine doing something that looks smart    │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │ MACHINE LEARNING                          │  │
│  │ Learns the rules from data                │  │
│  │                                           │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │ DEEP LEARNING                       │  │  │
│  │  │ ML with many-layered networks       │  │  │
│  │  │                                     │  │  │
│  │  │  ┌───────────────────────────────┐  │  │  │
│  │  │  │ GENERATIVE AI                 │  │  │  │
│  │  │  │ Deep learning that CREATES    │  │  │  │
│  │  │  └───────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

| Layer | What it means | Example |
|---|---|---|
| **AI** | Anything that appears intelligent, **including hand-written rules** | A chess engine following programmed strategy |
| **ML** | Learns the rules from data | Spam filter, loan approval |
| **DL** | ML using neural networks with many layers | Face recognition, speech-to-text |
| **GenAI** | Deep learning that produces **new** content | ChatGPT, Stable Diffusion |

> **The catch that surprises people:** a rule-based chess engine is AI but **not** ML. AI is the older, wider word.

## 📘 Examples

**Example 1 — the same task, at each layer**

| Task | Approach | Which layer |
|---|---|---|
| "Reject loans where credit score < 600" | A written rule | AI (not ML) |
| "Learn from 10,000 past decisions" | Random Forest | ML |
| "Read the scanned application form" | Convolutional network | Deep Learning |
| "Explain the rejection in plain English" | An LLM | Generative AI |

**Example 2 — predictive versus generative**

```python
# PREDICTIVE: choose among options that already exist
model.predict(applicant)      # -> "Approved"  (one of two labels)

# GENERATIVE: produce something that did not exist before
llm.generate("Explain this rejection kindly")
# -> a paragraph nobody wrote in advance
```

**Example 3 — where the real systems sit**

```python
systems = {
    "Thermostat that switches on below 18C": "Rule (AI, not ML)",
    "Netflix recommendations":                "ML",
    "Face unlock on your phone":              "Deep Learning",
    "ChatGPT writing an email":               "Generative AI",
    "Spam filter":                            "ML",
    "Self-driving car vision":                "Deep Learning",
    "Chess engine (classic)":                 "Rule (AI, not ML)",
}
for name, kind in systems.items():
    print(f"{name:<42} {kind}")
```

## ✏️ Practice

Classify each as **Rule / ML / DL / GenAI**, and say why:

1. A calculator app
2. Gmail suggesting the rest of your sentence
3. A bank flagging an unusual transaction
4. Google Photos grouping pictures of the same person
5. A traffic light that changes on a fixed timer

<details><summary>Solutions</summary>

```text
1. Rule       - the arithmetic is exact and known. Not even AI.
2. GenAI      - it produces text that did not exist before.
3. ML         - learned what "unusual" looks like from past transactions.
4. DL         - face recognition from pixels needs a deep network.
5. Rule (AI)  - programmed behaviour, nothing learned from data.
```
</details>

## ❓ MCQs

**Q1.** Which statement is correct?
- (a) AI is a subset of ML
- (b) ML is a subset of AI
- (c) They are the same
- (d) DL contains ML

**Q2.** A chess engine that follows programmed strategy is…
- (a) Machine Learning  (b) AI, but not ML  (c) Deep Learning  (d) Not AI at all

**Q3.** What makes Generative AI *generative*?
- (a) It runs on a GPU
- (b) It produces new content rather than choosing among existing options
- (c) It uses more data
- (d) It is always correct

**Q4.** Face recognition from raw pixels is usually which layer?
- (a) Rules  (b) Classical ML  (c) Deep Learning  (d) Generative AI

**Q5.** Which is the **oldest and widest** of the four terms?
- (a) Generative AI  (b) Deep Learning  (c) Machine Learning  (d) Artificial Intelligence

<details><summary>Answers</summary>

**A1 — (b).** ML sits inside AI, DL inside ML, GenAI inside DL.

**A2 — (b).** It appears intelligent, but nothing was learned from data. **AI is the wider, older word.**

**A3 — (b).** Predictive models pick a label from a fixed set; generative models produce something new.

**A4 — (c).** Learning from raw pixels is what deep networks are for.

**A5 — (d) Artificial Intelligence.** The term dates from 1956.
</details>

## 🎯 Tasks

**Task 1 — Map your phone.** List ten features on your phone and place each in the diagram. For any you are unsure about, write down **what you would need to know** to decide — that uncertainty is the useful part.

**Task 2 — The boundary cases.** Find three systems that are genuinely hard to classify, and argue both sides for each. *Is autocomplete on your keyboard ML or GenAI? Is a recommendation engine that uses a hand-tuned formula ML?*

---

# 3. Types of Machine Learning

| Type | You give it | It learns to | Example here |
|---|---|---|---|
| **Supervised** | Inputs **and** correct answers | Predict the answer for new inputs | Loan approval, salary |
| **Unsupervised** | Inputs only, no answers | Find structure and groups | Customer segments |
| **Reinforcement** | An environment and a reward | Act to maximise reward | Game playing, robotics |
| **Generative** | Huge amounts of text/images | Produce new content | Gemini, Llama |

**Supervised splits in two:**

- **Regression** — the answer is a **number**. *How much will this car sell for?*
- **Classification** — the answer is a **category**. *Will this loan be approved: yes or no?*

> **Decide whether your target is a number or a category and you have made the single most important modelling decision.** It determines which models you can use, which metrics you must report, and how you evaluate everything.

## 📘 Examples

**Example 1 — supervised: labels given**

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

# REGRESSION - the answer is a number
LinearRegression().fit(X_train, y_price)      # y_price = 450000, 320000, ...

# CLASSIFICATION - the answer is a category
LogisticRegression().fit(X_train, y_approved) # y_approved = 1, 0, 1, ...
```

**Example 2 — unsupervised: no labels at all**

```python
from sklearn.cluster import KMeans

# Notice: no y. There are no answers to learn from.
groups = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
print(groups[:10])   # [2 0 1 1 0 ...] - group numbers it invented
```

**Example 3 — deciding, on real questions**

```python
questions = [
    ("How much will this house sell for?",        "Supervised - Regression"),
    ("Will this customer churn?",                 "Supervised - Classification"),
    ("What natural groups exist in our customers?","Unsupervised - Clustering"),
    ("Write a product description",               "Generative"),
    ("How many visitors tomorrow?",               "Supervised - Regression"),
    ("Which of 5 plans will they choose?",        "Supervised - Classification"),
]
for q, a in questions:
    print(f"{q:<45} {a}")
```

## ✏️ Practice

Number (**N**) or category (**C**) — or neither?

1. How much will this used car sell for?
2. Will this student pass the semester?
3. What natural groups exist among our customers?
4. How many minutes until the bus arrives?
5. Write a summary of this article.

<details><summary>Solutions</summary>

```text
1. N - regression
2. C - classification
3. Neither - unsupervised clustering; there is no answer to predict
4. N - regression
5. Neither - generative; the output is new content, not a label or number
```
</details>

## ❓ MCQs

**Q1.** You have inputs but **no** answers. Which type of learning?
- (a) Supervised  (b) Unsupervised  (c) Reinforcement  (d) None is possible

**Q2.** Predicting tomorrow's temperature is…
- (a) Classification  (b) Regression  (c) Clustering  (d) Generation

**Q3.** What is the first and most important modelling decision?
- (a) Which library to use
- (b) Whether the target is a number or a category
- (c) How many rows you have
- (d) Which scaler to apply

**Q4.** Customer segmentation with no predefined groups is…
- (a) Supervised classification  (b) Unsupervised clustering  (c) Regression  (d) Reinforcement

**Q5.** Which type learns by receiving rewards for its actions?
- (a) Supervised  (b) Unsupervised  (c) Reinforcement  (d) Generative

<details><summary>Answers</summary>

**A1 — (b) Unsupervised.** It finds structure without an answer key.

**A2 — (b) Regression.** Temperature is a number.

**A3 — (b).** It determines your models, your metrics and your whole evaluation.

**A4 — (b).** Nobody told the algorithm what the segments are — it discovered them.

**A5 — (c) Reinforcement learning.** Used for game playing and robotics; not covered further in this course.
</details>

## 🎯 Tasks

**Task 1 — Twenty questions.** Write twenty questions your own college could ask of its data. Classify each as regression, classification, clustering or generation, and for each say **what data you would need**. Mark any where you are unsure — several will be genuinely ambiguous.

**Task 2 — The ambiguous target.** *"What rating out of 5 will this user give?"* can be modelled as regression (a number 1–5) or classification (five categories). Build **both** on the same data, compare, and write a paragraph on which you would ship and why. **Real teams argue about exactly this.**

---

# 4. The Machine Learning workflow

Every project — including your capstone — follows these steps in this order.

```text
1. Define the problem   -> Number or category?
2. Collect data         -> Where from? Am I allowed to use it?
3. Explore (EDA)        -> shape, info, missing, target balance
4. Preprocess           -> clean, encode, split, scale        <- most of your time
5. Choose a model       -> start with the simplest that could work
6. Train                -> model.fit(X_train, y_train)
7. Evaluate             -> on data the model has never seen
8. Improve              -> features, cross-validation, tuning
9. Deploy               -> a Streamlit app, an API
10. Monitor             -> does it still work on next month's data?
```

> **Roughly 70% of real ML work is steps 2 to 4.** Nobody puts that in the demo video.

## 📘 Examples

**Example 1 — the four methods that never change**

```python
model = SomeModel(...)         # 1. create, with settings
model.fit(X_train, y_train)    # 2. learn from training data
model.predict(X_test)          # 3. predict on new data
model.score(X_test, y_test)    # 4. a quick default score
```

Learn these four names once and **every** model in scikit-learn works the same way. Classifiers add `model.predict_proba(X)` for confidence rather than just a decision.

**Example 2 — the whole workflow, end to end**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv")          # 2. collect
print(df.shape, df["loan_status"].value_counts().to_dict())   # 3. explore

df = df.dropna().reset_index(drop=True)               # 4. preprocess
for c in df.select_dtypes(include="object").columns:
    df[c] = LabelEncoder().fit_transform(df[c])

X, y = df.drop(columns=["loan_status"]), df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)   # 5
model.fit(X_train, y_train)                                        # 6
print(classification_report(y_test, model.predict(X_test)))        # 7
```

**Example 3 — always start with a baseline**

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("baseline:", round(baseline.score(X_test, y_test), 4))
print("model   :", round(model.score(X_test, y_test), 4))
```

**If your model cannot clearly beat a model that always guesses the majority class, it has learned nothing.** That check costs three lines.

## ✏️ Practice

1. List the ten workflow steps from memory. Check against the list above.
2. Which step usually takes the most time, and why?
3. Run the end-to-end example. What accuracy do you get?
4. Add a `DummyClassifier` baseline. By how much does the real model beat it?
5. What would you check in step 10 (monitor) that you cannot check in step 7?

<details><summary>Solutions</summary>

```python
# 1, 2 - Preprocessing (step 4). Real data is messy; roughly 70% of the work.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in df.select_dtypes(include="object").columns:
    df[c] = LabelEncoder().fit_transform(df[c])
X, y = df.drop(columns=["loan_status"]), df["loan_status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

m = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
b = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
print("model   :", round(m.score(Xte, yte), 4))          # 3
print("baseline:", round(b.score(Xte, yte), 4))          # 4
print("gain    :", round(m.score(Xte, yte) - b.score(Xte, yte), 4))

# 5 - whether the WORLD changed: new customer types, shifted price ranges,
#     a changed form field. Step 7 only tests today's held-out data.
```
</details>

## ❓ MCQs

**Q1.** Which workflow step usually takes the most time?
- (a) Choosing a model  (b) Preprocessing  (c) Training  (d) Deploying

**Q2.** Which four methods does almost every scikit-learn model share?
- (a) `open`, `read`, `write`, `close`
- (b) `create`, `train`, `test`, `deploy`
- (c) constructor, `fit`, `predict`, `score`
- (d) `load`, `clean`, `plot`, `save`

**Q3.** Why compute a `DummyClassifier` baseline?
- (a) It is faster than a real model
- (b) It gives the score any real model must beat
- (c) scikit-learn requires it
- (d) It replaces cross-validation

**Q4.** Evaluation must be done on…
- (a) The training data
- (b) Data the model has never seen
- (c) All the data at once
- (d) A random sample of the training data

**Q5.** What does step 10, *monitoring*, catch that step 7 cannot?
- (a) Syntax errors
- (b) The world changing after deployment — new data unlike the training data
- (c) Overfitting
- (d) Missing values

<details><summary>Answers</summary>

**A1 — (b) Preprocessing.** Roughly 70% of real work, and where most projects succeed or fail.

**A2 — (c).** Learn them once and every model works the same way.

**A3 — (b).** If you cannot clearly beat "always guess the majority class", you have learned nothing.

**A4 — (b).** Otherwise you are measuring memorisation, not learning.

**A5 — (b).** Your test set is from the same period as your training set. Next year's customers may differ.
</details>

## 🎯 Tasks

**Task 1 — Walk the whole workflow.** Pick a dataset from [`datasets/`](../../../datasets/) nobody used in class and go through all ten steps, writing **one paragraph per step** — including step 10, where you say what you would monitor and how you would notice it going wrong.

**Task 2 — The baseline habit.** For three different datasets, compute the `DummyClassifier` baseline **before** training anything. Record baseline, your model's score, and the gain. **Was there a dataset where the gain was disappointingly small? What would you do about it?**

---

# 5. ML & AI APIs

You do not implement algorithms from scratch. You use libraries. **Choosing the right one is a real decision.**

🧠 **Analogy: choosing a vehicle.** A bicycle for the corner shop. A van for moving house. A truck for freight. **All three "move things", and picking the truck for a loaf of bread is a mistake.**

| Library | Built for | Use it when | Not for |
|---|---|---|---|
| **scikit-learn** | Classical ML on **tables** | Rows and columns, under ~1M rows | Images, text, deep networks |
| **TensorFlow** | Deep learning at production scale | Large deployments, mobile, TF ecosystem | Small tabular problems |
| **Keras** | A friendly **front end** for TensorFlow | Building networks quickly and readably | Very unusual architectures |
| **PyTorch** | Deep learning, research-first | Most new deep learning work; full control | Simple tabular problems |

> **Keras is not a competitor to TensorFlow — it runs on top of it.** You write Keras; TensorFlow does the work underneath. Since Keras 3 it can also run on PyTorch and JAX.

## 📘 Examples

**Example 1 — the same task in scikit-learn**

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

Three lines. **For a table of numbers, this is almost always the right choice.**

**Example 2 — the same shape in Keras**

```python
# Not run here - shown for comparison
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(13,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

More code — but **you control every layer**, which matters for images and text.

**Example 3 — and in PyTorch**

```python
# Not run here - shown for comparison
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(13, 16), nn.ReLU(),
    nn.Linear(16, 8),  nn.ReLU(),
    nn.Linear(8, 1),   nn.Sigmoid(),
)
# ...and you write the training loop yourself
```

**Most verbose, most control.** The training loop being explicit is exactly why researchers prefer it.

## How to choose

```text
Is your data a table of rows and columns?
   YES -> scikit-learn.  Stop here.
   NO  |
       Images, audio, video, or text?
          YES -> a deep learning framework
                    Learning / prototyping -> Keras
                    Research / full control -> PyTorch
                    Large production deployment -> TensorFlow
```

> ⚠️ **On tabular data a Random Forest usually beats a neural network and trains in seconds.** Do not reach for deep learning because it sounds more advanced.

## ✏️ Practice

1. Which library for predicting house prices from 12 numeric columns?
2. Which for classifying 50,000 photographs?
3. What is the relationship between Keras and TensorFlow?
4. Train an `MLPClassifier` on the loan data. How does it compare with a Random Forest?
5. Name one reason a researcher might prefer PyTorch over Keras.

<details><summary>Solutions</summary>

```python
# 1 - scikit-learn. A table of numbers.
# 2 - a deep learning framework: Keras to prototype, PyTorch for control.
# 3 - Keras is a high-level front end; TensorFlow does the work underneath.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in df.select_dtypes(include="object").columns:
    df[c] = LabelEncoder().fit_transform(df[c])
X, y = df.drop(columns=["loan_status"]), df["loan_status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
s = StandardScaler(); Xtr_s, Xte_s = s.fit_transform(Xtr), s.transform(Xte)

nn = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
print("MLP          :", round(nn.fit(Xtr_s, ytr).score(Xte_s, yte), 4))       # 4
rf = RandomForestClassifier(n_estimators=200, random_state=42)
print("RandomForest :", round(rf.fit(Xtr, ytr).score(Xte, yte), 4))
# The forest usually wins on tabular data - and trains far faster.

# 5 - PyTorch's training loop is explicit, so you can change any step of it.
```
</details>

## ❓ MCQs

**Q1.** You have a table of 20 numeric columns and 50,000 rows. Which library?
- (a) TensorFlow  (b) PyTorch  (c) scikit-learn  (d) Keras

**Q2.** What is the relationship between Keras and TensorFlow?
- (a) Competitors
- (b) Keras is a high-level front end that runs on top of TensorFlow
- (c) TensorFlow runs on top of Keras
- (d) They are unrelated

**Q3.** Why might a researcher prefer PyTorch?
- (a) It is the only one that supports GPUs
- (b) The training loop is explicit, so every step can be changed
- (c) It requires less code than Keras
- (d) It is the only one that supports images

**Q4.** On tabular data, a neural network versus a Random Forest usually…
- (a) Wins clearly
- (b) Loses, and takes far longer to train
- (c) Is identical
- (d) Cannot be used at all

**Q5.** Which library would you use to deploy a deep model to a mobile phone?
- (a) scikit-learn  (b) TensorFlow  (c) NumPy  (d) Pandas

<details><summary>Answers</summary>

**A1 — (c) scikit-learn.** Rows and columns under about a million: stop there.

**A2 — (b).** You write Keras; TensorFlow does the work. Since Keras 3 it can also run on PyTorch and JAX.

**A3 — (b).** Full control over each step is exactly what research needs.

**A4 — (b).** Do not reach for deep learning because it sounds more advanced.

**A5 — (b) TensorFlow**, via TensorFlow Lite — its deployment ecosystem is its main strength.
</details>

## 🎯 Tasks

**Task 1 — The framework decision memo.** For three scenarios — a hospital predicting readmission from 30 tabular columns; a startup classifying 200,000 product photos; a research group trying a new attention mechanism — recommend a library and justify it in a paragraph each. **Name the one fact that would change your mind.**

**Task 2 — Same problem, two libraries.** Solve the loan problem with `MLPClassifier` **and** `RandomForestClassifier`. Compare accuracy, training time, and lines of code. **Which would you ship, and why?** Include a sentence on what you gave up.

---

# ✅ Before you move on

- [ ] I can state the ML-versus-programming difference in one sentence
- [ ] I can place AI, ML, DL and GenAI correctly inside one another
- [ ] I know a rule-based chess engine is AI but not ML
- [ ] I can decide regression / classification / clustering / generation from a question
- [ ] I can list the ten workflow steps, and know which takes longest
- [ ] I always compute a baseline before celebrating a score
- [ ] I know the four methods every scikit-learn model shares
- [ ] I can choose between scikit-learn, Keras, PyTorch and TensorFlow, and justify it

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-04-intro-ml-ai.ipynb) | Every example above, runnable |
| [Teachable Machine](https://teachablemachine.withgoogle.com/) | Train a real image classifier with no code |
