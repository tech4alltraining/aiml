# ML/AI & GenAI Internship

## Student Hands-on Handbook

**Beginner-friendly practical material for the taught modules and the capstone project phase**
**Python | NumPy & Pandas | Visualisation | Machine Learning | Deep Learning | Generative AI | Prompt Engineering | Gemini API | Hugging Face | Streamlit | Responsible AI**

---

## How to use this handbook

This handbook is written for students with little or no previous experience in Machine Learning. It is the book you keep open next to your editor. Work through it in order.

For every practical example:

1. Read the short explanation **before** running the code.
2. Type or paste the code into the filename given above the code block.
3. Run it once **without changing anything**, and check that you get output.
4. Change **one** thing, run it again, and compare the two results.
5. Write down, in one sentence, what changed and why.
6. Only then move on.

Step 4 is where the learning happens. Reading code teaches you nothing; breaking code and fixing it teaches you a lot.

> **Important:** AI models — both the ML models you train and the Generative AI models you call — can be confidently wrong. Always check important results yourself.

---

## Table of contents

1. [Start here: no experience needed](#start-here-no-experience-needed)
2. [The analogy bank](#the-analogy-bank)
3. [What you will be able to do](#what-you-will-be-able-to-do)
4. [Programme map](#programme-map)
5. [The datasets you will use](#the-datasets-you-will-use)
6. [Accounts and software you need](#accounts-and-software-you-need)
7. [Environment setup: the `genai` conda environment](setup-guide.md)
8. [Option B: Google Colab setup](setup-guide.md#google-colab-no-installation-at-all)
9. [Project folder and filenames](#project-folder-and-filenames)
10. [Module 1 — Python refresher and data handling](#module-1--python-refresher-and-data-handling)
11. [Module 2 — Visualisation, preprocessing and supervised learning](#module-2--visualisation-preprocessing-and-supervised-learning)
12. [Module 3 — Feature engineering and model improvement](#module-3--feature-engineering-and-model-improvement)
13. [Module 4 — Deep learning, clustering and Generative AI](#module-4--deep-learning-clustering-and-generative-ai)
14. [Module 5 — Open-source models, Hugging Face and app development](#module-5--open-source-models-hugging-face-and-app-development)
15. [Activity index](#activity-index)
16. [Capstone project guide](#capstone-project-guide)
17. [Responsible AI checklist](#responsible-ai-checklist)
18. [Troubleshooting](#troubleshooting)
19. [Self-check questions](#self-check-questions)
20. [Glossary](#glossary)
21. [Where to go next](#where-to-go-next)

---

# Start here: no experience needed

If you have never trained a model, never called an AI API, and are not sure you remember Python — **you are the reader this handbook was written for.** Nothing here assumes prior knowledge.

## Three habits that will carry you through

**1. Run the code before you understand it.**
It feels backwards, but it works. Run the example, see real output on your screen, *then* read the explanation. The output gives the explanation something to attach to.

**2. Change exactly one thing.**
After every example, change one number, one word, or one setting. Run it again. If you change three things at once and the result is different, you have learned nothing about which change mattered.

**3. Read error messages. All of them.**
Beginners see a red wall of text and panic. Experienced programmers read the **last line first** — that line names the actual problem. Errors are not failure; they are the computer telling you precisely what it needs.

## What a "day" looks like

Each module in this handbook follows the same rhythm, and it always starts with the easiest thing:

```text
Idea in plain words   →  An analogy you already understand
        ↓
A tiny activity       →  Usually on paper, no computer
        ↓
Small code example    →  10-20 lines, runs in seconds
        ↓
Change one thing      →  You break it, you fix it
        ↓
Bigger example        →  Real dataset, real result
        ↓
Exit task             →  Proof you can do it alone
```

Do not skip the paper activities because they look too easy. They are where the idea actually lands.

## The symbols used in this handbook

| Symbol | Meaning |
|---|---|
| 🧠 **Analogy** | An everyday comparison for the idea being introduced |
| ✏️ **Activity** | Something for you to do |
| 🔁 **Change one thing** | A small edit to make, to see what breaks |
| ⚠️ **Watch out** | A mistake beginners commonly make |
| ✅ **Check yourself** | A question to answer before moving on |

## If you get stuck

1. Read the **last line** of the error message.
2. Check that your terminal prompt shows `(genai)`.
3. Look in [Troubleshooting](#troubleshooting) — the ten most common errors are listed there with fixes.
4. Ask. Being stuck for twenty minutes is learning; being stuck for two hours is waste.

---

# The analogy bank

Every technical idea in this course has an everyday twin. Come back to this table whenever a term stops making sense.

| Technical idea | 🧠 Everyday analogy |
|---|---|
| **Traditional programming** | Following a recipe someone wrote for you |
| **Machine Learning** | Learning to cook by tasting hundreds of dishes until you work out the recipe yourself |
| **Training data** | The past exam papers you study from |
| **Test data** | The actual exam — questions you have never seen |
| **Model** | The rules you worked out from studying |
| **Label / target** | The answer key |
| **Feature** | A clue you use to make your guess |
| **Feature engineering** | Combining two clues into one better clue — height and weight into BMI |
| **Underfitting** | Not studying enough — you fail both the practice papers and the exam |
| **Overfitting** | Memorising last year's answer sheet — perfect on it, lost in the real exam |
| **Cross-validation** | Sitting five different practice exams instead of one, and averaging your marks |
| **Hyperparameter tuning** | Adjusting oven temperature and time until the cake comes out right |
| **Data leakage** | Studying with the answer key open — your practice score means nothing |
| **Scaling / normalisation** | Converting prices in rupees and weights in kilograms onto a common 0–10 scale so neither shouts louder |
| **Classification** | Sorting laundry into whites, colours and delicates |
| **Regression** | Guessing the price of a house |
| **Clustering** | Sorting a pile of unlabelled photos into groups without being told what the groups are |
| **Precision** | Of the fish in your net, how many are the fish you wanted |
| **Recall** | Of all the fish in the lake, how many you caught |
| **Confusion matrix** | A fire-alarm report: real fires caught, real fires missed, false alarms |
| **Neural network** | A relay race — each runner passes a slightly refined message to the next |
| **Large Language Model** | The world's most well-read autocomplete |
| **Token** | A syllable — the model reads in chunks, not whole words |
| **Context window** | How much of the conversation the model can hold in its head at once |
| **Temperature** | A spice dial: 0 is plain and predictable, 1 is bold and unpredictable |
| **Prompt** | Briefing a brilliant new intern who has no idea what your company does |
| **System instruction** | The intern's job description — it applies to every task, not just today's |
| **Hallucination** | A confident student who did not read the chapter but answers anyway |
| **Grounding / RAG** | Making that student answer with the textbook open in front of them |
| **Guardrail** | The fence at the edge of a cliff path |
| **API** | Ordering food by phone — you do not need to know how the kitchen works |
| **Streamlit** | Turning your Python script into a website without learning web design |
| **ML + GenAI together** | The doctor makes the diagnosis; the receptionist explains it in words you understand |

## ✏️ Activity 0.1 — Match the analogy

Cover the right-hand column. For each of these, say the analogy out loud before you look:

1. Overfitting
2. Precision
3. Temperature
4. Data leakage
5. Clustering

If you can do five of these on Module 1, the rest of the course will feel much easier.

---

# What you will be able to do

By the end of the programme you should be able to:

1. Write Python that loads, cleans and explores a real dataset with NumPy and Pandas.
2. Draw the right chart for the question you are asking, using Matplotlib and Seaborn.
3. Explain the difference between supervised, unsupervised and generative learning.
4. Train regression and classification models with scikit-learn and read their evaluation metrics correctly.
5. Recognise overfitting and underfitting, and fix them with cross-validation and hyperparameter tuning.
6. Engineer and reduce features, and explain why that changed the score.
7. Explain what a Large Language Model is and what it is not.
8. Write structured prompts (zero-shot, one-shot, few-shot, chain-of-thought) and predict which one a task needs.
9. Call the Gemini API from Python and control `temperature`, `top_p` and `top_k`.
10. Run open-source models from the Hugging Face Hub with `pipeline()`.
11. Build and run a Streamlit web app that serves a trained ML model.
12. Combine an ML model with a Generative AI model so that one predicts and the other explains.
13. Apply Responsible AI practice: check for bias, keep secrets out of code, and keep a human in the loop.

---

# Programme map

| Module | Focus | You finish with |
|---|---|---|
| **Module 1** | Machine Learning concepts, Python refresher, NumPy, Pandas, EDA | A notebook that loads a CSV and answers five questions about it |
| **Module 2** | Visualisation, preprocessing, regression, classification, ML workflow, metrics | A trained regression model and a trained classification model, both evaluated |
| **Module 3** | Data augmentation, feature engineering, feature reduction, overfitting, cross-validation, hyperparameter tuning | A tuned model that scores better than your Module 2 model |
| **Module 4** | Deep learning intro, clustering, Generative AI, LLMs, prompt engineering, Gemini API | Working prompts of all four types and your first Gemini API call |
| **Module 5** | Open-source GenAI models, Hugging Face, ML + GenAI integration, Streamlit, capstone planning | A running Streamlit app and a chosen capstone topic |
| **Weeks 2–4** | Capstone project with online reviews | A submitted project and a final presentation |

The tutorials referenced in each module are in the [`tutorials/`](tutorials) folder of this repository.

---

# The datasets you will use

All of these live in the [`datasets/`](../../datasets/) folder of this repository. You will meet them in this order — smallest and simplest first.

| Dataset | Rows | What it holds | Used for | Path |
|---|---:|---|---|---|
| **pre_data** | 12 | Country, age, salary, purchased | Learning preprocessing — small enough to read with your eyes | [`datasets/prepreprocessing/pre_data.csv`](../../datasets/prepreprocessing/pre_data.csv) |
| **iris** | 150 | Flower petal and sepal measurements | Your first classification model | [`datasets/classification/iris.csv`](../../datasets/classification/iris.csv) |
| **salary_data** | 375 | Years of experience, salary | Your first regression model | [`datasets/regression/salary_data.csv`](../../datasets/regression/salary_data.csv) |
| **advertising** | 200 | TV, radio, newspaper spend, sales | Regression with several inputs | [`datasets/regression/advertising.csv`](../../datasets/regression/advertising.csv) |
| **Mall_Customers** | 200 | Age, income, spending score | Clustering | [`datasets/clustering/Mall_Customers.csv`](../../datasets/clustering/Mall_Customers.csv) |
| **titanic** | 891 | Passenger details, survived | Classification with messy real data | [`datasets/classification/archive/titanic.csv`](../../datasets/classification/archive/titanic.csv) |
| **cardekho** | 15,411 | Used car details, selling price | EDA and visualisation | [`datasets/regression/cardekho_dataset.csv`](../../datasets/regression/cardekho_dataset.csv) |
| **loan_data_10k** | 10,000 | Applicant and loan details, approved | The main project dataset — used all week | [`datasets/loan_data_10k.csv`](../../datasets/loan_data_10k.csv) |
| **heart_failure** | 299 | Patient measurements, death event | Classification where recall matters | [`datasets/classification/heart_failure_raw.csv`](../../datasets/classification/heart_failure_raw.csv) |
| **diabetes_prediction** | 100,000 | Health indicators, diabetes | Larger classification, imbalanced classes | [`datasets/classification/diabetes_prediction_dataset.csv`](../../datasets/classification/diabetes_prediction_dataset.csv) |
| **bbc-text** | 2,225 | News articles, category | Text classification, GenAI summarising | [`datasets/nlp/bbc-text.csv`](../../datasets/nlp/bbc-text.csv) |

## Two ways to load a dataset

**If you cloned this repository**, use a relative path from your working folder:

```python
import pandas as pd

df = pd.read_csv("../datasets/classification/iris.csv")
```

**If you are on Colab, or have not cloned it**, load straight from the URL — no download step:

```python
import pandas as pd

BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "classification/iris.csv")
```

The examples in this handbook use the URL form so that they run anywhere. If you have the repository locally, swapping to the file path is faster.

## ✏️ Activity 0.2 — Meet your data

Before Module 1 begins, open **three** of the CSV files above in Excel, Google Sheets, or a text editor. Just look at them. For each one, write down:

1. How many columns are there?
2. Which column looks like the "answer" someone would want to predict?
3. Can you spot anything odd — a blank cell, a strange value, a repeated row?

Do this with your eyes, not with code. A person who has *looked* at their data makes far better decisions later than a person who has only ever run `df.head()`.

---

# Accounts and software you need

Create these **before Module 1**. Do not leave them for the morning of the session.

| What | Why | Where |
|---|---|---|
| Google account | Colab and Google AI Studio | [accounts.google.com](https://accounts.google.com) |
| Gemini API key | Module 4 and Module 5 GenAI code | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Hugging Face account | Module 5 open-source models and Spaces | [huggingface.co/join](https://huggingface.co/join) |
| GitHub account | Submitting the capstone project | [github.com/signup](https://github.com/signup) |
| Miniconda or Anaconda | Running code on your own machine | [docs.conda.io](https://docs.conda.io/projects/miniconda/en/latest/) |
| Visual Studio Code | Editing `.py` files | [code.visualstudio.com](https://code.visualstudio.com) |
| VS Code **Python** extension | Running and debugging Python in VS Code | Install from inside VS Code |

> **About API cost and limits:** Gemini API keys have a free tier with rate limits. If you send the same large prompt in a loop you will hit the limit and your code will start failing. Send prompts deliberately, not in loops, unless you mean to.

---

# Environment setup

**Full installation instructions for Windows, Ubuntu and macOS — using either `venv` or `conda` — are in the [Environment Setup Guide](setup-guide.md).**

Do it before you start Module 1.

## The short version

Whatever your operating system, you end up with an environment named **`genai`**.

**With conda:**

```bash
conda create -n genai python=3.12 -y
```

```bash
conda activate genai
```

**With venv** (Ubuntu/macOS):

```bash
python3 -m venv genai && source genai/bin/activate
```

**With venv** (Windows PowerShell):

```powershell
python -m venv genai
```

```powershell
.\genai\Scripts\Activate.ps1
```

Then, on every platform:

```bash
pip install -r requirements.txt
```

```bash
python check_setup.py
```

> ⚠️ **Your terminal prompt must show `(genai)` before you install or run anything.** If it does not, packages go somewhere else and nothing works. This is the single most common problem in the whole course, and you must re-activate in every new terminal.

## What the guide covers

| Topic | Where |
|---|---|
| Windows, Ubuntu and macOS instructions | [Setup Guide](setup-guide.md) |
| `venv` vs `conda` — which to choose | [Which route?](setup-guide.md#which-route-should-you-take) |
| Every package, and what it is for | [Step 3](setup-guide.md#step-3-install-the-packages) |
| Verifying it worked | [Step 4](setup-guide.md#step-4-verify-the-installation) |
| Jupyter kernel registration | [Step 5](setup-guide.md#step-5-register-the-environment-with-jupyter) |
| VS Code configuration | [Step 6](setup-guide.md#step-6-set-up-vs-code) |
| Google Colab (no installation) | [Colab](setup-guide.md#google-colab-no-installation-at-all) |
| Storing your API key safely | [API keys](setup-guide.md#storing-your-gemini-api-key-safely) |
| The pre-Day-1 checklist | [Checklist](setup-guide.md#setup-checklist) |

---


# Google Colab

**No installation at all.** Colab runs Python in your browser and already has NumPy, Pandas, Matplotlib, Seaborn and scikit-learn installed.

**Every demo in this course is available as a ready-to-run notebook:** see [`notebooks/`](notebooks). Each one has an **Open in Colab** badge — one click and you are running it.

```python
# Colab already has the data libraries. Install the GenAI SDK when you need it:
!pip install -q -U google-genai
```

```python
# Never paste your API key into a cell - it gets saved in the notebook.
# getpass hides what you type.
import os
from getpass import getpass

os.environ["GEMINI_API_KEY"] = getpass("Enter your Gemini API key: ")
```

**What Colab cannot do:** Streamlit apps need a local server. Run everything in [`streamlit-apps-collection.md`](tutorials/apps/streamlit-apps-collection.md) on your own machine.

Full Colab guidance — saving to Drive, uploading data, runtime limits — is in the [Setup Guide](setup-guide.md#google-colab-no-installation-at-all).

---

# Project folder and filenames

Create one folder for the whole programme and keep to these filenames. The instructor will refer to them by name.

```bash
mkdir mlai-genai-internship-work
cd mlai-genai-internship-work
code .
```

```text
mlai-genai-internship-work/
├── .gitignore
├── .env                          # your API key (never committed)
├── check_setup.py                # Module 0
├── day1_pandas_practice.py
├── day2_regression.py
├── day2_classification.py
├── day3_tuning.py
├── day4_first_genai.py
├── day4_prompt_types.py
├── day4_temperature.py
├── day5_huggingface.py
├── loan_app/                     # Streamlit ML app
│   ├── train_model.py
│   ├── app.py
│   ├── rf_model.joblib
│   └── .streamlit/secrets.toml
└── capstone/
```

---

# Module 1 — Python refresher and data handling

**Session 1.1:** Machine Learning concepts and a Python refresher.
**Session 1.2:** NumPy, Pandas, data loading and Exploratory Data Analysis.

## 1.1 What Machine Learning actually is

### 🧠 Analogy: the recipe and the chef

Imagine two ways of getting a good curry.

**The recipe.** Someone hands you exact instructions: 200g onion, fry 8 minutes, add 2 teaspoons of chilli. You follow them. If the onions are unusually sweet today, the recipe does not care — it cannot adapt. **That is traditional programming.**

**The chef.** Someone cooks a thousand curries, tastes each one, and gradually works out *for themselves* what makes a good curry. Nobody wrote the rules down. The chef learned them from examples. **That is Machine Learning.**

Everything in this course follows from that difference.

### The same idea in code

In traditional programming **you** write the rules:

```python
def is_spam(email_text):
    if "lottery" in email_text.lower():
        return True
    if "free money" in email_text.lower():
        return True
    return False
```

This breaks the moment a spammer writes "fr€€ m0ney". You cannot write a rule for every spelling.

In Machine Learning you supply **examples with answers**, and the algorithm finds the rules:

```text
Traditional programming:  Data + Rules   →  Answers
Machine Learning:         Data + Answers →  Rules
```

The "rules" that come out are called a **model**.

### ✏️ Activity 1.1 — Sort the mail (no computer)

Your instructor reads out eight subject lines. For each one, call out **spam** or **not spam**:

```text
1. "You have WON a free iPhone - claim now!!!"
2. "Meeting moved to 3pm, room 204"
3. "URGENT: your account will be closed"
4. "Notes from yesterday's lecture attached"
5. "Congratulations, you are our 1,000,000th visitor"
6. "Re: your internship application"
7. "Cheap m3dicine, no prescription needed"
8. "Fee payment reminder - due Friday"
```

Now answer these three questions as a class:

1. **Nobody taught you these rules. So how did you know?** (You learned from thousands of emails you have seen.)
2. **Write down the rule you used.** Most people cannot. That is the point — you learned a pattern, not a rule you can state.
3. **Which one was hardest?** Number 7 uses `m3dicine` to dodge a keyword filter. A rule-based system misses it. A model trained on examples catches it, because the *shape* of the message is still spam-like.

You have just done, by hand, what a spam classifier does.

### ✏️ Activity 1.2 — Number or category?

For each question below, decide whether the answer is a **number** (regression) or a **category** (classification). Write N or C.

```text
1.  How much will this used car sell for?
2.  Will this student pass the semester?
3.  How many customers will visit tomorrow?
4.  Is this email spam?
5.  What will the temperature be at 6pm?
6.  Which of three plans will this customer choose?
7.  How many minutes until the bus arrives?
8.  Does this X-ray show pneumonia?
9.  What rating out of 5 will this user give?
10. Is this transaction fraudulent?
```

<details>
<summary>Answers</summary>

1 N · 2 C · 3 N · 4 C · 5 N · 6 C · 7 N · 8 C · 9 N (or C if you treat 1–5 as fixed categories) · 10 C

Number 9 is deliberately ambiguous — a rating can be modelled either way, and real projects argue about exactly this. If you spotted that, you are thinking like a data scientist.
</details>

This one decision — number or category — determines which models you can use, which metrics you must report, and how you evaluate everything. Get it right first, every time.

### The three types you will meet

| Type | You give it | It learns to | Example in this course |
|---|---|---|---|
| **Supervised** | Inputs **and** correct answers | Predict the answer for new inputs | Loan approval, salary prediction |
| **Unsupervised** | Inputs only, no answers | Find structure and groups | Mall customer segmentation |
| **Generative** | Huge amounts of text/images | Produce new content | Gemini, Llama, Stable Diffusion |

Supervised learning splits into two:

- **Regression** — the answer is a number. *How much will this car sell for?*
- **Classification** — the answer is a category. *Will this loan be approved: yes or no?*

If you can decide whether your target is a number or a category, you have already made the single most important modelling decision.

### ✏️ Activity 1.3 — Train a model with no code at all

Open [Teachable Machine](https://teachablemachine.withgoogle.com/) and choose **Image Project → Standard image model**.

**Step by step:**

1. Rename Class 1 to `pen` and Class 2 to `no pen`.
2. Click **Webcam** under `pen`, hold a pen up, and hold the record button for about 5 seconds (that captures roughly 30 images).
3. Do the same for `no pen` with an empty hand.
4. Click **Train Model** and wait. This takes about 30 seconds.
5. Hold your pen up in front of the webcam and watch the confidence bars move.

**You have just trained a real image classifier.** Now break it on purpose:

| 🔁 Change one thing | What to watch for |
|---|---|
| Give `pen` 60 images and `no pen` only 5 | The model becomes biased towards `pen` — this is **class imbalance** |
| Test it in a darker corner of the room | Accuracy drops — the model learned your lighting too |
| Add a third class, `phone`, with 30 images | Watch the confidence split three ways |
| Hold up a pencil instead of a pen | Does it generalise, or did it memorise your specific pen? |

**✅ Check yourself:** did you, at any point, write down a rule describing what a pen looks like? No. You gave it examples and it found the rule. That single sentence is the whole of Machine Learning.

## 1.2 Python refresher

You need these six things fluently. Nothing more, for now.

### 🧠 Analogy: the four containers

Python's four collection types are four kinds of container, and choosing the right one is like choosing between a shelf, a sealed box, a labelled drawer and a bag of unique tokens.

| Type | Analogy | Ordered? | Can you change it? | Duplicates? |
|---|---|---|---|---|
| **list** `[ ]` | A shelf of books — you can add, remove, reorder | Yes | Yes | Yes |
| **tuple** `( )` | A sealed box of GPS coordinates — fixed once packed | Yes | **No** | Yes |
| **dict** `{k: v}` | A labelled drawer — you find things by label, not position | Yes | Yes | Keys must be unique |
| **set** `{ }` | A bag of raffle tickets — no order, no duplicates allowed | No | Yes | **No** |

Nine times out of ten in this course, the answer is a **list** or a **dict**.

**`day1_python_refresher.py`**

```python
# 1. Variables and data types
name = "Asha"            # str
age = 21                 # int
score = 87.5             # float
is_enrolled = True       # bool

print(type(name), type(age), type(score), type(is_enrolled))

# 2. Collections
marks = [78, 92, 65, 88]                       # list  - ordered, changeable
coordinates = (10.5, 20.3)                     # tuple - ordered, fixed
student = {"name": "Asha", "marks": 87}        # dict  - key/value pairs
subjects = {"Math", "Physics", "Math"}         # set   - no duplicates

print(marks[0])            # 78   (indexing starts at 0)
print(marks[-1])           # 88   (negative counts from the end)
print(marks[1:3])          # [92, 65]  (slice: start included, stop excluded)
print(student["name"])     # Asha
print(subjects)            # {'Math', 'Physics'} - the duplicate is gone

# 3. Conditionals
average = sum(marks) / len(marks)

if average >= 90:
    grade = "A"
elif average >= 75:
    grade = "B"
else:
    grade = "C"

print(f"Average: {average:.2f}, Grade: {grade}")

# 4. Loops
for mark in marks:
    print("Mark:", mark)

for index, mark in enumerate(marks):
    print(f"Subject {index + 1}: {mark}")

# 5. Functions
def calculate_percentage(obtained, total=100):
    """Return the percentage, rounded to 2 decimal places."""
    return round((obtained / total) * 100, 2)

print(calculate_percentage(87))
print(calculate_percentage(87, total=120))

# 6. List comprehension - you will read this constantly in ML code
squared = [m ** 2 for m in marks]
passed = [m for m in marks if m >= 75]

print(squared)
print(passed)
```

Run it:

```bash
conda activate genai
python day1_python_refresher.py
```

### 🔁 Change one thing

Try each of these, one at a time, and read the result:

| Change | What you should learn |
|---|---|
| In `calculate_percentage`, delete `total=100` from the definition, then run | What a **default argument** does — read the error's last line |
| Change `marks[0]` to `marks[4]` | Python counts from 0, so index 4 does not exist in a 4-item list |
| Change `marks[1:3]` to `marks[1:4]` | A slice **includes** the start and **excludes** the stop |
| Add `"Math"` to `subjects` again and print it | A set silently refuses duplicates |

### ✏️ Activity 1.4 — Fix the broken program

This program is meant to print each student's average and grade. It has **four** bugs. Find and fix them all.

```python
students = {
    "Asha": [78, 92, 65],
    "Ravi": [88, 74, 91],
    "Meera": [95, 89, 97]
}

def average(marks)
    return sum(marks) / len(mark)

for name in students:
    avg = average(students[name])
    if avg > 90:
        grade = "A"
    if avg > 75:
        grade = "B"
    else:
        grade = "C"
    print(name + " scored " + avg + " grade " + grade)
```

<details>
<summary>Hints (look only after ten minutes)</summary>

1. Line 7 is missing a character that every `def` line needs.
2. Line 8 uses a variable name that does not exist — look very closely.
3. The second `if` should be something else, or Meera gets the wrong grade.
4. You cannot join a number to a string with `+`. Use an f-string.
</details>

<details>
<summary>Working version</summary>

```python
students = {
    "Asha": [78, 92, 65],
    "Ravi": [88, 74, 91],
    "Meera": [95, 89, 97]
}

def average(marks):                      # fix 1: missing colon
    return sum(marks) / len(marks)       # fix 2: 'mark' -> 'marks'

for name in students:
    avg = average(students[name])
    if avg > 90:
        grade = "A"
    elif avg > 75:                       # fix 3: 'if' -> 'elif'
        grade = "B"
    else:
        grade = "C"
    print(f"{name} scored {avg:.1f} grade {grade}")   # fix 4: f-string
```
</details>

Debugging is not a separate skill from programming — it *is* programming. Expect to spend more time reading errors than writing new lines.

### Practice

- [W3Schools Python tutorial](https://www.w3schools.com/python/) — work through *Python Lists* to *Python Functions*.
- [Exercise 1.1: Python problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/python-exercises.ipynb)

## 1.3 NumPy: fast arrays

### 🧠 Analogy: the shopping bag and the egg tray

A Python **list** is a shopping bag. You can throw anything in — apples, a book, your keys. Flexible, but to do anything to everything you must pull items out one at a time.

A NumPy **array** is an egg tray. Every slot holds the same kind of thing, in a fixed grid. Because the computer knows every slot is identical, it can act on **all of them at once** instead of looping.

That is why this works:

```python
prices = np.array([100, 200, 300])
prices * 2        # [200 400 600]   - every element doubled
```

...while a plain list does something completely different:

```python
prices = [100, 200, 300]
prices * 2        # [100, 200, 300, 100, 200, 300]  - the bag was duplicated!
```

On a million numbers, the egg tray is roughly fifty times faster than the shopping bag. That is the entire reason NumPy exists.

### The basics

**`day1_numpy_basics.py`**

```python
import numpy as np

# Creating arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Shape of a:", a.shape)      # (5,)     - 5 elements, 1 dimension
print("Shape of b:", b.shape)      # (2, 3)   - 2 rows, 3 columns
print("Data type:", a.dtype)

# Vectorised maths - no loop needed
print(a * 2)          # [ 2  4  6  8 10]
print(a + 10)         # [11 12 13 14 15]
print(a ** 2)         # [ 1  4  9 16 25]

# This is the operation you cannot do with a plain Python list:
# [1, 2, 3] * 2 gives [1, 2, 3, 1, 2, 3], not [2, 4, 6].

# Useful generators
print(np.zeros(3))                 # [0. 0. 0.]
print(np.ones((2, 2)))
print(np.arange(0, 10, 2))         # [0 2 4 6 8]
print(np.linspace(0, 1, 5))        # [0.   0.25 0.5  0.75 1.  ]

# Statistics
data = np.array([23, 45, 12, 67, 34, 89, 21])
print("Mean:  ", data.mean())
print("Median:", np.median(data))
print("Std:   ", data.std().round(2))
print("Min/Max:", data.min(), data.max())

# Boolean masking - the idea behind Pandas filtering
print(data[data > 30])             # [45 67 34 89]
print((data > 30).sum(), "values are above 30")

# Reshaping - you will need this for images and model inputs
matrix = np.arange(12).reshape(3, 4)
print(matrix)
print("Transposed:\n", matrix.T)

# Reproducible random numbers
rng = np.random.default_rng(seed=42)
print(rng.integers(1, 100, size=5))
```

> **Why `seed=42`?** Random number generators are only pseudo-random. Fixing the seed means you and your instructor get the *same* random numbers, so your results can be compared. Almost every ML example you will see sets a seed for this reason.

### Practice

- [NumPy tutorial](https://www.w3schools.com/python/numpy/default.asp)
- [Exercise 1.2: NumPy problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/numpy-exercises.ipynb)

### ✏️ Activity 1.5 — The marks calculator

Write `activity_1_5.py` yourself. Given this array of marks out of 100 for one class:

```python
import numpy as np

marks = np.array([78, 92, 65, 88, 45, 97, 55, 71, 83, 60])
```

Produce each of these using **NumPy only — no `for` loops**:

1. The class average, rounded to 1 decimal place.
2. The highest and lowest mark.
3. How many students scored 75 or above.
4. A list of only the marks below 60.
5. Everyone's mark boosted by 5, but capped at 100. (Hint: `np.minimum`)
6. Each mark converted to a percentage of the highest mark in the class.

<details>
<summary>Solution</summary>

```python
import numpy as np

marks = np.array([78, 92, 65, 88, 45, 97, 55, 71, 83, 60])

print("1. Average      :", round(marks.mean(), 1))
print("2. Highest/lowest:", marks.max(), marks.min())
print("3. Scored 75+   :", (marks >= 75).sum())
print("4. Below 60     :", marks[marks < 60])
print("5. Boosted      :", np.minimum(marks + 5, 100))
print("6. As % of top  :", (marks / marks.max() * 100).round(1))
```
</details>

**Why no loops?** Because in real ML you will be working with a million rows, not ten. The habit of thinking in whole arrays instead of one element at a time is the habit you are building here.

## 1.4 Pandas: tables of data

### 🧠 Analogy: a spreadsheet that takes orders

A `DataFrame` is a spreadsheet — rows, columns, headers. The difference is that instead of clicking and dragging, you give it instructions:

| What you would do in Excel | What you type in Pandas |
|---|---|
| Scroll to the top to check the data | `df.head()` |
| Look at the status bar for the row count | `df.shape` |
| Filter → "Price greater than 100000" | `df[df["price"] > 100000]` |
| Sort descending by Price | `df.sort_values("price", ascending=False)` |
| Insert a PivotTable of average price by fuel | `df.groupby("fuel")["price"].mean()` |
| Find blanks with conditional formatting | `df.isnull().sum()` |
| Add a formula column | `df["new"] = df["a"] / df["b"]` |

Everything you already know how to do in a spreadsheet has a one-line Pandas equivalent. You are not learning a new idea — you are learning new words for an idea you have.

### The basics

**`day1_pandas_practice.py`**

```python
import pandas as pd

# Load directly from a URL - no download step needed
URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/regression/cardekho_dataset.csv")

df = pd.read_csv(URL)

# --- Always run these five lines first on any new dataset ---
print(df.shape)          # (rows, columns)
print(df.head())         # first 5 rows - do the values look sensible?
print(df.info())         # column names, data types, non-null counts
print(df.describe())     # min, max, mean, quartiles for numeric columns
print(df.isnull().sum()) # how many missing values per column

# --- Selecting ---
print(df["selling_price"].head())          # one column  -> Series
print(df[["brand", "selling_price"]].head())  # many columns -> DataFrame
print(df.iloc[0])          # first row, by position
print(df.iloc[0:5, 0:3])   # first 5 rows, first 3 columns

# --- Filtering ---
expensive = df[df["selling_price"] > 1000000]
print("Cars above 10 lakh:", len(expensive))

# Two conditions: use & for AND, | for OR, and wrap each in brackets
recent_petrol = df[(df["vehicle_age"] < 5) & (df["fuel_type"] == "Petrol")]
print("Recent petrol cars:", len(recent_petrol))

# --- Grouping: the most useful single Pandas skill ---
print(df.groupby("fuel_type")["selling_price"].mean().round(0))
print(df.groupby("brand")["selling_price"].agg(["count", "mean", "max"]).head())

# --- Sorting ---
print(df.sort_values("selling_price", ascending=False).head(3))

# --- Handling missing values ---
print("Missing before:", df.isnull().sum().sum())
df_clean = df.dropna()                              # drop rows with any NaN
# or fill instead of dropping:
# df["km_driven"] = df["km_driven"].fillna(df["km_driven"].median())
print("Rows kept:", len(df_clean), "of", len(df))

# --- Creating a new column ---
df["price_lakhs"] = (df["selling_price"] / 100000).round(2)
print(df[["selling_price", "price_lakhs"]].head())

# --- Value counts: how many of each category? ---
print(df["fuel_type"].value_counts())
print(df["fuel_type"].value_counts(normalize=True).round(3))  # as proportions
```

Run it:

```bash
conda activate genai
python day1_pandas_practice.py
```

### ✏️ Activity 1.6 — Twelve rows you can see

Before touching a dataset with 10,000 rows, work with one you can hold in your head. `pre_data.csv` has **twelve rows**. Open it in a text editor first and just read it:

```text
Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
,30,54000,No
Spain,38,61000,No
Germany,40,,Yes
France,35,58000,Yes
Spain,,52000,No
France,48,79000,Yes
Germany,50,83000,No
France,37,67000,Yes
Germany,50,83000,No
Spain,50,1500000,No
```

**Part A — with your eyes only.** Write down every problem you can spot. There are at least four. Take three minutes before reading on.

<details>
<summary>What is wrong with this data</summary>

1. **Row 3 has a missing Country** — the line begins with a comma.
2. **Row 5 has a missing Salary.**
3. **Row 7 has a missing Age.**
4. **Rows 9 and 11 are identical** — Germany, 50, 83000, No. A duplicate.
5. **Row 12 has a salary of 1,500,000** — over 18× the next highest. An outlier, or a typo where someone added zeros.

If you found three of the five, that is a good result for a first attempt.
</details>

**Part B — now confirm it with code.** Write `activity_1_6.py`:

```python
import pandas as pd

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/prepreprocessing/pre_data.csv")
df = pd.read_csv(URL)

print("--- The whole dataset (all 12 rows) ---")
print(df)

print("\n--- Missing values per column ---")
print(df.isnull().sum())

print("\n--- Duplicate rows ---")
print(df[df.duplicated(keep=False)])

print("\n--- Numeric summary: look at max vs 75% ---")
print(df.describe())
```

**✅ Check yourself — two things in the `describe()` output:**

1. Compare `75%` to `max` for Salary. The 75th percentile is **81,000**; the max is **1,500,000**. **A max that is many times the 75th percentile is the signature of an outlier.** You now have a numeric test for something you first spotted by eye.

2. Compare `mean` to `50%` (the median) for Salary. The mean is about **196,091**; the median is **67,000**. One single row dragged the average up by nearly 130,000. This is the concrete reason the next section tells you to fill missing numbers with the **median**, not the mean — and now you have seen it happen rather than been told it.

**Why start this small?** Because on twelve rows you can verify that the code agrees with what you saw. When you later run `df.isnull().sum()` on 100,000 rows you will trust it, because you have checked it once on data you could read.

> **Why median, not mean, for filling missing numbers?** The mean is dragged around by extreme values. If one car in the dataset has 900,000 km on the clock, the mean km is wrong for every other car. The median is not affected by that one value.

## 1.5 Exploratory Data Analysis (EDA)

### 🧠 Analogy: the doctor's check-up

A doctor does not prescribe medicine the moment you walk in. First comes the check-up: temperature, blood pressure, a few questions. Only then a treatment.

EDA is the check-up for your dataset. You look at it, measure it, and find out what is wrong **before** you decide what to do about it. Students who skip EDA and jump straight to `model.fit()` are prescribing medicine without examining the patient — and then wondering why it did not work.

Answer these five questions about every dataset, in this order:

1. **How big is it?** `df.shape`
2. **What is each column, and what type?** `df.info()`
3. **What is missing?** `df.isnull().sum()`
4. **What are the ranges and outliers?** `df.describe()`
5. **What is the target, and is it balanced?** `df["target"].value_counts()`

**`day1_eda.py`**

```python
import pandas as pd

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")

df = pd.read_csv(URL)

print("=== 1. Size ===")
print(df.shape)

print("\n=== 2. Columns and types ===")
df.info()

print("\n=== 3. Missing values ===")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n=== 4. Numeric summary ===")
print(df.describe().T[["mean", "min", "50%", "max"]].round(2))

print("\n=== 5. Target balance ===")
print(df["loan_status"].value_counts())
print(df["loan_status"].value_counts(normalize=True).round(3))

print("\n=== 6. Which numeric columns relate to the target? ===")
correlations = df.corr(numeric_only=True)["loan_status"].sort_values()
print(correlations.round(3))
```

Reading the correlation output: values near `+1` or `-1` mean the column moves with the target; values near `0` mean it does not, on its own. A correlation of `-0.3` is **not** weak in real data — it is worth investigating.

> **Correlation is not causation.** Ice cream sales correlate with drowning deaths. Neither causes the other; hot weather causes both.

### Practice: Dataset 1.2

- Dataset: [cardekho_dataset.csv](https://github.com/tech4alltraining/aiml/blob/main/datasets/regression/cardekho_dataset.csv)
- Notebook: [Notebook 1.3 — EDA](https://colab.research.google.com/drive/1dPXOKWHW2GlQ35-uoi5AlHIMtqtlwftj?usp=sharing)
- [Exercise 1.3: Pandas problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/pandas-exercises.ipynb)

### ✏️ Activity 1.7 — Data detective

Work in pairs. Each pair takes **one** dataset from the table in [The datasets you will use](#the-datasets-you-will-use) — not the loan one, which the whole class shares.

Run the five EDA commands on it, then fill in this card and present it to the class in **60 seconds**:

```text
DATASET NAME : ______________________
ROWS x COLS  : ______ x ______
THE TARGET   : ______________________  (number / category)
MISSING      : ______________________
BIGGEST SURPRISE:
_____________________________________
ONE QUESTION THIS DATA COULD ANSWER:
_____________________________________
```

Hearing eight different datasets described in eight minutes teaches you something no single dataset can: **every dataset is broken in its own particular way.**

### ✅ Module 1 exit task

Using the cardekho dataset, answer in a notebook:

1. How many cars are in the dataset, and how many columns describe each one?
2. Which fuel type has the highest average selling price?
3. Which brand appears most often?
4. Are there missing values? In which columns?
5. What is the price of the most expensive car, and is it plausible or a data-entry error?

Write each answer as **one sentence in plain English**, not as a code output. "The dataset has 15,411 cars described by 14 columns" is an answer. A screenshot of `df.shape` is not.

---

# Module 2 — Visualisation, preprocessing and supervised learning

**Session 2.1:** Data visualisation, preprocessing, regression and classification.
**Session 2.2:** The ML workflow, the scikit-learn API and evaluation metrics.

## 2.1 Choosing the right chart

### 🧠 Analogy: charts are questions, not decoration

Think of a chart the way you think of a camera lens. You do not pick a lens because it looks nice — you pick it because of what you are trying to photograph. A wide lens for a landscape, a macro lens for an insect.

Same with charts. **You do not pick a chart because it looks nice. You pick it because of the question you are asking.** If you cannot say what question your chart answers, delete it.

Pick the chart from the question:

| Your question | Chart | Code |
|---|---|---|
| How is one numeric column spread out? | Histogram | `sns.histplot(df["price"])` |
| Are there outliers in this column? | Box plot | `sns.boxplot(x=df["price"])` |
| Do two numeric columns move together? | Scatter plot | `sns.scatterplot(x="km", y="price", data=df)` |
| How do categories compare? | Bar plot | `sns.barplot(x="fuel", y="price", data=df)` |
| How many of each category? | Count plot | `sns.countplot(x="fuel", data=df)` |
| Which columns relate to which? | Heatmap | `sns.heatmap(df.corr(numeric_only=True), annot=True)` |
| How does something change over time? | Line plot | `sns.lineplot(x="year", y="sales", data=df)` |

**`day2_visualisation.py`**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/regression/cardekho_dataset.csv")
df = pd.read_csv(URL)

# A 2x2 grid of the four charts you will use most
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1. Distribution: where do most prices sit?
sns.histplot(df["selling_price"], bins=40, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of selling price")
axes[0, 0].set_xlabel("Selling price")

# 2. Outliers: is anything absurdly far from the rest?
sns.boxplot(x=df["km_driven"], ax=axes[0, 1])
axes[0, 1].set_title("Kilometres driven - outlier check")

# 3. Relationship: does age push the price down?
sns.scatterplot(x="vehicle_age", y="selling_price",
                hue="fuel_type", data=df, alpha=0.5, ax=axes[1, 0])
axes[1, 0].set_title("Vehicle age vs selling price")

# 4. Comparison: which fuel type sells highest?
sns.barplot(x="fuel_type", y="selling_price", data=df,
            errorbar=None, ax=axes[1, 1])
axes[1, 1].set_title("Average price by fuel type")
axes[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("day2_charts.png", dpi=120)   # saves the figure to a file
plt.show()                                 # opens a window
```

Run it:

```bash
conda activate genai
python day2_visualisation.py
```

> **If no window opens** when you run a `.py` file, the figure was still saved to `day2_charts.png` — open that. In Jupyter and Colab, charts appear inline automatically and you do not need `plt.show()`.

**Read your own histogram:** if the price histogram has a long tail to the right, the data is *right-skewed*. That matters, because a linear regression will be pulled towards those few expensive cars.

### ✏️ Activity 2.1 — The chart chooser

For each question, name the chart you would draw. No code — just the chart name.

```text
1.  Do students who study more hours get better marks?
2.  How many students chose each elective subject?
3.  Is anyone's electricity bill absurdly higher than everyone else's?
4.  Have monthly sales gone up or down over two years?
5.  Which of our five branches has the highest average revenue?
6.  Are most salaries clustered low, or spread evenly?
7.  Which pairs of columns in this table move together?
```

<details>
<summary>Answers</summary>

1 Scatter plot · 2 Count plot · 3 Box plot · 4 Line plot · 5 Bar plot · 6 Histogram · 7 Heatmap

Questions 5 and 2 are the pair people mix up. **Count plot** = how many rows in each category. **Bar plot** = the average of some *other* column, per category.
</details>

### ✏️ Activity 2.2 — Draw it before you code it

Take the four charts your script just produced and open `day2_charts.png`.

For each one, write **one sentence** stating what it tells you. Not what it is — what it *tells you*.

| Chart | ❌ Not an answer | ✅ An answer |
|---|---|---|
| Histogram | "It shows the distribution of price" | "Most cars sell under ₹10 lakh, but a long tail runs up to ₹40 lakh" |
| Box plot | "It shows km_driven" | "A handful of cars have over 300,000 km — those need checking before modelling" |
| Scatter | "Age vs price" | "Price falls sharply for the first 5 years, then flattens out" |
| Bar plot | "Price by fuel type" | "Diesel cars sell for noticeably more on average than petrol" |

**🔁 Change one thing:** in the histogram, change `bins=40` to `bins=5`, then to `bins=200`. Run it each time. With 5 bins the shape disappears; with 200 it becomes noise. The number of bins is a *choice you make*, and it changes the story your chart tells. Charts can mislead — including by accident, including yours.

### Practice

- [Matplotlib tutorial](https://www.w3schools.com/python/matplotlib_intro.asp) · [Seaborn tutorial](https://www.geeksforgeeks.org/python-seaborn-tutorial/)
- [Exercise 2.1: Matplotlib & Seaborn problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/matplotlib-seaborn-exercises.ipynb)

## 2.2 Data preprocessing

### 🧠 Analogy: chopping before cooking

No chef throws a whole unwashed onion into the pan. Before any cooking happens there is peeling, washing, chopping, measuring. It is unglamorous and it takes longer than the cooking itself.

Preprocessing is that. Models cannot handle the word "France", cannot handle a blank cell, and get confused when one column runs 0–1 and another runs 0–1,500,000. **Roughly 70% of real ML work is preprocessing.** Nobody puts it in the demo video, and it is where most projects succeed or fail.

### The problems and their fixes

| Problem | Fix | scikit-learn / Pandas |
|---|---|---|
| Missing values | Drop or fill them | `dropna()`, `fillna()`, `SimpleImputer` |
| Text categories | Turn into numbers | `LabelEncoder`, `OneHotEncoder`, `pd.get_dummies()` |
| Very different scales | Rescale to a common range | `StandardScaler`, `MinMaxScaler` |
| Extreme outliers | Cap or remove | IQR rule, domain judgement |
| Unbalanced classes | Resample or reweight | `class_weight="balanced"` |

**`day2_preprocessing.py`**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL)

print("Before:", df.shape)

# --- 1. Missing values ---
print("Missing per column:\n", df.isnull().sum()[df.isnull().sum() > 0])
df = df.dropna().reset_index(drop=True)     # only 3 rows here, safe to drop
print("After dropping:", df.shape)

# --- 2. Encode text columns ---
# LabelEncoder assigns 0, 1, 2 ... in ALPHABETICAL order of the values.
categorical = ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]

for column in categorical:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print(f"{column}: {mapping}")

# --- 3. Split BEFORE scaling ---
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # keep 20% completely unseen
    random_state=42,      # same split every run
    stratify=y            # keep the class balance in both halves
)

# --- 4. Scale: fit on TRAIN only, then apply to both ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform on train
X_test_scaled = scaler.transform(X_test)         # transform ONLY on test

print("Train:", X_train_scaled.shape, " Test:", X_test_scaled.shape)
print("Train mean after scaling (should be ~0):",
      X_train_scaled.mean().round(6))
```

### ✏️ Activity 2.3 — Preprocess twelve rows by hand

Go back to `pre_data.csv` from Activity 1.6 — the twelve-row one. Fix every problem in it, one step at a time, checking after each step.

**`activity_2_3.py`**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/prepreprocessing/pre_data.csv")
df = pd.read_csv(URL)

print("=== STEP 0: the raw data ===")
print(df)

# --- STEP 1: remove the duplicate row ---
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n=== STEP 1: after removing duplicates -> {len(df)} rows ===")

# --- STEP 2: fill the missing Country with the most common value ---
most_common_country = df["Country"].mode()[0]
df["Country"] = df["Country"].fillna(most_common_country)
print(f"=== STEP 2: filled missing Country with '{most_common_country}' ===")

# --- STEP 3: fill missing numbers with the MEDIAN (not the mean!) ---
for column in ["Age", "Salary"]:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)
    print(f"=== STEP 3: filled missing {column} with {median_value} ===")

# --- STEP 4: deal with the outlier salary ---
# Cap anything above the 95th percentile. This is called "winsorising".
cap = df["Salary"].quantile(0.95)
print(f"=== STEP 4: capping salaries above {cap:,.0f} ===")
df["Salary"] = df["Salary"].clip(upper=cap)

# --- STEP 5: turn text into numbers ---
df["Purchased"] = LabelEncoder().fit_transform(df["Purchased"])   # No=0, Yes=1
df = pd.get_dummies(df, columns=["Country"], dtype=int)  # one column per country

print("\n=== STEP 5: after encoding ===")
print(df)

# --- STEP 6: put Age and Salary on the same 0-1 scale ---
scaler = MinMaxScaler()
df[["Age", "Salary"]] = scaler.fit_transform(df[["Age", "Salary"]])

print("\n=== STEP 6: fully preprocessed and ready for a model ===")
print(df.round(3))
```

**✅ Check yourself after running it:**

1. How many rows are left? (11 — the duplicate went.)
2. Are there any `NaN` left? Add `print(df.isnull().sum().sum())` — it should print `0`.
3. Why did `get_dummies` turn one Country column into **three**? Because Germany, France and Spain have no natural order. Numbering them 0, 1, 2 would tell a linear model that Spain is "more" than France, which is nonsense. One column each, holding 0 or 1, says only "is it this country: yes or no".
4. After step 6, what is the smallest value in the Age column, and the largest? (0.0 and 1.0 — that is what MinMaxScaler does.)
5. **Look hard at the scaled Salary column.** Almost every value sits between 0.00 and 0.05, with one lonely 1.00. Capping at the 95th percentile on eleven rows barely touched the outlier — it is still 10× everything else, so after scaling it flattens all the real variation into nothing. **A preprocessing step that runs without error is not the same as a preprocessing step that worked.** Always look at the output. (What would fix it here? A tighter cap, dropping the row, or scaling `log(Salary)` instead.)

**🔁 Change one thing:** in Step 3, swap `.median()` for `.mean()`, run again, and look at the filled Salary. Because the 1,500,000 outlier is still present at that point, the mean fills in a wildly wrong value. You proved this numerically in Activity 1.6; now you have seen it damage a real pipeline.

### 🧠 Analogy: data leakage is studying with the answer key

Imagine practising for an exam with the answer key open beside you. You score 100% on every practice paper. You walk into the real exam confident — and fail.

Your practice score was meaningless because information from the answers had leaked into your practice.

That is **data leakage**, and the most common way students cause it is by scaling *before* splitting. When `fit_transform` sees the whole dataset, it learns the mean and range of the test rows too — and those numbers get baked into the training data. Your test score comes out beautifully high and completely fake.

> **The single most important rule on this page:** fit the scaler on the **training data only**. If you fit it on the whole dataset, information from the test set leaks into training, your test score becomes falsely high, and the model disappoints in production. This mistake is called **data leakage** and it is the most common serious error made by beginners.

**When to use which encoder:**

- `LabelEncoder` → categories with a natural order (`Low`, `Medium`, `High`), or tree-based models like Random Forest which do not care about the numeric ordering.
- `OneHotEncoder` / `pd.get_dummies()` → categories with no order (`Petrol`, `Diesel`, `CNG`) when using linear models, which would otherwise wrongly assume `CNG(0) < Diesel(1) < Petrol(2)`.

### Practice

- Dataset: [pre_data.csv](https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/prepreprocessing/pre_data.csv)
- Notebook: [Notebook 2.1 — Preprocessing](https://colab.research.google.com/drive/1MsdDPnB3WE3qZZUcZU7QRkJcl0kddz6e?usp=sharing)

## 2.3 Regression: predicting a number

### 🧠 Analogy: the straight road through scattered houses

Picture a scatter of dots on a page — each dot is one person: years of experience across, salary up.

Now lay a ruler on the page and slide it until it sits as fairly as possible through the middle of the cloud. Some dots end up above your line, some below, but overall it is as close to all of them as you can get.

**That line is linear regression.** The whole algorithm is: find the line where the total distance to all the dots is smallest. Once you have it, predicting is trivial — go across to 5.5 years, go up to the line, read off the salary.

The line has two numbers that describe it completely:

```text
Salary  =  slope × Experience  +  intercept
           ↑                      ↑
    "each extra year         "what you'd earn
     is worth this much"      with zero years"
```

When you run the code below, it prints exactly those two numbers. That is the entire model — two numbers.

### ✏️ Activity 2.4 — Beat the model

**Before running any code**, look at these five rows from the salary dataset:

```text
Experience   Salary
   5.0       90,000
   3.0       65,000
  15.0      150,000
   7.0       60,000
  20.0      200,000
```

On paper, answer:

1. Roughly how much does **one extra year** of experience seem to be worth? (Divide a salary difference by an experience difference.)
2. Using your number, what would you predict for someone with **5.5 years**?
3. Write your prediction down. Commit to it before you see the model's.

Now run `day2_regression.py`. It prints the equation the model learned and its prediction for 5.5 years.

**✅ Check yourself:** how close was your guess? Most people land within about 15% of the model. That is the point — **the model is not doing anything mystical.** It is doing the same arithmetic you just did, only over all 375 rows at once and choosing the line that minimises the total error.

Also notice row 4: seven years of experience but only 60,000, less than the person with five years. The model cannot explain that, and neither can you — because experience alone does not determine salary. That leftover, unexplainable variation is precisely what R² measures.

**`day2_regression.py`**

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/regression/salary_data.csv")
df = pd.read_csv(URL)
print(df.head())

# This dataset has a few missing values. LinearRegression cannot handle
# NaN, so remove those rows first. (Try commenting this line out to see
# the error message scikit-learn gives you - it is worth reading once.)
df = df.dropna().reset_index(drop=True)

# X must be 2-D (a table), y is 1-D (a column)
X = df[["Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- The learned equation ---")
print(f"Salary = {model.coef_[0]:.2f} * Experience + {model.intercept_:.2f}")

print("\n--- Evaluation ---")
print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2  :", round(r2_score(y_test, y_pred), 4))

# Predict for a new person
new_candidate = pd.DataFrame({"Experience": [5.5]})
print("\nPredicted salary for 5.5 years:",
      round(model.predict(new_candidate)[0], 2))
```

### Reading regression metrics

| Metric | What it means in words | Good value |
|---|---|---|
| **MAE** | On average, how far off am I? Same unit as the target. | As low as possible |
| **RMSE** | Like MAE, but punishes large mistakes much more heavily. | As low as possible |
| **R²** | What fraction of the variation did the model explain? | Closer to 1 is better; 0 means no better than always guessing the mean |

If MAE is 5,000 on salaries around 80,000, your model is typically off by about 6%. State it that way in your report — a bare "MAE = 5000" tells the reader nothing.

**A negative R² is possible** and means the model performs worse than simply predicting the average every time. If you see one, something is wrong with your features or your split.

### 🧠 Analogy: what R² actually measures

Suppose you had to guess everyone's salary but were given **no information at all**. Your best strategy is to guess the average every time. You would be wrong a lot — call that total wrongness 100%.

Now you are told each person's years of experience. Your guesses improve. **R² is the fraction of that original wrongness you managed to remove.**

- **R² = 0.90** → you removed 90% of the guessing error. Experience explains most of salary.
- **R² = 0.30** → you removed only 30%. Experience matters, but something bigger is going on that you are not measuring.
- **R² = 0** → knowing experience helped you not at all.
- **R² < 0** → you did *worse* than just guessing the average. Something is broken.

When you run `day2_regression.py` you should get an R² of about **0.90**. In plain words: **years of experience explains roughly 90% of the variation in salary in this dataset.** That is the sentence to put in your report — not "R2: 0.8991".

### Practice

- Dataset: [salary_data.csv](https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/salary_data.csv) · [Notebook 2.2 — Salary](https://colab.research.google.com/drive/14E7_EF2Eo4Bc-iz6mb0du-66TIwxj3ru?usp=sharing)
- Exercise: [Used car price prediction](https://colab.research.google.com/drive/1RC0GU70efQL-9dabTVIwv-HmGYOz7u3L?usp=sharing)

## 2.4 Classification: predicting a category

### 🧠 Analogy: sorting the laundry

Regression asks *how much*. Classification asks *which pile*.

You already sort laundry into whites, colours and delicates without consciously applying a rule. You look at a garment, weigh up several clues (colour, fabric, label), and it goes in a pile. A classifier does exactly that: takes several clues (features) and puts the row in a pile (class).

The one extra thing a classifier gives you that you do not usually give yourself is **a confidence number**. It does not just say "colours" — it says "colours, 87% confident". That is `predict_proba()`, and it is often more useful than the label itself.

### ✏️ Activity 2.5 — What does a mistake cost?

Every classifier makes two different kinds of mistake, and they almost never cost the same. Work in pairs. For each system, decide **which mistake is worse** and why.

| System | Mistake A: false alarm | Mistake B: a miss | Which is worse? |
|---|---|---|---|
| Cancer screening | Healthy person told to get a scan | Cancer missed entirely | |
| Spam filter | Real email sent to spam folder | Spam reaches the inbox | |
| Fraud detection | Genuine purchase blocked | Fraud goes through | |
| Loan approval | Good applicant rejected | Bad loan approved | |
| Fire alarm | Alarm with no fire | Fire with no alarm | |

<details>
<summary>Discussion</summary>

- **Cancer screening** — a miss is catastrophic; a false alarm costs one anxious afternoon. Maximise **recall**.
- **Spam filter** — a lost job offer is far worse than one spam email you delete. Maximise **precision**.
- **Fraud** — usually a miss is worse, but block too many genuine cards and customers leave. Genuinely contested.
- **Loan approval** — the bank says a bad loan is worse. The rejected applicant strongly disagrees. **Whose cost are you optimising?** This is where fairness enters ML, and it is not a technical question.
- **Fire alarm** — a miss can kill. Recall, overwhelmingly. This is why alarms go off when you burn toast.

</details>

**This is the most important idea on the page.** "Which metric should I use?" is not a maths question. It is a question about **what it costs to be wrong**, and you cannot answer it from the data alone.

**`day2_classification.py`**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL).dropna().reset_index(drop=True)

categorical = ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]
for column in categorical:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compare two models on identical data
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1 score :", round(f1_score(y_test, y_pred), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

### 🧠 Analogy: the fire alarm report

The confusion matrix looks intimidating and is genuinely simple. Imagine the end-of-year report from a building's fire alarm:

| | The alarm stayed quiet | The alarm went off |
|---|---|---|
| **There was no fire** | ✅ Correct silence (TN) | 🔔 False alarm (FP) |
| **There was a fire** | 💀 Missed fire (FN) | ✅ Correctly caught (TP) |

Two boxes are good news (the diagonal). Two are bad news — and they are bad in **completely different ways**. A false alarm annoys people. A missed fire kills people.

That is the whole confusion matrix. Everything else is arithmetic on those four boxes.

### Reading the confusion matrix

For a yes/no problem, scikit-learn lays it out like this:

```text
                    Predicted 0     Predicted 1
Actual 0            TN              FP
Actual 1            FN              TP
```

- **TN** — correctly said no
- **FP** — said yes, but it was no → a **false alarm**
- **FN** — said no, but it was yes → a **miss**
- **TP** — correctly said yes

### 🧠 Analogy: precision and recall are a fishing net

You are fishing for tuna in a lake that also contains other fish.

- **Precision** — you haul in the net. Of everything *in your net*, how much is actually tuna? A net full of tuna and nothing else is 100% precision.
- **Recall** — of all the tuna *in the entire lake*, how much did you catch? Catching every last tuna is 100% recall.

Now notice the tension:

- Use a tiny, very selective net → almost everything you catch is tuna (**high precision**) but you miss most of the lake (**low recall**).
- Drag a giant net across the whole lake → you catch every tuna (**high recall**) but also boots, weeds and other fish (**low precision**).

**You cannot maximise both.** Every classifier lives somewhere on that trade-off, and where you choose to sit depends entirely on Activity 2.5 — what does each kind of mistake cost you?

**F1 score** is the single number that summarises the compromise. When you cannot decide, report F1.

### Reading classification metrics

| Metric | Question it answers | Use it when |
|---|---|---|
| **Accuracy** | What fraction did I get right overall? | Classes are roughly balanced |
| **Precision** | When I said yes, how often was I right? | False alarms are costly (spam filter deleting real mail) |
| **Recall** | Of all the real yeses, how many did I catch? | Misses are costly (cancer screening, fraud detection) |
| **F1** | Balance of precision and recall | You care about both, or classes are imbalanced |

> **Why accuracy alone will mislead you.** Imagine 1,000 transactions, 10 of them fraud. A model that says "not fraud" every single time scores **99% accuracy** and catches **zero** fraud. Its recall is 0. This is why you must always print the full classification report, not just accuracy.

### ✏️ Activity 2.6 — Build the useless 99% model

Do not take the warning above on trust. Build the useless model yourself — it takes ten lines.

**`activity_2_6.py`**

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# 1,000 transactions. Only 10 are fraud (class 1).
y_true = np.array([0] * 990 + [1] * 10)
X = np.zeros((1000, 1))          # the features do not even matter

# This "model" has one strategy: always predict the most common class.
lazy_model = DummyClassifier(strategy="most_frequent")
lazy_model.fit(X, y_true)
y_pred = lazy_model.predict(X)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Recall  :", recall_score(y_true, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print("\nFraud cases caught:", ((y_pred == 1) & (y_true == 1)).sum(), "out of 10")
```

**✅ Check yourself:** it reports **99% accuracy** and catches **zero out of ten** frauds. If you showed only the accuracy in a presentation, it would look like an excellent model. It is worthless.

**The habit to build:** every time you train a classifier, ask *"what would a model that always guesses the majority class score?"* If your real model cannot clearly beat that, it has learned nothing. `DummyClassifier` gives you that baseline in three lines — always compute it.

### Practice

- Dataset: [loan_data_10k.csv](https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv) · [Notebook 2.3 — Loan approval](https://colab.research.google.com/drive/1mJQBMddfoi4P8EdgnPZVLo0tFPlAyBsH?usp=sharing)

## 2.5 The ML workflow

Every project you build, including your capstone, follows these steps in this order:

```text
1. Define the problem   → What am I predicting? A number or a category?
2. Collect data         → Where does it come from? Is it allowed to be used?
3. Explore (EDA)        → shape, info, missing values, target balance
4. Preprocess           → clean, encode, split, scale
5. Choose a model       → start with the simplest one that could work
6. Train                → model.fit(X_train, y_train)
7. Evaluate             → on the test set the model has never seen
8. Improve              → features, cross-validation, hyperparameters
9. Deploy               → Streamlit app
10. Monitor             → does it still work on next month's data?
```

### The scikit-learn API is the same for every model

Learn these four method names once and every model in the library works the same way:

```python
model = SomeModel(...)        # 1. create, with settings
model.fit(X_train, y_train)   # 2. learn from training data
model.predict(X_test)         # 3. predict on new data
model.score(X_test, y_test)   # 4. quick default score
```

Classifiers add `model.predict_proba(X)`, which returns the probability of each class instead of just the label. Use it when you need confidence, not just a decision.

### Module 2 exit task

Complete the [ML practice exercise](https://github.com/tech4alltraining/aiml/blob/main/assessments/ml_ai_practice.md). For each model you train, report accuracy **and** precision **and** recall, and write one sentence on which of the three matters most for that problem.

---

# Module 3 — Feature engineering and model improvement

**Session:** Data augmentation, feature engineering, feature reduction, overfitting, cross-validation and hyperparameter tuning.

## 3.1 Data augmentation

### 🧠 Analogy: recognising your friend

You can recognise a close friend from the side, in bad light, wearing a hat, or half-turned away. You did not need a thousand separate photos to learn that — but you *did* need to see them from many angles over time.

If you had only ever seen one photograph of your friend, taken from directly in front in bright sunlight, you would struggle to recognise them in a dim corridor.

Data augmentation gives your model those extra angles artificially: take the photos you have and rotate, flip, brighten and crop them.

### The idea

```text
Original photo  →  rotate 15°   →  a new, valid training image
                →  flip left-right
                →  brighten
                →  zoom slightly
```

A car photographed at a slight angle is still a car, so the label does not change — but the model now sees more variety and generalises better.

The key rule is that the transformation must **preserve the label**. Flipping a photo of a cat gives a cat. Flipping a photo of the digit "2" gives something that is not a "2" — so horizontal flips are wrong for digit recognition.

### ✏️ Activity 3.1 — Legal or illegal augmentation?

For each pair, say whether the transformation keeps the label correct. Answer **legal** or **illegal**.

```text
1. Task: cat vs dog          Transformation: flip left-right
2. Task: read handwritten digits    Transformation: flip left-right
3. Task: detect a road sign  Transformation: rotate 180°
4. Task: X-ray shows pneumonia or not   Transformation: flip left-right
5. Task: classify fruit      Transformation: change brightness
6. Task: read a car number plate    Transformation: add heavy blur
7. Task: detect a helmet on a rider  Transformation: crop 10% from the edges
```

<details>
<summary>Answers and reasoning</summary>

1. **Legal.** A mirrored cat is a cat.
2. **Illegal.** A flipped `2` is not a `2`, and a flipped `5` may look like a `2`. You would be teaching the model wrong answers.
3. **Illegal.** An upside-down STOP sign does not appear on real roads — you would be training on a situation that never occurs.
4. **Debatable, usually illegal.** Human organs are not symmetric. The heart is on the left. A mirrored chest X-ray shows an anatomy that does not exist, and radiologists use side as a diagnostic clue.
5. **Legal.** An apple in dim light is still an apple — and this genuinely helps, because your test photos will have varied lighting.
6. **Illegal if too heavy.** Mild blur is realistic and helps. Blur so heavy the digits are unreadable creates an image with a label nobody could justify.
7. **Legal, with care.** Cropping the edges is fine — unless the crop removes the helmet.

</details>

**The rule to carry away:** before every augmentation, ask *"would a human still give this the same label?"* If not, you are not augmenting your data — you are corrupting it.

- Practice: [Notebook 3.1 — Data augmentation](https://colab.research.google.com/drive/1bvfMkPtrSTILCFbxGvaUPr5iGwk65mTe?usp=sharing)

## 3.2 Feature engineering

### 🧠 Analogy: height, weight, and BMI

A doctor could look at "height: 170cm" and "weight: 85kg" separately. But what actually matters for health risk is the **relationship** between them — and the medical world long ago gave that relationship a name: BMI.

BMI is not new information. It is `weight / height²` — nothing that was not already in the two columns. But it puts the useful pattern **directly in front of** whoever is looking, instead of leaving them to infer it.

That is feature engineering: taking columns you already have and combining them into a column that makes the pattern obvious.

For loans, the equivalent is `loan_amount / income`. A ₹5 lakh loan means something completely different to someone earning ₹3 lakh a year than to someone earning ₹50 lakh. Neither column alone tells you that. The ratio does, immediately.

> **Feature engineering usually improves a model more than switching algorithms does.** Beginners spend hours trying Random Forest instead of Logistic Regression. Experienced practitioners spend those hours building better columns.

### Building new features

**`day3_feature_engineering.py`**

```python
import pandas as pd

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL).dropna().reset_index(drop=True)

# --- Ratio features: often stronger than either column alone ---
df["loan_to_income"] = (df["loan_amnt"] / df["person_income"]).round(4)
df["income_per_year_worked"] = (
    df["person_income"] / df["person_emp_exp"].replace(0, 1)
).round(2)

# --- Binning: turn a continuous column into meaningful groups ---
df["age_group"] = pd.cut(
    df["person_age"],
    bins=[0, 25, 35, 50, 100],
    labels=["Under 25", "25-35", "36-50", "Over 50"]
)

df["credit_band"] = pd.cut(
    df["credit_score"],
    bins=[0, 580, 670, 740, 850],
    labels=["Poor", "Fair", "Good", "Excellent"]
)

# --- Flag features: encode domain knowledge as a 0/1 column ---
df["high_interest"] = (df["loan_int_rate"] > 15).astype(int)

# Did the new features actually help? Check against the target.
print(df.groupby("credit_band", observed=True)["loan_status"].mean().round(3))
print(df.groupby("age_group", observed=True)["loan_status"].mean().round(3))
print("\nCorrelation of new features with the target:")
print(df[["loan_to_income", "high_interest", "loan_status"]]
      .corr()["loan_status"].round(3))
```

Common patterns worth trying on almost any dataset:

| Pattern | Example |
|---|---|
| Ratio | `loan_amount / income` |
| Difference | `current_year - year_built` → age |
| Binning | age → age group |
| Date parts | date → day of week, month, is_weekend |
| Aggregation | average purchase per customer |
| Flag | `is_first_time_buyer` (0/1) |
| Text length | number of words in a review |

> **Test every new feature.** Add it, re-train, compare the score. Features that do not help are not free — they add noise and slow training down.

### ✏️ Activity 3.2 — Invent three features

Work in pairs, on paper, for ten minutes. Here are the columns of the loan dataset:

```text
person_age                     loan_amnt
person_gender                  loan_intent
person_education               loan_int_rate
person_income                  loan_percent_income
person_emp_exp                 cb_person_cred_hist_length
person_home_ownership          credit_score
                               previous_loan_defaults_on_file
```

Invent **three** new columns you could compute from these. For each one write:

```text
NAME       : ______________________________
FORMULA    : ______________________________
WHY IT HELPS (one sentence, in plain English):
_____________________________________________
```

<details>
<summary>Some good answers — check yours against these</summary>

| Name | Formula | Why it helps |
|---|---|---|
| `loan_to_income` | `loan_amnt / person_income` | Affordability. The same loan is trivial for one applicant and impossible for another. |
| `monthly_repayment_burden` | `(loan_amnt × int_rate) / (income / 12)` | How much of a month's pay the interest eats. |
| `credit_per_year` | `credit_score / cb_person_cred_hist_length` | Distinguishes a good score built slowly from one built recently. |
| `started_working_age` | `person_age - person_emp_exp` | Someone working since 16 has a different profile from a recent graduate. |
| `is_young_borrower` | `person_age < 25` | Flags a group lenders genuinely treat differently. |
| `high_risk_combo` | `(defaults == 'Yes') & (credit_score < 600)` | Two red flags together are worse than either alone. |

</details>

**Now test one.** Add your best feature to the classification script from Module 2, re-train, and compare. Write down the before and after scores — including if it made things *worse*. A feature that does not help is a real result, and knowing that took you five minutes to discover.

- Practice: [Notebook 3.2 — Feature engineering](https://colab.research.google.com/drive/1xKoiS5WaH9_kNHjg1dNP-xuAgaTmZ0Wr?usp=sharing)

## 3.3 Feature reduction

Too many features cause slow training, overfitting and models nobody can explain. Reduce them.

**`day3_feature_reduction.py`**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL).dropna().reset_index(drop=True)

for column in ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Method 1: Feature importance from a tree model (most interpretable) ---
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)

importances = (pd.Series(forest.feature_importances_, index=X.columns)
                 .sort_values(ascending=False))
print("=== Feature importance ===")
print(importances.round(4))

# --- Method 2: Statistical selection - keep the best K columns ---
selector = SelectKBest(score_func=f_classif, k=6)
selector.fit(X_train, y_train)
print("\n=== SelectKBest kept ===")
print(list(X.columns[selector.get_support()]))

# --- Method 3: PCA - compress many columns into a few components ---
# PCA needs scaled data, and the components are NOT human-readable.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=0.95)          # keep components explaining 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)

print(f"\n=== PCA ===")
print(f"Original columns: {X_train.shape[1]}  ->  components: {X_train_pca.shape[1]}")
print("Variance explained by each:", pca.explained_variance_ratio_.round(3))
```

| Method | Keeps original column names? | Use when |
|---|---|---|
| Feature importance | Yes | You must explain the model to a person |
| SelectKBest | Yes | You want a quick statistical filter |
| PCA | **No** | You have very many correlated columns and do not need interpretability |

- Practice: [Notebook 3.3 — Feature reduction](https://colab.research.google.com/drive/1wKr-AwnHXF3HgPvb-6K89SjVxUxxHsEK?usp=sharing)

## 3.4 Overfitting and underfitting

This is the central problem of Machine Learning. If you understand only one idea from Module 3, make it this one.

### 🧠 Analogy: three students and an exam

Three students prepare for the same exam using last year's paper.

**Anmol barely studies.** He glances at the paper for ten minutes. He does badly on the practice paper *and* badly on the real exam. → **Underfitting.** Too simple. Did not learn the pattern at all.

**Bina studies the practice paper properly.** She works out *why* each answer is right, so she understands the underlying topics. She scores well on the practice paper and well on the real exam. → **Good fit.** She learned the pattern.

**Chandra memorises the answer sheet.** Question 4 is "C". Question 5 is "B". He scores 100% on the practice paper — perfect, better than Bina. Then the real exam has different questions and he is lost. → **Overfitting.** He memorised the *specific data*, not the pattern behind it.

**Here is the part that catches beginners out.** On the practice paper, Chandra looked like the best student in the room. Training accuracy alone would have told you he was the strongest. **This is exactly why you must never judge a model by its training score.**

### The three cases side by side

| | Training score | Test score | What is happening | Fix |
|---|---|---|---|---|
| **Underfitting** | Low | Low | Model is too simple to capture the pattern | More features, a more complex model, train longer |
| **Good fit** | High | High | Model learned the pattern | Nothing — ship it |
| **Overfitting** | Very high | Much lower | Model memorised the training rows, including their noise | More data, fewer features, regularisation, simpler model |

**`day3_overfitting.py`**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL).dropna().reset_index(drop=True)

for column in ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(columns=["loan_status"])
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"{'max_depth':>10} {'train':>8} {'test':>8}  {'gap':>8}")
for depth in [1, 2, 3, 5, 10, 20, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_score = tree.score(X_train, y_train)
    test_score = tree.score(X_test, y_test)
    label = str(depth) if depth else "unlimited"
    print(f"{label:>10} {train_score:>8.4f} {test_score:>8.4f}  "
          f"{train_score - test_score:>8.4f}")
```

### ✏️ Activity 3.3 — Watch a model overfit, live

Run `day3_overfitting.py` and copy its output into this table:

| max_depth | Train | Test | Gap | Which student is this? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 5 | | | | |
| 10 | | | | |
| 20 | | | | |
| unlimited | | | | |

Fill the last column with **Anmol** (underfitting), **Bina** (good fit) or **Chandra** (overfitting).

Then answer:

1. At `max_depth=1`, are both scores low? That is Anmol.
2. At `max_depth=None`, the training score races towards 1.0. What happens to the **test** score?
3. **Which depth gives the best test score?** That is your answer — not the depth with the best training score.
4. At which depth does the gap first exceed 0.05? That is roughly where memorising begins.

**✅ The habit this builds:** always print training **and** test scores together. A single number tells you nothing about whether the model generalises. The *gap between them* is the diagnosis.

**🔁 Change one thing:** set `max_depth=None` but add `min_samples_leaf=50`. This forces every leaf of the tree to cover at least 50 rows, so it cannot carve out a branch for one memorised row. Watch the gap shrink. You have just applied **regularisation** — deliberately limiting a model's freedom so it generalises better.

- Practice: [Notebook 3.4 — Overfitting & underfitting](https://colab.research.google.com/drive/1NJSLQ3slItQTyvYYQkkzP36h9vKwKT2z?usp=sharing)

## 3.5 K-Fold cross-validation

### 🧠 Analogy: one practice exam is not enough

You sit one practice exam and score 82%. Are you an 82% student?

Maybe. Or maybe that paper happened to cover the two chapters you knew best. Sit five different practice papers and you might score 82, 71, 79, 68, 80 — an average of 76%, with a spread that tells you how *reliable* your performance is.

A single train/test split is one practice paper. The score depends on which rows happened to land in the test set. **Cross-validation sits five papers and averages the marks.**

```text
K-Fold with k=5: split the data into 5 equal parts.

Round 1:  [TEST ][train][train][train][train]  -> score 1
Round 2:  [train][TEST ][train][train][train]  -> score 2
Round 3:  [train][train][TEST ][train][train]  -> score 3
Round 4:  [train][train][train][TEST ][train]  -> score 4
Round 5:  [train][train][train][train][TEST ]  -> score 5

Report the mean and the standard deviation of the five scores.
```

**`day3_cross_validation.py`**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL).dropna().reset_index(drop=True)

for column in ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# StratifiedKFold keeps the class balance in every fold - always use it
# for classification.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name}")
    print(f"  fold scores: {scores.round(4)}")
    print(f"  mean {scores.mean():.4f}  std {scores.std():.4f}\n")
```

**Read the standard deviation, not just the mean.** A model scoring `0.89 ± 0.002` is dependable. A model scoring `0.89 ± 0.06` gives wildly different results depending on the data it sees, and you should not trust it.

### ✏️ Activity 3.4 — Prove that one split is unreliable

Run the same model with five *different* random splits and watch the score move.

**`activity_3_4.py`**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL).dropna().reset_index(drop=True)

for column in ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# --- Five single splits, differing ONLY in random_state ---
print("Five single train/test splits (same model, same data):")
single_scores = []
for seed in [0, 1, 2, 3, 4]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    score = model.score(X_te, y_te)
    single_scores.append(score)
    print(f"  random_state={seed}: {score:.4f}")

print(f"\n  Lowest : {min(single_scores):.4f}")
print(f"  Highest: {max(single_scores):.4f}")
print(f"  Spread : {max(single_scores) - min(single_scores):.4f}")

# --- Now the honest answer: cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=42),
                            X, y, cv=cv)

print(f"\n5-fold cross-validation:")
print(f"  fold scores: {cv_scores.round(4)}")
print(f"  mean {cv_scores.mean():.4f}  std {cv_scores.std():.4f}")
```

**✅ Check yourself:**

1. How far apart are the highest and lowest single-split scores? That spread came from **nothing but luck** — same model, same data, same settings, different random split.
2. If you reported only the highest one, would that be honest?
3. The cross-validation mean sits inside that range and comes with a standard deviation. **Report the mean and the std.** That is a claim you can defend in your project review.

**⚠️ Watch out:** if a classmate's score is suddenly much higher than everyone else's on the same dataset, the first question is not "what model did you use?" It is **"did you cross-validate, or did you get a lucky split?"**

- Practice: [Notebook 3.5 — K-Fold cross validation](https://colab.research.google.com/drive/1_MHdhg7Y1x5BnelDgPn9KB3d4NpwYwpf?usp=sharing)

## 3.6 Hyperparameter tuning

### 🧠 Analogy: the oven dials

A cake recipe has two kinds of number.

**Things the cake works out for itself:** how brown the crust gets, how far it rises. You do not set those — they emerge from baking. Those are **parameters**, and the model learns them during `fit()`.

**Things you set before it goes in:** oven temperature 180°C, , shelf position middle. Those are **hyperparameters**, and no amount of baking will choose them for you.

If the cake comes out burnt on top and raw inside, you do not blame the flour. You change the dials and bake another one. **Hyperparameter tuning is systematically trying dial settings and keeping the best cake.**

- **GridSearchCV** = try every combination of dial settings. Thorough, slow.
- **RandomizedSearchCV** = try twenty random combinations. Much faster, usually finds something nearly as good.

### The mechanics

**`day3_tuning.py`**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, StratifiedKFold)

URL = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/loan_data_10k.csv")
df = pd.read_csv(URL).dropna().reset_index(drop=True)

for column in ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(columns=["loan_status"])
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# --- GridSearchCV: tries EVERY combination ---
# 3 x 3 x 2 = 18 combinations x 3 folds = 54 model fits. Keep grids small.
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,        # use all CPU cores
    verbose=1,
)
grid.fit(X_train, y_train)

print("\n=== GridSearchCV ===")
print("Best parameters:", grid.best_params_)
print("Best CV score  :", round(grid.best_score_, 4))
print("Test score     :", round(grid.score(X_test, y_test), 4))

# --- RandomizedSearchCV: samples N random combinations. Much faster on
# --- large search spaces, and usually finds something nearly as good.
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions={
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    n_iter=10,             # try only 10 random combinations
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)
random_search.fit(X_train, y_train)

print("\n=== RandomizedSearchCV ===")
print("Best parameters:", random_search.best_params_)
print("Best CV score  :", round(random_search.best_score_, 4))
print("Test score     :", round(random_search.score(X_test, y_test), 4))
```

> **This takes a few minutes to run.** That is normal and it is the point: `GridSearchCV` fits `combinations × folds` models. Start with a small grid, see which values look promising, then search more finely around them.

### Hyperparameters worth knowing

| Model | Hyperparameter | Effect |
|---|---|---|
| `RandomForest` | `n_estimators` | More trees → more stable, slower |
| `RandomForest` | `max_depth` | Deeper → fits more, overfits sooner |
| `DecisionTree` | `min_samples_split` | Higher → simpler tree |
| `LogisticRegression` | `C` | Lower → stronger regularisation, simpler model |
| `KNeighbors` | `n_neighbors` | Higher → smoother decision boundary |
| `SVM` | `kernel`, `C`, `gamma` | Shape and tightness of the boundary |

- Practice: [Notebook 3.6 — Hyperparameter tuning](https://colab.research.google.com/drive/1-eyZakV1mK4C-_9nvLzV66lqsJBLS0JO?usp=sharing)

### Module 3 exit task

Take your Module 2 classification model and improve it. Report a small table:

| Version | What changed | CV mean | CV std | Test score |
|---|---|---|---|---|
| Baseline | Module 2 model | | | |
| + features | Added `loan_to_income` | | | |
| + tuning | GridSearchCV best params | | | |

If a change made the score worse, keep it in the table and say so. Negative results are results.

---

# Module 4 — Deep learning, clustering and Generative AI

**Session 4.1:** Deep learning intro, AI ethics, unsupervised learning, clustering.
**Session 4.2:** Generative AI, LLMs, prompt engineering, the Gemini API.

## 4.1 Deep learning in one page

### 🧠 Analogy: the relay race

Imagine a relay race where each runner is allowed to slightly *improve* the message before passing it on.

Runner 1 receives raw scribbles and passes on "these look like edges and lines". Runner 2 receives that and passes on "these edges form circles and curves". Runner 3 passes on "these shapes look like eyes and a nose". The final runner says "it's a face".

No single runner understands faces. Each one does something small and passes it along, and the *chain* produces something none of them could alone.

That is a neural network. Each **layer** is a runner. **Deep** learning just means the relay has many runners rather than one.

### The mechanics

```text
Input layer      Hidden layer(s)         Output layer
  [age]  ─┐
  [income]─┼──▶  ( ) ( ) ( )  ──▶  ( ) ( )  ──▶  [approve?]
  [score] ─┘      weights            weights
```

**Deep** simply means more than one hidden layer.

Training is a loop:

```text
1. Forward pass    - push the input through, get a prediction
2. Loss            - measure how wrong the prediction was
3. Backpropagation - work out how each weight contributed to the error
4. Update          - nudge every weight to reduce the error
5. Repeat for many epochs
```

**When to use deep learning instead of classical ML:**

| Situation | Use |
|---|---|
| Table of rows and columns, under ~100k rows | **Classical ML** (Random Forest, Gradient Boosting) |
| Images, audio, video | Deep learning (CNN) |
| Text and language | Deep learning (Transformers) |
| Very large datasets, complex patterns | Deep learning |

On tabular data, a Random Forest usually beats a neural network and trains in seconds. Do not reach for deep learning just because it sounds more advanced.

## 4.2 Clustering: unsupervised learning

### 🧠 Analogy: the box of old photographs

Someone hands you a box of 500 unlabelled family photographs and asks you to organise them. Nobody tells you what the categories are.

You start laying them out and groups emerge on their own: beach photos, wedding photos, photos of one particular child growing up. **You did not decide the categories in advance — you discovered them from the photos themselves.**

That is clustering. Every model so far in this course was given the right answers to learn from. Clustering gets no answers at all, and finds structure anyway.

**One warning that follows directly from the analogy:** two different people organising the same box will produce different piles, and neither is "wrong". Clustering has no answer key, so it has **no accuracy score**. You judge it by whether the groups are *useful*, and that judgement is yours to make and defend.

**`day4_clustering.py`**

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

URL = ("https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/"
       "master/Datasets/Mall_Customers.csv")
df = pd.read_csv(URL)
print(df.head())

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = StandardScaler().fit_transform(X)

# --- Step 1: how many clusters? Use the elbow method ---
inertias, silhouettes = [], []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(k_range, inertias, marker="o")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (lower is tighter)")

axes[1].plot(k_range, silhouettes, marker="o", color="darkorange")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Score (higher is better)")
plt.tight_layout()
plt.savefig("day4_choosing_k.png", dpi=120)
plt.show()

# --- Step 2: fit with the chosen k and interpret the groups ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

print("\nCustomers per segment:")
print(df["segment"].value_counts().sort_index())

print("\nAverage profile of each segment:")
print(df.groupby("segment")[["Annual Income (k$)",
                             "Spending Score (1-100)"]].mean().round(1))

plt.figure(figsize=(8, 6))
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"],
            c=df["segment"], cmap="viridis", s=60)
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score")
plt.title("Customer segments found by K-Means")
plt.colorbar(label="segment")
plt.savefig("day4_segments.png", dpi=120)
plt.show()
```

### ✏️ Activity 4.1 — Cluster the classroom (no computer)

Before running K-Means, do it with people.

1. Everyone in the class stands up.
2. The instructor calls out two axes — for example **"how far you travel to campus"** (left = near, right = far) and **"how early you wake up"** (front = early, back = late).
3. Everyone physically positions themselves in the room accordingly.
4. Now the instructor asks: **"Looking at this room, how many natural groups do you see?"**

Discuss:

- Did people agree on the number of groups? Usually not — some see 3, some see 5. **That disagreement is exactly the difficulty of choosing `k`.**
- Was anyone standing alone, far from everyone? That person is an **outlier**, and K-Means will either drag a whole cluster towards them or give them their own.
- If the instructor had chosen different axes — say, "hours of sleep" and "favourite subject" — would the groups be the same people? No. **Which features you choose determines which clusters you find.**

Now run the code, and notice you are doing precisely what the room just did — only with income and spending score as the two axes.

### ✏️ Activity 4.2 — Choose k, then defend it

Run `day4_clustering.py` and open `day4_choosing_k.png`.

1. On the **elbow** chart, find where the curve stops dropping steeply and starts flattening. That bend is the "elbow". Which k is it?
2. On the **silhouette** chart, which k scores highest?
3. **Do the two charts agree?** Often they do not. Which do you trust, and why?
4. Change `n_clusters=5` to your chosen k, re-run, and read the segment profile table.
5. **Give every segment a business name** — not "cluster 3" but "high income, low spenders: cautious professionals".

**✅ Check yourself:** could you explain your chosen k to a shop manager who knows no maths? If your only justification is "the silhouette score was highest", you have not finished. The real justification is "we found five kinds of customer, and here is what each one is like."

**The step everyone skips:** after clustering, *name* the groups. Look at the average profile table and describe each segment in business language — "high income, low spending: cautious wealthy customers". A cluster number with no name is useless to anybody who has to act on it.

- Practice: [Dataset 4.1 — Mall customers](https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv) · [Notebook 4.1](https://colab.research.google.com/drive/1jXCcXXQbSmYnxjulhgtgJB9ueQ76u6Pj?usp=sharing)

## 4.3 What a Large Language Model actually is

### 🧠 Analogy: the world's most well-read autocomplete

Open the messaging app on your phone and type "I will be". Look at the three words it suggests. Keep tapping the middle suggestion, over and over. You will get a sentence — grammatical, plausible, and completely generic.

Your phone learned those suggestions from the messages you and millions of others have typed.

**An LLM is that, scaled up almost unimaginably.** Instead of learning from your text messages, it learned from a substantial fraction of everything ever written. Instead of suggesting three words, it weighs a hundred thousand possible next chunks with a probability on each.

Nothing else changed. **It is still autocomplete.** Every impressive thing an LLM does — writing code, explaining photosynthesis, translating Tamil — comes out of doing autocomplete extraordinarily well.

### Why this analogy matters

Hold onto it, because it explains the two things that confuse people most:

**"Why does it make things up?"** Your phone does not check whether "I will be late" is true. It suggests what is *likely to come next*. An LLM does the same. Ask it for a citation and it produces text shaped exactly like a real citation — right author style, right journal-sounding name, right year format — because that is what is probable there. Whether the paper exists was never part of the calculation. **That is a hallucination**, and it is not a bug; it is what the machine does.

**"Why does it forget what I just said?"** Your phone does not remember yesterday's conversation. Neither does an LLM between calls. Each request starts blank. If a chatbot appears to remember, it is because the program is quietly re-sending the whole conversation every single time.

### The mechanics

```text
"The capital of France is" →  Paris  87%
                              a      4%
                              known  3%
                              ...
```

That is the whole mechanism. Everything else — reasoning, translation, summarising, writing code — emerges from doing this extremely well on an enormous amount of text.

Two consequences follow immediately, and they explain almost every strange thing an LLM does:

1. **It does not look anything up.** It has no database of facts. It produces what is *linguistically probable*, which is usually also true — but not always. This is what a **hallucination** is.
2. **It has no memory between calls.** Each API request is independent. If you want a conversation, *you* must send the previous messages back every time.

### Vocabulary you need

| Term | Meaning |
|---|---|
| **Token** | A chunk of text, roughly ¾ of a word. "unbelievable" ≈ `un` + `bel` + `iev` + `able` |
| **Prompt** | Everything you send to the model |
| **Context window** | The maximum tokens the model can consider at once |
| **Temperature** | Randomness dial: 0 = most probable token always, 1+ = adventurous |
| **Top-p** | Consider only the most likely tokens whose probabilities sum to *p* |
| **Top-k** | Consider only the *k* most likely tokens |
| **Hallucination** | Confident, fluent, wrong |
| **Grounding** | Giving the model the source text so it does not have to guess |
| **System instruction** | A standing instruction that shapes every reply |

### ✏️ Activity 4.3 — Be the language model

Do this before any code. It takes five minutes and it makes everything afterwards obvious.

**Round 1 — the easy one.** The instructor writes on the board:

```text
"The capital of France is ___"
```

Everyone shouts the next word. Almost certainly unanimous: *Paris*. Write on the board:

```text
Paris        ~95%
a            ~2%
located      ~1%
```

**Round 2 — the hard one.**

```text
"The weather today is really ___"
```

Now the class splits: *hot, cold, nice, bad, humid, unpredictable*. Count the votes and write the top five with their percentages.

**Round 3 — build a sentence together.** Start with one word and go round the class. Each person adds only the single word they think most likely follows. Write the sentence as it grows.

**The debrief — this is the important part:**

1. **Round 1 vs Round 2.** In Round 1 almost everyone agreed; in Round 2 you scattered. **That difference is exactly what `temperature` controls.** At temperature 0 the model always takes the top choice — it would say "hot" every single time. At temperature 1 it samples from the whole spread, so it might say "humid".
2. **In Round 3, did anyone know what the sentence would be?** No. Each person saw only what came before and added one likely word. **The sentence had no author and no plan.** That is precisely how an LLM generates a paragraph.
3. **Did anyone check whether the sentence was true?** No. You were each choosing what *sounds right next*, not what is correct. Now you understand hallucination from the inside.

### ✏️ Activity 4.4 — Count the tokens

Models do not read words; they read tokens. Guess before you check.

Guess how many tokens each of these becomes:

```text
1. "cat"
2. "unbelievable"
3. "internationalisation"
4. "Thiruvananthapuram"
5. "AI"
6. "🙂"
```

Then check at the [OpenAI tokenizer](https://platform.openai.com/tokenizer) or [Google AI Studio's token counter](https://aistudio.google.com/).

**What you will notice:** common short words are one token. Long or unusual words split into several. Names and non-English words split the most. An emoji can cost several tokens on its own.

**Why you should care:** you are billed per token, and the context window is measured in tokens. A prompt full of long unusual names costs meaningfully more than the same length of ordinary English. And it explains a classic puzzle — ask a model how many `r`s are in "strawberry" and it often gets it wrong, because it never saw the individual letters. It saw two or three chunks.

## 4.4 Prompt engineering

### 🧠 Analogy: briefing a brilliant new intern

Imagine an intern who has read almost every book ever written, works instantly, never gets tired — and has been at your organisation for exactly zero minutes. They do not know who you are, who the work is for, or what "good" looks like here.

Now compare two briefings.

**Briefing A:** *"Write something about our product."*
You will get back something bland and generic, because you gave them nothing to work with. That is not the intern being unhelpful. That is you not briefing them.

**Briefing B:** *"You're writing for our website. The readers are parents of school-age children who are not technical. Explain what our app does and why it's safe. Friendly tone, under 100 words, no jargon, end with a call to action."*
Now they can actually do the job.

**A prompt is a briefing.** Every complaint people have about AI output — "too generic", "wrong tone", "too long" — is almost always a briefing problem, not a model problem.

### The five parts of a good prompt

### The five parts of a good prompt

| Part | Question it answers | Example |
|---|---|---|
| **Role** | Who should the model be? | "You are a career counsellor for engineering students." |
| **Task** | What exactly should it do? | "Review this CV and list its three biggest weaknesses." |
| **Context** | What does it need to know? | "The student is a final-year CS undergraduate applying for data roles." |
| **Constraints** | What are the limits? | "Be specific. Do not invent experience. Under 150 words." |
| **Format** | What should the output look like? | "Return a numbered list. One sentence per point." |

Compare:

```text
Weak:   Write about machine learning.

Strong: You are a teacher explaining to first-year students who know
        basic Python but no statistics.
        Explain what overfitting is.
        Use one everyday analogy and one concrete example.
        Keep it under 150 words. Do not use the word "variance".
```

The weak prompt has no role, no audience, no length, no format. You get back an encyclopaedia entry. The strong prompt has all five parts, and you get something you can actually put on a slide.

### ✏️ Activity 4.5 — The prompt makeover

Here are five weak prompts. Rewrite each one using all five parts. Then run **both** versions and compare.

```text
1. "Write about climate change."
2. "Give me some interview questions."
3. "Summarise this article."
4. "Write code to sort a list."
5. "Explain machine learning."
```

Use this scaffold — fill in all five lines before you type anything into the model:

```text
ROLE       : You are a ______________________
TASK       : ______________________________
CONTEXT    : The reader is ______________________
CONSTRAINTS: ______________________________
FORMAT     : ______________________________
```

<details>
<summary>Example makeover for number 2</summary>

**Weak:** "Give me some interview questions."

**Strong:**
```text
You are a technical interviewer at a mid-size analytics company.

Write 8 interview questions for a final-year engineering student applying
for a junior data analyst internship.

The student knows Python and Pandas but has never worked on a real project.
The interview is 30 minutes.

Do not ask questions that need deep statistics or production experience.
Mix 4 conceptual questions with 4 practical ones.

Return a numbered list. After each question, add one line starting with
"Looking for:" describing a good answer.
```

</details>

**✅ Check yourself:** put the two outputs side by side. Which one could you use *without editing*? That difference took you two extra minutes of writing. This is the highest-return skill in the whole of Module 4.

### The four prompt types

**Zero-shot** — task only, no examples. Use for straightforward tasks.

```text
Extract the name, occupation, and city from the following sentence and
output it as JSON:
"My name is Sarah, I work as a mechanical engineer, and I just moved to Seattle."
```

**One-shot** — one worked example. Use when the *format* must be exact.

```text
Extract the flight details into a pipe-separated format.

Input: "I'm flying on Delta flight 402 from JFK to LAX on Tuesday."
Output: Delta | 402 | JFK | LAX | Tuesday

Input: "Book me on United 88 departing from ORD and arriving at SFO tomorrow."
Output:
```

**Few-shot** — three to five examples. Use for classification and for patterns that are hard to describe in words.

```text
Classify the customer support ticket as [BILLING], [TECH_ISSUE], or [SALES].

Ticket: "My screen is cracked and the touch sensor won't work."
Category: [TECH_ISSUE]

Ticket: "Do you offer enterprise discounts for teams of 50 or more?"
Category: [SALES]

Ticket: "I was double-charged on my credit card this month."
Category: [BILLING]

Ticket: "How do I upgrade my account from basic to premium?"
Category:
```

**Chain-of-thought** — ask for the reasoning before the answer. Use for maths, logic and multi-constraint problems.

```text
Solve the following logic puzzle. Before giving the final answer, break
down your reasoning step-by-step.

Puzzle: If it takes 5 machines 5 minutes to make 5 widgets, how long
would it take 100 machines to make 100 widgets?
```

Run that last one both with and without the chain-of-thought instruction. Without it, models frequently answer "100 minutes" — the same knee-jerk mistake humans make. The answer is 5 minutes.

### 🧠 Analogy: teaching someone your filing system

The four prompt types are four ways of explaining a task to a new colleague.

| Type | How you would explain it to a person |
|---|---|
| **Zero-shot** | "File these by date." — they know what dates are; just tell them. |
| **One-shot** | "File these like this one." — you show them a single finished example, because the exact format matters. |
| **Few-shot** | "Here are four already filed. See the pattern?" — the rule is hard to state in words but obvious from examples. |
| **Chain-of-thought** | "Work through it out loud so I can follow your reasoning." — you want to catch mistakes in the middle, not just at the end. |

### Choosing a type

```text
Is the task simple and the format flexible?          → Zero-shot
Does the output format have to be exact?             → One-shot
Is it classification, or a hard-to-describe pattern? → Few-shot
Does it need arithmetic, logic, or many constraints? → Chain-of-thought
```

### ✏️ Activity 4.6 — All four on one task

Take **one** task and write it four ways. Use: *classify a student's feedback comment as Positive, Negative or Suggestion.*

1. **Zero-shot** — just the instruction and the comment.
2. **One-shot** — add one worked example.
3. **Few-shot** — add three worked examples, one per category.
4. **Chain-of-thought** — ask it to explain its reasoning before giving the label.

Test all four on this deliberately awkward comment:

```text
"The lab sessions were fine but honestly three hours is too long,
maybe split it into two."
```

**Then compare, using this table:**

| Version | Label it gave | Extra words around the label? | Would this work in code? |
|---|---|---|---|
| Zero-shot | | | |
| One-shot | | | |
| Few-shot | | | |
| Chain-of-thought | | | |

**✅ What you should find:**

- Zero-shot often wraps the answer in chat: *"This comment appears to be a Suggestion..."* — you cannot use that directly in a program.
- One-shot and few-shot return the bare label, because you showed them the shape you wanted.
- Chain-of-thought gives the best *reasoning* and the worst *format* — it writes a paragraph.
- The comment is genuinely ambiguous (positive **and** a suggestion). Different prompt types may disagree. **That is real: when humans would disagree on a label, models disagree too.**

**The lesson:** more examples is not automatically better. Chain-of-thought is not automatically better. **Match the prompt type to what the task actually needs.**

The full set of demo prompts is in [`prompts.md`](prompts.md).

## 4.5 Your first Gemini API call

Make sure the environment is ready:

```bash
conda activate genai
pip install google-genai python-dotenv
```

**`day4_first_genai.py`**

```python
"""Your first call to a Generative AI model. Run it, then change the prompt."""

import os
from dotenv import load_dotenv
from google import genai

# Load GEMINI_API_KEY from the .env file in this folder
load_dotenv()

MODEL_NAME = "gemini-3.5-flash"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model=MODEL_NAME,
    contents="Explain overfitting to a first-year student in exactly three sentences."
)

print(response.text)
```

Run it:

```bash
conda activate genai
python day4_first_genai.py
```

If you see an authentication error, your key is missing or wrong — check [Troubleshooting](#troubleshooting).

### Controlling the output

**`day4_temperature.py`**

```python
"""Same prompt, different temperature. Run it and read the difference."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3.5-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = "Write a tagline for a coffee shop. Return only the tagline."

for temperature in [0.0, 1.0]:
    print(f"\n{'=' * 50}")
    print(f"TEMPERATURE = {temperature}")
    print("=" * 50)

    for run in range(1, 4):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=PROMPT,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=50,
            ),
        )
        print(f"Run {run}: {response.text.strip()}")
```

At `temperature=0.0` the three runs should be identical or nearly so. At `temperature=1.0` they should differ noticeably. That difference *is* the parameter.

### 🧠 Analogy: the spice dial

`temperature` is a spice dial on a dish.

- **Turned to 0** — plain, safe, exactly the same every time. Perfect for a hospital canteen where consistency matters more than excitement.
- **Turned to 1** — bold and varied. Sometimes wonderful, sometimes inedible. Right for a chef experimenting.

`top_p` and `top_k` are a different control: they limit **which spices are on the shelf at all**, before the dial is even turned.

- `top_k=1` — only one spice on the shelf. Whatever the dial says, you get the same dish.
- `top_k=40` — forty spices available; the dial decides how adventurously you pick among them.
- `top_p=0.9` — keep adding spices to the shelf, most-likely first, until they account for 90% of the probability. Drop the rest.

**Practical advice:** change `temperature` and leave the other two alone until you have a specific reason. Adjusting all three at once makes it impossible to tell which one caused what.

### The three sampling parameters

| Parameter | Range | Effect | Use for |
|---|---|---|---|
| `temperature` | 0.0 – 2.0 | How willing the model is to pick a less likely token | 0.0–0.3 for facts, code, extraction; 0.7–1.0 for creative writing |
| `top_p` | 0.0 – 1.0 | Consider only tokens whose probabilities sum to `p` | 0.9 is a sensible default |
| `top_k` | 1 – 100 | Consider only the `k` most likely tokens | 40 is a sensible default |

**`day4_parameters.py`**

```python
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

settings = {
    "Factual (deterministic)": dict(temperature=0.0, top_p=0.1, top_k=1),
    "Balanced (default-ish)":  dict(temperature=0.7, top_p=0.9, top_k=40),
    "Creative (adventurous)":  dict(temperature=1.2, top_p=1.0, top_k=100),
}

PROMPT = "Describe a rainy evening in one sentence."

for name, params in settings.items():
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=PROMPT,
        config=types.GenerateContentConfig(max_output_tokens=80, **params),
    )
    print(f"\n--- {name} ---")
    print(f"    {params}")
    print(response.text.strip())
```

### ✏️ Activity 4.7 — The temperature dial

Run `day4_temperature.py` and record what you see.

| Run | temperature = 0.0 | temperature = 1.0 |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

**✅ Check yourself:**

1. Were the three outputs at `0.0` identical or nearly so? **Why?** (At 0 the model always takes the single most probable token — there is no randomness left to produce a difference.)
2. At `1.0`, which run produced the *best* tagline? Which produced the *worst*?
3. This is the trade-off in one line: **high temperature gives you the highest highs and the lowest lows.** Low temperature gives you consistently acceptable and consistently dull.

**Now choose the setting for each real job:**

```text
a. Extracting dates from 500 invoices          → temperature ____
b. Brainstorming names for a college fest      → temperature ____
c. Translating a legal notice                  → temperature ____
d. Writing a poem for a farewell card          → temperature ____
e. Generating a quiz from lecture notes        → temperature ____
```

<details>
<summary>Answers</summary>

a **0.0** — you want the same date extracted every time, with zero creativity.
b **0.9–1.2** — you want variety; that is the entire point.
c **0.0–0.2** — accuracy is everything; invention is dangerous.
d **0.8–1.0** — a predictable poem is a bad poem.
e **0.3–0.5** — mostly factual, but you do not want the same five questions every run.

</details>

### ✏️ Activity 4.8 — Top-k in the extreme

The fastest way to feel what `top_k` does is to set it to 1.

```python
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = "Suggest a name for a new coffee shop. Reply with the name only."

for label, config in {
    "top_k=1  (only the single most likely token)": dict(temperature=1.0, top_k=1),
    "top_k=40 (forty candidates available)":        dict(temperature=1.0, top_k=40),
}.items():
    print(f"\n--- {label} ---")
    for run in range(3):
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=PROMPT,
            config=types.GenerateContentConfig(max_output_tokens=20, **config),
        )
        print("  ", response.text.strip())
```

**✅ Check yourself:** with `top_k=1`, the three runs are identical **even though temperature is 1.0**. Why? Because temperature decides *how adventurously to choose among the candidates*, and with `top_k=1` there is only one candidate. **A high temperature cannot create variety that `top_k` has already removed.** That is the relationship between the two dials, and it is much easier to feel than to read.

### Adding a system instruction

A system instruction shapes every reply without being repeated in each prompt.

```python
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="What is a p-value?",
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a statistics tutor for students who have never studied "
            "statistics. Never use jargon without defining it first. "
            "Always give one concrete example."
        ),
        temperature=0.3,
    ),
)
print(response.text)
```

### Getting structured output you can use in code

Free-form text is hard to parse. Ask for JSON and enforce it:

**`day4_json_output.py`**

```python
import os, json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=(
        "Extract the name, occupation and city from this sentence: "
        "'My name is Sarah, I work as a mechanical engineer, and I just "
        "moved to Seattle.'"
    ),
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
    ),
)

# Because we asked for JSON, this parses cleanly instead of needing
# string surgery on "Here is your JSON: ..."
data = json.loads(response.text)
print(data)
print("City:", data.get("city"))
```

> This is the difference between a demo and an application. An app needs `data["city"]`, not a paragraph that happens to mention Seattle.

### Inspecting the raw response — the JSON Treasure Hunt

```python
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Write a short haiku about a robot.",
)

print("--- The text ---")
print(response.text)

print("\n--- The bill ---")
print("Prompt tokens :", response.usage_metadata.prompt_token_count)
print("Output tokens :", response.usage_metadata.candidates_token_count)
print("Total tokens  :", response.usage_metadata.total_token_count)

print("\n--- The full object ---")
print(response)
```

Token counts are how API usage is billed. Print them once so you know they exist, and remember that a long prompt costs money on every single call.

### Building conversation memory

The API is stateless. To hold a conversation you send the history back each time — or use the SDK's chat helper, which does it for you:

**`day4_chat.py`**

```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chat = client.chats.create(model="gemini-3.5-flash")

print("Type 'quit' to exit.\n")

while True:
    user_message = input("You: ")
    if user_message.lower() in {"quit", "exit"}:
        break

    response = chat.send_message(user_message)
    print("AI :", response.text.strip(), "\n")
```

Ask it "What is Python?" then "Who created it?" — the second question only makes sense because the history was carried over.

### ✏️ Activity 4.9 — Prove the model has no memory

Run this and watch a chatbot fail, then succeed, on the same two questions.

**`activity_4_9.py`**

```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3.5-flash"

# --- PART 1: two separate calls. No memory between them. ---
print("=== WITHOUT memory (two independent calls) ===")

first = client.models.generate_content(
    model=MODEL_NAME, contents="What is Python?")
print("Q1: What is Python?")
print("A1:", first.text.strip()[:150], "...\n")

second = client.models.generate_content(
    model=MODEL_NAME, contents="Who created it?")     # 'it' means nothing here
print("Q2: Who created it?")
print("A2:", second.text.strip()[:200], "\n")

# --- PART 2: same two questions, in a chat that keeps history ---
print("=== WITH memory (a chat session) ===")

chat = client.chats.create(model=MODEL_NAME)

print("Q1: What is Python?")
print("A1:", chat.send_message("What is Python?").text.strip()[:150], "...\n")

print("Q2: Who created it?")
print("A2:", chat.send_message("Who created it?").text.strip()[:200])
```

**✅ Check yourself:** in Part 1, the model has no idea what "it" refers to — it will ask for clarification or guess something unrelated. In Part 2 it answers *Guido van Rossum* immediately.

**Nothing about the model changed between the two parts.** The only difference is that `chats.create()` re-sends the earlier messages with every request. **Memory is not a property of the model. It is a feature your program provides.** Every chatbot you have ever used works this way.

**🔁 Change one thing:** keep the chat going for ten more questions, then ask about something from the first message. Eventually the history exceeds the **context window** and the earliest messages fall out. Real chatbots handle this by summarising old messages rather than sending them all.

## 4.6 Group activities

The activities above are ones you do alone at your keyboard. These four are run by your instructor with the whole class.

| ID | Activity | What it teaches |
|---|---|---|
| **Activity 4.10** | The Prompting Tournament | Steering a model without using the obvious words |
| **Activity 4.11** | The AI Fact-Checker | Catching hallucination and bias in the act |
| **Activity 4.12** | The Red Team Challenge | Why text guardrails are so easy to break |
| **Activity 4.13** | The JSON Treasure Hunt | That a model response is just a nested dictionary |

### ✏️ Activity 4.14 — Your own useful prompt

Now make it real. Pick a task you genuinely do every week — summarising lecture notes, drafting a message to a professor, generating practice questions, explaining a topic to a friend.

Write it twice:

1. A weak, one-line prompt.
2. A strong prompt with all five parts.

Run both. Paste both outputs into your notes side by side, and write one sentence on what improved.

**Then do the part that matters:** actually use the strong prompt this week. A prompt you saved and reuse is worth more than fifty you tried once.

### ✅ Module 4 exit task

You should be able to show:

1. A working Gemini API call from your own machine, in the `genai` environment.
2. The temperature table from Activity 4.7, filled in with your real outputs.
3. Your before-and-after prompt from Activity 4.14.
4. A one-sentence answer to: **"Why does an LLM hallucinate?"** — in terms of how it works, not "it makes mistakes".

---

# Module 5 — Open-source models, Hugging Face and app development

**Session 5.1:** Open-source GenAI models, the Hugging Face ecosystem, integrating ML with GenAI.
**Session 5.2:** Streamlit app development and capstone planning.

## 5.1 Open-source models vs API models

| | API model (Gemini, GPT, Claude) | Open-source model (Llama, Mistral, Qwen, Gemma) |
|---|---|---|
| Where it runs | The provider's servers | Your machine, your server, or a host you choose |
| Cost | Per token | Free weights; you pay for the hardware |
| Setup | An API key | Download the weights, have enough RAM/GPU |
| Data privacy | Data leaves your network | Data can stay entirely local |
| Customisation | Prompting only | Full fine-tuning possible |
| Best for | Getting started, best quality, no infrastructure | Privacy, cost at scale, offline use, research |

Full survey with model families and sizes: [`tutorials/open-source-gen-ai.md`](tutorials/concepts/open-source-gen-ai.md).

## 5.2 Hugging Face

### 🧠 Analogy: the app store for AI models

Nobody writes their own camera app from scratch. You open the app store, search, read the reviews, check the size, and install one.

Hugging Face is that for AI models. Half a million of them, free, searchable by task. You do not train a sentiment model — you install one that someone else trained on far more data than you have access to, and you have it running in three lines.

The **model card** on each page is the store listing: what it does, what it was trained on, how big it is, what licence, and — crucially — **what it is bad at**. Read it. A model with no card is a model nobody will vouch for.

Install the libraries (do this the evening before — it is a large download):

```bash
conda activate genai
pip install transformers torch datasets evaluate gradio
```

### The `pipeline()` function

`pipeline()` handles tokenising, running the model and decoding the output in one call. It is the fastest way to use any model on the Hub.

**`day5_huggingface.py`**

```python
from transformers import pipeline

# --- 1. Sentiment analysis ---
# Always name the model explicitly. If you omit it you get a default that
# may change between library versions, and your results become
# unreproducible.
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

sentences = [
    "I love machine learning.",
    "This movie was very boring.",
    "The product quality is excellent.",
    "I am not happy with this service.",
]

for sentence, result in zip(sentences, sentiment(sentences)):
    print(f"{result['label']:<8} {result['score']:.3f}  {sentence}")

# --- 2. Text generation ---
generator = pipeline("text-generation", model="distilgpt2")

output = generator(
    "Artificial Intelligence is useful because",
    max_new_tokens=40,        # how many NEW tokens to add
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8,
)
print("\n", output[0]["generated_text"])

# --- 3. Question answering (extractive: the answer is IN the context) ---
qa = pipeline("question-answering",
              model="distilbert-base-cased-distilled-squad")

context = """
Artificial Intelligence is a branch of computer science that enables machines
to perform tasks that normally require human intelligence. These tasks include
learning, reasoning, problem-solving, understanding language, and recognising
images.
"""

answer = qa(question="What is Artificial Intelligence?", context=context)
print("\nAnswer:", answer["answer"])
print("Confidence:", round(answer["score"], 3))

# --- 4. Summarisation ---
summariser = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
long_text = context * 3
print("\nSummary:", summariser(long_text, max_length=60,
                               min_length=20)[0]["summary_text"])
```

Run it:

```bash
conda activate genai
python day5_huggingface.py
```

> **The first run is slow.** Each model is downloaded once (a few hundred MB) and cached in `~/.cache/huggingface`. Later runs start immediately. If you are on classroom wifi, expect the first run to take several minutes.

### ✏️ Activity 5.1 — Two models, one sentence

Different models trained on different data give different answers. Prove it.

**`activity_5_1.py`**

```python
from transformers import pipeline

# Two sentiment models trained on different kinds of text.
model_a = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english")
model_b = pipeline("sentiment-analysis",
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentences = [
    "The lab session was great.",
    "The lab session was fine I guess.",
    "Oh brilliant, another three-hour lab.",     # sarcasm
    "not bad at all",                            # double negative
    "The food was ok but the service was terrible.",
]

for sentence in sentences:
    a = model_a(sentence)[0]
    b = model_b(sentence)[0]
    print(f"\n{sentence}")
    print(f"  Model A (movie reviews): {a['label']:<10} {a['score']:.3f}")
    print(f"  Model B (tweets)      : {b['label']:<10} {b['score']:.3f}")
```

**✅ Check yourself:**

1. Do both models agree on every sentence? Where do they disagree?
2. **Sentence 3 is sarcasm.** Did either model catch it? Almost certainly not — sarcasm depends on tone and context that plain text does not carry.
3. **Sentence 4** is a double negative. Which model handled it better?
4. **Sentence 5** is genuinely both. Notice that a two-label model is *forced* to choose one, and its confidence tells you it is struggling.
5. Model B was trained on tweets; Model A on film reviews. **Which one would you deploy for student feedback, and why?**

**The lesson:** "which model is best?" is the wrong question. **"Best for what text?"** is the right one. Check what a model was trained on before you trust it on your data.

### ✏️ Activity 5.2 — Read a model card

Open the Hugging Face page for [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) and find each of these:

```text
Who made it?              _______________________
What task does it do?     _______________________
What data was it trained on?  ___________________
What licence?             _______________________
How many downloads this month?  _________________
What limitations or biases does the card admit to?
_______________________________________________
```

**Then decide, and justify in one sentence each:**

- Could you use this model in a **medical triage** system?
- Could you use it to **screen job applications**?
- Could you use it in a **commercial product**?

<details>
<summary>Discussion</summary>

The card for this model explicitly warns about biased predictions on certain demographic and country terms. That single line changes the answers: fine for a classroom demo or ranking film reviews; **not** fine for anything affecting a person's healthcare or employment.

**This is what Responsible AI looks like in practice.** It is not a lecture at the end of a course. It is five minutes reading a page before you use somebody else's model.
</details>

### Extractive QA is not a chatbot

The `question-answering` pipeline finds the span of text in the context that answers the question. It **cannot** answer something the context does not contain. That is a feature: it cannot hallucinate. This is exactly the mechanism behind **RAG** (Retrieval-Augmented Generation) — find the relevant text first, then answer only from it.

The full walkthrough with nine hands-on activities is in [`tutorials/hugging-face-ecosystem.md`](tutorials/concepts/hugging-face-ecosystem.md).

## 5.3 Streamlit: turning a model into an app

### 🧠 Analogy: the script that became a website

Normally, putting something on the web means learning three more languages: HTML for structure, CSS for appearance, JavaScript for behaviour. That is weeks of work before your model reaches a single user.

Streamlit removes all of it. You write ordinary Python top to bottom — and `st.title(...)` becomes a heading, `st.slider(...)` becomes a slider. **The script is the website.**

### 🧠 Analogy: the whiteboard that gets rewiped

Here is the one Streamlit behaviour that confuses everyone at first.

Imagine a lecturer who, every time a student asks a question, **wipes the whiteboard and rewrites the entire lecture from the beginning** — incorporating the new question.

That is Streamlit. Move a slider, and your **whole script runs again from line 1**. Nothing survives unless you deliberately save it.

Two consequences, and they are the source of nearly every beginner Streamlit bug:

- **`st.session_state`** is the notebook the lecturer keeps beside the whiteboard. Anything that must survive a rewipe — chat history, a running total — goes there.
- **`@st.cache_resource`** means "you already loaded this model, do not load it again". Without it, your model is read from disk on **every single click**, and your app crawls.

```bash
conda activate genai
pip install streamlit
```

**`hello_streamlit.py`**

```python
import streamlit as st

st.set_page_config(page_title="My First App", page_icon="👋")

st.title("👋 Hello, Streamlit")
st.write("This whole page is one Python file.")

name = st.text_input("What is your name?")
age = st.slider("Your age", 15, 60, 21)
course = st.selectbox("Course", ["CSE", "ECE", "EEE", "Mechanical"])

if st.button("Greet me"):
    if name.strip() == "":
        st.warning("Please enter your name first.")
    else:
        st.success(f"Hello {name}! Age {age}, studying {course}.")
        st.balloons()
```

Run it:

```bash
conda activate genai
streamlit run hello_streamlit.py
```

Your browser opens at `http://localhost:8501`. Stop the server with `Ctrl+C`.

### The widgets you will actually use

| Widget | Purpose |
|---|---|
| `st.title()` / `st.header()` / `st.write()` | Text |
| `st.text_input()` / `st.text_area()` | Typed input |
| `st.number_input()` / `st.slider()` | Numbers |
| `st.selectbox()` / `st.radio()` / `st.multiselect()` | Choices |
| `st.file_uploader()` | Uploads |
| `st.button()` / `st.form()` | Trigger an action |
| `st.dataframe()` / `st.table()` | Show a DataFrame |
| `st.line_chart()` / `st.pyplot()` | Charts |
| `st.success()` / `st.warning()` / `st.error()` | Coloured messages |
| `st.spinner()` | "Working…" while something slow runs |
| `st.chat_input()` / `st.chat_message()` | Chat interfaces |
| `st.session_state` | Remember values across re-runs |
| `@st.cache_resource` | Load a model **once**, not on every re-run |

> **`@st.cache_resource` is not optional.** Without it, your trained model is loaded from disk on every single click. With it, the model loads once and stays in memory.

### The four tutorials to work through

| Tutorial | What you build |
|---|---|
| [`streamlit-app-simple.md`](tutorials/apps/streamlit-app-simple.md) | A Gemini text generator and a chatbot |
| [`streamlit-app-advanced.md`](tutorials/apps/streamlit-app-advanced.md) | A YouTube summariser and an image-input diagnostic helper |
| [`loan-app.md`](tutorials/apps/loan-app.md) | A Streamlit app serving a **trained ML model** |
| [`ml_gen_ai.md`](tutorials/apps/ml_gen_ai.md) | ML **and** GenAI together: the model predicts, the LLM explains |

Do them in that order. Each one adds exactly one new idea.

## 5.4 Integrating ML with GenAI

### 🧠 Analogy: the doctor and the receptionist

A hospital has two people who both talk to you, and they are good at completely different things.

The **doctor** reads your test results and makes the diagnosis. Trained for years, works from measurable evidence, and is accountable for the decision.

The **receptionist** explains what happens next in language you actually understand, answers your worried questions, and is warm about it. They are excellent at communicating — and they do **not** diagnose you.

- **ML model = the doctor.** Trained on your data, measurable, reproducible. **It makes the decision.**
- **GenAI model = the receptionist.** Fluent, patient, endlessly good at explaining. **It explains the decision.**

A hospital where the receptionist diagnoses patients is a disaster. So is an application where the LLM decides the loan.

This is the most important idea of Module 5, and the strongest thing you can put in a capstone project.

```text
Structured input  →  ML model      →  prediction  ┐
(age, income,        (Random           (Approved)  ├→  GenAI  →  plain-English
 credit score)        Forest)                      ┘   (Gemini)   explanation
```

**Each part does what it is good at:**

- The **ML model** makes the decision. It is trained on your data, it is measurable, and it is reproducible.
- The **GenAI model** explains the decision in language a customer understands. It does **not** make the decision.

Getting this backwards is the classic mistake. Never let the LLM decide the loan, the diagnosis or the grade. Let it explain what the measurable model decided.

The prompt that connects them looks like this:

```python
prompt = f"""
You are a helpful loan officer assistant.

A machine learning model reviewed this application and predicted: {prediction}

Applicant details:
- Age: {age}
- Annual income: {income}
- Credit score: {credit_score}
- Loan amount: {loan_amount}
- Interest rate: {interest_rate}%
- Previous defaults: {previous_defaults}

Write a short, respectful explanation for the applicant:
1. State the decision in one sentence.
2. Give the two or three factors that most likely influenced it.
3. If rejected, give two specific, actionable steps to improve.

Do not invent information that is not in the details above.
Do not present this as financial advice.
Keep it under 150 words.
"""
```

Notice the last three lines. Those are **guardrails** — they are as much a part of the prompt as the task is.

Full working code: [`tutorials/ml_gen_ai.md`](tutorials/apps/ml_gen_ai.md).

### ✏️ Activity 5.3 — Build the loan app end to end

This is the capstone rehearsal. Follow [`tutorials/loan-app.md`](tutorials/apps/loan-app.md) and get a working app in four steps:

1. **Train and save the model** — `python train_model.py` produces `rf_model.joblib`.
2. **Run the app** — `streamlit run app.py`.
3. **Enter an applicant** who should clearly be approved, and one who should clearly be rejected. Does the app agree with your intuition?
4. **Find a case where the model surprises you.** Change one field at a time until the prediction flips.

**✅ Check yourself:** which single field, changed on its own, flipped the prediction most easily? Compare with a classmate. Then check that field against the feature-importance output from section 3.3 — **do the two agree?** If a field flips predictions easily but has low importance, something is worth investigating.

### ✏️ Activity 5.4 — Add the explanation layer

Now upgrade the app you just built into an ML + GenAI app, following [`tutorials/ml_gen_ai.md`](tutorials/apps/ml_gen_ai.md).

Then deliberately test the boundary between the two components:

1. Get a **rejection** and read the GenAI explanation. Is it accurate about *why*?
2. **Does the explanation ever contradict the model?** (For example, praising a credit score the model treated as low.) If so, your prompt is not receiving enough of the applicant's details.
3. Remove `"Do not invent information that is not in the details above."` from the prompt, run again, and look for invented facts. **Put it back.**
4. Add a guardrail of your own — for example, `"Never state or imply a specific probability."` — and check it holds.

**✅ Check yourself:** in one sentence, what would go wrong if you deleted the ML model and asked Gemini to approve the loan directly? (You would have replaced a measurable, testable, auditable decision with an unmeasurable one — and you could not tell a rejected applicant why.)

### ✅ Module 5 exit task

Have a Streamlit app running on your own machine, in the `genai` environment, that either serves a trained ML model or calls the Gemini API. Screenshot it. That screenshot is your Module 5 evidence.

---

# Activity index

Every activity in this handbook, in order. Tick them off as you go — the ✏️ ones are the course.

| # | Activity | Computer needed? | Teaches |
|---|---|---|---|
| **0.1** | [Match the analogy](#the-analogy-bank) | No | The vocabulary of the whole course |
| **0.2** | [Meet your data](#the-datasets-you-will-use) | Spreadsheet only | Looking at data before coding |
| **1.1** | [Sort the mail](#module-1--python-refresher-and-data-handling) | No | What “learning from examples” means |
| **1.2** | Number or category? | No | Regression vs classification |
| **1.3** | Teachable Machine | Browser | Training a real model with no code |
| **1.4** | Fix the broken program | Yes | Reading error messages |
| **1.5** | The marks calculator | Yes | NumPy without loops |
| **1.6** | Twelve rows you can see | Yes | Spotting data problems |
| **1.7** | Data detective | Yes | EDA on an unfamiliar dataset |
| **2.1** | The chart chooser | No | Choosing a chart from a question |
| **2.2** | Draw it before you code it | Yes | Reading your own charts |
| **2.3** | Preprocess twelve rows by hand | Yes | The full preprocessing pipeline |
| **2.4** | Beat the model | Paper first | What regression actually computes |
| **2.5** | What does a mistake cost? | No | Choosing precision vs recall |
| **2.6** | Build the useless 99% model | Yes | Why accuracy misleads |
| **3.1** | Legal or illegal augmentation? | No | Label-preserving transformations |
| **3.2** | Invent three features | Paper first | Feature engineering |
| **3.3** | Watch a model overfit, live | Yes | The train/test gap |
| **3.4** | Prove one split is unreliable | Yes | Why cross-validate |
| **4.1** | Cluster the classroom | No — stand up | What clustering does, and choosing k |
| **4.2** | Choose k, then defend it | Yes | Elbow and silhouette |
| **4.3** | Be the language model | No | Next-token prediction, hallucination |
| **4.4** | Count the tokens | Browser | Tokens, cost, context |
| **4.5** | The prompt makeover | Yes | The five parts of a prompt |
| **4.6** | All four prompt types on one task | Yes | Choosing a prompt type |
| **4.7** | The temperature dial | Yes | What `temperature` does |
| **4.8** | Top-k in the extreme | Yes | How `top_k` and temperature interact |
| **4.9** | Prove the model has no memory | Yes | Statelessness and chat history |
| **4.10** | The Prompting Tournament | Browser | Semantic steering |
| **4.11** | The AI Fact-Checker | Browser | Hallucination and bias |
| **4.12** | The Red Team Challenge | Browser | Guardrails and jailbreaks |
| **4.13** | The JSON Treasure Hunt | Yes | A response is just a dictionary |
| **4.14** | Your own useful prompt | Yes | Applying it to real work |
| **5.1** | Two models, one sentence | Yes | Model choice matters |
| **5.2** | Read a model card | Browser | Responsible model selection |
| **5.3** | Build the loan app end to end | Yes | Serving an ML model |
| **5.4** | Add the explanation layer | Yes | ML + GenAI integration |

Activities **4.10** to **4.13** are instructor-led.

## If you are short of time

Doing all 38 takes the full week. If you have to choose, these ten carry the most weight:

**1.1** (what learning from examples means) · **1.6** (see data problems) · **2.3** (preprocessing) · **2.5** (cost of a mistake) · **2.6** (why accuracy lies) · **3.3** (overfitting) · **4.3** (be the language model) · **4.5** (prompt makeover) · **4.7** (temperature) · **5.3** (build the app)

---

# Capstone project guide

## Choosing a topic

A good capstone topic satisfies all four:

1. **A dataset exists** and you have already opened it and looked at it.
2. **The target is clear** — you can say in one sentence what you are predicting.
3. **It is finishable in three weeks** alongside your other work.
4. **You can explain to a non-technical person why it matters.**

Topic ideas by track:

| Track | Project ideas |
|---|---|
| **Classical ML** | Diabetes risk prediction · Credit-score classification · Customer churn · House price prediction · Music genre classification |
| **Clustering** | Customer segmentation · Student performance grouping · Retail basket segments |
| **GenAI** | Study-notes summariser · Quiz generator from a syllabus · Interview practice bot · Regional-language explainer |
| **ML + GenAI** | Loan approval with explanations · Health risk score with lifestyle advice · Resume screener with feedback |
| **Computer vision** | Helmet detection · Fruit classification · Attendance by face recognition |

Datasets are available in the [`datasets/`](../../datasets/) folder of this repository.

## Required deliverables

```text
capstone/
├── README.md              # problem, data, approach, results, how to run
├── notebooks/
│   └── 01_eda.ipynb
│   └── 02_modelling.ipynb
├── app/
│   ├── app.py
│   └── requirements.txt
├── models/
│   └── model.joblib
└── report.pdf
```

Your `README.md` must contain:

1. **Problem statement** — one paragraph. What decision does this support, and for whom?
2. **Dataset** — source, licence, number of rows and columns, and how you cleaned it.
3. **Approach** — which models you tried, and why you chose the one you did.
4. **Results** — a metrics table. Include the models that *lost*, not only the winner.
5. **How to run** — exact commands, starting with `conda activate genai`.
6. **Limitations** — where does your model fail? Who could it treat unfairly?

Section 6 is not optional. A project that claims no limitations has not been examined honestly.

## Review milestones

| Milestone | Format | You present |
|---|---|---|
| **Review 1** | Online | Problem statement, dataset, EDA findings |
| **Review 2** | Online | Baseline model and evaluation metrics |
| **Review 3** | Online | Improved model, app demo |
| **Submission** | After Review 3 | Repository and report |
| **Final presentation** | After submission | Presentation and questions |

## Marking guide

| Area | Weight | What earns marks |
|---|---|---|
| Problem framing and EDA | 15% | Clear question, honest look at the data |
| Preprocessing | 15% | Correct handling of missing values, encoding, **no leakage** |
| Modelling and evaluation | 25% | Right metrics for the problem, models compared fairly |
| Improvement | 15% | Feature engineering, cross-validation, tuning — with before/after evidence |
| Application | 15% | Working Streamlit app |
| Responsible AI | 10% | Bias considered, limitations stated, secrets kept out of the repo |
| Presentation | 5% | Clear, on time, honest about what did not work |

## Common mistakes that cost marks

- Scaling before splitting → **data leakage** → an inflated test score.
- Reporting only accuracy on an imbalanced dataset.
- Committing `.env` or `secrets.toml` with a live API key.
- A `README` with no run instructions, so nobody can reproduce your result.
- Claiming a number you cannot reproduce when asked in the review.
- Letting the LLM make the decision instead of explaining it.

---

# Responsible AI checklist

Run through this before every submission and every demo.

## Data

- Am I allowed to use this dataset? Check the licence.
- Have I removed personal identifiers I do not need?
- Does the data represent all the groups the model will be used on?
- Do I know when it was collected and whether it is still current?

## Model

- Have I checked accuracy **separately for each group** (gender, age band, region)?
- Do I know which features drive the prediction?
- Have I stated the model's failure modes in the README?
- Is a human making the final decision on anything consequential?

## Generative AI

- Have I verified the facts in anything I present as true?
- Have I told users that the content is AI-generated?
- Am I keeping personal or confidential data out of prompts?
- Are my guardrails in the prompt (what the model must not do)?
- Have I tested what happens when a user tries to misuse it?

## Security

- Is my API key out of the code and in `.env` or `secrets.toml`?
- Is `.gitignore` in place **before** my first commit?
- Have I checked my repository for keys already committed?

> **If you ever push a key by accident**, revoke it immediately at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and generate a new one. Deleting the commit is **not** enough — the key is already in the Git history and in anyone's clone.

## Three principles to carry out of this course

1. **The model is a tool, not an authority.** It supports a human decision; it does not replace one.
2. **Bias in, bias out.** A model trained on unfair historical decisions will reproduce them, efficiently and at scale.
3. **If you cannot explain it, do not deploy it.** For anything affecting a person's money, health, education or freedom, explainability is a requirement, not a nice-to-have.

---

# Troubleshooting

**Every error you are likely to hit, with its fix, is in the [Troubleshooting Guide](troubleshooting.md).**

Before you look anything up, check these four. Nine times in ten, one of them is the answer:

1. **Does my prompt show `(genai)`?** If not, activate the environment.
2. **Am I in the right folder?**
3. **Did I run the training script first?** Apps that load a `.joblib` need it to exist.
4. **Is VS Code using the right interpreter?** `Ctrl+Shift+P` → **Python: Select Interpreter**.

| Your problem | Section |
|---|---|
| `command not found`, `ModuleNotFoundError`, wrong Python | [Environment problems](troubleshooting.md#environment-problems) |
| `API key not valid`, `429`, empty responses | [API problems](troubleshooting.md#api-problems) |
| `could not convert string to float`, suspicious accuracy | [Modelling problems](troubleshooting.md#modelling-problems) |
| App loses state, port in use, `FileNotFoundError` | [Streamlit problems](troubleshooting.md#streamlit-problems) |
| Kernel issues, `NameError` for a variable you defined | [Notebook problems](troubleshooting.md#notebook-problems) |

> **How to read an error.** Read the **last line first** — that line names the actual problem. Everything above it is just the path the computer took to get there.

---


# Self-check questions

Answer these without looking anything up. If you cannot, go back to that section.

## Module 1

1. In one sentence, how does Machine Learning differ from traditional programming?
2. When would you choose regression over classification?
3. Which five commands do you run first on any new dataset?
4. Why fill missing numbers with the median rather than the mean?

## Module 2

5. A model has 99% accuracy on fraud detection. Why might it still be worthless?
6. What exactly is the difference between precision and recall?
7. Why must you fit the scaler on the training set only?
8. What does R² = 0 mean? What would a negative R² mean?

## Module 3

9. Training accuracy 0.99, test accuracy 0.71. What is happening and what are two fixes?
10. Why is 5-fold cross-validation more trustworthy than a single split?
11. What is the difference between a parameter and a hyperparameter?
12. When would you use `RandomizedSearchCV` instead of `GridSearchCV`?

## Module 4

13. Why does an LLM hallucinate? Answer in terms of how it works, not "it makes mistakes".
14. You need output in a fixed format. Which prompt type, and why?
15. What does `temperature=0` do, and when would you want it?
16. Your chatbot forgets the previous question. Why, and what fixes it?

## Module 5

17. Give one situation where an open-source model beats an API model, and one where the reverse is true.
18. Why must a model load be wrapped in `@st.cache_resource`?
19. In an ML + GenAI app, which component makes the decision, and why that one?
20. Name three things that must never be committed to your Git repository.

---

# Glossary

| Term | Meaning |
|---|---|
| **Accuracy** | Fraction of predictions that were correct |
| **API** | An interface that lets one program call another over the network |
| **Chain-of-thought** | Prompting the model to show its reasoning before answering |
| **Classification** | Predicting a category |
| **Clustering** | Grouping unlabelled data by similarity |
| **Cross-validation** | Training and testing over several different splits and averaging |
| **Data leakage** | Test information reaching the training process; inflates scores |
| **Deep learning** | Neural networks with several hidden layers |
| **Epoch** | One full pass through the training data |
| **Feature** | An input column |
| **Feature engineering** | Creating new, more informative columns |
| **Few-shot** | A prompt containing several worked examples |
| **Fine-tuning** | Further training a pretrained model on your own data |
| **Grounding** | Supplying source text so the model does not have to guess |
| **Guardrail** | A rule constraining what the model may output |
| **Hallucination** | Fluent, confident, incorrect output |
| **Hyperparameter** | A setting you choose before training |
| **Inference** | Using a trained model to make a prediction |
| **Label** | The correct answer for a training example |
| **LLM** | Large Language Model — a very large next-token predictor |
| **Model** | The rules learned from data |
| **Normalisation** | Rescaling values to a common range |
| **Overfitting** | Memorising the training data instead of the pattern |
| **PCA** | Compressing many correlated columns into fewer components |
| **Pipeline** | Chained preprocessing and modelling steps |
| **Precision** | Of the positives I predicted, how many were right |
| **Prompt** | Everything you send to a generative model |
| **RAG** | Retrieval-Augmented Generation — retrieve source text, then answer from it |
| **Recall** | Of the real positives, how many I found |
| **Regression** | Predicting a number |
| **Regularisation** | Penalising complexity to reduce overfitting |
| **Session state** | Streamlit's store for values that survive a re-run |
| **Supervised learning** | Learning from labelled examples |
| **System instruction** | A standing instruction shaping every model reply |
| **Target** | The column you are predicting |
| **Temperature** | Randomness of token selection |
| **Token** | A chunk of text, roughly ¾ of a word |
| **Top-k** | Sample only from the k most likely tokens |
| **Top-p** | Sample only from the top tokens summing to probability p |
| **Training set** | Data the model learns from |
| **Test set** | Held-out data used to measure real performance |
| **Underfitting** | Model too simple to capture the pattern |
| **Unsupervised learning** | Learning structure from unlabelled data |
| **Zero-shot** | A prompt with no examples |

---

# Where to go next

## Course material in this repository

| File | What it is |
|---|---|
| [`README.md`](README.md) | Index of everything here |
| [`setup-guide.md`](setup-guide.md) | Install for Windows, Ubuntu, macOS — venv and conda |
| [`troubleshooting.md`](troubleshooting.md) | Every error and its fix |
| [`notebooks/`](notebooks) | Nine Colab/Jupyter notebooks, outputs included |
| [`exercises-assignments.md`](exercises-assignments.md) | Exercises per concept and five graded assignments |
| [`tutorials/apps/`](tutorials/apps) | Build something — 15 Streamlit apps plus four guides |
| [`tutorials/concepts/`](tutorials/concepts) | Read about something — three in-depth tutorials |
| [`prompts.md`](prompts.md) | Copy-paste prompt library |

## Practice

- [Python exercises](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/python-exercises.ipynb)
- [NumPy exercises](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/numpy-exercises.ipynb)
- [Pandas exercises](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/pandas-exercises.ipynb)
- [Matplotlib & Seaborn exercises](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/matplotlib-seaborn-exercises.ipynb)
- [ML practice assessment](https://github.com/tech4alltraining/aiml/blob/main/assessments/ml_ai_practice.md)

## Official documentation — read the source, not a blog

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Gemini API documentation](https://ai.google.dev/gemini-api/docs)
- [Hugging Face documentation](https://huggingface.co/docs)

## Learn more

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn](https://www.kaggle.com/learn) — short, practical, free
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

---

**Final word.** You will not remember every method in this handbook, and you are not expected to. What you should carry away is the *shape* of the work: look at your data before you model it, split before you scale, choose the metric that matches the cost of being wrong, check whether the model generalises, and never let a system decide something important without a human who can explain the decision. The libraries will change. That will not.
