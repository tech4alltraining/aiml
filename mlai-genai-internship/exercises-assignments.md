# Exercises & Assignments

**Practice for every concept in the programme — simple to advanced, plus scenario problems and graded assignments.**

| Related | For |
|---|---|
| [Student Handbook](student-handbook.md) | The concepts these exercises practise |
| [Notebooks](notebooks) | Runnable versions of the demos |
| [Streamlit Apps](tutorials/apps/streamlit-apps-collection.md) | 15 apps, each with its own exercises |
| [Setup Guide](setup-guide.md) · [Troubleshooting](troubleshooting.md) | Getting unstuck |

---

## How this file is organised

Each module has four tiers. **Do them in order.** Module 0 is optional — skip it if you already know Python. The tiers get longer as you go.

| Tier | What it is |
|---|---|
| 🟢 **Warm-up** | One idea, one answer. Often no computer |
| 🔵 **Practice** | Write code that works |
| 🟠 **Challenge** | Combine several ideas |
| 🔴 **Scenario** | A realistic situation with no single right answer. Judgement required |

**Scenario problems are the ones that matter.** They are how the work actually arrives: ambiguous, with a stakeholder who wants something, and a right answer that depends on what it costs to be wrong.

### Answering rules

1. **Attempt before you look.** Solutions are in `<details>` blocks — opening one early costs you the exercise.
2. **Every number needs a sentence.** "MAE = 12094" is not an answer. "The model is typically off by about ₹12,000, which is 15% of the average salary" is.
3. **Report what failed.** An exercise where your idea did not work is still a completed exercise. Say so.

---

# Module 0 — Python from scratch

**If you have never written Python, do this before anything else.**

The three foundation notebooks contain **35 short exercises, 7 quizzes and 20 tasks**, all with solutions:

| Notebook | Covers | Exercises |
|---|---|---|
| [00a — Basics](notebooks/00a_python_foundations.ipynb) | printing, variables, types, operators, strings | 20 |
| [00b — Structures](notebooks/00b_python_foundations_2.ipynb) | collections, conditions, loops, functions | 15 |
| [00c — Scenarios](notebooks/00c_python_scenarios.ipynb) | ten real-world problems, task by task | 70 tasks |

**The ten scenarios in 00c**, in order of difficulty:

| # | Problem | Practises |
|---|---|---|
| 1 | Student Score Analyser | dict, list, loops, conditions |
| 2 | Grocery Billing System | dict, list, set, membership |
| 3 | Hospital Triage | dict of dicts, nested conditions |
| 4 | Attendance Calculator | percentages, formatting |
| 5 | Library Management | functions, returning values |
| 6 | Text Message Analyser | strings, splitting, counting |
| 7 | Temperature Station | functions, min/max |
| 8 | Cricket Score Tracker | aggregation, sorting |
| 9 | Expense Tracker | dict aggregation, percentages |
| 10 | **Simple ML Data Prep** | **everything — the bridge to Module 1** |

> **Scenario 10 is the important one.** You clean a messy dataset by hand in plain Python — duplicates, missing values, inconsistent categories, encoding, train/test split. Tomorrow Pandas does all of it in six lines. **Students who understand the why find Pandas easy; students who only learn the six lines get stuck the first time their data is different.**

More drills: [`python-internship/`](../python-internship/) has topic-by-topic exercise notebooks.

---

# Module 1 — Python, NumPy, Pandas, EDA

## 🟢 Warm-up

**1.1** Number or category? Answer N or C:

```text
a. How much will this used car sell for?
b. Will this student pass the semester?
c. How many customers will visit tomorrow?
d. Is this X-ray showing pneumonia?
e. What rating out of 5 will this user give?
```

<details><summary>Answers</summary>

a N · b C · c N · d C · e N *or* C — a rating can be modelled either way, and real teams argue about exactly this. If you spotted the ambiguity, you are thinking like a data scientist.
</details>

**1.2** What does each line print?

```python
marks = [78, 92, 65, 88]
print(marks[0])
print(marks[-1])
print(marks[1:3])
print(len(marks[1:3]))
```

<details><summary>Answers</summary>

`78` · `88` · `[92, 65]` · `2` — a slice includes the start and excludes the stop.
</details>

**1.3** Without running it, say what is wrong:

```python
df = pd.read_csv("data.csv")
df.dropna()
print(df.isnull().sum())
```

<details><summary>Answer</summary>

`dropna()` returns a **new** DataFrame; it does not modify `df`. The missing values are still there. Fix: `df = df.dropna()`.

This catches almost everybody once. Most Pandas methods return a copy.
</details>

## 🔵 Practice

**1.4** Using NumPy only — **no `for` loops** — on `marks = np.array([78, 92, 65, 88, 45, 97, 55, 71, 83, 60])`:

- a. The average to 1 decimal place
- b. How many scored 75 or above
- c. Only the marks below 60
- d. Everyone boosted by 5, capped at 100
- e. Each mark as a percentage of the class maximum

<details><summary>Solution</summary>

```python
print("a:", round(marks.mean(), 1))
print("b:", (marks >= 75).sum())
print("c:", marks[marks < 60])
print("d:", np.minimum(marks + 5, 100))
print("e:", (marks / marks.max() * 100).round(1))
```
</details>

**1.5** Load `datasets/regression/cardekho_dataset.csv` and answer with **one sentence each**:

- a. How many cars, and how many columns?
- b. Which fuel type has the highest average selling price?
- c. Which brand appears most often?
- d. Which columns have missing values?
- e. What is the most expensive car, and is that price plausible?

**1.6** On `datasets/classification/archive/titanic.csv`:

- a. What fraction of passengers survived?
- b. Survival rate by `sex`?
- c. Survival rate by `class`?
- d. What is the average `age`, and how many rows are missing it?
- e. Write one sentence: who was most likely to survive?

## 🟠 Challenge

**1.7 — The data quality report.** Pick any dataset from [`datasets/`](../datasets/) that was not used in class. Produce a one-page report containing:

```text
1. Shape, and what one row represents
2. Every column: type, and whether it is a feature or the target
3. Missing values: which columns, how many, and your recommended fix for each
4. Outliers: at least one column checked with the IQR rule, with your judgement
5. Duplicates: how many, and whether they are genuine or errors
6. The target: balanced or imbalanced, and what that means for your metric choice
7. Three questions this dataset could answer
```

**1.8 — Reimplement `describe()`.** Write `my_describe(df)` returning count, mean, std, min, 25%, 50%, 75% and max for every numeric column — **without** calling `.describe()`. Compare against the real thing.

<details><summary>Hint</summary>

`df.select_dtypes(include="number").columns` gives you the numeric columns. `series.quantile(0.25)` gives a percentile. Build a dict of dicts and hand it to `pd.DataFrame`.
</details>

## 🔴 Scenario

**1.9 — The suspicious spreadsheet.**

> *You join a project. A colleague sends `sales_data.csv` and says "it's clean, just build the model". You open it and find: 3% of `revenue` values are negative, `region` contains both `"North"` and `"north"`, 12% of `customer_id` values are duplicated, and `date` is text in three different formats.*

Write a short memo (under 400 words) covering:

1. Which problems you can fix yourself, and how.
2. Which need your colleague's input, and the exact questions you would ask.
3. What you would do if they replied "just drop the bad rows".
4. Whether you would proceed. Justify it.

> **What this is really testing:** whether you push back. "Just drop the bad rows" discards 12% of the data and may bias everything downstream. Negative revenue might be refunds — genuine and meaningful — or a data error. **You cannot tell from the data alone, and guessing is the wrong move.**

---

# Module 2 — Visualisation, preprocessing, supervised learning

## 🟢 Warm-up

**2.1** Which chart answers each question?

```text
a. Do students who study more get better marks?
b. How many students chose each elective?
c. Is anyone's bill absurdly higher than everyone else's?
d. Have monthly sales risen over two years?
e. Which of our five branches earns most on average?
```

<details><summary>Answers</summary>

a Scatter · b Count plot · c Box plot · d Line plot · e Bar plot

**b and e are the pair people confuse.** Count plot = how many rows per category. Bar plot = the average of some *other* column, per category.
</details>

**2.2** A fraud model reports **99.2% accuracy**. Give two reasons this might be worthless.

<details><summary>Answer</summary>

1. **Class imbalance.** If 99% of transactions are legitimate, always predicting "not fraud" scores 99% and catches nothing. Recall would be 0.
2. **Data leakage.** If a column like `investigation_opened` leaked into the features, the model is reading the answer.

Ask for the **confusion matrix**, and for **recall on the fraud class**.
</details>

**2.3** Why must the scaler be fitted on the training set only?

<details><summary>Answer</summary>

Fitting on everything lets the test set's mean and range leak into training. Your test score becomes inflated and meaningless — you were **studying with the answer key**. It is called data leakage, and it is the most common serious beginner error.
</details>

## 🔵 Practice

**2.4** On the cardekho dataset, produce a 2×2 figure with:
- Histogram of `selling_price` · Box plot of `km_driven` · Scatter of `vehicle_age` vs `selling_price` · Bar plot of average price by `fuel_type`

Then write **one sentence per panel** saying what it *tells you* — not what it is.

**2.5** Take `datasets/prepreprocessing/pre_data.csv` (12 rows) and preprocess it fully: remove duplicates, fill missing values sensibly, handle the outlier, encode the text, scale the numbers. **Print the data after every step** and justify each choice in a comment.

**2.6** Train a linear regression on `advertising.csv` predicting `Sales` from `TV`, `Radio` and `Newspaper`.

- a. Report MAE, RMSE and R², each with a plain-English sentence.
- b. Which channel has the largest coefficient?
- c. Drop `Newspaper` and retrain. Did the score change much? What does that tell you?

**2.7** Train a classifier on `heart_failure_raw.csv` predicting `DEATH_EVENT`.

- a. Report accuracy, precision, recall and F1.
- b. Print the confusion matrix and say, in words, what each of the four numbers means *for a patient*.
- c. **Which metric matters most here, and why?**

## 🟠 Challenge

**2.8 — Beat the baseline.** On `diabetes_prediction_dataset.csv`:

1. Compute the `DummyClassifier` baseline first. Write it down.
2. Train three different models.
3. Produce a comparison table: accuracy, precision, recall, F1.
4. **State clearly which models actually beat the baseline** on recall, not just accuracy.
5. Recommend one model and defend the choice in three sentences.

**2.9 — The threshold dial.** A classifier's default cut-off is 0.5. Using `predict_proba`:

1. Compute precision and recall at thresholds 0.1, 0.3, 0.5, 0.7, 0.9.
2. Plot both against the threshold.
3. Answer: for a **cancer screening** tool, which threshold would you ship, and what does that cost you?

<details><summary>Hint</summary>

```python
probabilities = model.predict_proba(X_test)[:, 1]
y_pred = (probabilities >= threshold).astype(int)
```

You will see precision rise and recall fall as the threshold rises. **There is no setting that improves both** — that is the whole point.
</details>

## 🔴 Scenario

**2.10 — The hiring filter.**

> *A company asks you to build a model that screens CVs, trained on ten years of their past hiring decisions. Historically 85% of the people they hired were men. The model reaches 91% accuracy at predicting who they would have hired.*

Answer in a memo:

1. What has this model actually learned?
2. Why is 91% accuracy the wrong thing to celebrate?
3. What would you measure to check whether it is fair? Be specific about the metric.
4. Removing the `gender` column is proposed as a fix. Why is that insufficient?
5. Would you build it? If yes, under what conditions; if no, what would you propose instead?

<details><summary>What a strong answer covers</summary>

- The model learned **past decisions**, not job performance. If hiring was biased, the model reproduces it — faster and at larger scale.
- Accuracy measures agreement with the historical decision, so a perfectly biased model scores 100%.
- Measure **selection rate and recall per group** (gender, age band, university). Compare them.
- Removing `gender` does not work because of **proxies**: names, university, sports, gaps in employment, even postcode. The model reconstructs the protected attribute from correlated columns.
- The strongest answers question the framing: a model that predicts *past decisions* is the wrong target. A model predicting *on-the-job performance* would need performance data — which the company may not have, and which is itself measured by potentially biased managers.
</details>

**2.11 — The 40% model.**

> *Your regression model gets R² = 0.40 on house prices. Your project partner says "that's terrible, let's try a neural network".*

1. Is R² = 0.40 necessarily bad? Give a case where it would be acceptable.
2. Name three things you would try **before** changing the algorithm.
3. What would you check to find out whether the ceiling is the model or the data?

---

# Module 3 — Feature engineering and model improvement

## 🟢 Warm-up

**3.1** Legal or illegal augmentation?

```text
a. Cat vs dog          -> flip left-right
b. Handwritten digits  -> flip left-right
c. Road sign detection -> rotate 180°
d. Chest X-ray         -> flip left-right
e. Fruit classifier    -> change brightness
```

<details><summary>Answers</summary>

a Legal · b **Illegal** (a flipped 2 is not a 2) · c **Illegal** (upside-down signs do not occur) · d **Illegal** (organs are not symmetric; the heart is on the left, and radiologists use side diagnostically) · e Legal, and genuinely helpful

**The rule:** would a human still give this the same label?
</details>

**3.2** Training accuracy 0.99, test accuracy 0.71. Diagnose it and give two fixes.

<details><summary>Answer</summary>

**Overfitting.** The model memorised the training rows including their noise.

Fixes: reduce complexity (`max_depth`, `min_samples_leaf`), get more data, remove features, add regularisation, or use cross-validation to pick better hyperparameters.
</details>

**3.3** Parameter or hyperparameter?

```text
a. The slope of a linear regression
b. n_estimators in a Random Forest
c. The split points inside a decision tree
d. max_depth
e. The learning rate
```

<details><summary>Answers</summary>

a Parameter · b Hyperparameter · c Parameter · d Hyperparameter · e Hyperparameter

**Parameters are learned during `fit()`. Hyperparameters you choose before it.**
</details>

## 🔵 Practice

**3.4** Invent three features for the loan dataset. For each: name, formula, and one sentence on why it should help. Then **test one** — train with and without it, and report both cross-validation scores.

**3.5** On the loan dataset, run a decision tree at `max_depth` = 1, 2, 3, 5, 10, 20, None. Build a table of train score, test score and gap. State the depth you would ship and why.

**3.6** Prove a single split is unreliable: train the same model with `random_state` = 0, 1, 2, 3, 4. Report the spread, then report the 5-fold cross-validation mean and std. Write one sentence on which number you would put in a report.

**3.7** Run `GridSearchCV` on a Random Forest with at least 12 combinations. Report the best parameters, the CV score, and the test score. Then run `RandomizedSearchCV` with `n_iter=10` and compare **both the score and the number of fits**.

## 🟠 Challenge

**3.8 — The improvement log.** Take your Module 2 classifier and improve it in stages. Produce this table with real numbers:

| # | Change | CV mean | CV std | Test | Better? |
|---|---|---|---|---|---|
| 0 | Baseline (DummyClassifier) | | | | — |
| 1 | Module 2 model, untouched | | | | |
| 2 | + engineered features | | | | |
| 3 | + feature selection | | | | |
| 4 | + tuned hyperparameters | | | | |

**Rules:** change one thing per row. Keep the rows that made things *worse*. Finish with three sentences on which change gave the most improvement per unit of effort.

**3.9 — PCA trade-off.** On a dataset with 10+ numeric columns:

1. Train a model on all features. Record the score and the fit time.
2. Apply PCA keeping 95% of variance. Retrain. Record both again.
3. Apply PCA keeping 80%. Retrain.
4. Produce a table: components, score, time.
5. **Answer: would you use PCA here?** Include interpretability in your reasoning, not just the score.

## 🔴 Scenario

**3.10 — The score that vanished.**

> *Your model scores 94% in your notebook. You hand it to a teammate; they run it on next month's data and get 71%. Nothing in the code changed.*

List **five** possible causes, and for each say exactly what you would check to confirm or rule it out.

<details><summary>Causes worth having on your list</summary>

1. **Data leakage** — a feature in training that is unavailable, or computed differently, at prediction time. Check whether any column is derived from the target.
2. **A lucky split** — 94% was one fortunate `random_state`. Check by cross-validating.
3. **Distribution shift** — next month's customers genuinely differ. Compare feature distributions between the two periods.
4. **Preprocessing mismatch** — the scaler or encoder was refitted on the new data instead of reused. Check that the fitted objects were saved and loaded.
5. **Column order or naming drift** — the new CSV has the same columns in a different order. This fails **silently** and is nastier than an error.
</details>

**3.11 — The 0.3% improvement.**

> *After two days of tuning you improve accuracy from 89.1% to 89.4%. Your manager asks whether it is worth deploying. Cross-validation std is 0.008.*

1. Is a 0.3% improvement meaningful given that std? Show your reasoning.
2. What would you need to be confident it is real?
3. What is the *cost* of deploying a new model, beyond the code?
4. What do you tell your manager, in three sentences?

---

# Module 4 — Clustering and Generative AI

## 🟢 Warm-up

**4.1** Why does clustering have no accuracy score?

<details><summary>Answer</summary>

There is no answer key. Nobody labelled the "correct" groups, so there is nothing to be accurate against. You judge clustering by whether the groups are **useful and defensible**, using guides like silhouette score — and by whether you can name them.
</details>

**4.2** Explain in one sentence, *in terms of how it works*, why an LLM hallucinates.

<details><summary>Answer</summary>

It predicts what text is **probable** next, not what is **true** — it has no database of facts to check against, so a plausible-sounding fabricated citation is exactly as likely to be produced as a real one.

"It makes mistakes" is not an answer. The mechanism is the answer.
</details>

**4.3** Which prompt type for each task?

```text
a. Translate this sentence to French
b. Output must be exactly "Name | Age | City"
c. Sort tickets into 5 categories with fuzzy boundaries
d. A multi-step logic puzzle
```

<details><summary>Answers</summary>

a Zero-shot · b One-shot · c Few-shot · d Chain-of-thought
</details>

**4.4** `temperature=1.0` but `top_k=1`. Will three runs differ? Why?

<details><summary>Answer</summary>

**No — they will be identical.** `top_k=1` allows only one candidate token, and temperature only controls how adventurously you choose *among the candidates*. **A high temperature cannot create variety that `top_k` already removed.**
</details>

## 🔵 Practice

**4.5** On `Mall_Customers.csv`: run K-Means for k = 2 to 10, plot both the elbow and silhouette curves, choose a k, and **give every segment a business name** based on its profile table. Justify your k in one sentence a shop manager would accept.

**4.6** Take one weak prompt and rewrite it with all five parts (role, task, context, constraints, format). Run both. Paste both outputs and write one sentence on what improved.

**4.7** Run the same prompt at `temperature` 0.0 and 1.0, three times each. Record all six outputs in a table and answer: which setting would you use for extracting invoice dates, and which for naming a college fest?

**4.8** Write the *same* classification task four ways — zero-shot, one-shot, few-shot, chain-of-thought. Test all four on this deliberately awkward input:

```text
"The lab sessions were fine but honestly three hours is too long,
maybe split it into two."
```

Build a table: what label, how much filler around it, and **would this work in code?**

## 🟠 Challenge

**4.9 — The grounded answerer.** Write a script that:

1. Reads a text file.
2. Takes a question.
3. Answers **only** from the file, replying `"Not stated in the provided document."` otherwise.
4. Quotes the supporting sentence.
5. Prints the token count.

Test it with three questions that *are* answerable and three that are not. **Report how many times it correctly refused.**

**4.10 — Structured output.** Build a script that extracts, from any block of text, a JSON object with keys `people`, `places`, `dates`, `organisations` — each a list. Use `response_mime_type="application/json"` and `json.loads`. Run it on five different paragraphs and report how often the JSON parsed first time.

## 🔴 Scenario

**4.11 — The confident citation.**

> *A classmate submits a report citing "Kumar & Sharma (2019), Journal of Applied Machine Learning, 14(3), pp. 221–239". You cannot find it anywhere. They insist ChatGPT gave it to them.*

1. Explain, mechanically, how the model produced that citation.
2. Why does it look *so* convincingly real?
3. What should your classmate have done?
4. Design a rule for your project team that prevents this. It must be checkable by someone else.

**4.12 — The support bot.**

> *A company wants an LLM chatbot to answer customer questions about refund policy. Their policy is a 12-page PDF that changes monthly.*

1. Why is fine-tuning a poor fit here?
2. Sketch a grounded (RAG-style) architecture instead.
3. What happens when a customer asks something the PDF does not cover? Write the exact prompt line that handles it.
4. Name three failure modes and a mitigation for each.
5. What should a human still be responsible for?

---

# Module 5 — Open-source models, Hugging Face, apps

## 🟢 Warm-up

**5.1** One case where an open-source model beats an API model, and one where the reverse is true.

<details><summary>Answer</summary>

**Open-source wins:** patient records that cannot legally leave the hospital network; or millions of calls per day where per-token cost dominates.

**API wins:** a two-week project where you need the best quality now and have no GPU or infrastructure.
</details>

**5.2** Why must a Streamlit model load be wrapped in `@st.cache_resource`?

<details><summary>Answer</summary>

Streamlit re-runs the whole script on **every** interaction. Without caching, the model is read from disk on every click and the app crawls.
</details>

**5.3** In an ML + GenAI app, which component makes the decision, and why that one?

<details><summary>Answer</summary>

**The ML model.** It is trained on your data, measurable, reproducible and auditable. The LLM explains the decision — it is fluent but not reproducible and cannot be evaluated with a confusion matrix.

**The doctor diagnoses; the receptionist explains.**
</details>

## 🔵 Practice

**5.4** Run the same five sentences through two different sentiment models. Build a comparison table and answer: which would you deploy for student feedback, and why?

**5.5** Open a Hugging Face model card and record: who made it, training data, licence, and **what limitations it admits to**. Then answer whether you could use it for medical triage, CV screening, or a commercial product — one sentence each.

**5.6** Build app **A1** (salary predictor) from the [apps collection](tutorials/apps/streamlit-apps-collection.md) and complete its five exercises.

**5.7** Build app **A3** (iris classifier). Find the exact petal length at which the prediction flips from versicolor to virginica. **That is the decision boundary** — report the value.

## 🟠 Challenge

**5.8 — The batch tool.** Extend app **A5** so it:

1. Lets the user set the low-confidence threshold with a slider.
2. Shows the runner-up class and its probability.
3. **Warns when uploaded values fall far outside the training range.**
4. Refuses gracefully on a malformed CSV.

Point 3 is the important one: a file in millimetres instead of centimetres produces confident nonsense, and **the model cannot tell you the units are wrong**.

**5.9 — The explanation layer.** Build app **C1**, then improve it: pass the model's **actual top-three feature importances** into the prompt so the explanation reflects what the model really weighted, rather than what the LLM guesses usually matters in loans.

Compare the explanations before and after. **Which is more honest?**

## 🔴 Scenario

**5.10 — "Why was I rejected?"**

> *Your loan app rejects an applicant. They phone and ask: "Why was I rejected when my friend, who earns less than me, was approved?"*

1. Can your current app answer that? Precisely what stops it?
2. What would it need in order to?
3. Is the GenAI explanation sufficient as an answer? Why not?
4. In several jurisdictions applicants have a legal right to an explanation of an automated decision. Does your app satisfy that?
5. Redesign the app so it can answer this question honestly.

**5.11 — The 96% classifier.**

> *You classify 10,000 support tickets at 96% accuracy. The GenAI summary reports "most complaints concern billing". Management wants to move staff to the billing team.*

1. 96% accuracy means roughly 400 tickets are misclassified. Could that change the conclusion?
2. How would you verify "most complaints concern billing" before repeating it?
3. What would you check about the **confidence distribution**?
4. Write the two sentences you would add to the report so management understands the uncertainty.

---

# Assignments

Longer, graded work. Each one is a deliverable, not an exercise.

## Assignment 1 — The dataset investigation
**After Module 2 · Weight: 15%**

Choose a dataset from [`datasets/`](../datasets/) that was **not** used in class.

**Deliverable:** a notebook plus a one-page summary.

| Requirement | Marks |
|---|---|
| Shape, columns, and what one row represents | 10 |
| Missing values found, with a justified fix for each | 15 |
| At least four charts, each answering a stated question | 20 |
| Outlier analysis with your judgement, not just detection | 15 |
| The target identified, with its balance reported | 10 |
| Three questions the data could answer | 10 |
| **One thing you found that surprised you** | 10 |
| Written clearly, with each number explained in a sentence | 10 |

## Assignment 2 — The honest model comparison
**After Module 3 · Weight: 25%**

Build the best classifier you can for a dataset of your choice.

| Requirement | Marks |
|---|---|
| Baseline (`DummyClassifier`) computed and reported first | 10 |
| At least four models compared on identical splits | 15 |
| Correct metrics for the problem, with justification | 15 |
| Cross-validation, reporting mean **and** std | 15 |
| Feature engineering, with before/after evidence | 15 |
| Hyperparameter tuning, with before/after evidence | 10 |
| **An improvement log that includes changes that failed** | 10 |
| A recommendation, defended in under 200 words | 10 |

> **Marks are deducted for:** scaling before splitting, reporting only accuracy on imbalanced data, or a results table with no failures in it.

## Assignment 3 — The prompt portfolio
**After Module 4 · Weight: 15%**

Five prompts that solve real problems **you actually have**.

For each: the weak version, the strong five-part version, both outputs, the prompt type you chose and why, and one guardrail line with an explanation of what it prevents.

At least one must use structured JSON output and be parsed in Python.

## Assignment 4 — The application
**After Module 5 · Weight: 20%**

One working Streamlit app, from any category.

| Requirement | Marks |
|---|---|
| It runs from a clean checkout following your README | 20 |
| Input validation, and it fails helpfully | 15 |
| `@st.cache_resource` used correctly | 10 |
| Confidence or uncertainty shown, not just a bare answer | 15 |
| A visible responsible-AI notice appropriate to the domain | 10 |
| No secrets in the repository (checked) | 10 |
| README with exact run commands, starting from environment activation | 10 |
| **A stated limitation section** | 10 |

## Assignment 5 — The capstone
**The capstone project · Weight: 25%**

Full requirements, deliverables and marking are in the [Capstone project guide](student-handbook.md#capstone-project-guide).

---

# Self-assessment

Tick honestly. Anything unticked is where to spend your next hour.

## Module 1
- I can load a CSV and describe it in five sentences without help
- I know why `df.dropna()` alone changes nothing
- I can filter a DataFrame on two conditions
- I can explain why the median beats the mean for filling gaps

## Module 2
- I choose a chart from the question, not from habit
- I can explain data leakage with an analogy
- I always split before I scale
- I compute a baseline before celebrating any score
- I can say which of precision and recall matters for a given problem, and why

## Module 3
- I can spot overfitting from train and test scores
- I report cross-validation std, not just the mean
- I know the difference between a parameter and a hyperparameter
- I have engineered a feature and measured whether it helped

## Module 4
- I can explain hallucination mechanically
- I write prompts with all five parts
- I can pick the right prompt type for a task
- I know what temperature, top-p and top-k each do
- I know why a chatbot needs the history resent

## Module 5
- I can run a Hugging Face model with `pipeline()`
- I read the model card before trusting a model
- I have a Streamlit app running on my own machine
- I can say which component decides and which explains — and why

## Throughout
- My API key has never been in a file I committed
- I read the **last line** of an error first
- I have said "I do not know" at least once, and then found out
