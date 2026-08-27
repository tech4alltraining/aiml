# Session Plan

**ML/AI & GenAI Internship Program — 5 classroom days + 3 project weeks**

This is the trainer's time-blocked plan. Students should use the [Student Handbook](../student-handbook.md) instead; the activity IDs below match it exactly.

Each classroom day runs **two sessions of roughly three hours**, with a break in the middle of each.

| Column | Meaning |
|---|---|
| **Mode** | 🗣️ Talk · 💻 Live code · ✏️ Student activity · 👥 Group work |
| **Ref** | Where the material lives |

---

## Day 0 — Python pre-work (optional, self-paced)

**For students who have never written Python.** Set this as pre-course work at least a week before Day 1.

| Notebook | Covers | Time |
|---|---|---|
| [00a — Basics](../notebooks/00a_python_foundations.ipynb) | printing, variables, types, operators, strings | 2 hours |
| [00b — Structures](../notebooks/00b_python_foundations_2.ipynb) | collections, conditions, loops, functions | 2 hours |
| [00c — Scenarios](../notebooks/00c_python_scenarios.ipynb) | ten real-world problems, task by task | 3-4 hours |

**Ask students to complete Scenario 10 in particular.** It walks them through cleaning a messy dataset by hand — duplicates, missing values, inconsistent categories, encoding, train/test split — which is exactly what Session 1.2 and Session 2.1 then do with Pandas. Students who have done it by hand understand *why*; students who have not just copy six lines.

If a student arrives on Day 1 without having done this, point them at 00a and 00b during the Python refresher block.

---

## Day 1 — Machine Learning foundations and data handling

### Session 1.1 — Machine Learning concepts and Python refresher

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:15 | Welcome, programme map, what students will build by Friday | 🗣️ | [Handbook: Programme map](../student-handbook.md#programme-map) |
| 0:15–0:30 | Environment check — everyone runs `check_setup.py` | ✏️ | [Handbook: Setup](../setup-guide.md) |
| 0:30–0:45 | AI vs ML vs Deep Learning vs GenAI. The recipe-and-chef analogy | 🗣️ | Handbook 1.1 |
| 0:45–0:55 | **Activity 1.1** — Sort the mail (no computer) | ✏️ | Handbook 1.1 |
| 0:55–1:10 | Types of ML: supervised, unsupervised, generative | 🗣️ | Handbook 1.1 |
| 1:10–1:20 | **Activity 1.2** — Number or category? | ✏️ | Handbook 1.1 |
| 1:20–1:40 | **Activity 1.3** — Teachable Machine, train a model with no code | ✏️ | [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com/) |
| 1:40–1:55 | **Break** | | |
| 1:55–2:10 | Real-world ML and GenAI applications | 🗣️ | Handbook 1.1 |
| 2:10–2:25 | Google Colab tour: cells, runtime, uploading data | 💻 | Handbook: Colab setup |
| 2:25–2:55 | Python refresher: variables, collections, conditionals, loops, functions | 💻 | Handbook 1.2 |
| 2:55–3:00 | **Activity 1.4** — Fix the broken program (set as homework if short) | ✏️ | Handbook 1.2 |

### Session 1.2 — NumPy, Pandas and Exploratory Data Analysis

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:10 | Why libraries exist. The shopping-bag / egg-tray analogy | 🗣️ | Handbook 1.3 |
| 0:10–0:35 | NumPy: arrays, vectorised maths, statistics, boolean masking | 💻 | Handbook 1.3 |
| 0:35–0:50 | **Activity 1.5** — The marks calculator | ✏️ | Handbook 1.3 |
| 0:50–1:05 | Pandas: the spreadsheet-that-takes-orders analogy | 🗣️💻 | Handbook 1.4 |
| 1:05–1:35 | Loading, selecting, filtering, grouping, sorting | 💻 | Handbook 1.4 |
| 1:35–1:50 | **Break** | | |
| 1:50–2:10 | **Activity 1.6** — Twelve rows you can see (`pre_data.csv`) | ✏️ | Handbook 1.4 |
| 2:10–2:35 | EDA: the doctor's-check-up analogy, the five questions, correlation | 💻 | Handbook 1.5 |
| 2:35–2:55 | **Activity 1.7** — Data detective, in pairs, 60-second presentations | 👥 | Handbook 1.5 |
| 2:55–3:00 | Set the Day 1 exit task | 🗣️ | Handbook 1.5 |

**Day 1 outcome:** every student can load a CSV and answer five questions about it.

---

## Day 2 — Visualisation, preprocessing and supervised learning

### Session 2.1 — Visualisation and preprocessing

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:10 | Day 1 recap; review the exit task | 🗣️ | |
| 0:10–0:25 | Charts are questions, not decoration. The chart-chooser table | 🗣️ | Handbook 2.1 |
| 0:25–0:35 | **Activity 2.1** — The chart chooser | ✏️ | Handbook 2.1 |
| 0:35–1:05 | Matplotlib and Seaborn: histogram, box, scatter, bar, heatmap | 💻 | Handbook 2.1 |
| 1:05–1:20 | **Activity 2.2** — Draw it before you code it | ✏️ | Handbook 2.1 |
| 1:20–1:35 | **Break** | | |
| 1:35–1:50 | Preprocessing: the chopping-before-cooking analogy | 🗣️ | Handbook 2.2 |
| 1:50–2:20 | Missing values, encoding, scaling, train/test split | 💻 | Handbook 2.2 |
| 2:20–2:45 | **Activity 2.3** — Preprocess twelve rows by hand | ✏️ | Handbook 2.2 |
| 2:45–3:00 | **Data leakage** — the studying-with-the-answer-key analogy | 🗣️ | Handbook 2.2 |

> Spend the full fifteen minutes on data leakage. It is the single most common serious mistake in student projects.

### Session 2.2 — Supervised learning and evaluation

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:15 | Regression: the straight-road-through-scattered-houses analogy | 🗣️ | Handbook 2.3 |
| 0:15–0:30 | **Activity 2.4** — Beat the model (predict by hand first) | ✏️ | Handbook 2.3 |
| 0:30–0:50 | Linear regression in scikit-learn; MAE, RMSE, R² | 💻 | Handbook 2.3 |
| 0:50–1:05 | Classification: the sorting-the-laundry analogy | 🗣️ | Handbook 2.4 |
| 1:05–1:20 | **Activity 2.5** — What does a mistake cost? | 👥 | Handbook 2.4 |
| 1:20–1:35 | **Break** | | |
| 1:35–2:00 | Logistic Regression and Random Forest on the loan dataset | 💻 | Handbook 2.4 |
| 2:00–2:20 | Confusion matrix (fire-alarm analogy); precision and recall (fishing-net analogy) | 🗣️ | Handbook 2.4 |
| 2:20–2:30 | **Activity 2.6** — Build the useless 99% model | ✏️ | Handbook 2.4 |
| 2:30–2:50 | The ML workflow; the four scikit-learn methods | 🗣️ | Handbook 2.5 |
| 2:50–3:00 | Set the [ML practice exercise](https://github.com/tech4alltraining/aiml/blob/main/assessments/ml_ai_practice.md) | 🗣️ | |

**Day 2 outcome:** every student has trained and evaluated one regression and one classification model.

---

## Day 3 — Feature engineering and model improvement

### Session 3.1 — Augmentation, feature engineering, feature reduction

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:15 | Day 2 recap; walk through the practice exercise | 🗣️ | |
| 0:15–0:30 | Data augmentation: the recognising-your-friend analogy | 🗣️ | Handbook 3.1 |
| 0:30–0:40 | **Activity 3.1** — Legal or illegal augmentation? | ✏️ | Handbook 3.1 |
| 0:40–0:55 | Augmentation demo | 💻 | [Notebook 3.1](https://colab.research.google.com/drive/1bvfMkPtrSTILCFbxGvaUPr5iGwk65mTe?usp=sharing) |
| 0:55–1:15 | Feature engineering: the height-weight-BMI analogy; ratios, bins, flags | 💻 | Handbook 3.2 |
| 1:15–1:30 | **Break** | | |
| 1:30–1:45 | **Activity 3.2** — Invent three features, then test one | 👥 | Handbook 3.2 |
| 1:45–2:20 | Feature reduction: importance, SelectKBest, PCA | 💻 | Handbook 3.3 |
| 2:20–2:45 | Which reduction method when, and the interpretability trade-off | 🗣️ | Handbook 3.3 |
| 2:45–3:00 | Buffer / questions | | |

### Session 3.2 — Overfitting, cross-validation, tuning

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:20 | Overfitting: the three-students-and-an-exam analogy | 🗣️ | Handbook 3.4 |
| 0:20–0:45 | **Activity 3.3** — Watch a model overfit, live | ✏️ | Handbook 3.4 |
| 0:45–1:00 | Model improvement strategies; regularisation | 🗣️ | Handbook 3.4 |
| 1:00–1:20 | Cross-validation: the one-practice-exam-is-not-enough analogy | 💻 | Handbook 3.5 |
| 1:20–1:35 | **Break** | | |
| 1:35–1:50 | **Activity 3.4** — Prove that one split is unreliable | ✏️ | Handbook 3.5 |
| 1:50–2:10 | Hyperparameter tuning: the oven-dials analogy | 🗣️ | Handbook 3.6 |
| 2:10–2:45 | GridSearchCV and RandomizedSearchCV (start the run, then talk over it) | 💻 | Handbook 3.6 |
| 2:45–3:00 | Set the Day 3 exit task: the before/after improvement table | 🗣️ | Handbook 3.6 |

**Day 3 outcome:** every student has a tuned model that beats their Day 2 baseline, with evidence.

---

## Day 4 — Deep learning, clustering and Generative AI

### Session 4.1 — Deep learning, ethics and unsupervised learning

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:15 | Day 3 recap; review improvement tables | 🗣️ | |
| 0:15–0:40 | Neural networks: the relay-race analogy; the training loop | 🗣️ | Handbook 4.1 |
| 0:40–0:55 | When to use deep learning — and when not to | 🗣️ | Handbook 4.1 |
| 0:55–1:20 | AI ethics and Responsible AI: bias, privacy, accountability | 🗣️👥 | Handbook: Responsible AI checklist |
| 1:20–1:35 | **Break** | | |
| 1:35–1:45 | Clustering: the box-of-old-photographs analogy | 🗣️ | Handbook 4.2 |
| 1:45–1:55 | **Activity 4.1** — Cluster the classroom (everyone stands up) | 👥 | Handbook 4.2 |
| 1:55–2:25 | K-Means on Mall Customers; elbow and silhouette | 💻 | Handbook 4.2 |
| 2:25–2:50 | **Activity 4.2** — Choose k, then defend it. Name every segment | ✏️ | Handbook 4.2 |
| 2:50–3:00 | Buffer | | |

### Session 4.2 — Generative AI, LLMs and prompt engineering

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:15 | **Activity 4.3** — Be the language model (three rounds on the board) | ✏️ | Handbook 4.3 |
| 0:15–0:35 | What an LLM is: the world's-most-well-read-autocomplete analogy | 🗣️ | Handbook 4.3 |
| 0:35–0:45 | **Activity 4.4** — Count the tokens | ✏️ | Handbook 4.3 |
| 0:45–1:00 | Prompt engineering: the briefing-a-new-intern analogy; the five parts | 🗣️ | Handbook 4.4 |
| 1:00–1:20 | **Activity 4.5** — The prompt makeover | ✏️ | Handbook 4.4 |
| 1:20–1:35 | **Break** | | |
| 1:35–2:00 | Zero-shot, one-shot, few-shot, chain-of-thought | 🗣️💻 | Handbook 4.4 · [activities.md](activities.md) |
| 2:00–2:20 | **Activity 4.6** — All four prompt types on one task | ✏️ | Handbook 4.4 |
| 2:20–2:40 | First Gemini API call; **Activity 4.7** — the temperature dial | 💻✏️ | Handbook 4.5 |
| 2:40–2:50 | **Activity 4.10** — The Prompting Tournament | 👥 | [activities.md](activities.md) |
| 2:50–3:00 | **Activity 4.12** — The Red Team Challenge (or set as homework) | 👥 | [activities.md](activities.md) |

> **Time is tight in Session 4.2.** If you must cut, drop Activity 4.6 and set it as homework — but never cut Activity 4.3. It is the one that makes everything else on Day 4 make sense.

**Day 4 outcome:** every student has called the Gemini API and can explain why an LLM hallucinates.

---

## Day 5 — Open-source models, Hugging Face and app development

### Session 5.1 — Open-source GenAI, Hugging Face, ML + GenAI

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:15 | Day 4 recap; a few students read out their prompt makeovers | 🗣️ | |
| 0:15–0:35 | Open-source vs API models; when each wins | 🗣️ | [open-source-gen-ai.md](../tutorials/concepts/open-source-gen-ai.md) |
| 0:35–0:50 | Hugging Face: the app-store-for-AI-models analogy; tour of the Hub | 🗣️💻 | [hugging-face-ecosystem.md](../tutorials/concepts/hugging-face-ecosystem.md) |
| 0:50–1:20 | `pipeline()`: sentiment, text generation, question answering | 💻 | Handbook 5.2 |
| 1:20–1:35 | **Break** | | |
| 1:35–1:55 | **Activity 5.1** — Two models, one sentence | ✏️ | Handbook 5.2 |
| 1:55–2:10 | **Activity 5.2** — Read a model card | ✏️ | Handbook 5.2 |
| 2:10–2:35 | ML + GenAI: the doctor-and-receptionist analogy | 🗣️💻 | Handbook 5.4 · [ml_gen_ai.md](../tutorials/apps/ml_gen_ai.md) |
| 2:35–3:00 | AI-powered application concepts and architecture | 🗣️ | [ai-powered-apps.md](../tutorials/concepts/ai-powered-apps.md) |

### Session 5.2 — Streamlit and capstone planning

| Time | Topic | Mode | Ref |
|---:|---|---|---|
| 0:00–0:20 | Streamlit: the script-that-became-a-website and the rewiped-whiteboard analogies | 🗣️💻 | Handbook 5.3 |
| 0:20–0:45 | Build `hello_streamlit.py` together; widgets, session state, caching | 💻 | Handbook 5.3 |
| 0:45–1:20 | **Activity 5.3** — Build the loan app end to end | ✏️ | [loan-app.md](../tutorials/apps/loan-app.md) |
| 1:20–1:35 | **Break** | | |
| 1:35–2:10 | **Activity 5.4** — Add the GenAI explanation layer | ✏️ | [ml_gen_ai.md](../tutorials/apps/ml_gen_ai.md) |
| 2:10–2:30 | Capstone: topic selection, group formation | 👥 | Handbook: Capstone guide |
| 2:30–2:45 | Deliverables, marking guide, review milestones | 🗣️ | Handbook: Capstone guide |
| 2:45–3:00 | Internship guidelines, mentoring plan, close | 🗣️ | |

**Day 5 outcome:** every student has a Streamlit app running locally and a chosen capstone topic.

---

## Weeks 2–4 — Project phase

| Week | Milestone | Format | Students present |
|---|---|---|---|
| **Week 2** | Review Update 01 | Online, 10 min per group | Problem statement, dataset, EDA findings |
| **Week 3** | Review Update 02 | Online, 10 min per group | Baseline model, evaluation metrics, honest limitations |
| **Week 4** | Review Update 03 | Online, 10 min per group | Improved model, working app demo |
| **End of Week 4** | Project submission | Repository + report | |
| **End of Week 4** | Final presentation | 8–10 min + questions | |

### Questions to ask at every review

1. What is your **target**, and is it a number or a category?
2. Did you **split before you scaled**?
3. What does a **baseline** score on this dataset — and do you beat it?
4. Did you **cross-validate**, or is that a single lucky split?
5. Which metric are you optimising, and **what does a mistake cost**?
6. Where does your model **fail**, and who could it treat unfairly?

Question 6 is the one students prepare for least and learn from most.

---

## Trainer preparation checklist

**A week before:**

- [ ] Send students the setup instructions and ask them to create the `genai` environment
- [ ] Ask them to create their Gemini API key and Hugging Face account
- [ ] Warn them that the Day 5 Hugging Face install is 2–3 GB — do it the evening before

**The day before each session:**

- [ ] Run every code example yourself, in the `genai` environment
- [ ] Check that all Colab notebook links still open
- [ ] Verify the dataset URLs load (they are fetched live during the session)

**On the day:**

- [ ] Have the datasets downloaded locally as a fallback for poor classroom wifi
- [ ] Start long-running cells (GridSearchCV, Hugging Face downloads) *before* explaining them
- [ ] Keep [Troubleshooting](../student-handbook.md#troubleshooting) open in a tab

---

## Related material

| File | Purpose |
|---|---|
| [student-handbook.md](../student-handbook.md) | The student's book — concepts, code, all 38 activities |
| [gen-ai-curriculum.md](curriculum.md) | Topic-by-topic delivery plan with notebook links |
| [activities.md](activities.md) | Full facilitation notes for the instructor-led activities |
| [prompts.md](../prompts.md) | Copy-paste prompt library for the Day 4 demos |
| [tutorials/streamlit-apps-collection.md](../tutorials/apps/streamlit-apps-collection.md) | 15 runnable apps: ML, GenAI, ML+GenAI |
| [notebooks/](../notebooks) | Six Colab/Jupyter notebooks with outputs |
| [exercises-assignments.md](../exercises-assignments.md) | Exercises and graded assignments |
| [setup-guide.md](../setup-guide.md) | Environment setup for all platforms |
| [troubleshooting.md](../troubleshooting.md) | Every error and its fix |
| [tutorials/](../tutorials) | Eight in-depth tutorials |
