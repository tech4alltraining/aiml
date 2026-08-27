# Teaching Sequence

**ML/AI & GenAI Internship Program — trainer notes**

The order to teach in, and what to do at each step. **No timings are given deliberately** — run each module at whatever pace your group needs, over whatever schedule suits you.

Students should use the [Student Handbook](../../mlai-genai-internship/student-handbook.md) instead; the activity numbers below match it exactly.

| Column | Meaning |
|---|---|
| **Mode** | 🗣️ Talk · 💻 Live code · ✏️ Student activity · 👥 Group work |
| **Ref** | Where the material lives |

---

## Module 0 — Python from scratch *(optional pre-work, self-paced)*

**For students who have never written code.** Set this as pre-work before Module 1.

| Notebook | Covers |
|---|---|
| [00a — Basics](../../mlai-genai-internship/notebooks/00a_python_foundations.ipynb) | printing, variables, types, operators, strings |
| [00b — Structures](../../mlai-genai-internship/notebooks/00b_python_foundations_2.ipynb) | collections, conditions, loops, functions |
| [00c — Scenarios](../../mlai-genai-internship/notebooks/00c_python_scenarios.ipynb) | ten real-world problems, task by task |

**Ask students to complete Scenario 10 in particular.** It walks them through cleaning a messy dataset by hand — duplicates, missing values, inconsistent categories, encoding, train/test split — which is exactly what Modules 1 and 2 then do with Pandas. Students who have done it by hand understand *why*; students who have not just copy six lines.

If a student arrives without having done this, point them at 00a and 00b during the Python refresher.

---

## Module 1 — Machine Learning foundations and data handling

### Part 1 — ML concepts and Python refresher

| Topic | Mode | Ref |
|---|---|---|
| Welcome, programme map, what students will build | 🗣️ | [Handbook: Programme map](../../mlai-genai-internship/student-handbook.md#programme-map) |
| Environment check — everyone runs `check_setup.py` | ✏️ | [Setup Guide](../../mlai-genai-internship/setup-guide.md) |
| AI vs ML vs Deep Learning vs GenAI. The recipe-and-chef analogy | 🗣️ | Handbook 1.1 |
| **Activity 1.1** — Sort the mail (no computer) | ✏️ | Handbook 1.1 |
| Types of ML: supervised, unsupervised, generative | 🗣️ | Handbook 1.1 |
| **Activity 1.2** — Number or category? | ✏️ | Handbook 1.1 |
| **Activity 1.3** — Teachable Machine, train a model with no code | ✏️ | [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com/) |
| Real-world ML and GenAI applications | 🗣️ | Handbook 1.1 |
| Google Colab tour: cells, runtime, uploading data | 💻 | Setup Guide: Colab |
| Python refresher: variables, collections, conditionals, loops, functions | 💻 | Handbook 1.2 |
| **Activity 1.4** — Fix the broken program (set as homework if short) | ✏️ | Handbook 1.2 |

### Part 2 — NumPy, Pandas and EDA

| Topic | Mode | Ref |
|---|---|---|
| Why libraries exist. The shopping-bag / egg-tray analogy | 🗣️ | Handbook 1.3 |
| NumPy: arrays, vectorised maths, statistics, boolean masking | 💻 | Handbook 1.3 |
| **Activity 1.5** — The marks calculator | ✏️ | Handbook 1.3 |
| Pandas: the spreadsheet-that-takes-orders analogy | 🗣️💻 | Handbook 1.4 |
| Loading, selecting, filtering, grouping, sorting | 💻 | Handbook 1.4 |
| **Activity 1.6** — Twelve rows you can see (`pre_data.csv`) | ✏️ | Handbook 1.4 |
| EDA: the doctor's-check-up analogy, the five questions, correlation | 💻 | Handbook 1.5 |
| **Activity 1.7** — Data detective, in pairs, short presentations | 👥 | Handbook 1.5 |
| Set the Module 1 exit task | 🗣️ | Handbook 1.5 |

**Outcome:** every student can load a CSV and answer five questions about it.

---

## Module 2 — Visualisation, preprocessing and supervised learning

### Part 1 — Visualisation and preprocessing

| Topic | Mode | Ref |
|---|---|---|
| Module 1 recap; review the exit task | 🗣️ | |
| Charts are questions, not decoration. The chart-chooser table | 🗣️ | Handbook 2.1 |
| **Activity 2.1** — The chart chooser | ✏️ | Handbook 2.1 |
| Matplotlib and Seaborn: histogram, box, scatter, bar, heatmap | 💻 | Handbook 2.1 |
| **Activity 2.2** — Draw it before you code it | ✏️ | Handbook 2.1 |
| Preprocessing: the chopping-before-cooking analogy | 🗣️ | Handbook 2.2 |
| Missing values, encoding, scaling, train/test split | 💻 | Handbook 2.2 |
| **Activity 2.3** — Preprocess twelve rows by hand | ✏️ | Handbook 2.2 |
| **Data leakage** — the studying-with-the-answer-key analogy | 🗣️ | Handbook 2.2 |

> Do not rush data leakage. It is the single most common serious mistake in student projects.

### Part 2 — Supervised learning and evaluation

| Topic | Mode | Ref |
|---|---|---|
| Regression: the straight-road-through-scattered-houses analogy | 🗣️ | Handbook 2.3 |
| **Activity 2.4** — Beat the model (predict by hand first) | ✏️ | Handbook 2.3 |
| Linear regression in scikit-learn; MAE, RMSE, R² | 💻 | Handbook 2.3 |
| Classification: the sorting-the-laundry analogy | 🗣️ | Handbook 2.4 |
| **Activity 2.5** — What does a mistake cost? | 👥 | Handbook 2.4 |
| Logistic Regression and Random Forest on the loan dataset | 💻 | Handbook 2.4 |
| Confusion matrix (fire-alarm analogy); precision and recall (fishing-net analogy) | 🗣️ | Handbook 2.4 |
| **Activity 2.6** — Build the useless 99% model | ✏️ | Handbook 2.4 |
| The ML workflow; the four scikit-learn methods | 🗣️ | Handbook 2.5 |
| Set the [ML practice exercise](https://github.com/tech4alltraining/aiml/blob/main/assessments/ml_ai_practice.md) | 🗣️ | |

**Outcome:** every student has trained and evaluated one regression and one classification model.

---

## Module 3 — Feature engineering and model improvement

### Part 1 — Augmentation, feature engineering, feature reduction

| Topic | Mode | Ref |
|---|---|---|
| Module 2 recap; walk through the practice exercise | 🗣️ | |
| Data augmentation: the recognising-your-friend analogy | 🗣️ | Handbook 3.1 |
| **Activity 3.1** — Legal or illegal augmentation? | ✏️ | Handbook 3.1 |
| Augmentation demo | 💻 | [Augmentation notebook](https://colab.research.google.com/drive/1bvfMkPtrSTILCFbxGvaUPr5iGwk65mTe?usp=sharing) |
| Feature engineering: the height-weight-BMI analogy; ratios, bins, flags | 💻 | Handbook 3.2 |
| **Activity 3.2** — Invent three features, then test one | 👥 | Handbook 3.2 |
| Feature reduction: importance, SelectKBest, PCA | 💻 | Handbook 3.3 |
| Which reduction method when, and the interpretability trade-off | 🗣️ | Handbook 3.3 |

### Part 2 — Overfitting, cross-validation, tuning

| Topic | Mode | Ref |
|---|---|---|
| Overfitting: the three-students-and-an-exam analogy | 🗣️ | Handbook 3.4 |
| **Activity 3.3** — Watch a model overfit, live | ✏️ | Handbook 3.4 |
| Model improvement strategies; regularisation | 🗣️ | Handbook 3.4 |
| Cross-validation: the one-practice-exam-is-not-enough analogy | 💻 | Handbook 3.5 |
| **Activity 3.4** — Prove that one split is unreliable | ✏️ | Handbook 3.5 |
| Hyperparameter tuning: the oven-dials analogy | 🗣️ | Handbook 3.6 |
| GridSearchCV and RandomizedSearchCV (start the run, then talk over it) | 💻 | Handbook 3.6 |
| Set the exit task: the before/after improvement table | 🗣️ | Handbook 3.6 |

**Outcome:** every student has a tuned model that beats their Module 2 baseline, with evidence.

---

## Module 4 — Deep learning, clustering and Generative AI

### Part 1 — Deep learning, ethics and unsupervised learning

| Topic | Mode | Ref |
|---|---|---|
| Module 3 recap; review improvement tables | 🗣️ | |
| Neural networks: the relay-race analogy; the training loop | 🗣️ | Handbook 4.1 |
| When to use deep learning — and when not to | 🗣️ | Handbook 4.1 |
| AI ethics and Responsible AI: bias, privacy, accountability | 🗣️👥 | [Handbook: Responsible AI](../../mlai-genai-internship/student-handbook.md#responsible-ai-checklist) |
| Clustering: the box-of-old-photographs analogy | 🗣️ | Handbook 4.2 |
| **Activity 4.1** — Cluster the classroom (everyone stands up) | 👥 | Handbook 4.2 |
| K-Means on Mall Customers; elbow and silhouette | 💻 | Handbook 4.2 |
| **Activity 4.2** — Choose k, then defend it. Name every segment | ✏️ | Handbook 4.2 |

### Part 2 — Generative AI, LLMs and prompt engineering

| Topic | Mode | Ref |
|---|---|---|
| **Activity 4.3** — Be the language model (three rounds on the board) | ✏️ | Handbook 4.3 |
| What an LLM is: the world's-most-well-read-autocomplete analogy | 🗣️ | Handbook 4.3 |
| **Activity 4.4** — Count the tokens | ✏️ | Handbook 4.3 |
| Prompt engineering: the briefing-a-new-intern analogy; the five parts | 🗣️ | Handbook 4.4 |
| **Activity 4.5** — The prompt makeover | ✏️ | Handbook 4.4 |
| Zero-shot, one-shot, few-shot, chain-of-thought | 🗣️💻 | Handbook 4.4 · [activities.md](activities.md) |
| **Activity 4.6** — All four prompt types on one task | ✏️ | Handbook 4.4 |
| First Gemini API call; **Activity 4.7** — the temperature dial | 💻✏️ | Handbook 4.5 |
| **Activity 4.10** — The Prompting Tournament | 👥 | [activities.md](activities.md) |
| **Activity 4.12** — The Red Team Challenge (or set as homework) | 👥 | [activities.md](activities.md) |

> **This part is dense.** If you must cut, drop Activity 4.6 and set it as homework — but never cut Activity 4.3. It is the one that makes everything else make sense.

**Outcome:** every student has called the Gemini API and can explain why an LLM hallucinates.

---

## Module 5 — Open-source models, Hugging Face and app development

### Part 1 — Open-source GenAI, Hugging Face, ML + GenAI

| Topic | Mode | Ref |
|---|---|---|
| Module 4 recap; a few students read out their prompt makeovers | 🗣️ | |
| Open-source vs API models; when each wins | 🗣️ | [open-source-gen-ai](../../mlai-genai-internship/tutorials/concepts/open-source-gen-ai.md) |
| Hugging Face: the app-store-for-AI-models analogy; tour of the Hub | 🗣️💻 | [hugging-face-ecosystem](../../mlai-genai-internship/tutorials/concepts/hugging-face-ecosystem.md) |
| `pipeline()`: sentiment, text generation, question answering | 💻 | Handbook 5.2 |
| **Activity 5.1** — Two models, one sentence | ✏️ | Handbook 5.2 |
| **Activity 5.2** — Read a model card | ✏️ | Handbook 5.2 |
| ML + GenAI: the doctor-and-receptionist analogy | 🗣️💻 | Handbook 5.4 · [ml_gen_ai](../../mlai-genai-internship/tutorials/apps/ml_gen_ai.md) |
| AI-powered application concepts and architecture | 🗣️ | [ai-powered-apps](../../mlai-genai-internship/tutorials/concepts/ai-powered-apps.md) |

### Part 2 — Streamlit and capstone planning

| Topic | Mode | Ref |
|---|---|---|
| Streamlit: the script-that-became-a-website and rewiped-whiteboard analogies | 🗣️💻 | Handbook 5.3 |
| Build `hello_streamlit.py` together; widgets, session state, caching | 💻 | Handbook 5.3 |
| **Activity 5.3** — Build the loan app end to end | ✏️ | [loan-app](../../mlai-genai-internship/tutorials/apps/loan-app.md) |
| **Activity 5.4** — Add the GenAI explanation layer | ✏️ | [ml_gen_ai](../../mlai-genai-internship/tutorials/apps/ml_gen_ai.md) |
| Capstone: topic selection, group formation | 👥 | [Capstone guide](../../mlai-genai-internship/student-handbook.md#capstone-project-guide) |
| Deliverables, marking guide, review milestones | 🗣️ | [Capstone guide](../../mlai-genai-internship/student-handbook.md#capstone-project-guide) |
| Internship guidelines, mentoring plan, close | 🗣️ | |

**Outcome:** every student has a Streamlit app running locally and a chosen capstone topic.

---

## Capstone phase

| Milestone | Format | Students present |
|---|---|---|
| **Review Update 01** | Online, per group | Problem statement, dataset, EDA findings |
| **Review Update 02** | Online, per group | Baseline model, evaluation metrics, honest limitations |
| **Review Update 03** | Online, per group | Improved model, working app demo |
| **Project submission** | Repository + report | |
| **Final presentation** | Presentation and questions | |

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

**Before the programme starts:**

- [ ] Send students the [Setup Guide](../../mlai-genai-internship/setup-guide.md) and ask them to create the `genai` environment
- [ ] Ask them to create their Gemini API key and Hugging Face account
- [ ] Warn them the Module 5 Hugging Face install is 2–3 GB — do it well in advance
- [ ] Point absolute beginners at the [Python foundation notebooks](../../mlai-genai-internship/notebooks/)

**Before each session:**

- [ ] Run every code example yourself, in the `genai` environment
- [ ] Check that all Colab notebook links still open
- [ ] Verify the dataset URLs load — they are fetched live during the session

**During the session:**

- [ ] Have the datasets downloaded locally as a fallback for poor classroom wifi
- [ ] Start long-running cells (GridSearchCV, Hugging Face downloads) *before* explaining them
- [ ] Keep [Troubleshooting](../../mlai-genai-internship/troubleshooting.md) open in a tab

---

## Related material

| File | Purpose |
|---|---|
| [curriculum.md](curriculum.md) | Topic-by-topic coverage with notebook links |
| [activities.md](activities.md) | Full facilitation notes for instructor-led activities |
| [Student Handbook](../../mlai-genai-internship/student-handbook.md) | The student's book — concepts, code, all activities |
| [Slide decks](../../mlai-genai-internship/latex/) | Six projector-ready decks (build with `make`) |
