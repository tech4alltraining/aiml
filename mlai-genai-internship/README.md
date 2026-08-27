# ML/AI & GenAI Internship Program

Everything for the taught modules and the capstone project phase.

> **New here?** Two things, in this order:
> 1. **[Set up your environment](setup-guide.md)** — Windows, Ubuntu or macOS.
> 2. **[Open the Student Handbook](student-handbook.md)** — written for beginners, assumes no prior Machine Learning.
>
> **Never written Python?** Start with the [Python foundation notebooks](notebooks/) (`00a`, `00b`, `00c`) before Module 1.

---

## Start here

| If you are a… | Go to |
|---|---|
| **Starting out, nothing installed** | [**Setup Guide**](setup-guide.md) — venv or conda, all three platforms |
| **Student who has never written Python** | [**Python Foundations**](notebooks/00a_python_foundations.ipynb) — from `print()` upwards |
| **Student** | [**Student Handbook**](student-handbook.md) — concepts, analogies, 38 activities |
| **Student who wants to run code now** | [**Notebooks**](notebooks/) — one click to Colab, outputs included |
| **Student building an app** | [**Streamlit Apps**](tutorials/apps/streamlit-apps-collection.md) — 15 runnable apps |
| **Student practising** | [**Exercises & Assignments**](exercises-assignments.md) — warm-up to scenario-based |
| **Anyone with an error** | [**Troubleshooting**](troubleshooting.md) |
| **Anyone needing a demo prompt** | [**Prompt Library**](prompts.md) |

---

## How this folder is organised

```text
mlai-genai-internship/
│
├── README.md                    ← you are here
├── setup-guide.md               install for Windows / Ubuntu / macOS
├── student-handbook.md          the main student book
├── exercises-assignments.md     practice + five graded assignments
├── troubleshooting.md           every error and its fix
├── prompts.md                   copy-paste prompt library
├── requirements.txt             the package list
│
├── notebooks/                   ← run these
│   ├── 00a 00b 00c              Python from scratch
│   └── 01 → 06                  Module 1 to Module 5
│
├── tutorials/
│   ├── apps/                    build something (15 Streamlit apps + 4 guides)
│   └── concepts/                read about something
│
└── archive/                     superseded versions
```

---

## The modules

| Module | Topics | Notebook | Handbook |
|---|---|---|---|
| **0** | Python from scratch — *optional, for absolute beginners* | [00a](notebooks/00a_python_foundations.ipynb) · [00b](notebooks/00b_python_foundations_2.ipynb) · [00c](notebooks/00c_python_scenarios.ipynb) | — |
| **1** | ML concepts, Python refresher, NumPy, Pandas, EDA | [01](notebooks/01_module1_python_data.ipynb) | [Module 1](student-handbook.md#module-1--python-refresher-and-data-handling) |
| **2** | Visualisation, preprocessing, regression, classification, metrics | [02](notebooks/02_module2_ml_basics.ipynb) | [Module 2](student-handbook.md#module-2--visualisation-preprocessing-and-supervised-learning) |
| **3** | Features, reduction, overfitting, cross-validation, tuning | [03](notebooks/03_module3_model_improvement.ipynb) | [Module 3](student-handbook.md#module-3--feature-engineering-and-model-improvement) |
| **4** | Clustering, LLMs, prompting, the Gemini API | [04a](notebooks/04_module4a_clustering.ipynb) · [04b](notebooks/05_module4b_genai_api.ipynb) | [Module 4](student-handbook.md#module-4--deep-learning-clustering-and-generative-ai) |
| **5** | Open-source models, Hugging Face, ML+GenAI, Streamlit, capstone | [05](notebooks/06_module5_huggingface.ipynb) | [Module 5](student-handbook.md#module-5--open-source-models-hugging-face-and-app-development) |

Each module is self-contained. Work through them in order.

---

## Notebooks

Every demo is a runnable notebook. **Notebooks 00a–04a show their charts and results directly on GitHub** — you can read the whole lesson without running anything.

| # | Notebook | Needs an API key? | Outputs saved? |
|---|---|---|---|
| 00a | [Python Foundations — Basics](notebooks/00a_python_foundations.ipynb) | No | ✅ |
| 00b | [Python Foundations — Collections, Loops, Functions](notebooks/00b_python_foundations_2.ipynb) | No | ✅ |
| 00c | [Python Scenario Worksheets](notebooks/00c_python_scenarios.ipynb) | No | Write your own |
| 01 | [Module 1 — Python, NumPy, Pandas, EDA](notebooks/01_module1_python_data.ipynb) | No | ✅ |
| 02 | [Module 2 — Visualisation and supervised learning](notebooks/02_module2_ml_basics.ipynb) | No | ✅ |
| 03 | [Module 3 — Model improvement](notebooks/03_module3_model_improvement.ipynb) | No | ✅ |
| 04a | [Module 4A — Clustering and how LLMs work](notebooks/04_module4a_clustering.ipynb) | No | ✅ |
| 04b | [Module 4B — The Gemini API](notebooks/05_module4b_genai_api.ipynb) | **Yes** | Run it yourself |
| 05 | [Module 5 — Hugging Face](notebooks/06_module5_huggingface.ipynb) | No (downloads models) | Run it yourself |

**Every concept follows the same shape**, so you always know what comes next:

```text
📘 Examples       2-3 worked examples you run and read
✏️ Practice now   5 short exercises, with solutions
🧠 Quick quiz     5 questions to check you understood
🎯 Tasks          2-3 longer pieces for after the session
```

Charts come with a **📊 How to read this chart** guide — because drawing a chart is easy and reading one is the skill.

More on running them, and on the `.ipynb` vs Quarto question: [notebooks/README.md](notebooks/README.md)

---

## Setup

**Full instructions for Windows, Ubuntu and macOS, with both `venv` and `conda`: [Setup Guide](setup-guide.md).** Do it before you start Module 1.

The short version, once you have Python or Miniconda:

```bash
conda create -n genai python=3.12 -y && conda activate genai
```

```bash
pip install -r requirements.txt
```

```bash
python check_setup.py
```

> ⚠️ **Your prompt must show `(genai)` before you install or run anything.** If it does not, packages go elsewhere and nothing works. You must re-activate in every new terminal.

Module 5 additionally needs the Hugging Face libraries — 2–3 GB, so install them the evening before:

```bash
conda activate genai && pip install transformers torch datasets evaluate gradio
```

---

## Accounts to create before Module 1

- [Google account](https://accounts.google.com) — Colab and Google AI Studio
- [Gemini API key](https://aistudio.google.com/apikey) — all Module 4 and Module 5 GenAI code
- [Hugging Face account](https://huggingface.co/join) — Module 5
- [GitHub account](https://github.com/signup) — submitting your capstone

---

## Reviews and milestones

| Milestone | Format | Deliverable |
|---|---|---|
| Review Update 01 | Online | Problem statement, dataset, EDA findings |
| Review Update 02 | Online | Baseline model and evaluation metrics |
| Review Update 03 | Online | Improved model and app demo |
| Project submission | After Review 03 | Repository and report |
| Final presentation | After submission | Presentation and questions |

Requirements and marking guide: [Capstone project guide](student-handbook.md#capstone-project-guide)

---

## Elsewhere in this repository

| Location | Contents |
|---|---|
| [`datasets/`](../datasets/) | All course datasets — see [the dataset table](student-handbook.md#the-datasets-you-will-use) |
| [`python-internship/`](../python-internship/) | Extra Python drill exercises and topic notebooks |
| [`mlai-internship/`](../mlai-internship/) | NumPy, Pandas and plotting exercise notebooks |
| [`assessments/`](../assessments/) | Practice exercise and final assessment |

---

## Stuck?

1. Read the **last line** of the error message — it names the actual problem.
2. Check that your terminal prompt shows `(genai)`.
3. Look in [Troubleshooting](troubleshooting.md) — every common error is listed with its fix.
4. Ask your instructor. Twenty minutes stuck is learning; two hours is waste.
