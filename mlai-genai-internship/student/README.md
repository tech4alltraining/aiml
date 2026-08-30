# ML/AI & GenAI Internship Program

Everything for the taught sessions and the capstone project phase.

> **New here?** Two things, in this order:
> 1. **[Set up your environment](setup-guide.md)** — Windows, Ubuntu or macOS.
> 2. **[Start Session 1](sessions/session-01-python-refresher.md)** — assumes you have never written a line of Python.

---

## Start here

| If you are… | Go to |
|---|---|
| **Following the course** | [**The twelve sessions**](sessions/) — this is the course |
| **Starting out, nothing installed** | [**Setup Guide**](setup-guide.md) — venv or conda, all three platforms |
| **Never written Python** | [**Session 1**](sessions/session-01-python-refresher.md) — from `print()` to classes |
| **Want to run code now** | [**Notebooks**](notebooks/) — one click to Colab, outputs included |
| **Building an app** | [**Streamlit Apps**](tutorials/apps/streamlit-apps-collection.md) — 15 runnable apps |
| **Practising** | [**Exercises & Assignments**](exercises-assignments.md) — warm-up to scenario-based |
| **Anyone with an error** | [**Troubleshooting**](troubleshooting.md) |
| **Anyone needing a demo prompt** | [**Prompt Library**](prompts.md) |

---

## The twelve sessions

**Work through them in order.** Each session has a guide (concepts, examples, exercises, MCQs, tasks) and a notebook (the same code, runnable, with outputs already saved so you can read it on GitHub without running anything).

| # | Session | Guide | Notebook |
|---|---|---|---|
| 1 | Python Refresher — syntax to classes | [guide](sessions/session-01-python-refresher.md) | [open](notebooks/session-01-python-refresher.ipynb) |
| 2 | Python Libraries: NumPy & Pandas | [guide](sessions/session-02-numpy-pandas.md) | [open](notebooks/session-02-numpy-pandas.ipynb) |
| 3 | EDA & Data Preprocessing | [guide](sessions/session-03-eda-preprocessing.md) | [open](notebooks/session-03-eda-preprocessing.ipynb) |
| 4 | Introduction to Machine Learning & AI | [guide](sessions/session-04-intro-ml-ai.md) | [open](notebooks/session-04-intro-ml-ai.ipynb) |
| 5 | Regression | [guide](sessions/session-05-regression.md) | [open](notebooks/session-05-regression.ipynb) |
| 5B | Classification | [guide](sessions/session-05b-classification.md) | [open](notebooks/session-05b-classification.ipynb) |
| 5C | Model Deployment & Streamlit | [guide](sessions/session-05c-deployment.md) | [open](notebooks/session-05c-deployment.ipynb) |
| 6 | Data Augmentation & Feature Engineering | [guide](sessions/session-06-augmentation-feature-engg-red.md) | [open](notebooks/session-06-augmentation-feature-engg-red.ipynb) |
| 7 | Unsupervised Learning | [guide](sessions/session-07-unsupervised.md) | [open](notebooks/session-07-unsupervised.ipynb) |
| 8 | Model Evaluation & Improvement | [guide](sessions/session-08-evaluation-tuning.md) | [open](notebooks/session-08-evaluation-tuning.ipynb) |
| 9 | Deep Learning | [guide](sessions/session-09-deep-learning.md) | [open](notebooks/session-09-deep-learning.ipynb) |
| 10 | Generative AI & Large Language Models | [guide](sessions/session-10-genai-llms.md) | [open](notebooks/session-10-genai-llms.ipynb) |
| 11 | AI-Powered Applications | [guide](sessions/session-11-ai-apps.md) | [open](notebooks/session-11-ai-apps.ipynb) |
| 12 | Open Source, Hugging Face & Responsible AI | [guide](sessions/session-12-opensource-ethics.md) | [open](notebooks/session-12-opensource-ethics.ipynb) |

**Every topic follows the same shape**, so you always know what comes next:

```text
🧠 Analogy        the everyday comparison for the idea
📘 Examples       worked examples you run and read
✏️ Practice       short exercises, with solutions
❓ MCQs           questions to check you understood, with answers and why
🎯 Tasks          longer pieces for after the session
```

Charts come with a **📊 How to read this chart** guide — because drawing a chart is easy and reading one is the skill.

---

## How this folder is organised

```text
student/
│
├── README.md                    ← you are here
├── setup-guide.md               install for Windows / Ubuntu / macOS
├── exercises-assignments.md     practice + five graded assignments
├── troubleshooting.md           every error and its fix
├── prompts.md                   copy-paste prompt library
├── requirements.txt             the package list
│
├── sessions/                    ← THE COURSE. Twelve sessions, in order
│
├── notebooks/                   ← run these
│   ├── session-01 → session-12  one per session
│   └── 00a–00c, 01–06           extra practice (see below)
│
├── tutorials/
│   ├── apps/                    build something (15 Streamlit apps + 4 guides)
│   └── concepts/                read about something
│
└── student-handbook.md          reference book (see below)
```

---

## Extra practice and reference

**These are not part of the twelve-session path.** They cover the same ground in a different order, and are useful for revision or if you want more drill on one area.

| What | Use it for |
|---|---|
| [Student Handbook](student-handbook.md) | A single continuous reference book, organised by module rather than session |
| [00a](notebooks/00a_python_foundations.ipynb) · [00b](notebooks/00b_python_foundations_2.ipynb) · [00c](notebooks/00c_python_scenarios.ipynb) | Extra Python drill — 35 exercises and 70 scenario tasks |
| [01](notebooks/01_module1_python_data.ipynb) → [06](notebooks/06_module5_huggingface.ipynb) | The earlier module-based notebooks |

> **If the handbook and a session disagree, follow the session.** The sessions are the current syllabus.

---

## Setup

**Full instructions for Windows, Ubuntu and macOS, with both `venv` and `conda`: [Setup Guide](setup-guide.md).** Do it before you start Session 1.

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

Session 12 additionally needs the Hugging Face libraries — 2–3 GB, so install them the evening before:

```bash
conda activate genai && pip install transformers torch datasets evaluate gradio
```

---

## Accounts to create before Session 10

- [Google account](https://accounts.google.com) — Colab and Google AI Studio
- [Gemini API key](https://aistudio.google.com/apikey) — all the GenAI code from Session 10 onwards
- [Hugging Face account](https://huggingface.co/join) — Session 12
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

Planning your project: [Session 12, topic 4](sessions/session-12-opensource-ethics.md#4-project-grouping-capstone-planning-and-mentoring) · Requirements and marking: [Capstone project guide](student-handbook.md#capstone-project-guide)

---

## Elsewhere in this repository

| Location | Contents |
|---|---|
| [`datasets/`](../../datasets/) | All course datasets — see [the dataset table](student-handbook.md#the-datasets-you-will-use) |
| [`python-internship/`](../../python-internship/) | Extra Python drill exercises and topic notebooks |
| [`mlai-internship/`](../../mlai-internship/) | NumPy, Pandas and plotting exercise notebooks |
| [`assessments/`](../../assessments/) | Practice exercise and final assessment |

---

## Stuck?

1. Read the **last line** of the error message — it names the actual problem.
2. Check that your terminal prompt shows `(genai)`.
3. Look in [Troubleshooting](troubleshooting.md) — every common error is listed with its fix.
4. Ask your instructor. Twenty minutes stuck is learning; two hours is waste.
