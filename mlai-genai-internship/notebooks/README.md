# Notebooks

**Every demo in this course, as a runnable notebook.** Open in Colab with one click, or run locally in Jupyter or VS Code.

| # | Notebook | Topic | Needs API key? | Outputs saved? |
|---|---|---|---|---|
| 00a | [00a_python_foundations.ipynb](00a_python_foundations.ipynb) | Python from scratch: printing, variables, types, operators, strings | No | ✅ Yes |
| 00b | [00b_python_foundations_2.ipynb](00b_python_foundations_2.ipynb) | Collections, conditions, loops, functions | No | ✅ Yes |
| 00c | [00c_python_scenarios.ipynb](00c_python_scenarios.ipynb) | Ten scenario worksheets, task by task | No | You write them |
| 1 | [01_module1_python_data.ipynb](01_module1_python_data.ipynb) | Python, NumPy, Pandas, EDA | No | ✅ Yes |
| 2 | [02_module2_ml_basics.ipynb](02_module2_ml_basics.ipynb) | Visualisation, preprocessing, regression, classification | No | ✅ Yes |
| 3 | [03_module3_model_improvement.ipynb](03_module3_model_improvement.ipynb) | Features, overfitting, cross-validation, tuning | No | ✅ Yes |
| 4 | [04_module4a_clustering.ipynb](04_module4a_clustering.ipynb) | K-Means, and how LLMs actually work | No | ✅ Yes |
| 5 | [05_module4b_genai_api.ipynb](05_module4b_genai_api.ipynb) | Gemini API, prompting, temperature, JSON, memory | **Yes** | ❌ Run it yourself |
| 6 | [06_module5_huggingface.ipynb](06_module5_huggingface.ipynb) | `pipeline()`, open-source models, model cards | No (downloads models) | ❌ Run it yourself |

**Notebooks 00a–04 show their outputs directly on GitHub** — charts, tables and printed results are saved inside the file. You can read the whole lesson without running anything.

**Notebooks 5 and 6 ship deliberately empty.** Notebook 5 calls a live API with *your* key; notebook 6 downloads models. Their outputs would be misleading, and running them yourself is the point.

---

## Running them

### Option 1: Google Colab — nothing to install

Click the **Open in Colab** badge at the top of any notebook. Colab already has NumPy, Pandas, Matplotlib, Seaborn and scikit-learn.

> ⚠️ Colab resets when the runtime disconnects. Re-run the `!pip install` cell and **File → Save a copy in Drive** to keep your work.

### Option 2: Locally

```bash
conda activate genai
```

```bash
jupyter notebook
```

Then choose **Kernel → Change kernel → Python (genai)**. Setup instructions: [Setup Guide](../setup-guide.md).

### Option 3: VS Code

Open the `.ipynb` file directly. Select the `genai` interpreter with `Ctrl+Shift+P` → **Python: Select Interpreter**.

---

## Never written Python?

Start with **00a**, then **00b**, then **00c**. They assume nothing — 00a begins with `print("Hello, World!")`.

Every concept in every notebook follows the same shape:

```text
📘 Examples       2-3 worked examples you run and read
✏️ Practice now   5 short exercises, with solutions hidden until you try
🧠 Quick quiz     5 questions to check you understood
🎯 Tasks          2-3 longer pieces for after the session
```

Charts come with a **📊 How to read this chart** guide.

---

## How to work through a notebook

1. **Run each cell in order** with `Shift+Enter`.
2. **Read the explanation before the cell**, not after.
3. **Look at the output.** If it surprises you, that is the interesting part.
4. **Change one thing and run it again.** This is where the learning happens.
5. Before submitting anything: **Kernel → Restart & Run All.**

> ⚠️ **Always restart and run all before submitting.** A notebook that only works in the order *you happened* to click is not reproducible — and markers run it top to bottom.

---

## What about Streamlit?

**Streamlit apps do not run in a notebook.** They need a local web server.

All 15 apps are in [`../tutorials/streamlit-apps-collection.md`](../tutorials/apps/streamlit-apps-collection.md) as `.py` files, with the exact command to run each one.

| Medium | Use it for |
|---|---|
| **Notebook** (`.ipynb`) | Learning, experimenting, showing your working |
| **Script** (`.py`) | Streamlit apps, training scripts, anything reusable |

---

## Showing code *and* output on GitHub

**Three ways, in order of effort.**

### 1. Committed `.ipynb` — what this folder uses ✅

GitHub renders `.ipynb` files natively, **including saved outputs**: printed text, DataFrames as tables, and Matplotlib charts as images. Nothing to build, nothing to configure.

- ✅ Renders on GitHub with outputs
- ✅ One-click **Open in Colab**
- ✅ Students edit the same file they read
- ⚠️ Diffs are noisy (JSON with embedded base64 images)
- ⚠️ Easy to commit an API key by accident — see the warning below

**This is the right default, and it is why the course uses it.**

### 2. Quarto → GitHub-Flavoured Markdown

**You asked whether Quarto `.qmd` renders on GitHub. It does not** — GitHub shows `.qmd` as plain text, because Quarto has to *execute* the code first.

The fix is to keep `.qmd` as your **source** and commit the **rendered** `.md`:

```yaml
---
title: "Module 1 — Pandas"
format: gfm          # GitHub-Flavoured Markdown
jupyter: python3
---
```

```bash
quarto render module1.qmd --to gfm
```

That produces `module1.md` plus a `day1_files/` folder of PNGs, referenced by relative path. **GitHub renders the result perfectly**, with charts.

- ✅ Clean, readable diffs on the `.qmd` source
- ✅ Same source can also produce HTML, PDF and slides
- ❌ Requires installing Quarto
- ❌ Two files to keep in sync — commit both, or the `.md` goes stale
- ❌ No "Open in Colab" — students cannot edit and run it in place

**Worth it if** you also want a course website or PDF handouts. **Not worth it** just to show output on GitHub — `.ipynb` already does that.

### 3. Quarto → HTML → GitHub Pages

The best-looking option: a proper website with a sidebar, search and cross-references.

```bash
quarto publish gh-pages
```

Quarto renders `.ipynb` files directly, so you can **keep the notebooks exactly as they are** and add a website on top:

```yaml
# _quarto.yml
project:
  type: website
  render:
    - notebooks/*.ipynb
website:
  title: "ML/AI & GenAI Internship"
  sidebar:
    contents: notebooks/*.ipynb
```

- ✅ Best reading experience
- ✅ Reuses the notebooks you already have — no duplicate source
- ❌ A publish step to remember
- ❌ Students still go to the `.ipynb` to actually run things

### The recommendation

| Your goal | Use |
|---|---|
| **Students read and run the same file** | `.ipynb` — what this folder does |
| Students run it in Colab | `.ipynb` with a Colab badge |
| You want a polished course website too | Keep `.ipynb`, add Quarto → GitHub Pages on top |
| You want PDF handouts | Quarto `.qmd` → PDF |
| Clean git diffs matter more than one-click running | Quarto `.qmd` → `gfm`, commit both files |

**For this course, `.ipynb` is the answer**, because the file students read is the file they run — and nothing beats one click to Colab. If you later want a website, add Quarto pointing at these same notebooks; you do not have to choose.

---

## ⚠️ Before you commit a notebook

**Notebook outputs are saved inside the file.** If you printed your API key, it is now in the file, and it will go to GitHub.

Check before every commit:

```bash
grep -l "AIza" notebooks/*.ipynb
```

If that finds anything, **revoke the key immediately** at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and clear the output before committing.

To strip all outputs from a notebook:

```bash
jupyter nbconvert --clear-output --inplace mynotebook.ipynb
```

**Never print a key, and always use `getpass` or Colab Secrets.**

---

## Related material

| File | For |
|---|---|
| [Student Handbook](../student-handbook.md) | The full course, with all the explanation |
| [Setup Guide](../setup-guide.md) | Installing everything |
| [Troubleshooting](../troubleshooting.md) | When something goes wrong |
| [Streamlit Apps](../tutorials/apps/streamlit-apps-collection.md) | 15 runnable apps |
| [Exercises](../exercises-assignments.md) | Practice, simple to advanced |
