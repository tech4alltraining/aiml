# Trainer Materials

Everything for delivering the ML/AI & GenAI Internship Program.

| What | For | When you use it |
|---|---|---|
| [session-plan.md](session-plan.md) | The teaching sequence, step by step | Planning and running each module |
| [curriculum.md](curriculum.md) | Module-by-module topic list with every notebook, dataset and exercise link | Checking coverage |
| [activities.md](activities.md) | Full facilitation notes for instructor-led demos and activities | Running a class activity |
| [latex/](latex/) | Six projector-ready slide decks — build with `make` | Presenting |
| [archive/](archive/) | Superseded versions of the curriculum | Reference only |

**Students never see this folder.** Everything they need is in [`../student/`](../student/), and the activity numbers match exactly — call out "Activity 2.5" and everyone is on the same page.

---

## The modules at a glance

| Module | Parts | Students finish with |
|---|---|---|
| **1** | ML concepts, Python refresher, NumPy, Pandas, EDA | A notebook that loads a CSV and answers five questions about it |
| **2** | Visualisation, preprocessing, regression, classification, metrics | A trained regression model and a trained classifier, both evaluated |
| **3** | Features, reduction, overfitting, cross-validation, tuning | A tuned model that beats their Module 2 baseline, with evidence |
| **4** | Deep learning, clustering, GenAI, prompting, Gemini API | Working prompts of all four types and their first API call |
| **5** | Open-source models, Hugging Face, ML+GenAI, Streamlit, capstone | A running Streamlit app and a chosen capstone topic |

## Before you teach

**Before the programme starts:**

- Send students the [Setup Guide](../student/setup-guide.md) and ask them to build the `genai` environment
- Ask them to create a Gemini API key and a Hugging Face account
- Warn them the Module 5 Hugging Face install is 2–3 GB — do it well in advance
- Point absolute beginners at the [Python foundation notebooks](../student/notebooks/) (`00a`, `00b`, `00c`)

**Before each session:**

- Run every code example yourself in the `genai` environment
- Check the Colab notebook links still open
- Verify the dataset URLs load — they are fetched live during the session

**During the session:**

- Have the datasets downloaded locally as a fallback for poor classroom wifi
- Start long-running cells (GridSearchCV, model downloads) *before* explaining them
- Keep [Troubleshooting](../student/troubleshooting.md) open in a tab

## Questions to ask at every project review

1. What is your **target**, and is it a number or a category?
2. Did you **split before you scaled**?
3. What does a **baseline** score on this dataset — and do you beat it?
4. Did you **cross-validate**, or is that a single lucky split?
5. Which metric are you optimising, and **what does a mistake cost**?
6. Where does your model **fail**, and who could it treat unfairly?

Question 6 is the one students prepare for least and learn from most.


---

## How the material is produced

Notes for whoever maintains this course.

### Notebooks

The nine student notebooks live in [`../student/notebooks/`](../student/notebooks/). Notebooks `00a`–`04` are committed **with their outputs saved**, so GitHub renders the charts and results and a student can read the whole lesson without running anything. Notebooks `05` and `06` ship deliberately empty — one calls a live API with the student's own key, the other downloads models.

Before committing an executed notebook, check it carries no API key:

```bash
grep -l "AIza" ../student/notebooks/*.ipynb
```

### Slide decks

[`latex/`](latex/) holds six Beamer decks and three build checks. `make` builds them all; `make check` also reports any slide whose content runs into the footer. See [latex/README.md](latex/README.md).

### Why `.ipynb` and not Quarto

`.qmd` does **not** render on GitHub — it is shown as plain text, because Quarto has to execute the code first. Committed `.ipynb` files render natively *with outputs*, and carry a one-click **Open in Colab** badge, so the file a student reads is the file they run.

If you later want a course website or PDF handouts, Quarto renders `.ipynb` directly — keep the notebooks exactly as they are and add Quarto on top, rather than maintaining a second source.
