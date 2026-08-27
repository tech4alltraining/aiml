# Trainer Materials

Everything for delivering the ML/AI & GenAI Internship Program.

| File | What it is | When you use it |
|---|---|---|
| [session-plan.md](session-plan.md) | The teaching sequence, step by step | Planning and running each module |
| [curriculum.md](curriculum.md) | Module-by-module topic list with every notebook, dataset and exercise link | Checking coverage |
| [activities.md](activities.md) | Full facilitation notes for instructor-led demos and activities | Running a class activity |

Students should use the [Student Handbook](../../mlai-genai-internship/student-handbook.md) instead — activity numbers match exactly across all four documents.

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

- Send students the [Setup Guide](../../mlai-genai-internship/setup-guide.md) and ask them to build the `genai` environment
- Ask them to create a Gemini API key and a Hugging Face account
- Warn them the Module 5 Hugging Face install is 2–3 GB — do it well in advance
- Point absolute beginners at the [Python foundation notebooks](../../mlai-genai-internship/notebooks/) (`00a`, `00b`, `00c`)

**Before each session:**

- Run every code example yourself in the `genai` environment
- Check the Colab notebook links still open
- Verify the dataset URLs load — they are fetched live during the session

**During the session:**

- Have the datasets downloaded locally as a fallback for poor classroom wifi
- Start long-running cells (GridSearchCV, model downloads) *before* explaining them
- Keep [Troubleshooting](../../mlai-genai-internship/troubleshooting.md) open in a tab

## Questions to ask at every project review

1. What is your **target**, and is it a number or a category?
2. Did you **split before you scaled**?
3. What does a **baseline** score on this dataset — and do you beat it?
4. Did you **cross-validate**, or is that a single lucky split?
5. Which metric are you optimising, and **what does a mistake cost**?
6. Where does your model **fail**, and who could it treat unfairly?

Question 6 is the one students prepare for least and learn from most.
