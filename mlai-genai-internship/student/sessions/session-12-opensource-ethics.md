# Session 12 — Open Source, Hugging Face & Responsible AI

**Open Source Generative AI Models · Hugging Face Ecosystem · AI Ethics & Responsible AI · Project Grouping, Capstone Planning & Mentoring**

| | |
|---|---|
| **Notebook** | [session-12-opensource-ethics.ipynb](../notebooks/session-12-opensource-ethics.ipynb) |
| **Previous** | [Session 11 — AI-Powered Applications](session-11-ai-apps.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **The last session, and the one that matters most for your career.** Sessions 1–11 taught you to build things that work. This one is about building things that should exist — and then planning what you will actually build.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Decide between an open and a closed model, with reasons beyond quality
2. Read a model licence and say whether you may use it commercially
3. Navigate the Hugging Face Hub and load a model in three lines
4. **Measure fairness in a model you have trained**
5. Explain why removing a sensitive column does *not* remove bias
6. Write a model card for your own work
7. Choose a capstone project that is achievable and worth doing

---

## The four topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Open source GenAI](#1-open-source-generative-ai-models) | Licence and data residency decide before quality does |
| 2 | [Hugging Face](#2-the-hugging-face-ecosystem) | Three lines gets you a working model |
| 3 | [Ethics & Responsible AI](#3-ai-ethics-and-responsible-ai) | **Dropping the sensitive column does not fix bias** |
| 4 | [Capstone planning](#4-project-grouping-capstone-planning-and-mentoring) | Narrow and finished beats broad and abandoned |

---

# 1. Open Source Generative AI Models

You met the open/closed split in Session 10. **Here is what actually running an open model involves.**

🧠 **Analogy: renting a taxi versus buying a car.** The taxi is instant, you pay per trip, and you go where the driver is allowed to go. The car costs money up front, needs a garage and maintenance — and it is yours, it goes where you like, and nobody sees your journeys. **Neither is the right answer; they answer different questions.**

## The comparison, honestly

| | Closed API (Gemini, GPT, Claude) | Open weights (Llama, Mistral, Falcon) |
|---|---|---|
| Time to first result | **Minutes** | Hours |
| Hardware needed | None | A GPU, for anything substantial |
| Cost shape | Per token, forever | Up front, then near-zero |
| **Your data** | Leaves your machine | **Never leaves** |
| Fine-tuning | Limited | **Full control** |
| Works offline | ❌ | ✅ |
| Reproducible in 5 years | Not guaranteed | **Yes — you have the weights** |
| Quality at the top end | Generally ahead | Closing, and close enough for most tasks |

> **Two of these rows decide most real projects, and neither is quality:** *your data never leaves*, and *reproducible in five years*. A hospital, a bank or a research group will often pick open before anyone has compared benchmarks.

## Licences — read them, they genuinely differ

> ⚠️ **"Open weights" does not mean "do whatever you like".** It means you can download the file. What you may then *do* with it is the licence's business.

| Licence | What it usually permits |
|---|---|
| **Apache 2.0 / MIT** | Almost anything, including commercial use |
| **Llama Community Licence** | Commercial use, with conditions above a very large user count |
| **Research-only** | **No commercial use at all** |
| **Non-commercial (CC-BY-NC)** | Study and experiment only |

**Before you build a product on a model, check three things:**

1. May you use it **commercially**?
2. Must you **credit** the creator or publish changes?
3. Are there **restrictions on use** (no medical advice, no surveillance)?

## What size model can you actually run?

| Parameters | Rough memory | Runs on |
|---|---|---|
| ~1B | ~2 GB | A laptop, slowly |
| ~7B | ~14 GB (or ~4 GB quantised) | A good laptop or a single GPU |
| ~13B | ~26 GB | One large GPU |
| ~70B | ~140 GB | Several GPUs |

**Quantisation** stores the weights at lower precision — 4-bit instead of 16-bit — cutting memory by roughly four times for a modest quality loss. **It is what makes a 7B model run on a laptop at all.**

## 📘 Examples

**Example 1 — the decision, with reasons**

```python
# A hospital classifying patient notes; data cannot leave the building
#   -> OPEN, self-hosted. The licence and data residency decide it.
#
# A student demo due on Friday, no GPU
#   -> CLOSED API. Free tier, working in minutes.
#
# A research paper that must be reproducible in 2031
#   -> OPEN. You keep the weights; an API version can change or vanish.
#
# A startup with 50 users and no ML engineer
#   -> CLOSED. Self-hosting is an engineering job you do not need yet.
```

**Example 2 — the licence check, before you write any code**

```python
# Ask three questions of every model you are considering:
#   1. Commercial use allowed?
#   2. Attribution or share-alike required?
#   3. Any prohibited uses (medical, surveillance, minors)?
#
# The answers live on the model's Hub page under "License",
# and in the LICENSE file in the repository. Read both.
```

**Example 3 — the running cost, worked out**

```python
tokens_per_request = 1500
requests_per_day = 2000
days = 365

total_tokens = tokens_per_request * requests_per_day * days
print(f"{total_tokens:,} tokens per year")
# At a published rate you can now compare this against the cost of
# a GPU that runs the same workload for free after purchase.
```

## ✏️ Practice

1. For three scenarios of your own, decide open or closed and give the deciding reason.
2. Look up the licence of Llama, Mistral and Falcon. Which allows commercial use most freely?
3. Work out roughly what memory a 7B model needs, and whether your laptop could run it quantised.
4. Name two reasons to choose open weights that have nothing to do with model quality.
5. Estimate the annual token cost of an app serving 500 requests a day at 1,200 tokens each.

<details><summary>Solutions</summary>

```python
# 1 - The deciding reason is usually one of: data residency, licence,
#     reproducibility, offline operation, or engineering capacity.
#     Quality is rarely the deciding factor for a student project.

# 2 - Apache 2.0 (several Mistral models, Falcon) is the most permissive.
#     The Llama Community Licence allows commercial use but adds
#     conditions above a very large monthly-user threshold.
#     ALWAYS check the specific model, not the family: licences differ
#     between releases from the same organisation.

params_billion = 7                                                     # 3
print(f"16-bit: about {params_billion * 2} GB")     # 14 GB
print(f" 4-bit: about {params_billion * 0.5} GB")   # 3.5 GB - a laptop can
# Quantisation is what makes a 7B model run on a laptop at all.

# 4 - Data never leaves your machine; reproducibility (you keep the
#     weights); offline operation; no per-token cost at scale;
#     full freedom to fine-tune.

tokens = 1200 * 500 * 365                                              # 5
print(f"{tokens:,} tokens per year")     # 219,000,000
# Multiply by the published per-million rate to get a figure you can
# compare against buying a GPU.
```
</details>

## ❓ MCQs

**Q1.** "Open weights" means…
- (a) Free for any use  (b) You can download and run the model yourself  (c) The training data is public  (d) There is no licence

**Q2.** Which reason most often decides in favour of an open model?
- (a) It is more accurate  (b) The data never leaves your machine  (c) It is easier to set up  (d) It is faster

**Q3.** What does quantisation do?
- (a) Makes the model more accurate  (b) Stores weights at lower precision, cutting memory roughly fourfold  (c) Speeds up training  (d) Removes layers

**Q4.** Roughly how much memory does a 7B model need at 16-bit?
- (a) 700 MB  (b) 7 GB  (c) 14 GB  (d) 70 GB

**Q5.** Before building a product on an open model you must check…
- (a) Only the accuracy  (b) The licence: commercial use, attribution, and prohibited uses  (c) Only the size  (d) Nothing

**Q6.** A research group needs results reproducible in 2031. They should use…
- (a) A closed API  (b) An open model whose weights they keep  (c) Either  (d) Neither

<details><summary>Answers</summary>

**A1 — (b).** What you may *do* with it is a separate question, answered by the licence.

**A2 — (b).** **Data residency decides before quality is even discussed.**

**A3 — (b).** It is what makes a 7B model run on a laptop.

**A4 — (c) ~14 GB.** Roughly 2 bytes per parameter at 16-bit.

**A5 — (b).** All three questions, every time.

**A6 — (b).** An API version can change or be withdrawn; weights you hold cannot.
</details>

## 🎯 Tasks

**Task 1 — The licence audit.** Pick three open models on the Hugging Face Hub. For each, record the licence, whether commercial use is permitted, and any prohibited uses. **Present it as a table you would show a manager.**

**Task 2 — The build-or-buy memo.** For an application idea of your own, write one page comparing an open self-hosted model against a closed API. **Include a rough annual cost for both** and name the single fact that decides it.

---

# 2. The Hugging Face ecosystem

**Hugging Face is where the open model world lives.** Models, datasets, demos and the libraries to use them, in one place.

🧠 **Analogy: an app store for models.** You browse, read the description, check the licence and reviews, and install with one line. **The alternative — finding a paper, hunting for weights, writing your own loading code — is what people did before it existed.**

| Part | What it is | Address |
|---|---|---|
| **Hub** | Hundreds of thousands of models | `huggingface.co/models` |
| **Datasets** | Ready-to-use datasets | `huggingface.co/datasets` |
| **Spaces** | Live demos, many built with Streamlit | `huggingface.co/spaces` |
| **`transformers`** | The library that loads and runs models | `pip install transformers` |
| **`datasets`** | Loads datasets in one line | `pip install datasets` |

## `pipeline()` — the three-line version

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("This course is genuinely useful"))
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Three lines. It downloads a model, loads it, and runs it.** The first run downloads weights (a few hundred MB); after that it is cached locally.

| Task name | Does |
|---|---|
| `"sentiment-analysis"` | Positive / negative |
| `"zero-shot-classification"` | Classify into labels **you** supply |
| `"summarization"` | Long text → short text |
| `"question-answering"` | Answer from a supplied passage |
| `"ner"` | Find names, places, organisations |
| `"text-generation"` | Continue text |
| `"translation"` | Between languages |
| `"fill-mask"` | Predict a hidden word |

## Naming a specific model

```python
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
```

> **Always name the model in real work.** The default can change between library versions, and then your results change without you touching anything.

## Tokenisation, which you can inspect directly

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("Machine learning models predict the next token.")
print(len(ids), "tokens")
print([tok.decode([i]) for i in ids])
```

**This is the Session 10 lesson made concrete** — you can see the leading spaces and the sub-word splits for yourself.

## Model cards — and why you will write one

**Every model on the Hub has a card** describing what it is, what it was trained on, how it performs, and where it should not be used. **Your capstone needs one too.**

```markdown
# Loan Approval Classifier

## What it does
Predicts whether a loan application will be approved.

## Training data
10,000 historical applications, 2019-2024. 50/50 approved/declined.

## Performance
Random Forest. Test accuracy 0.891, ROC-AUC 0.964,
5-fold CV 0.8948 ± 0.0090, bootstrap 95% CI [0.881, 0.908].

## Limitations
- Trained on one region; may not transfer elsewhere.
- Recall is lower for applicants under 25 (see fairness audit).
- NOT validated for automated decisions without human review.

## Intended use
A decision-support tool for loan officers. Not an autonomous approver.
```

> **The Limitations section is the one that matters.** Anyone can report an accuracy. Stating clearly where your model should not be trusted is what marks out serious work.

## 📘 Examples

**Example 1 — sentiment in three lines**

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
for text in ["This course is genuinely useful",
             "The delivery was late and the box was damaged"]:
    print(classifier(text))
```

**Example 2 — zero-shot classification with your own labels**

```python
classifier = pipeline("zero-shot-classification")

result = classifier(
    "The wifi in the hostel has been down for three days",
    candidate_labels=["accommodation", "academics", "food", "transport"],
)
print(result["labels"][0])      # accommodation
```

**No training data, and the labels are yours.** This is the Session 10 zero-shot idea, running locally.

**Example 3 — loading a dataset**

```python
from datasets import load_dataset

data = load_dataset("imdb", split="train[:100]")
print(data[0]["text"][:200])
print(data[0]["label"])
```

## ✏️ Practice

1. Run a sentiment pipeline over five sentences of your own.
2. Use zero-shot classification with four labels relevant to your college.
3. Tokenise a sentence with the GPT-2 tokenizer and count the tokens against the words.
4. Find a summarisation model on the Hub and note its licence and size.
5. Write a model card for a model you built in an earlier session.

<details><summary>Solutions</summary>

```python
# needs-download: these fetch model weights on first run (a few hundred MB).
# Run them in Colab or locally. See troubleshooting.md if you hit
# "RuntimeError: Numpy is not available" - that is a torch/NumPy version
# clash, not a mistake in this code.
from transformers import pipeline, AutoTokenizer

clf = pipeline("sentiment-analysis",                                   # 1
               model="distilbert-base-uncased-finetuned-sst-2-english")
for s in ["This course is genuinely useful",
          "The hostel food has got worse",
          "It is fine I suppose"]:
    print(s, "->", clf(s)[0])

zs = pipeline("zero-shot-classification")                              # 2
print(zs("The wifi in the hostel has been down for three days",
         candidate_labels=["accommodation", "academics", "food", "transport"]))

tok = AutoTokenizer.from_pretrained("gpt2")                            # 3
text = "Machine learning models predict the next token, repeatedly."
ids = tok.encode(text)
print(f"{len(text.split())} words -> {len(ids)} tokens")
print([tok.decode([i]) for i in ids])

# 4 - e.g. facebook/bart-large-cnn or sshleifer/distilbart-cnn-12-6.
#     Check the "License" field on the model page, and the file sizes
#     under "Files and versions" before you download.

# 5 - Use the template above. The LIMITATIONS section is the one that
#     distinguishes serious work from a demo.
```
</details>

## ❓ MCQs

**Q1.** What does `pipeline("sentiment-analysis")` do?
- (a) Trains a model  (b) Downloads, loads and runs a pretrained model  (c) Creates an empty model  (d) Uploads your data

**Q2.** Why name the model explicitly rather than using the default?
- (a) It is faster  (b) The default can change between library versions, changing your results silently  (c) It is required  (d) Defaults are always wrong

**Q3.** Zero-shot classification requires…
- (a) Thousands of labelled examples  (b) No training data — you supply the labels  (c) Fine-tuning  (d) A GPU

**Q4.** Which section of a model card matters most?
- (a) The title  (b) Limitations — where it should not be trusted  (c) The accuracy  (d) The author

**Q5.** What are Hugging Face Spaces?
- (a) Storage  (b) Live hosted demos, many built with Streamlit  (c) Datasets  (d) GPUs

**Q6.** The first `pipeline()` call is slow because…
- (a) The model is training  (b) It is downloading model weights, which are then cached  (c) The internet is slow  (d) It is compiling

<details><summary>Answers</summary>

**A1 — (b).** Three lines from nothing to a working model.

**A2 — (b).** **Reproducibility.** Your results should not change because a library updated.

**A3 — (b).** The Session 10 idea, running locally.

**A4 — (b) Limitations.** Anyone can report an accuracy.

**A5 — (b).** A good place to publish your capstone.

**A6 — (b).** Subsequent runs load from the local cache.
</details>

## 🎯 Tasks

**Task 1 — The pipeline tour.** Try four different pipeline tasks on text relevant to your college. **For each, record what it did well and one case where it failed** — the failures are what you learn from.

**Task 2 — Write your model card.** Take the best model you built in Sessions 5–8 and write its full card, using the template above. **The Limitations section must contain at least three genuine limitations** with evidence from your own measurements.

---

# 3. AI Ethics and Responsible AI

**A model that is accurate can still be wrong to deploy.** This topic is about telling the difference.

🧠 **Analogy: a hiring manager who has only ever hired one kind of person.** They are not malicious. They have simply learned, from experience, what a "good candidate" looks like — and their experience was narrow. **A model trained on historical data learns exactly the same way, from exactly the same evidence.**

## The five concerns

| Concern | The question | An example |
|---|---|---|
| **Bias & fairness** | Does it work equally well for everyone? | Lower recall for one group |
| **Transparency** | Can you explain a decision? | "The model said no" is not a reason |
| **Privacy** | Whose data is it, and did they agree? | Training on scraped personal data |
| **Accountability** | Who is answerable when it is wrong? | Nobody, is the wrong answer |
| **Environmental cost** | What did training consume? | Large models cost real energy |

## Measuring fairness — the part most courses skip

**You cannot manage what you do not measure.** Here is a real audit, on the census income dataset.

**Step 1 — look at the data before you model anything:**

| Group | Rows | Earn over 50K |
|---|---|---|
| Male | 20,380 | **31.4%** |
| Female | 9,782 | **11.4%** |

**The disparity is in the world, and therefore in the data.** The model has not done anything yet.

**Step 2 — train a Random Forest and look at overall accuracy:**

```text
Overall accuracy: 0.8621        <- looks fine
```

**Step 3 — split that same number by group:**

| Group | Accuracy | **Recall** | Precision | Predicted over 50K |
|---|---|---|---|---|
| Male | 0.8341 | **0.6512** | 0.7817 | 26.0% |
| Female | **0.9211** | **0.5523** | 0.6955 | 9.1% |

**Read those two bolded columns together, because they tell opposite stories.**

Accuracy is **higher** for women (0.9211 vs 0.8341). By that measure the model serves them better. But **recall is lower** (0.5523 vs 0.6512): of the women who genuinely earn over 50K, the model finds **fewer** of them.

**Why both at once?** Because most women in the data are labelled "under 50K", and the model can score well simply by saying "under" more often for women. **It is accurate *because* it under-predicts them.**

> **A single overall accuracy hid all of this.** If this model gated loan offers or job adverts, high-earning women would be systematically overlooked — and the dashboard would show 86% and a green tick.

**Step 4 — the fix everyone tries first, and why it fails.** Remove `sex` from the features. Also remove the obvious proxies, `relationship` and `marital.status`:

| | Recall (Male) | Recall (Female) | Overall accuracy |
|---|---|---|---|
| With those columns | 0.6512 | 0.5523 | 0.8621 |
| **Without them** | 0.5481 | 0.4477 | 0.8350 |

**The gap did not close. It widened slightly — and overall accuracy fell.**

> ⚠️ **This is the single most important lesson in the topic. "Fairness through unawareness" does not work.** Occupation, hours per week, education and industry all correlate with sex. The model reconstructs the pattern from whatever remains, and you have lost accuracy *and* the ability to measure what it is doing.
>
> **You cannot fix a bias you have made yourself blind to.** Keep the sensitive attribute so you can *audit* with it, and address fairness deliberately — with thresholds per group, reweighting, or by changing what you deploy.

## Transparency

```python
# A Random Forest gives you feature importances - use them
importances = dict(zip(feature_names, model.feature_importances_))

# For an individual decision, explain it in English (Session 11's pattern)
# "Declined. The largest factors were loan-to-income (0.31) and
#  credit score (0.24). Your loan-to-income of 2.4 is above the
#  approved median of 0.9."
```

**"The model said no" is not a reason.** Under laws like the GDPR, people subject to automated decisions have a right to meaningful information about the logic involved.

## A checklist for your capstone

- [ ] Where did this data come from, and was it collected with consent?
- [ ] Does it contain personal information? Should it?
- [ ] Which groups does my data under-represent?
- [ ] Have I measured performance **per group**, not just overall?
- [ ] Can I explain any individual prediction?
- [ ] What is the cost of my model being wrong — and who pays it?
- [ ] Who is accountable when it fails?
- [ ] Have I written the limitations down honestly?

## 📘 Examples

**Example 1 — the audit function you should reuse**

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score

def fairness_audit(y_true, y_pred, groups):
    """Report metrics separately for each group."""
    for g in sorted(set(groups)):
        k = (groups == g)
        print(f"{g:<10} n={k.sum():>6}  acc {accuracy_score(y_true[k], y_pred[k]):.4f}"
              f"  recall {recall_score(y_true[k], y_pred[k]):.4f}"
              f"  predicted-positive {y_pred[k].mean():.1%}")
```

**Ten lines. Run it on every model you deploy.**

**Example 2 — the three fairness definitions, which genuinely conflict**

```python
# Demographic parity  - each group gets positive predictions at the same RATE
# Equal opportunity   - each group gets the same RECALL
# Predictive parity   - each group gets the same PRECISION
#
# You generally CANNOT satisfy all three at once when the base rates
# differ between groups. This is a mathematical result, not an
# engineering failure.
#
# So you must CHOOSE which one your application needs, and say why.
```

**Example 3 — documenting the decision**

```markdown
## Fairness

Audited by sex on a held-out test set (n=7,541).

| Group  | n     | Accuracy | Recall | Predicted positive |
|--------|-------|----------|--------|--------------------|
| Male   | 5,119 | 0.8341   | 0.6512 | 26.0%              |
| Female | 2,422 | 0.9211   | 0.5523 |  9.1%              |

Recall is 9.9 points lower for women. Removing `sex`, `relationship`
and `marital.status` did NOT close the gap (0.5481 vs 0.4477) and cost
2.7 points of overall accuracy.

We therefore do not deploy this model for automated decisions.
It is used only as a ranking aid with human review, and the disparity
is stated on the reviewer's screen.
```

**That last paragraph is what responsible deployment looks like:** a measured problem, an honest statement that the easy fix failed, and a deployment decision that accounts for it.

## ✏️ Practice

1. Compute the base rate of the positive class for each group in a dataset.
2. Train a model and report accuracy, recall and precision **per group**.
3. Remove the sensitive column, retrain, and check whether the gap closes.
4. Explain in your own words why removing the column does not remove the bias.
5. Write the fairness section of a model card using your numbers.

<details><summary>Solutions</summary>

```python
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

A = pd.read_csv(BASE + "classification/adult.csv").replace("?", np.nan)
A = A.dropna().reset_index(drop=True)
A["target"] = (A["income"].str.strip() == ">50K").astype(int)
sex = A["sex"].str.strip()

for g in ["Male", "Female"]:                                           # 1
    m = sex == g
    print(f"{g:<8} n={m.sum():>6}  earn >50K: {A.loc[m, 'target'].mean():.1%}")
# The disparity is in the WORLD, and therefore in the data, before the
# model does anything at all.

X = A.drop(columns=["income", "target"]).copy()
for c in X.select_dtypes(include="object").columns:
    X[c] = LabelEncoder().fit_transform(X[c])
y = A["target"]
Xtr, Xte, ytr, yte, _, ste = train_test_split(
    X, y, sex, test_size=.25, random_state=42, stratify=y)

m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
p = m.predict(Xte)
print(f"\\nOVERALL accuracy {accuracy_score(yte, p):.4f}   <- looks fine")

for g in ["Male", "Female"]:                                           # 2
    k = (ste == g).values
    print(f"{g:<8} acc {accuracy_score(yte[k], p[k]):.4f}"
          f"  recall {recall_score(yte[k], p[k]):.4f}"
          f"  predicted>50K {p[k].mean():.1%}")
# Accuracy is HIGHER for women but recall is LOWER. The model scores well
# on them by saying "under 50K" more often. It is accurate BECAUSE it
# under-predicts them.

X2 = X.drop(columns=["sex", "relationship", "marital.status"])         # 3
X2tr, X2te, _, _ = train_test_split(X2, y, test_size=.25, random_state=42, stratify=y)
p2 = RandomForestClassifier(n_estimators=200, random_state=42,
                            n_jobs=-1).fit(X2tr, ytr).predict(X2te)
print(f"\\nAfter dropping sex/relationship/marital.status "
      f"(accuracy {accuracy_score(yte, p2):.4f}):")
for g in ["Male", "Female"]:
    k = (ste == g).values
    print(f"{g:<8} recall {recall_score(yte[k], p2[k]):.4f}")
# The gap did NOT close - it widened slightly, and accuracy fell.

# 4 - Occupation, hours per week, education and industry all correlate
#     with sex. The model reconstructs the pattern from whatever is left.
#     You have lost accuracy AND the ability to measure what it is doing.
#     You cannot fix a bias you have made yourself blind to.

# 5 - See the model card example above. State the measurement, state that
#     the easy fix failed, and state what you decided to do about it.
```
</details>

## ❓ MCQs

**Q1.** A model has 86% overall accuracy. What can you conclude about fairness?
- (a) It is fair  (b) Nothing — you must measure per group  (c) It is unfair  (d) Accuracy measures fairness

**Q2.** Accuracy was **higher** for women but recall was **lower**. Why?
- (a) A bug  (b) Most women are labelled negative, so predicting negative scores well while missing the positives  (c) Women's data is noisier  (d) It is impossible

**Q3.** Removing the `sex` column from the features…
- (a) Removes the bias  (b) Does not remove the bias — proxies remain — and costs you the ability to measure it  (c) Always improves accuracy  (d) Is legally required

**Q4.** "Fairness through unawareness" means…
- (a) Auditing carefully  (b) Deleting sensitive attributes and hoping the bias goes with them  (c) Using more data  (d) Explaining decisions

**Q5.** Demographic parity, equal opportunity and predictive parity…
- (a) Are the same thing  (b) Generally cannot all hold at once when base rates differ  (c) Are always achievable together  (d) Are not real

**Q6.** "The model said no" as an explanation is…
- (a) Sufficient  (b) Not a reason, and may not satisfy laws granting a right to meaningful information  (c) Required  (d) Best practice

**Q7.** The most important section of a model card is…
- (a) The accuracy  (b) The limitations  (c) The author  (d) The training time

<details><summary>Answers</summary>

**A1 — (b) Nothing.** **A single overall number hides everything that matters here.**

**A2 — (b).** It is accurate *because* it under-predicts them.

**A3 — (b).** **Measured:** the gap widened from 9.9 points to 10.0 and accuracy fell 2.7 points.

**A4 — (b).** It does not work, and this is the most important lesson in the topic.

**A5 — (b).** A mathematical result, not an engineering failure. **You must choose which one your application needs and say why.**

**A6 — (b).** People subject to automated decisions have a right to know the logic involved.

**A7 — (b) Limitations.** Anyone can report an accuracy.
</details>

## 🎯 Tasks

**Task 1 — Audit your own model.** Take your best model from Sessions 5–8 and audit it across a group of your choice. **Report the per-group table.** If you find no disparity, say what you checked and why you are confident.

**Task 2 — Prove the unawareness failure yourself.** Reproduce the experiment: measure the gap, drop the sensitive column and its proxies, and measure again. **Report both, and write a paragraph on what a team should do instead.**

**Task 3 — The ethics review.** For your capstone idea, answer all eight checklist questions in writing. **Any question you cannot answer is work you have not done yet.**

---

# 4. Project grouping, capstone planning and mentoring

**Everything so far was practice. The capstone is the thing you will show people.**

## Choosing a topic

🧠 **Analogy: cooking for guests for the first time.** You do not attempt a five-course banquet. You cook one dish you can make well, and you make it properly. **A narrow project, finished, beats an ambitious one abandoned — every single time.**

**A good capstone has all four of these:**

| Property | Test |
|---|---|
| **A real question** | Can you state it in one sentence, with a number or category as the answer? |
| **Data you actually have** | Have you already downloaded it and opened it? |
| **A measurable outcome** | What metric, and what would count as good? |
| **Something to show** | A Streamlit app, a notebook, a report |

> ⚠️ **The commonest failure is choosing a topic before checking the data exists.** Download it first. Open it. Look at the columns. *Then* decide.

## Project shapes that work

| Shape | Sessions used | Example |
|---|---|---|
| **Predict & explain** | 3, 5, 8, 11 | Predict student dropout, explain each case |
| **Segment & act** | 3, 7 | Cluster customers, recommend an action per segment |
| **Extract & structure** | 10, 11 | Turn CVs or forms into structured data |
| **Ask your documents** | 11 | RAG over college handbooks |
| **Compare & report** | 5, 6, 8 | Rigorous comparison with CV and confidence intervals |
| **Audit** | 12 | Fairness audit of a public model or dataset |

**The strongest projects combine two shapes** — usually *predict & explain*, which is exactly the Session 11 pattern.

## Group working

| Role | Owns |
|---|---|
| **Data** | Collection, cleaning, EDA, the data section of the report |
| **Modelling** | Training, evaluation, tuning, the fairness audit |
| **Application** | Streamlit app, integration, deployment |
| **Documentation** | Report, model card, presentation |

> **Rotate at least once.** The point is that everyone learns everything, not that everyone specialises early.

**Two rules that prevent most group failures:**

1. **Everyone runs the notebook from a clean start in week one.** Environment problems found in week one are an inconvenience; found the night before submission they are a disaster.
2. **Agree the metric before you build anything.** Arguments about "which model is better" dissolve once the metric is fixed in advance.

## A plan that fits the course

| Stage | Do | Deliverable |
|---|---|---|
| **1. Scope** | Pick the question. **Download the data.** | One paragraph + the data on your disk |
| **2. Explore** | EDA, missing values, target balance | A notebook with charts |
| **3. Baseline** | `DummyClassifier`, then the simplest real model | A number to beat |
| **4. Model** | Two or three models, cross-validated | A comparison table with mean ± std |
| **5. Evaluate** | Bootstrap CI, per-group fairness audit | Honest metrics |
| **6. Build** | Streamlit app, ML + GenAI if it fits | A running app |
| **7. Document** | Report, model card, README | Something a stranger could follow |
| **8. Present** | Demo and defend it | Slides + the live app |

> **Stage 3 is the one people skip, and it is the cheapest insurance in the course.** A baseline takes three lines and tells you whether anything you do afterwards is worth reporting.

## What good work looks like

| Level | Looks like |
|---|---|
| **Adequate** | A model, a score, a notebook |
| **Good** | Cross-validated, baselined, a working app, honest limitations |
| **Excellent** | All of that, plus a fairness audit, a confidence interval, a model card, and **a clear statement of what you would do next** |

> **The gap between "good" and "excellent" is almost entirely honesty and documentation** — not a better model. Nobody is expecting you to beat the state of the art. They are looking for someone who measures carefully and reports faithfully.

## Getting the most from mentoring

**Bring these to a mentoring session:**

1. **What you tried** — including what failed
2. **What you measured** — actual numbers, not impressions
3. **The specific question you are stuck on**
4. **What you think the answer might be**

> **"It doesn't work" is not a question.** *"My recall is 0.55 for one group and 0.65 for the other, I tried dropping the sensitive column and it got worse — what should I try next?"* is a question your mentor can answer in two minutes.

## 📘 Examples

**Example 1 — a scope paragraph that would be approved**

```markdown
## Question
Can we predict which first-year students are at risk of dropping out,
early enough to offer support?

## Data
Anonymised records for 2,400 students, 2020-2025: attendance,
internal marks, fee payment status, distance from home.
ALREADY DOWNLOADED and opened. 14 columns, 2,400 rows, 6% missing.

## Target
Binary: continued / did not continue. Base rate 11%. IMBALANCED,
so we will report recall and F1, not accuracy.

## Success
Beat the majority-class baseline on recall by a clear margin, with
a per-group fairness audit by gender and home district.

## Deliverable
A Streamlit tool where a mentor enters a student's details and sees
a risk score with a plain-English explanation of the main factors.
```

**Notice what makes this good:** the data is already downloaded, the imbalance is spotted before modelling, the metric is chosen in advance, and a fairness audit is planned rather than added at the end.

**Example 2 — a scope paragraph that would not**

```markdown
## Question
Use AI to improve the college.

## Data
We will find some.

## Success
Get high accuracy.
```

**Three failures:** the question is not answerable, the data does not exist yet, and "high accuracy" is meaningless without a baseline or a metric.

**Example 3 — the week-one checklist**

```python
# Before you write ANY modelling code:
#   [ ] The data file is on my disk and opens
#   [ ] I have printed .shape, .info() and .describe()
#   [ ] I know the target column and whether it is balanced
#   [ ] I have computed a DummyClassifier baseline
#   [ ] Every group member has run the notebook from a clean start
#   [ ] We have agreed the metric in writing
#
# Six checkboxes. They prevent most project failures.
```

## ✏️ Practice

1. Write a scope paragraph for your project using the template above.
2. List the data you need and confirm you can actually get it. Download it today.
3. Choose your metric and justify it from the class balance.
4. Compute a baseline before doing anything else.
5. Write the eight-stage plan with a deliverable for each.

<details><summary>Solutions</summary>

```python
# 1, 2 - Use the Example 1 template. The test of a good scope paragraph is
#        whether someone else could start work from it tomorrow.

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in df.select_dtypes(include="object").columns:
    df[c] = LabelEncoder().fit_transform(df[c])
X, y = df.drop(columns=["loan_status"]), df["loan_status"]

print("class balance:", y.value_counts(normalize=True).round(3).to_dict())   # 3
# Balanced -> accuracy is meaningful here.
# If it were 90/10 you would report RECALL and F1 instead.

base = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=5)  # 4
print(f"baseline: {base.mean():.4f}")
print("Anything you build must clearly beat this, or it has learned nothing.")

# 5 - Scope, Explore, Baseline, Model, Evaluate, Build, Document, Present.
#     Each with a deliverable you could show someone.
```
</details>

## ❓ MCQs

**Q1.** What is the commonest capstone failure?
- (a) A weak model  (b) Choosing a topic before checking the data exists  (c) Poor slides  (d) Too much data

**Q2.** Which should you do before any modelling?
- (a) Tune hyperparameters  (b) Compute a baseline  (c) Build the app  (d) Write the report

**Q3.** Your target is 11% positive. Which metric should you report?
- (a) Accuracy  (b) Recall and F1  (c) Training time  (d) R²

**Q4.** What separates "good" from "excellent" work?
- (a) A better model  (b) Honesty and documentation — fairness audit, confidence intervals, stated limitations  (c) More data  (d) A prettier app

**Q5.** What should you bring to a mentoring session?
- (a) "It doesn't work"  (b) What you tried, what you measured, and a specific question  (c) Nothing  (d) The finished project

**Q6.** Why should every group member run the notebook in week one?
- (a) Practice  (b) Environment problems found in week one are an inconvenience; found the night before, a disaster  (c) It is required  (d) To split the work

<details><summary>Answers</summary>

**A1 — (b).** **Download the data first. Open it. Then decide.**

**A2 — (b) A baseline.** Three lines, and it tells you whether anything afterwards is worth reporting.

**A3 — (b).** Accuracy would be 89% for a model that finds nobody — Session 5's lesson.

**A4 — (b).** **Nobody expects you to beat the state of the art.** They are looking for careful measurement and faithful reporting.

**A5 — (b).** A specific question gets a specific answer in two minutes.

**A6 — (b).** The cheapest risk reduction available to a group.
</details>

## 🎯 Tasks

**Task 1 — Your project proposal.** Write the full scope paragraph, the eight-stage plan with dates, and the role assignment for your group. **Confirm in writing that the data is downloaded and opens.**

**Task 2 — Week one, done properly.** Complete all six checkboxes from Example 3 and produce the EDA notebook and the baseline number. **Bring both to your first mentoring session.**

**Task 3 — The ethics section, up front.** Write the fairness and limitations sections of your report **before you build the model.** You will revise them with real numbers later — but writing them first changes what you build.

---

# ✅ Before you finish the course

**This session**

- [ ] I can choose open or closed weights, with a reason beyond quality
- [ ] I check a licence before building on a model
- [ ] I can load a model from Hugging Face in three lines
- [ ] I can write a model card, including honest limitations
- [ ] **I audit fairness per group, never on overall accuracy alone**
- [ ] I know that dropping the sensitive column does not remove bias
- [ ] I have a capstone question, and the data is already on my disk
- [ ] I compute a baseline before anything else

**The whole course**

- [ ] I can write Python: variables through classes *(Session 1)*
- [ ] I can load, explore and clean real data *(Sessions 2–3)*
- [ ] I can choose the right kind of model for a problem *(Session 4)*
- [ ] I can train, evaluate and save a model *(Session 5)*
- [ ] I can engineer features and know what each technique costs *(Session 6)*
- [ ] I can cluster and find structure without labels *(Session 7)*
- [ ] I report mean ± std, and never a gain smaller than my noise *(Session 8)*
- [ ] I have written a neural network from scratch *(Session 9)*
- [ ] I can prompt an LLM deliberately and get structured output *(Session 10)*
- [ ] I can build an app where the model decides and the LLM explains *(Session 11)*
- [ ] **I can say where my model should not be trusted** *(Session 12)*

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-12-opensource-ethics.ipynb) | Every example above, runnable |
| [Open source GenAI](../tutorials/concepts/open-source-gen-ai.md) | More depth on open models |
| [Hugging Face ecosystem](../tutorials/concepts/hugging-face-ecosystem.md) | The Hub, in detail |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
| [All sessions](README.md) | Back to the index |
