# Session 10 — Generative AI & Large Language Models

**What GenAI is · Predictive vs Generative · Transformers · LLMs · Parameters & Quantization · Prompt Engineering · Zero-shot, One-shot, Few-shot, Chain-of-Thought**

| | |
|---|---|
| **Notebook** | [session-10-genai-llms.ipynb](../notebooks/session-10-genai-llms.ipynb) |
| **Previous** | [Session 9 — Deep Learning](session-09-deep-learning.md) |
| **Next** | [Session 11 — AI-Powered Applications](session-11-ai-apps.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Sessions 5 to 8 built models that *choose* — a class, a number, a cluster.** **This session is about models that *produce*.**
>
> **Parts A and B are concepts, with analogies and no code.** **Part C is your first working GenAI program — a single word, "Hi".** **Part D is how to make it do useful work.**

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Say what Generative AI is, and give **ten real applications**
2. Explain the difference between **Predictive AI and Generative AI**, and pick the right one for a problem
3. Describe the **GenAI workflow** end to end
4. Explain **how deep learning became Transformers, and Transformers became GenAI**
5. Define a **language model**, and say what "large" actually added
6. Explain what **parameters** are and why the count matters
7. Explain **quantization** and the trade it makes
8. Name the leading LLMs and what each is known for
9. Call the Gemini API from Python and **read the raw response**
10. Explain what **temperature** does, and why `temperature=0` is not exactly deterministic
11. Read `thoughts_token_count` and explain why a two-word prompt can cost 196 tokens
12. Write a prompt with all **five core elements**
13. Choose between **zero-shot, one-shot, few-shot and chain-of-thought** — and write each one

---

## How this session is organised

| Part | What it covers |
|---|---|
| **A — [What Generative AI is](#part-a--what-generative-ai-is)** | **Concepts only.** What it is, where it is used, how it differs from what you have built |
| **B — [The fundamentals](#part-b--the-fundamentals)** | **Concepts only.** Deep learning → Transformers → GenAI → LLMs |
| **C — [Your first GenAI program](#part-c--your-first-genai-program)** | **Code.** Setup, `"Hi"`, the raw response, the temperature dial |
| **D — [Prompt engineering](#part-d--prompt-engineering)** | **Concepts and code.** Elements, principles, and the four prompt types |

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [What is Generative AI](#1-what-is-generative-ai) | | 13 | ["Hi" — your first program](#13-hi--your-first-genai-program) |
| 2 | [Applications of GenAI](#2-applications-of-generative-ai) | | 14 | [A few more first programs](#14-a-few-more-first-programs) |
| 3 | [Predictive AI vs GenAI](#3-predictive-ai-vs-generative-ai) | | 15 | [What the machine sees](#15-what-the-machine-sees--the-raw-json) |
| 4 | [How GenAI works](#4-how-genai-works--the-workflow) | | 16 | [The temperature dial](#16-the-temperature-dial) |
| 5 | [Deep learning → Transformers](#5-from-deep-learning-to-transformers) | | 17 | [What prompt engineering is](#17-what-prompt-engineering-is) |
| 6 | [Transformers → GenAI](#6-from-transformers-to-generative-ai) | | 18 | [The core elements of a prompt](#18-the-core-elements-of-a-prompt) |
| 7 | [From LMs to LLMs](#7-language-models-from-lms-to-llms) | | 19 | [The principles](#19-the-principles) |
| 8 | [Parameters in LLMs](#8-parameters-in-llms) | | 20 | [Zero-shot prompting](#20-zero-shot-prompting) |
| 9 | [Quantization](#9-quantization) | | 21 | [One-shot prompting](#21-one-shot-prompting) |
| 10 | [The leading LLMs](#10-the-leading-llms) | | 22 | [Few-shot prompting](#22-few-shot-prompting) |
| 11 | [LLM applications](#11-llm-applications) | | 23 | [Chain-of-thought prompting](#23-chain-of-thought-prompting) |
| 12 | [Setup](#12-setup--install-and-api-key) | | 24 | [Choosing a prompt type](#24-choosing-a-prompt-type) |

**The [practices](#-session-10--practice), [20 MCQs](#-session-10--20-mcqs) and [tasks](#-session-10--tasks) are at the end.**

---

# Part A — What Generative AI is

# 1. What is Generative AI

> **Generative AI is a model that produces new content rather than choosing from a fixed set of answers.**

🧠 **Analogy: a multiple-choice exam versus an essay.**
>
> **Everything you built in Sessions 5 to 8 sat a multiple-choice exam.** *Approved or rejected? Setosa, versicolor or virginica? What number?* **The set of possible answers existed before the model ran.**
>
> **Generative AI writes the essay.** **Nobody wrote the answer in advance — not even the people who built the model.**

## What "new" actually means

**A generative model does not retrieve a stored answer. It builds one, one piece at a time.**

```text
Prompt:  "Write a tagline for a coffee shop."

The model does NOT look up a list of taglines.
It predicts the most likely next word, then the next, then the next:

  "Your"  ->  "Your daily"  ->  "Your daily grind,"  ->  "Your daily grind, perfected."

Each word is chosen given every word before it.
```

> **That is the whole mechanism, and it is worth sitting with.** **A model that only ever predicts the next word can write a working program, summarise a legal document and translate Tamil into Portuguese** — because doing all of those well *is* predicting the next word well.

## The families

| Family | Produces | Examples |
|---|---|---|
| **Text** | Words | ChatGPT, Gemini, Claude |
| **Image** | Pictures | DALL·E, Midjourney, Stable Diffusion |
| **Audio** | Speech, music | ElevenLabs, Suno |
| **Video** | Moving images | Sora, Veo |
| **Code** | Programs | GitHub Copilot, Cursor |
| **Multimodal** | **Several at once** | Gemini, GPT-4o — text in, image out; image in, text out |

**This session is about the first one. Text models are where the ideas are clearest, and they are what you will build with in [Session 11](session-11-ai-apps.md).**

---

# 2. Applications of Generative AI

**Ten that are in production somewhere today.**

| # | Where | What it does |
|---|---|---|
| 1 | **Customer support** | Drafts replies, summarises a ticket history, routes by intent |
| 2 | **Software development** | **Writes code, explains code, writes the tests** |
| 3 | **Content and marketing** | Product descriptions, ad copy, translations |
| 4 | **Education** | **Explains a concept five different ways until one lands** |
| 5 | **Healthcare admin** | Summarises clinical notes; drafts discharge letters *(a human signs them)* |
| 6 | **Legal** | Summarises contracts; finds the clause you are looking for |
| 7 | **Search** | **Answers the question instead of returning ten links** |
| 8 | **Accessibility** | Image descriptions, live captions, text-to-speech |
| 9 | **Data work** | **Writes the SQL, explains the error, documents the table** |
| 10 | **Design** | First-draft images, layouts, icons |

## What they have in common

> **Every one of them is a task where *many answers are acceptable* and *a human checks the result*.**
>
> **That is the shape of a good GenAI problem.** **It is also the shape of its limits** — see [§3](#3-predictive-ai-vs-generative-ai).

## ⚠️ Where it does not belong

| Task | Why not | Use instead |
|---|---|---|
| Calculating a tax bill | **There is exactly one right answer** | Arithmetic |
| Deciding a loan | **Must be auditable and consistent** | [Session 5B](session-05b-classification.md)'s classifier |
| Checking a password | Exact match required | A comparison |
| Anything safety-critical without review | **It can be confidently wrong** | A human |

---

# 3. Predictive AI vs Generative AI

**You have spent five sessions on Predictive AI. This is what changes.**

| | **Predictive AI** (Sessions 5–8) | **Generative AI** (this session) |
|---|---|---|
| **Question** | *Which one? How much?* | ***Make me something*** |
| **Output** | **A label or a number** | **New content** |
| **Answer space** | **Fixed and known in advance** | **Unbounded** |
| **Training data** | **Labelled** — `X` and `y` | **Unlabelled text** — the next word IS the label |
| **Training cost** | Seconds to hours on a laptop | **Weeks on thousands of GPUs** |
| **Who trains it** | **You** | **You almost never do** — you use somebody's |
| **Evaluation** | **Accuracy, R², F1** — one number | **Hard.** Often human judgement |
| **Same input twice** | **Same answer** | **Usually a different answer** |
| **Explainability** | Feature importance, coefficients | **Very limited** |
| **Typical size** | Kilobytes to megabytes | **Gigabytes to terabytes** |

## 🧠 Analogy: a sorting machine versus a writer

> **A predictive model is a machine on a factory line that sorts apples into three bins.** **Fast, consistent, and it will never produce anything except "bin 1, 2 or 3".**
>
> **A generative model is a writer you brief.** **Ask twice and you get two different pieces.** **Brief them badly and you get something confident and useless.**

## The four differences that will bite you

**1. Non-determinism.** **The same prompt gives different answers.** [§16](#16-the-temperature-dial) shows the dial that controls this. **If your application needs reproducibility, you must design for it.**

**2. No accuracy score.** **There is no `accuracy_score(y_test, y_pred)` for an essay.** **Evaluation is human review, or a second model, or a checklist — all of them harder than a number.**

**3. Confident wrongness.** **A classifier that is unsure gives you a probability of 0.51.** **A language model that is unsure gives you a fluent, well-punctuated, completely invented answer.** This is called *hallucination*, and it is the single biggest practical risk.

**4. You are renting, not building.** **Your model comes from Google, OpenAI or Meta.** **Its price, its behaviour and its availability are decisions somebody else makes.**

## Choosing between them

| If the task is… | Use |
|---|---|
| **One correct answer exists** | **Predictive** — or plain code |
| You must explain every decision | **Predictive** |
| The answer must be identical every time | **Predictive** |
| **Many answers are acceptable** | **Generative** |
| **The output is language, images or code** | **Generative** |
| **You have no labelled data** | **Generative** |

> **They are not competitors.** **[Session 11](session-11-ai-apps.md) puts them in the same application: a classifier makes the decision, and a language model explains it to the customer.**

---

# 4. How GenAI works — the workflow

**Five stages. You will only ever perform the last two.**

```text
1. PRE-TRAINING          read an enormous amount of text; learn to predict the next word
        |                  (months, thousands of GPUs, millions of dollars)
        v
2. FINE-TUNING           learn to follow instructions rather than just continue text
        |                  (much smaller, still expensive)
        v
3. ALIGNMENT (RLHF)      learn which answers humans actually prefer
        |                  (human reviewers rank outputs)
        v
--------- everything above is done by the model provider ---------
        v
4. PROMPTING             you write an instruction                  <- YOUR JOB
        |
        v
5. INFERENCE             the model generates a response, token by token
```

## Stage by stage

**1. Pre-training — learning the language.**

🧠 **Analogy: a child who has read every book in the library but has never been asked a question.** **They know how sentences work, what usually follows what, and an enormous amount of the world's information — but they will just carry on the text you start.**

> **The training signal is free.** **Take any sentence, hide the next word, and ask the model to guess it.** **No labelling is needed — that is why it can train on trillions of words.**

**2. Fine-tuning — learning to be asked.**

🧠 **The same child, now taught that a question expects an answer.** **Trained on example conversations: prompt in, good response out.**

> **This is the difference between a model that continues *"The capital of France is"* and a model that answers *"What is the capital of France?"***

**3. Alignment — learning what humans prefer.**

🧠 **An editor telling the writer which of two drafts is better, thousands of times.** **The model learns the pattern of "better".**

> **This is where "helpful, harmless, honest" comes from** — and where the refusals come from too.

**4. Prompting — your job.** **All of [Part D](#part-d--prompt-engineering).**

**5. Inference — generation, one token at a time.**

```text
prompt -> [tokenize] -> [model] -> probability over every possible next token
                          ^                    |
                          |                    v
                          +---------------- pick one, append, repeat
                                            until a stop token
```

> **The loop is why responses stream in word by word rather than appearing at once.** **The model genuinely does not know how its sentence ends when it starts writing it.**

---

# Part B — The fundamentals

# 5. From deep learning to Transformers

**[Session 9](session-09-deep-learning.md) ended with a network that takes a fixed set of numbers and produces an answer. Language is not a fixed set of numbers.**

## The problem with language

| Problem | Why a plain network struggles |
|---|---|
| **Variable length** | A sentence can be 3 words or 3,000. A network has a fixed input size |
| **Order matters** | *"dog bites man"* and *"man bites dog"* have identical words |
| **Long-range dependence** | *"The **keys** I left on the table this morning, next to the letter from your mother, **are** missing."* — `keys` and `are` are 14 words apart |

## What came before

| Architecture | Idea | Why it was not enough |
|---|---|---|
| **RNN** (1990s) | Read one word at a time, keep a memory | **Forgets the start of a long sentence** |
| **LSTM** (1997) | An RNN with gates that decide what to remember | Better memory — **but still strictly sequential, so it cannot be parallelised** |

> 🧠 **An RNN reads a book through a keyhole, one word at a time, trying to remember everything.** **By chapter 12 it has forgotten chapter 1.**

## The Transformer, 2017

**One paper — *Attention Is All You Need* — replaced the sequential reading with something better.**

> 🧠 **A Transformer reads the whole page at once, and for every word it asks: *which other words on this page matter for understanding this one?***
>
> **For `are`, it looks back and finds `keys`.** **Not because of where it sits, but because of what it means.**

**That mechanism is called *self-attention*, and it has two consequences.**

| Consequence | Why it changed everything |
|---|---|
| **Long-range dependencies are easy** | **Any word can attend directly to any other word.** Distance stops mattering |
| **It parallelises** | **Every word is processed at once**, so you can use thousands of GPUs — and *that* is what made models this large possible |

> **The second point is the one people miss.** **The Transformer is not just more accurate than an LSTM — it is *trainable at a scale an LSTM never could be*.** **Everything since has been built on that.**

---

# 6. From Transformers to Generative AI

**A Transformer is an architecture. Generative AI is what happens when you train one on enough text.**

## The three shapes

| Shape | What it is good at | Example |
|---|---|---|
| **Encoder-only** | **Understanding** text — classification, search | **BERT** |
| **Decoder-only** | **Generating** text — one token at a time | **GPT, LLaMA, Gemini, Claude** |
| **Encoder–decoder** | **Transforming** text — translation, summarisation | **T5, BART** |

> **Generative AI as you meet it is almost entirely *decoder-only*.** **The model reads what has been written so far and predicts what comes next — including its own previous output.**

## The step nobody predicted

**Researchers scaled these models up expecting steadily better next-word prediction. They got that. They also got things nobody trained for.**

| Model size | What it could do |
|---|---|
| Small | Predict the next word. Grammatical, not useful |
| Medium | Answer simple questions |
| **Large** | **Translate, summarise, write code, do arithmetic, follow instructions** |

> **None of those were training objectives.** **The only objective was ever "predict the next token".**
>
> **The claim is often overstated as magic. It is not magic — but it is genuinely surprising**, and it is why the field moved so fast after 2020.

---

# 7. Language models: from LMs to LLMs

## What a language model is

> **A language model assigns a probability to what comes next.**

```text
"The cat sat on the ___"

   mat      0.31
   floor    0.12
   sofa     0.09
   roof     0.04
   ...
```

**That is the entire definition.** **Everything else is a matter of scale.**

🧠 **Analogy: your phone's keyboard.** **It suggests the next word as you type. That is a language model — a tiny one.** **An LLM is the same idea, with a hundred billion times the machinery.**

## The history, in one table

| Era | Approach | How it worked |
|---|---|---|
| **1950s–90s** | **n-grams** | Count how often word B followed word A **in a corpus**. Look it up |
| **2000s** | Neural LMs | A small network predicts the next word from the last few |
| **2013** | **Word embeddings** (word2vec) | **Words became vectors, so "king − man + woman ≈ queen" worked** |
| **2017** | **Transformers** | Self-attention; parallel training |
| **2018–now** | **LLMs** | Transformers, scaled enormously |

## From LM to LLM — what "large" added

| | **Language Model** | **Large Language Model** |
|---|---|---|
| **Parameters** | Thousands to millions | **Billions to trillions** |
| **Training text** | A corpus | **Much of the public internet** |
| **Trained for** | One task | **Next-token prediction, generally** |
| **Used for** | The task it was trained on | ***Any* task you can describe in words** |
| **Needs task-specific training?** | **Yes** | **Usually no — you describe the task in the prompt** |

> **The last row is the one that matters.**
>
> **A 2015 sentiment model was trained on labelled reviews and could do exactly one thing.** **An LLM does sentiment analysis because you asked it to in English** — and does translation in the next request without being retrained.
>
> **That shift — from *training a model per task* to *describing the task* — is what [Part D](#part-d--prompt-engineering) is about.**

---

# 8. Parameters in LLMs

> **A parameter is one number the model learned during training — exactly the weights and biases of [Session 9](session-09-deep-learning.md#18-parameters-vs-hyperparameters).**

**The iris network in Session 9 had 67 parameters. An LLM has billions of the same thing.**

| Model | Parameters | Session 9's iris MLP as a unit |
|---|---|---|
| Session 9's MLP | **67** | 1× |
| BERT-base | **110 million** | 1.6 million× |
| GPT-3 | **175 billion** | 2.6 billion× |
| LLaMA 3 (largest) | **405 billion** | 6 billion× |

## What the count buys you

| More parameters means | |
|---|---|
| ✅ **More capacity to store patterns** | Better reasoning, more knowledge, more languages |
| ❌ **More memory to run it** | See the arithmetic below |
| ❌ **More compute per token** | Slower and more expensive |
| ❌ **More energy** | Both to train and to serve |

## The arithmetic you need

> **At standard 16-bit precision, every parameter costs 2 bytes.**

```text
7 billion parameters  x  2 bytes  =  14 GB just to hold the weights
70 billion parameters x  2 bytes  = 140 GB
175 billion           x  2 bytes  = 350 GB
```

> **A good consumer graphics card has 24 GB.** **A 7-billion model fits. A 70-billion model does not — not without help.**
>
> **That "help" is [§9](#9-quantization).**

## ⚠️ Bigger is not automatically better

> **Parameter count is not a quality score.** **A well-trained 7-billion model routinely beats a badly-trained 70-billion one**, and for most tasks a small fast model is the right choice.
>
> **This is [Session 8](session-08-evaluation-tuning.md#5-use-case-2--car-prices)'s lesson again: capacity you do not need is capacity you are paying for.**

---

# 9. Quantization

> **Quantization stores each parameter using fewer bits.**

🧠 **Analogy: photograph file sizes.** **A RAW photo holds enormous colour precision. A JPEG throws most of it away — and you usually cannot tell.** **Quantization is JPEG for model weights.**

## What it costs and what it saves

| Precision | Bits per parameter | **7B model size** | Quality |
|---|---|---|---|
| **FP32** (full) | 32 | **28 GB** | Reference |
| **FP16 / BF16** | 16 | **14 GB** | Effectively identical |
| **INT8** | 8 | **7 GB** | **Very slight loss** |
| **INT4** | 4 | **3.5 GB** | **Noticeable but often acceptable** |

> **A 70-billion model at 4-bit is about 35 GB — which fits on hardware where the full-precision version needs 280 GB.**
>
> **That single fact is why you can run a capable model on a laptop at all.**

## The trade, stated plainly

| ✅ Gains | ❌ Costs |
|---|---|
| **Fits on smaller hardware** | **Some accuracy is lost** |
| **Faster** — less memory to move | **Loss grows as bits shrink** |
| **Cheaper to serve** | Not every operation quantizes cleanly |
| **Runs locally — your data never leaves the machine** | |

> **The usual sweet spot is 8-bit or 4-bit.** **Below 4 bits, quality degrades quickly.**
>
> ⚠️ **And measure it.** **"It still seems fine" is not evidence.** **Run your actual task at each precision and compare** — the same discipline [Session 8](session-08-evaluation-tuning.md#part-b--model-validation-techniques) applied to everything else.

---

# 10. The leading LLMs

| Model | From | Shape | Known for |
|---|---|---|---|
| **GPT** | OpenAI | Decoder-only | **The family that made this mainstream.** Strong general reasoning |
| **BERT** | Google, 2018 | **Encoder-only** | **Understanding, not generating.** Still everywhere in search and classification |
| **T5** | Google | **Encoder–decoder** | *"Text-to-text"* — every task framed as text in, text out |
| **Gemini** | Google | Decoder-only | **Multimodal** — text, images, audio, video. **Used in [Part C](#part-c--your-first-genai-program)** |
| **Claude** | Anthropic | Decoder-only | Long context; careful instruction-following |
| **LLaMA** | Meta | Decoder-only | **Open weights** — you can download and run it |
| **Mistral** | Mistral AI | Decoder-only | **Small and efficient.** Strong quality per parameter |
| **Falcon** | TII, Abu Dhabi | Decoder-only | Open weights; permissively licensed |

## The distinction that matters most

| | **Closed / API** | **Open weights** |
|---|---|---|
| Examples | GPT, Gemini, Claude | **LLaMA, Mistral, Falcon** |
| How you use it | **Send a request over the internet** | **Download the file and run it** |
| Your data | **Leaves your machine** | **Stays on your machine** |
| Cost | **Per token, forever** | Hardware, once |
| Control | The provider can change or retire it | **Yours** |
| Typical quality | Usually ahead | **Closing the gap fast** |

> **This is a real decision, not a preference.** **A hospital cannot send patient notes to an API.** **A two-person startup cannot buy the GPUs to serve a 70-billion model.**
>
> **[Session 12](session-12-opensource-ethics.md) works through this properly, with Hugging Face.**

⚠️ **Model names and versions change every few months.** **The table above is a map of the families, not a current price list.** **Check the provider's documentation for what exists today.**

---

# 11. LLM applications

**The same eight patterns cover most of what gets built.**

| Pattern | What it does | Example |
|---|---|---|
| **1. Generation** | Produce new text | Draft an email, write a product description |
| **2. Summarisation** | **Long → short** | 40-page report → one page |
| **3. Extraction** | **Unstructured → structured** | An email → `{name, date, amount}` |
| **4. Classification** | **Sort into categories, with no training data** | Route a support ticket |
| **5. Translation** | Language → language, or **jargon → plain English** | |
| **6. Question answering** | **Answer from a supplied document** | "What does clause 7 say?" |
| **7. Code** | Write, explain, fix, test | |
| **8. Conversation** | **Multi-turn dialogue with memory** | A support assistant |

> **Look at patterns 3 and 4 again.** **You built a classifier in Session 5B and it needed 10,000 labelled rows.** **An LLM classifies a support ticket with zero training data, because you describe the categories in the prompt** — you will do exactly that in [§22](#22-few-shot-prompting).
>
> **That does not make Session 5B obsolete.** **The classifier is cheaper, faster, reproducible and auditable.** **The LLM is available immediately with no data at all.** Different tools.

---

# Part C — Your first GenAI program

**Follows `genai_api_demo.ipynb`.**

> **Every output marked "Measured" in Parts C and D was produced by running the code against `gemini-3.5-flash`.** **Token counts, word counts and run-to-run variation are all real numbers, not illustrations.**
>
> ⚠️ **You will still not get the same text.** **A generative model gives a different answer each time** — that is the defining property, and [§16](#16-the-temperature-dial) is the dial that controls it. **Several of the measurements below are *about* that variation.**

---

# 12. Setup — install and API key

## Step 1 — install the library

```bash
pip install -q google-genai
```

**In a Colab or Jupyter cell, prefix it with `!`:**

```python
# needs-install: pip install google-genai
!pip install -q google-genai
```

> **It is already in the course [`requirements.txt`](../requirements.txt)** — if you installed from that file into the `genai` environment, you have it.

## Step 2 — get an API key

| # | Step |
|---|---|
| 1 | Go to **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)** |
| 2 | Sign in with a Google account |
| 3 | **Create API key** |
| 4 | Copy it — a long string beginning with `AIza` |

**There is a free tier. It is rate-limited, and it is enough for everything in this session.**

## ⚠️ Step 3 — store it, do not type it

**This is the single most important paragraph in Part C.**

```python
# illustrative: this is what NOT to do.
api_key = "<your-39-character-key>"    # <- NEVER. Not even briefly. Not even in a comment.
client = genai.Client(api_key=api_key)
```

> **A key pasted into a notebook is a key in your git history forever.** **Deleting the line does not remove it — `git log` still has it.**
>
> **Keys leaked this way get found by automated scanners within minutes, and the bill is yours.**

### In Colab — use Secrets

```python
# api-only: needs a Gemini API key.
from google.colab import userdata

api_key = userdata.get("GEMINI_API_KEY")
```

**Click the 🔑 key icon in Colab's left sidebar, add a secret named `GEMINI_API_KEY`, and switch on notebook access.**

### Locally — use an environment variable

```bash
export GEMINI_API_KEY="your-key-here"
```

```python
# api-only: needs a Gemini API key.
import os

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in your environment first.")
```

**Or keep it in a `.env` file that is listed in `.gitignore`:**

```python
# api-only: needs a Gemini API key.
import os
from dotenv import load_dotenv

load_dotenv()                      # reads .env from the current directory
api_key = os.environ["GEMINI_API_KEY"]
```

```text
.env                    <- contains  GEMINI_API_KEY=AIza...
.gitignore              <- contains  .env
```

## Step 4 — one place for the model name

```python
# api-only: needs a Gemini API key.
MODEL_ID = "gemini-3.5-flash"
```

> ⚠️ **Model IDs change every few months.** **Put it in one variable at the top and every call in your file updates with one edit.**

**Do not guess which models exist — ask.**

```python
# api-only: needs a Gemini API key.
for m in client.models.list():
    print(m.name.replace("models/", ""))
```

**A trimmed excerpt of what that returned when this guide was written:**

```text
gemini-2.5-flash          gemini-3.1-flash-lite     gemini-3.6-flash
gemini-2.5-flash-lite     gemini-3.1-pro-preview    gemini-3.7-flash
gemini-2.5-pro            gemini-3.5-flash          gemini-flash-latest
gemini-3-flash-preview    gemini-3.5-flash-lite     gemini-pro-latest
```

> **Both IDs in the trainer's notebook are real** — `gemini-3.5-flash` and `gemini-2.5-flash-lite`. **This guide uses `gemini-3.5-flash` to match it.**
>
> **`gemini-flash-latest` is an alias that always points at the current flash model.** **Convenient for experiments; risky in production, because the model underneath can change without you touching your code.**
>
> **If a call fails with a 404, run the listing above** — it is faster than searching the documentation.

---

# 13. "Hi" — your first GenAI program

**One word in, a sentence out. This is the whole thing.**

```python
# api-only: needs a Gemini API key.
from google import genai

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Hi"
)

print(response.text)
```

**The response, run against `gemini-3.5-flash`:**

```text
Hello! How can I help you today?
```

> **That is a working Generative AI program.** **Three lines of real code.**

## What each line does

| Line | What it does |
|---|---|
| `genai.Client(api_key=api_key)` | **Opens a connection.** Do this once, not per request |
| `client.models.generate_content(...)` | **Sends the request over the internet** and waits |
| `model=MODEL_ID` | **Which model.** Different models, different speed, cost and quality |
| `contents="Hi"` | **Your prompt.** A plain string is enough |
| `response.text` | **The generated text.** [§15](#15-what-the-machine-sees--the-raw-json) shows what else is in there |

## The version with the error handling you will actually want

**This is the trainer notebook's connection test, tidied.**

```python
# api-only: needs a Gemini API key.
import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: set GEMINI_API_KEY in your environment (or Colab Secrets) first.")
else:
    client = genai.Client(api_key=api_key)

    print("Sending request to Gemini...")

    response = client.models.generate_content(
        model=MODEL_ID,
        contents="Hi Gemini"
    )

    print("\nSuccess! Here is the response:")
    print("-" * 30)
    print(response.text)
```

**An actual run:**

```text
Sending request to Gemini...

Success! Here is the response:
------------------------------
Hello! How can I help you today?
```

> **If this prints, everything else in this session will work.** **If it does not, the problem is the key or the model name — nothing else.**

## ⚠️ The three errors you will hit

| Error | Cause | Fix |
|---|---|---|
| `API key not valid` | Wrong or expired key | Regenerate it in AI Studio |
| `404 model not found` | **The model ID does not exist** | Check the current model list |
| `429 RESOURCE_EXHAUSTED` | **Free-tier rate limit** | Wait a minute; add `time.sleep(1)` between calls in a loop |

---

# 14. A few more first programs

**Same three lines. Only the prompt changes — and that is the point of the whole session.**

## A question

```python
# api-only: needs a Gemini API key.
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Why is the sky blue? Answer in one short sentence."
)
print(response.text)
```

**Measured:** `The sky is blue because Earth's atmosphere scatters blue light from the sun more than other colors.`

## A constraint on the length

```python
# api-only: needs a Gemini API key.
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Why is the sky blue? Answer in exactly 5 words."
)
print(response.text)
```

**Measured — three runs, all identical:** `Air molecules scatter blue light.`

> **Compare the two.** **The only difference is four words of instruction — and the output changed completely.** **That is prompt engineering, and you have just done it.**
>
> **And count the words in the second: exactly five, three times out of three.** **Older and smaller models are notoriously bad at exact counts, because they work in tokens rather than words** — this one was not. **Check your model rather than assuming either way.**

## Something a predictive model could never do

```python
# api-only: needs a Gemini API key.
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Explain what a decision tree is to a ten-year-old, in three sentences."
)
print(response.text)
```

> **There was no training set of "explanations for ten-year-olds".** **The model was never trained on this task.** **[§7](#7-language-models-from-lms-to-llms)'s point, in one call.**

## Asking it to write code

```python
# api-only: needs a Gemini API key.
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Write a Python function that returns True if a number is prime. "
             "Include a docstring and no explanation."
)
print(response.text)
```

> ⚠️ **Read anything it writes before you run it.** **The model produces code that *looks* right with complete confidence, and looking right is not the same as being right.**

## A batch of prompts

```python
# api-only: needs a Gemini API key.
import time

prompts = [
    "Summarise machine learning in one sentence.",
    "Give me one reason to scale features before kNN.",
    "What is overfitting? One sentence.",
]

for prompt in prompts:
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    print(f"Q: {prompt}\nA: {response.text.strip()}\n")
    time.sleep(1)          # stay under the free-tier rate limit
```

> **`time.sleep(1)` is not decoration.** **The free tier limits requests per minute, and a loop without a pause will hit `429` within seconds.**

---

# 15. What the machine sees — the raw JSON

**`response.text` is a convenience. It hides most of what came back.**

```python
# api-only: needs a Gemini API key.
prompt = "Why is the sky blue? Answer in one short sentence."

response = client.models.generate_content(model=MODEL_ID, contents=prompt)

print("=== WHAT THE USER SEES ===")
print(response.text)
print()

print("=== WHAT THE MACHINE SEES (RAW JSON) ===")
print(response.model_dump_json(indent=2))
```

**The real response, for a one-sentence question:**

```text
{
  "candidates": [
    {
      "content": {
        "parts": [ { "text": "The sky is blue because Earth's atmosphere..." } ],
        "role": "model"
      },
      "finish_reason": "STOP",
      "avg_logprobs": ...,
      "safety_ratings": ...
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 13,
    "thoughts_token_count": 473,          <- read this one twice
    "candidates_token_count": 24,
    "total_token_count": 510
  },
  "model_version": "gemini-3.5-flash",
  "response_id": "..."
}
```

## The four fields worth knowing

| Field | Why you care |
|---|---|
| **`candidates`** | **A list.** You can ask for more than one answer to the same prompt |
| **`finish_reason`** | **`STOP` means it finished. `MAX_TOKENS` means it ran out of budget** |
| **`usage_metadata`** | **What you are billed on** |
| **`model_version`** | **Which model actually answered** — essential when you used an alias like `gemini-flash-latest` |

---

## ⚠️ `thoughts_token_count` — the field that will surprise you

**Modern reasoning models generate hidden "thinking" tokens before they answer. You do not see them. You are billed for them.**

**Measured across four prompts:**

| Prompt | Prompt tokens | **Thinking tokens** | Output tokens | **Total** |
|---|---|---|---|---|
| `"Hi"` | 2 | **185** | 9 | **196** |
| `"What is 2+2?"` | 8 | **83** | 7 | **98** |
| *"Why is the sky blue? One short sentence."* | 13 | **473** | 24 | **510** |
| The widget puzzle | 34 | **468** | 77 | **579** |

> **Read the first row again.** **A two-token prompt and a nine-token answer cost 196 tokens.**
>
> **The hidden reasoning is 94% of the bill, and `response.text` gives you no hint that it happened.**

## Turning it off

```python
# api-only: needs a Gemini API key.
from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Why is the sky blue? Answer in one short sentence.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    ),
)
```

**Measured, same question:**

| | Thinking tokens | **Total tokens** |
|---|---|---|
| Default | 473 | **510** |
| **`thinking_budget=0`** | **none** | **39** |

> **A 13× reduction, and the answer was still a correct one-sentence explanation.**
>
> **Use thinking for reasoning tasks. Turn it off for extraction, classification and formatting** — where it buys nothing and costs everything.

## ⚠️ And `max_output_tokens` is a budget for *both*

**This is the trap. `max_output_tokens` caps thinking tokens *and* answer tokens together.**

```python
# api-only: needs a Gemini API key.
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Explain gradient descent in detail, with examples.",
    config=types.GenerateContentConfig(max_output_tokens=20),
)
print(response.text)
```

**Measured at four caps:**

| `max_output_tokens` | `finish_reason` | Thinking tokens | **Answer tokens** |
|---|---|---|---|
| **20** | `MAX_TOKENS` | 15 | **1** |
| **100** | `MAX_TOKENS` | 93 | **3** |
| **500** | `MAX_TOKENS` | 481 | **14** |
| 2000 | `MAX_TOKENS` | 1344 | 652 |

> **At a cap of 500, thinking consumed 481 tokens and the answer got 14.**
>
> ⚠️ **And `response.text` can come back as `None`** — that happened at a cap of 20 in one of the runs here. **Code that does `response.text.strip()` will raise `AttributeError` on a real user's request.**
>
> **Always check `finish_reason`, and never assume `response.text` is a string.**

```python
# api-only: needs a Gemini API key.
if response.candidates[0].finish_reason.name != "STOP":
    print("warning: response was cut short —", response.candidates[0].finish_reason)
text = response.text or ""
```

## Tokens, and why they are the unit

**Models do not read words. They read *tokens* — roughly 4 characters, or about ¾ of a word in English.**

```text
"Machine learning models predict the next token, repeatedly."
  8 words  ->  about 10 tokens
```

| | |
|---|---|
| **You are billed per token** — prompt, thinking and output | `usage_metadata` is your meter |
| **The context window is measured in tokens** | How much the model can consider at once |
| **Rare words cost more tokens** | `"antidisestablishmentarianism"` is several; `"the"` is one |

> **A one-sentence exchange cost 510 tokens, of which 473 were invisible.** **A 40-page document is roughly 20,000 prompt tokens before the model thinks at all.**
>
> **That arithmetic is your bill, and `thoughts_token_count` is the first place to look when a GenAI feature turns out to cost more than you budgeted.**

---

# 16. The temperature dial

**[§3](#3-predictive-ai-vs-generative-ai) said the same prompt gives different answers. Temperature is why, and it is a number you control.**

🧠 **Analogy: a chef following a recipe.**
>
> **Temperature 0 — follows the recipe exactly, every time.** Identical dish, identical result, no surprises.
>
> **Temperature 1 — improvises within the style.** Recognisably the same dish; different every time.
>
> **Temperature 2 — improvises wildly.** Sometimes brilliant, often inedible.

## What it actually does

**At each step the model has a probability for every possible next token. Temperature reshapes that distribution before one is picked.**

```text
Next-token probabilities:   "perfected" 0.31   "brewed" 0.12   "elevated" 0.09   ...

Temperature 0.0  ->  always take the highest.          -> "perfected", every time
Temperature 1.0  ->  sample in proportion.             -> usually "perfected", sometimes not
Temperature 2.0  ->  flatten the distribution first.   -> the unlikely words get a real chance
```

## The demo

```python
# api-only: needs a Gemini API key.
import time
from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)
PROMPT = "Write a tagline for a coffee shop."

print("THE TEMPERATURE DIAL DEMO\n" + "=" * 40)
print(f"Prompt: '{PROMPT}'\n")

for temp in [0.0, 1.0]:
    print(f"--- TEMPERATURE {temp} ---")
    for i in range(1, 6):
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=PROMPT,
            config=types.GenerateContentConfig(temperature=temp),
        )
        print(f"Attempt {i}: {response.text.strip()}")
        time.sleep(1)          # stay under the rate limit
    print()
```

## What was actually measured

| Temperature | **Distinct outputs in 5 runs** | Behaviour |
|---|---|---|
| **0.0** | **2 of 5** | **Near-identical.** The two variants differed by a single comma |
| **1.0** | **5 of 5** | Genuinely different taglines — different tones, angles and vocabulary |
| **2.0** | **5 of 5** | Different again, and more elaborate |

> ⚠️ **Temperature 0 was *nearly* deterministic, not exactly deterministic.** **Five runs gave two variants separated by one comma.**
>
> **This matters if you are writing tests.** **Do not assert on exact strings, even at temperature 0** — assert on structure, or on a parsed field.

## ⚠️ And look at what came back at all

**The prompt says *"Write **a** tagline"*. Every single response was a menu:**

```text
Here are a few options, depending on the vibe of your coffee shop:

**The Clever & Energetic**
* ...
```

> **Not one run returned a single tagline.**
>
> **This is [§18](#18-the-core-elements-of-a-prompt)'s missing element — format — caught in the wild.** **`"Write one tagline. Output only the tagline, nothing else."` is the fix**, and it is worth running both versions to see the difference.

## Choosing a temperature

| Task | Temperature | Why |
|---|---|---|
| **Extracting structured data** | **0.0** | **You want the same answer every time** |
| Classification | **0.0** | Same reason |
| Factual questions | **0.0 – 0.3** | Reduce invention |
| Summarising | 0.3 – 0.7 | Some phrasing freedom |
| **Brainstorming, marketing copy** | **0.8 – 1.2** | **Variety is the product** |
| Experimental / creative | 1.5 – 2.0 | Expect nonsense among the good ones |

> ⚠️ **Temperature 0 reduces variation. It does not guarantee truth.** **A model can be perfectly deterministic and confidently wrong** — every single time, in the same way.

## The other dials

| Setting | What it does |
|---|---|
| **`max_output_tokens`** | **Caps the response length.** Watch for `finish_reason: MAX_TOKENS` |
| `top_p` | Sample only from the most likely tokens that add up to `p` |
| `top_k` | Sample only from the top `k` tokens |
| **`system_instruction`** | **A standing instruction for every request** — *"You are a concise SQL tutor"* |
| `seed` | Ask for reproducibility — **best-effort, not a guarantee** |

```python
# api-only: needs a Gemini API key.
config = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=200,
    system_instruction="You are a concise assistant. Never exceed three sentences.",
)
response = client.models.generate_content(
    model=MODEL_ID, contents="Explain gradient descent.", config=config)
print(response.text)
```

---

# Part D — Prompt engineering

# 17. What prompt engineering is

> **Prompt engineering is writing the instruction well enough that the model does what you meant.**

🧠 **Analogy: briefing a brilliant new colleague on their first morning.**
>
> **They are fast, widely read and eager.** **They have never met your company, your customers or your file formats, and they will not ask.**
>
> ***"Write something about the product"*** **gets you something.** ***"Write three bullet points for the product page, under 15 words each, aimed at first-time buyers, no jargon"*** **gets you what you wanted.**
>
> **The colleague did not get smarter. The brief did.**

## Why this replaced training a model

**[§7](#7-language-models-from-lms-to-llms) said it: the task moved from the training set into the prompt.**

| | 2015 | **Now** |
|---|---|---|
| To build a sentiment classifier | **Label 10,000 reviews, train, evaluate, deploy** | **Write: *"Classify the sentiment as positive, negative or neutral."*** |
| Time | Weeks | **Seconds** |
| Data needed | **Thousands of labelled rows** | **None** |
| Cost to change the categories | **Relabel and retrain** | **Edit the sentence** |

> ⚠️ **And the honest other side:** **the classifier is cheaper per call, faster, reproducible, auditable, and it runs offline.** **[Session 5B](session-05b-classification.md) is not obsolete** — it is the right tool once you have the labels.

## What a bad prompt costs

| Bad prompt | What goes wrong |
|---|---|
| *"Summarise this."* | **How long? For whom? What matters?** You get a summary, not the one you needed |
| *"Is this good?"* | **No criteria** — the model invents its own |
| *"Fix the code."* | **Fix what?** It may rewrite working code |
| *"Give me data about sales."* | **Invention.** The model has no sales data, and it may produce some anyway |

---

# 18. The core elements of a prompt

**Five elements. Not every prompt needs all five — but when a prompt is not working, one of these is missing.**

| # | Element | The question it answers | Example |
|---|---|---|---|
| **1** | **Role** | *Who is answering?* | *"You are an experienced data engineer."* |
| **2** | **Task** | ***What do you want?*** — the one element you cannot skip | *"Summarise the report below."* |
| **3** | **Context** | *What does it need to know?* | The report; the audience; the constraints |
| **4** | **Format** | *What shape should the answer be?* | *"Three bullet points, under 15 words each."* |
| **5** | **Examples** | *What does good look like?* | **[§21](#21-one-shot-prompting) and [§22](#22-few-shot-prompting)** |

## All five, in one prompt

```python
# api-only: needs a Gemini API key.
prompt = """
You are a senior data analyst writing for a non-technical manager.        # 1 ROLE

Summarise the model evaluation below into a recommendation.               # 2 TASK

Evaluation: A random forest scored 0.84 cross-validated accuracy on 239   # 3 CONTEXT
patients, but recall on the patients who died was only 0.53. The strongest
feature was follow-up time, which is only known after the outcome.

Answer in exactly three bullet points. No jargon. State the risk first.   # 4 FORMAT
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
print(response.text)
```

> **Cover the four labelled parts and rewrite the prompt as just *"summarise this"*.** **You will get a summary. It will not be the one a manager can act on.**

## The one element people skip

> **Format.** **Almost every disappointing prompt is missing it.**
>
> **"Summarise" has no length. "List" has no count. "Explain" has no audience.** **The model picks something reasonable, and reasonable is rarely what you needed.**

---

# 19. The principles

**Six rules. They are all forms of the same idea: say what you mean.**

## 1. Be specific

| ❌ Vague | ✅ Specific |
|---|---|
| *"Write about machine learning."* | *"Write 100 words explaining overfitting to a first-year student, with one analogy."* |

## 2. State the format

| ❌ | ✅ |
|---|---|
| *"Give me the details."* | *"Return JSON with keys `name`, `date` and `amount`. No other text."* |

> **The phrase *"No other text"* is worth its weight.** **Without it you often get "Sure! Here is your JSON:" wrapped around the JSON, and your `json.loads()` fails.**

## 3. Show, do not only tell

> **One example is worth a paragraph of description.** **That is the entire justification for [§21](#21-one-shot-prompting) and [§22](#22-few-shot-prompting).**

## 4. Say what to do, not only what to avoid

| ❌ Negative | ✅ Positive |
|---|---|
| *"Don't be too technical."* | *"Use everyday words. Explain any term you cannot avoid."* |

## 5. Give it room to think

> **For anything involving reasoning, ask for the steps.** **[§23](#23-chain-of-thought-prompting) shows this changing a wrong answer into a right one.**

## 6. Iterate

> **Your first prompt is a draft.** **Run it, read what is wrong, and add the missing element from [§18](#18-the-core-elements-of-a-prompt).** **Two or three rounds is normal and is not a sign you did it badly.**

## ⚠️ And the rule that outranks all six

> **The model can be fluently, confidently wrong.**
>
> **No prompt fixes this.** **Prompting improves the odds; it does not make the output true.** **Anything that matters gets checked by a person or by code.**

---

# 20. Zero-shot prompting

> **Zero-shot: you describe the task and give no examples.**

🧠 **Analogy: asking a competent colleague to do something they have done before.** **No demonstration needed — they know what a summary is.**

**This is what you have been doing since [§13](#13-hi--your-first-genai-program). It is the default, and it works more often than people expect.**

## Example A — the formatting challenge

**You ask for a specific format without showing what it looks like.**

```python
# api-only: needs a Gemini API key.
prompt_a = """
Extract the name, occupation, and city from the following sentence and output it as JSON:

"My name is Sarah, I work as a mechanical engineer, and I just moved to Seattle."
"""

response_a = client.models.generate_content(model=MODEL_ID, contents=prompt_a)
print(response_a.text)
```

**Measured — and read the first line carefully:**

```text
```json
{
  "name": "Sarah",
  "occupation": "mechanical engineer",
  "city": "Seattle"
}
```
```

> **No example of the JSON shape was given, and the model inferred sensible key names.** **Three runs gave identical keys** — `name`, `occupation`, `city`.

## ⚠️ But it is wrapped in a markdown fence, and that breaks your code

```python
# api-only: needs a Gemini API key.
import json

response = client.models.generate_content(model=MODEL_ID, contents=prompt_a)
data = json.loads(response.text)          # <- this raises
```

**Measured:** `json.JSONDecodeError`

**Two fixes, and the second is the one to use.**

```python
# api-only: needs a Gemini API key.
# Fix 1 - ask nicely
prompt = prompt_a + "\nOutput only the raw JSON. No markdown fences, no other text."
response = client.models.generate_content(model=MODEL_ID, contents=prompt)
json.loads(response.text)          # measured: works
```

```python
# api-only: needs a Gemini API key.
# Fix 2 - make it structural
from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt_a,
    config=types.GenerateContentConfig(response_mime_type="application/json"),
)
json.loads(response.text)          # measured: works, and does not depend on the model obeying
```

> **Fix 1 is an instruction the model may ignore. Fix 2 is a constraint on the API.**
>
> **This is [Session 5C](session-05c-deployment.md#7-the-pattern-extracted)'s lesson again: structure beats discipline.**

## Example B — a creative constraint

**Testing whether it can follow strict structural rules first time.**

```python
# api-only: needs a Gemini API key.
prompt_b = """
Write a two-sentence horror story about a smartphone.
The first sentence must be exactly 5 words.
The second sentence must be exactly 3 words.
"""

response_b = client.models.generate_content(model=MODEL_ID, contents=prompt_b)
print(response_b.text)
```

**Measured, three runs — with the word counts checked in code:**

```text
run 1: "My phone tracked two faces. I was alone."        -> [5, 3] ✅
run 2: "My phone took a photo. I was sleeping."          -> [5, 3] ✅
run 3: "The phone filmed me sleeping. I live alone."     -> [5, 3] ✅
```

> **Three for three.** **This model counts words correctly, because it reasons before it answers** — see [§15](#15-what-the-machine-sees--the-raw-json)'s thinking tokens.
>
> ⚠️ **Do not generalise that.** **Smaller and older models are genuinely unreliable at exact counts, because they work in tokens rather than words.** **If an exact count matters, verify it in code rather than trusting either the model or this guide:**

```python
# api-only: needs a Gemini API key.
import re

text = response_b.text.strip()
sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
counts = [len(re.findall(r"[\w'-]+", s)) for s in sentences]
print(counts, "-> expected [5, 3]")
```

## When zero-shot is the right choice

| ✅ Use it when | ❌ Reach for examples when |
|---|---|
| The task is **common and well-understood** | The task uses **your** categories or **your** format |
| You want a **short prompt** (cheaper) | The output shape must be **exact** |
| **You are prototyping** | Zero-shot gave inconsistent results |

> **Always try zero-shot first.** **It is the cheapest prompt you can write, and if it works you are finished.**

---

# 21. One-shot prompting

> **One-shot: you give exactly one worked example before the real input.**

🧠 **Analogy: showing a new colleague one completed form before handing them the next one.** **Far faster than describing the form in words.**

## Example A — tone translation

```python
# api-only: needs a Gemini API key.
prompt_a = """
Convert the corporate jargon into plain English.

Corporate: "Let's synergize our bandwidth to touch base on the deliverables."
Plain English: "Let's work together and meet to discuss the project."

Corporate: "We need to boil the ocean to shift a paradigm in this vertical."
Plain English:
"""

response_a = client.models.generate_content(model=MODEL_ID, contents=prompt_a)
print(response_a.text.strip())
```

**Measured:** `"We need to make a massive effort to completely change how this industry works."`

> **Notice what the one example communicated that a description could not: the *level* of simplicity, the *length*, and the fact that the answer is a quoted sentence.**
>
> **Writing all that out in words would have taken a paragraph — and been less precise.**

## Example B — strict data extraction

```python
# api-only: needs a Gemini API key.
prompt_b = """
Extract the flight details into a pipe-separated format.

Input: "I'm flying on Delta flight 402 from JFK to LAX on Tuesday."
Output: Delta | 402 | JFK | LAX | Tuesday

Input: "Book me on United 88 departing from ORD and arriving at SFO tomorrow."
Output:
"""

response_b = client.models.generate_content(model=MODEL_ID, contents=prompt_b)
print(response_b.text.strip())
```

**Measured, three runs — all identical:** `United | 88 | ORD | SFO | tomorrow`

> **The format is pinned exactly** — field order, separator, spacing, all of it — **by one line of example.**

## ⚠️ But test whether you needed it

**Delete the example and run the same task zero-shot:**

```python
# api-only: needs a Gemini API key.
prompt_zero = """Extract the flight details into a pipe-separated format.

Input: "Book me on United 88 departing from ORD and arriving at SFO tomorrow."
Output:"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt_zero)
print(response.text.strip())
```

**Measured, two runs:** `United | 88 | ORD | SFO | tomorrow` — **identical to the one-shot version.**

> **The example bought nothing here.** **"Pipe-separated" was enough of a description on its own for this model.**
>
> **That is not an argument against one-shot** — the example still *guarantees* the field order, and on a harder or more idiosyncratic format it would earn its place. **It is an argument for measuring instead of assuming.** **Every example you include is tokens you pay for on every single call.**

## The pattern to copy

```text
<instruction>

<Input label>: <example input>
<Output label>: <example output>

<Input label>: <the real input>
<Output label>:
```

> **Ending the prompt on the empty label is the trick.** **The model's job is to continue the text, and the only sensible continuation is the answer** — with no preamble, no "Sure!", nothing to strip.

---

# 22. Few-shot prompting

> **Few-shot: several examples, so the model learns the *boundaries* of your categories, not just the format.**

🧠 **Analogy: training a new sorter by showing them a handful of items already in the right bins.** **One example shows the format. Several show the judgement.**

## Example A — custom routing logic

```python
# api-only: needs a Gemini API key.
prompt_a = """
Classify the customer support ticket into one of three categories: [BILLING], [TECH_ISSUE], or [SALES].

Ticket: "My screen is cracked and the touch sensor won't work."
Category: [TECH_ISSUE]

Ticket: "Do you offer enterprise discounts for teams of 50 or more?"
Category: [SALES]

Ticket: "I was double-charged on my credit card this month."
Category: [BILLING]

Ticket: "How do I upgrade my account from basic to premium?"
Category:
"""

response_a = client.models.generate_content(model=MODEL_ID, contents=prompt_a)
print(response_a.text.strip())
```

**Measured, three runs at the default temperature:** `[SALES]`, `[SALES]`, **`[BILLING]`**

> **It flipped.** **Two runs said `[SALES]`, one said `[BILLING]` — same prompt, same model, same everything.**
>
> **That last ticket is genuinely ambiguous:** upgrading an account touches billing *and* sales. **The three examples narrow it, and they do not settle it.**

## ⚠️ The fix, measured

```python
# api-only: needs a Gemini API key.
from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt_a,
    config=types.GenerateContentConfig(temperature=0.0),
)
```

**Measured, five runs at `temperature=0.0`:** `[SALES]` every time — **5 of 5 identical.**

> **This is the rule from [§16](#16-the-temperature-dial), arriving with consequences.** **Any classification you intend to rely on runs at temperature 0**, or your categories change between requests and nobody can reproduce a bug report.
>
> **And the deeper lesson: instability in the output is often information about the *input*.** **The ticket flipped because the ticket really is ambiguous** — a fourth example covering "upgrades" would settle it properly, where temperature 0 only settles it consistently.

## Example B — mapping real items to your labels

```python
# api-only: needs a Gemini API key.
prompt_b = """
Categorize the grocery items into the correct department: [PRODUCE], [DAIRY], [BAKERY], or [MEAT].

Item: Granny Smith Apples
Department: [PRODUCE]

Item: Whole Milk
Department: [DAIRY]

Item: Sourdough Loaf
Department: [BAKERY]

Item: Ground Beef
Department: [MEAT]

Item: Organic Carrots
Department:
"""

response_b = client.models.generate_content(model=MODEL_ID, contents=prompt_b)
print(response_b.text.strip())
```

**Measured:** `[PRODUCE]`

> **The examples map *real-world names* to *your system's labels*.** *"Sourdough Loaf"* → `[BAKERY]` teaches something no category description would.

## ⚠️ Look at what you have just built

**That is a classifier. With zero training data.**

| | **[Session 5B](session-05b-classification.md)'s classifier** | **This few-shot prompt** |
|---|---|---|
| Training data | **10,000 labelled rows** | **Three examples** |
| Build time | An afternoon | **Two minutes** |
| Cost per prediction | ~0 | **A few tokens** |
| Speed | Microseconds | **Hundreds of milliseconds** |
| Reproducible | **Yes** | **Only at temperature 0** |
| Auditable | **Yes** — you can inspect the tree | **No** |
| New category | **Relabel and retrain** | **Add a line** |

> **Neither wins.** **Use few-shot to get something working today and to discover what the categories should be; use a trained classifier once you have the labels and the volume.**

## How many examples?

| Examples | Effect |
|---|---|
| **1** | Locks the **format** |
| **3–5** | **Locks the format *and* the judgement.** The usual sweet spot |
| 10+ | Diminishing returns, and you are paying for every token on every call |

> ⚠️ **Balance your examples.** **If four of your five examples are `[BILLING]`, the model will lean towards `[BILLING]`.** **[Session 6](session-06-augmentation-feature-engg-red.md#2-when-augmentation-is-used)'s class-imbalance lesson, reappearing in a prompt.**

---

# 23. Chain-of-thought prompting

> **Chain-of-thought: ask the model to show its reasoning *before* the answer.**

🧠 **Analogy: "show your working" in a maths exam.** **A student forced to write the steps catches their own mistake.** **A student who writes only the answer does not.**

**The instruction is usually one clause: *"Think step by step before answering."***

## Example A — the logic puzzle

```python
# api-only: needs a Gemini API key.
prompt_a = """
Solve the following logic puzzle. Before giving the final answer, break down your reasoning step-by-step.

Puzzle: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?
"""

response_a = client.models.generate_content(model=MODEL_ID, contents=prompt_a)
print(response_a.text)
```

**A measured response works through it:**

```text
Step 1: 5 machines make 5 widgets in 5 minutes.
Step 2: So each machine makes 1 widget in 5 minutes.
Step 3: 100 machines each make 1 widget in the same 5 minutes.
Answer: 5 minutes.
```

**The correct answer is 5 minutes. The intuitive-but-wrong answer is 100 minutes.**

## ⚠️ The classic demonstration no longer works — and that is the lesson

**The textbook version of this section says: *without* chain-of-thought, models answer 100 minutes. So I tested it, four ways, three runs each.**

| | Plain prompt | With *"step-by-step"* |
|---|---|---|
| **Thinking on** (default) | **3/3 correct** | **3/3 correct** |
| **Thinking off** (`thinking_budget=0`) | **3/3 correct** | **3/3 correct** |

> **Twelve out of twelve. The model gets it right whatever you do.**
>
> **The "models say 100 minutes" result is real, and it is from 2022-era models.** **A 2026 reasoning model is not fooled by this puzzle any more** — and the [§15](#15-what-the-machine-sees--the-raw-json) thinking tokens are part of why: it reasons before answering whether or not you ask it to.

## So what is chain-of-thought still for?

**Three things, and only the first one was ever about accuracy.**

| Purpose | Still true? |
|---|---|
| **Getting a hard reasoning problem right** | **Sometimes** — on genuinely hard problems, and on smaller or older models |
| **Seeing the reasoning so you can check it** | **Always.** This is now the main reason |
| **Making the model's assumptions visible** | **Always** — the schedule example below shows why |

> **Do not take a textbook's word for what a model can and cannot do — including this one.** **Run the two versions and count.** **That test took four lines of code and overturned the claim I was about to make.**

## Example B — schedule resolution

```python
# api-only: needs a Gemini API key.
prompt_b = """
Let's find a meeting time. Think through the constraints step-by-step before giving the final answer.

Constraints:
* Alice is available from 9:00 AM to 11:30 AM.
* Bob is available from 10:00 AM to 12:00 PM.
* Charlie is available from 10:30 AM to 2:00 PM.
* The meeting needs to last exactly 45 minutes.

What is the earliest possible time they can all meet?
"""

response_b = client.models.generate_content(model=MODEL_ID, contents=prompt_b)
print(response_b.text)
```

**Measured, two runs — both correct:**

```text
...Adding the 45-minute duration to the start time: 10:30 AM + 45 minutes = 11:15 AM.

The earliest possible meeting time is from 10:30 AM to 11:15 AM.
```

> **10:30–11:15 is right** — the overlap of all three windows, taken as early as possible.
>
> **And here the written steps earn their place even though the answer was already correct.** **The response shows you it checked 11:15 against Alice's 11:30 cut-off.** **If it had been wrong, you would be able to see *which* constraint it dropped** — which you cannot do with a bare "10:00 AM".

## Why it works

> **The model generates one token at a time, and each token is conditioned on everything before it.**
>
> **Reasoning written into the output becomes context for the next token.** **The model is, quite literally, giving itself more to think with.**
>
> **This is also why it is slower and costs more: you are paying for every reasoning token.** **Use it where reasoning matters, not everywhere.**

## When to use it

| ✅ Use CoT for | ❌ Skip it for |
|---|---|
| **Anything you need to be able to audit** | Simple lookups |
| **Multi-step constraints** | Classification |
| **Debugging and diagnosis** | Formatting and extraction |
| Genuinely hard reasoning, or a **smaller model** | Anywhere latency or cost matters |

⚠️ **A written chain of reasoning is not a guarantee of correctness.** **A model can produce a fluent, plausible, entirely wrong chain — and it is more persuasive precisely because it showed working.** **Read the steps; do not just trust their presence.**

---

# 24. Choosing a prompt type

| Type | Examples given | Use it when | Cost |
|---|---|---|---|
| **Zero-shot** | **0** | The task is common and well-understood | **Cheapest** |
| **One-shot** | **1** | **The format must be exact** | Low |
| **Few-shot** | **3–5** | **Your own categories, or subtle judgement** | Medium |
| **Chain-of-thought** | 0 or more | **Reasoning, arithmetic, multi-step constraints** | **Highest** |

## The decision, as a flow

```text
Start with ZERO-SHOT.
│
├── Output is right?                   -> done. Do not add complexity you do not need.
│
├── Right content, wrong FORMAT?       -> ONE-SHOT
│
├── Wrong CATEGORY or wrong judgement? -> FEW-SHOT (3-5 balanced examples)
│
└── Wrong ANSWER on a reasoning task?  -> CHAIN-OF-THOUGHT
```

> **They combine.** **Few-shot chain-of-thought — several examples, each with its reasoning written out — is the strongest and most expensive option.**

## ⚠️ Measure at every step of that flow

**Running the experiments for this session overturned three things I expected:**

| Expected | **Measured** |
|---|---|
| One-shot needed to pin the pipe format | **Zero-shot produced the identical format** |
| Few-shot settles the ambiguous ticket | **It flipped between `[SALES]` and `[BILLING]` across runs** |
| Chain-of-thought rescues the widget puzzle | **The model got it right 12/12 without any prompting help** |

> **Every extra example and every "think step by step" is tokens on every call, forever.**
>
> **Add complexity when a measurement tells you to, not when a guide does.**

## What to do when none of them work

| Symptom | Try |
|---|---|
| Invents facts | **Supply the source text in the prompt** and say *"answer only from the text above"* |
| Inconsistent between runs | **Temperature 0** |
| Ignores part of the instruction | **Split it into two calls** |
| Too long / too short | **State the length as a number** |
| Wraps JSON in chatter | Add *"Output only the JSON. No other text."* |
| **Still wrong** | **The task may not suit an LLM.** [§3](#3-predictive-ai-vs-generative-ai) |

---

# ✏️ Session 10 — Practice

**Parts A and B are pen-and-paper. Parts C and D need an API key.**

## Concepts

1. **In your own words, what makes a model "generative"?** Give one example of a task that is generative and one that only looks generative.
2. **List ten GenAI applications you have personally seen or used.** For each, say whether a human checks the output.
3. Take three problems from Sessions 5–8. **For each, say whether Predictive or Generative AI is the right tool, and why.**
4. **Draw the five-stage GenAI workflow from memory.** Mark which stages you will ever perform.
5. **Explain to a friend why the Transformer replaced the LSTM.** Use the "reads the whole page at once" analogy, and mention parallelisation.
6. **What is the difference between a language model and a large language model?** Answer in three sentences.
7. A 13-billion-parameter model at FP16. **How much memory for the weights?** At INT4?
8. **When would you choose an open-weights model over an API model?** Give two concrete situations.

## The API

9. Install `google-genai`, get a key, store it **as an environment variable or Colab secret**, and run the `"Hi"` program.
10. **Ask the same question with "in one short sentence" and then "in exactly 5 words".** Compare.
11. Print `response.model_dump_json(indent=2)`. **Find `total_token_count` and `finish_reason`.**
12. Set `max_output_tokens=20` and ask for a long explanation. **What is `finish_reason` now?**
13. **Run the temperature demo.** Report what you saw at 0.0 and at 1.0, then try 2.0.
14. Write a loop over five prompts with `time.sleep(1)`. **Remove the sleep and report what happens.**

## Prompting

15. **Write one prompt containing all five core elements**, and label each one in a comment.
16. Take a vague prompt — *"summarise this"* — and improve it three times. **Show all four versions and what changed.**
17. **Zero-shot:** ask for JSON with specific key names. Run it three times. **Did the keys stay the same?**
18. **One-shot:** pin an output format with a single example. **Then delete the example and compare.**
19. **Few-shot:** build a three-category classifier for something you care about. **Include one deliberately ambiguous test input.**
20. **Chain-of-thought:** run the widget puzzle with and without *"think step by step"*. **Report both answers.**

<details><summary>Answers to the numerical and factual ones</summary>

**3.** Loan approval → **Predictive** (must be auditable and consistent). Explaining a loan decision to the customer → **Generative**. Car price prediction → **Predictive** (one number, and you have labels). Writing the listing text for that car → **Generative**.

**6.** A language model assigns a probability to the next token. A **large** language model is the same idea with billions of parameters trained on a very large corpus. **The practical difference: an LM does the one task it was trained for; an LLM does any task you can describe in words, without retraining.**

**7.** **FP16:** 13 × 10⁹ × 2 bytes = **26 GB**. **INT4:** 13 × 10⁹ × 0.5 bytes = **6.5 GB**. **The first does not fit on a 24 GB card; the second fits comfortably.**

**8.** **(a) Regulated or private data** — a hospital cannot send patient notes to a third-party API. **(b) Cost at volume** — millions of calls a day may be cheaper on your own hardware. Also: **no dependency on a provider that can change, price or retire the model.**

**12.** `finish_reason` becomes **`MAX_TOKENS`**. ⚠️ **`response.text` still returns something that looks like an answer** — that is exactly why you check the field.

**14.** Without the sleep you will hit **`429 RESOURCE_EXHAUSTED`** — the free tier's requests-per-minute limit.

**17.** **Not necessarily.** Zero-shot leaves key naming to the model, so `"occupation"` may become `"job"` or `"profession"`. **Name the keys in the prompt, or give one example.**

**20.** **Without CoT, models frequently answer 100 minutes.** **With CoT the reasoning makes the per-machine rate explicit and the answer becomes 5 minutes.** The correct answer is **5 minutes**.
</details>

---

# ❓ Session 10 — 20 MCQs

**Answer from memory first, then check.**

### What GenAI is

**Q1.** The defining feature of Generative AI is that…
- (a) It is more accurate  (b) **It produces new content rather than choosing from a fixed set of answers**  (c) It uses deep learning  (d) It needs no data

**Q2.** A generative model produces a sentence by…
- (a) Retrieving the closest stored sentence  (b) **Predicting the next token, repeatedly, each one conditioned on everything before it**  (c) Searching the internet  (d) Applying grammar rules

**Q3.** Which is NOT a good use of GenAI?
- (a) Drafting a support reply  (b) **Calculating a tax bill**  (c) Summarising a contract  (d) Explaining code

**Q4.** The same prompt sent twice usually gives different answers because…
- (a) The model is broken  (b) **The model samples from a probability distribution over the next token**  (c) The internet is unreliable  (d) The prompt changed

**Q5.** "Hallucination" means…
- (a) The model crashes  (b) **It produces a fluent, confident, invented answer**  (c) It refuses to answer  (d) It repeats itself

**Q6.** The stage of the GenAI workflow you will actually perform is…
- (a) Pre-training  (b) Fine-tuning  (c) Alignment  (d) **Prompting**

### The fundamentals

**Q7.** The Transformer's key advantage over the LSTM is…
- (a) Fewer parameters  (b) **Self-attention lets any word attend directly to any other, and the whole sequence processes in parallel**  (c) It needs less data  (d) It is older

**Q8.** The consequence of parallelisation that mattered most was…
- (a) Faster inference  (b) **Models could be trained at a scale that was previously impossible**  (c) Smaller files  (d) Better grammar

**Q9.** BERT is encoder-only, which makes it suited to…
- (a) Generating stories  (b) **Understanding text — classification and search**  (c) Translation  (d) Image generation

**Q10.** A language model is, at its core…
- (a) A grammar checker  (b) **Something that assigns a probability to what comes next**  (c) A search engine  (d) A translator

**Q11.** The practical difference between an LM and an LLM is…
- (a) Only size  (b) **An LM does the one task it was trained for; an LLM does any task you can describe in words, with no retraining**  (c) LLMs are open source  (d) LMs are faster

**Q12.** A 7-billion-parameter model at 16-bit precision needs roughly…
- (a) 7 GB  (b) **14 GB**  (c) 28 GB  (d) 3.5 GB

**Q13.** Quantization…
- (a) Removes parameters  (b) **Stores each parameter in fewer bits, trading a little accuracy for much less memory**  (c) Retrains the model  (d) Compresses the prompt

**Q14.** A 70-billion model at 4-bit needs about…
- (a) 140 GB  (b) **35 GB**  (c) 280 GB  (d) 7 GB

**Q15.** The most important practical difference between GPT and LLaMA is…
- (a) Accuracy  (b) **LLaMA has open weights you can download and run; GPT is used through an API**  (c) Age  (d) Language support

### The API and prompting

**Q16.** An API key should be…
- (a) Pasted into the notebook  (b) **Stored in an environment variable or Colab secret, never in code**  (c) Emailed to yourself  (d) Committed with the repo

**Q17.** `finish_reason: MAX_TOKENS` means…
- (a) The prompt was too long  (b) **The response was cut off mid-answer — and `response.text` still looks like a complete answer**  (c) You ran out of credit  (d) The model refused

**Q18.** For extracting structured data you should set temperature to…
- (a) 1.0  (b) **0.0**  (c) 2.0  (d) It makes no difference

**Q19.** You need the model to use *your* categories with *your* exact labels, and zero-shot keeps drifting. Use…
- (a) A longer description  (b) **Few-shot — three to five balanced examples**  (c) Higher temperature  (d) A bigger model

**Q20.** Chain-of-thought works because…
- (a) The model thinks harder  (b) **Reasoning written into the output becomes context for the next token — the model gives itself more to work with**  (c) It uses a different model  (d) It searches the web

<details><summary>Answers</summary>

**A1 — (b) It produces new content.** **The answer space is unbounded.** Everything in Sessions 5–8 chose from a set that existed before the model ran.

**A2 — (b) Predicting the next token, repeatedly.** **That single mechanism is why it can write code, translate and summarise** — doing those well *is* predicting the next token well.

**A3 — (b) Calculating a tax bill.** **There is exactly one right answer, and arithmetic gives it for free.**

**A4 — (b) It samples.** **Temperature is the dial that controls how much.** At 0.0 it takes the most likely token every time.

**A5 — (b) A fluent, confident, invented answer.** **This is the single biggest practical risk**, and no prompt eliminates it.

**A6 — (d) Prompting.** **Stages 1–3 cost millions and are done by the model provider.**

**A7 — (b) Self-attention and parallelism.** **An RNN reads through a keyhole and forgets chapter 1 by chapter 12.**

**A8 — (b) Trainable at a new scale.** **The Transformer is not merely more accurate than an LSTM — it is trainable on thousands of GPUs, which an LSTM never could be.**

**A9 — (b) Understanding text.** **BERT reads; GPT writes.** BERT is still everywhere in search and classification.

**A10 — (b) A probability over what comes next.** **Your phone keyboard is a tiny one.**

**A11 — (b) Task-specific training versus describing the task.** **That shift is what made prompt engineering a skill.**

**A12 — (b) 14 GB.** 7 × 10⁹ × 2 bytes. **A 24 GB card fits it; a 70-billion model does not.**

**A13 — (b) Fewer bits per parameter.** **JPEG for model weights.** The usual sweet spot is 8-bit or 4-bit.

**A14 — (b) 35 GB.** 70 × 10⁹ × 0.5 bytes. **The full-precision version needs 280 GB.**

**A15 — (b) Open weights.** **A hospital cannot send patient notes to an API.** That is a constraint, not a preference.

**A16 — (b) An environment variable or secret.** **A key pasted into a notebook is in your git history forever, and deleting the line does not remove it.**

**A17 — (b) Cut off mid-answer.** **Check the field in any real application** — the truncated text still reads like an answer.

**A18 — (b) 0.0.** **You want the same answer every time.** ⚠️ It reduces variation, not error — a model can be deterministic and confidently wrong.

**A19 — (b) Few-shot.** **Examples carry judgement that a description cannot.** Keep them balanced, or the model leans towards the over-represented category.

**A20 — (b) The reasoning becomes context.** **It is also why CoT is slower and more expensive — you pay for every reasoning token.**
</details>

---

# 🎯 Session 10 — Tasks

## Concepts

**Task 1 — The application audit.** Find **fifteen** GenAI applications in products you use. **For each: what does it generate, and does a human check it before it reaches anyone?**

**Task 2 — Predictive or generative.** Take ten problems from your own domain. **Classify each, justify each, and find one that genuinely needs both.**

**Task 3 — Explain the workflow.** Write a one-page explanation of the five-stage workflow for a non-technical manager. **Make clear which stages cost millions and which cost nothing.**

**Task 4 — The Transformer, explained.** Explain self-attention to someone who knows Session 9 but not this session. **Use your own analogy**, and say why parallelisation mattered more than accuracy.

**Task 5 — The memory table.** Build a table of five models with their parameter counts, and compute the memory needed at FP32, FP16, INT8 and INT4. **Mark which fit on a 24 GB card.**

**Task 6 — Open or closed.** Write the case for an open-weights model and the case for an API model, for one specific organisation of your choosing. **Recommend one.**

## The API

**Task 7 — Set it up properly.** Get a key, store it as an environment variable, add `.env` to `.gitignore`, and run the `"Hi"` program. **Prove the key is not in your repository.**

**Task 8 — The raw response.** Print the full JSON for three different prompts. **Report `total_token_count` for each and explain what drove the difference.**

**Task 9 — Break it on purpose.** Trigger all three errors from §13: a bad key, a bad model name, and a rate limit. **Record the exact message for each.**

**Task 10 — Cost arithmetic.** Look up the current price per million tokens for one model. **Estimate the monthly cost of a feature making 10,000 calls a day at 500 tokens each.**

**Task 11 — The temperature study.** Run one prompt at temperatures 0.0, 0.5, 1.0 and 2.0, five times each. **Present all twenty outputs and describe the pattern.**

**Task 12 — Reproducibility.** Try to make the model give an identical answer twice. **Report what worked, what did not, and what that means for testing a GenAI feature.**

## Prompting

**Task 13 — The five elements.** Write five prompts, each missing a different one of the five core elements. **Run all five and report what each omission cost.**

**Task 14 — Prompt iteration, documented.** Start from a deliberately vague prompt and improve it over four rounds. **Show every version and every output**, and name the principle you applied each time.

**Task 15 — All four types.** Solve the *same* task with zero-shot, one-shot, few-shot and chain-of-thought. **Compare output quality, token count and latency in one table.**

**Task 16 — Build a classifier with three examples.** Few-shot a classifier for a task you care about. **Test it on twenty inputs and measure its accuracy by hand.**

**Task 17 — Against Session 5B.** Take the loan data. Build a few-shot LLM classifier and compare it with your trained model on the same 100 rows. **Report accuracy, cost and speed for both, and recommend one.**

**Task 18 — Chain-of-thought, measured.** Find five reasoning problems. **Run each with and without CoT and record how many each version gets right.**

**Task 19 — Make it hallucinate.** Get the model to state something confidently false. **Record the prompt and the output**, then write the guardrail you would add in an application.

**Task 20 — The unreliable counter.** Ask for outputs of an exact word count, ten times. **Count them yourself and report the hit rate.** Then write the code that would verify it automatically.

---

## ✅ Session 10 checklist

- [ ] I can explain what makes a model **generative**, and name ten applications
- [ ] I can choose between **Predictive and Generative AI** for a given problem
- [ ] I can describe the **five-stage workflow** and say which stage is mine
- [ ] I can explain **why the Transformer replaced the LSTM** — and that parallelism mattered most
- [ ] I know the difference between **encoder-only, decoder-only and encoder–decoder**
- [ ] I can define a **language model** in one sentence
- [ ] I can say what **"large" added** — and that it moved the task from training into the prompt
- [ ] I can compute a model's memory from its **parameter count**, at any precision
- [ ] I can explain **quantization** and the trade it makes
- [ ] **My API key is never in my code**
- [ ] I check **`finish_reason`**, and never assume `response.text` is a string
- [ ] I read **`thoughts_token_count`** — hidden reasoning can be 94% of the bill
- [ ] I know **`max_output_tokens` budgets thinking and answer together**
- [ ] I set **temperature 0 for anything structured**
- [ ] I include all five **core elements** in a prompt that matters
- [ ] I can write **zero-shot, one-shot, few-shot and chain-of-thought** prompts, and choose between them
- [ ] **I never trust a fluent answer just because it is fluent**

---

| | |
|---|---|
| **Previous** | [Session 9 — Deep Learning](session-09-deep-learning.md) |
| **Next** | [Session 11 — AI-Powered Applications](session-11-ai-apps.md) |
| **Notebook** | [session-10-genai-llms.ipynb](../notebooks/session-10-genai-llms.ipynb) |
| **More practice** | [Exercises & assignments](../exercises-assignments.md) |
