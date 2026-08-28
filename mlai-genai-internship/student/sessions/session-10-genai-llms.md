# Session 10 — Generative AI & Large Language Models

**Introduction to Generative AI · Language Models, LLM parameters, Leading LLMs, LLM Applications · Prompt Engineering Basics · Types of Prompts: Zero-shot, One-shot, Few-shot, Chain-of-Thought**

| | |
|---|---|
| **Notebook** | [session-10-genai-llms.ipynb](../notebooks/session-10-genai-llms.ipynb) |
| **Prompt library** | [prompts.md](../prompts.md) — every prompt here, ready to copy |
| **Previous** | [Session 9 — Deep Learning](session-09-deep-learning.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Session 9 ended with next-token prediction. This session is what that becomes at scale.** You will need a free [Google AI Studio](https://aistudio.google.com/) API key — see [setup-guide.md](../setup-guide.md).

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Say what makes an AI system *generative*
2. Explain tokens, parameters, context window and temperature — and what each costs you
3. Predict what temperature does to output, and demonstrate it with numbers
4. Place GPT, BERT, T5, LLaMA, Falcon, Mistral, Gemini and Claude in a sensible map
5. Explain why BERT cannot chat with you
6. Write a prompt with all five parts
7. Choose between zero-shot, one-shot, few-shot and chain-of-thought — with a reason
8. Call the Gemini API and get structured JSON back

---

## The five topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [What GenAI is](#1-what-generative-ai-is) | It creates; it does not choose from a list |
| 2 | [LLM anatomy](#2-large-language-models-tokens-parameters-context) | Tokens are the unit of everything, including cost |
| 3 | [Leading LLMs](#3-leading-llms-and-what-they-are-for) | Open vs closed matters more than "which is best" |
| 4 | [Prompt engineering](#4-prompt-engineering-basics) | Five parts: role, task, context, constraints, format |
| 5 | [Prompt types](#5-types-of-prompts) | Few-shot fixes format; CoT fixes reasoning |

---

# 1. What Generative AI is

Every model in Sessions 5–8 **chose from options that already existed**: approved or rejected, a number, a cluster number. **Generative models produce something that did not exist before.**

| | Predictive AI | Generative AI |
|---|---|---|
| Output | A label or a number | New text, images, audio, code |
| Question | *Which one?* | *Make me one* |
| Answer space | Fixed and known | Effectively unlimited |
| Your Session 5 model | ✅ | ❌ |

🧠 **Analogy: a music critic and a musician.** The critic listens and classifies — *this is jazz, this is three stars*. Useful, and bounded. The musician **writes a piece that has never been played.** Same domain, entirely different job.

## The families

| Type | Produces | Examples |
|---|---|---|
| **Text** | Prose, code, analysis | Gemini, GPT, Claude, Llama |
| **Image** | Pictures from descriptions | Stable Diffusion, Imagen, DALL·E |
| **Audio** | Speech, music | ElevenLabs, MusicGen |
| **Video** | Clips from descriptions | Veo, Sora |
| **Multimodal** | Handles several at once | Gemini, GPT-4o, Claude |

> **This course focuses on text**, because text models are where the immediate practical value is for the applications you will build in Sessions 11 and 12.

## 📘 Examples

**Example 1 — the same problem, both ways**

```python
# PREDICTIVE (Session 5): choose from a fixed set
model.predict(loan_application)          # -> 0 or 1

# GENERATIVE (this session): produce something new
llm("Explain in two kind sentences why this loan was declined.")
# -> a paragraph nobody wrote in advance
```

**Example 2 — your first API call**

```python
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Explain overfitting to a first-year student in three sentences.",
)
print(response.text)
```

**Example 3 — where each belongs**

| Task | Which |
|---|---|
| Will this customer churn? | Predictive |
| Write the email asking them to stay | Generative |
| Is this transaction fraud? | Predictive |
| Explain to the customer why their card was blocked | Generative |

> **The best applications use both.** Session 11 is entirely about joining them: a model that decides, and a model that explains.

## ✏️ Practice

1. List five generative tools you have used, and what each produces.
2. Classify five tasks from your own life as predictive or generative.
3. Get a Gemini API key and make your first call.
4. Ask the same question three times. Are the answers identical? Why not?
5. Name a task where a generative model would be the **wrong** choice.

<details><summary>Solutions</summary>

```python
# api-only: needs a Gemini API key; run this yourself in Colab or locally
# 1 - ChatGPT/Gemini (text), GitHub Copilot (code), DALL-E (images),
#     ElevenLabs (speech), Google Translate (text).

# 2 - Predictive: will it rain, is this spam, is this face mine.
#     Generative: write my email, summarise this chapter, make an image.

import os
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])         # 3
r = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Explain overfitting to a first-year student in three sentences.")
print(r.text)

for i in range(3):                                                   # 4
    print(client.models.generate_content(
        model="gemini-3.5-flash", contents="Name one colour.").text)
# Not identical. The model SAMPLES from a probability distribution over
# the next token, so different runs take different paths. Set temperature
# to 0 to make it (nearly) deterministic.

# 5 - Anything requiring a guaranteed-correct, auditable answer:
#     computing a tax bill, checking a password, calculating interest.
#     Use a rule or a calculation. Session 4's lesson, still true.
```
</details>

## ❓ MCQs

**Q1.** What makes an AI system *generative*?
- (a) It uses a neural network  (b) It produces new content rather than choosing from fixed options  (c) It is large  (d) It runs on a GPU

**Q2.** Your Session 5 loan classifier is…
- (a) Generative  (b) Predictive  (c) Both  (d) Neither

**Q3.** Asking a model the same question twice gives different answers because…
- (a) A bug  (b) It samples from a probability distribution over the next token  (c) The internet changed  (d) Different models were used

**Q4.** Which task should **not** use a generative model?
- (a) Writing a summary  (b) Computing a tax bill from published bands  (c) Drafting an email  (d) Explaining a decision

**Q5.** A multimodal model is one that…
- (a) Has many parameters  (b) Handles several input or output types  (c) Runs on many GPUs  (d) Uses several languages

<details><summary>Answers</summary>

**A1 — (b).** The answer space is effectively unlimited rather than a fixed list.

**A2 — (b) Predictive.** It picks one of two existing labels.

**A3 — (b).** Set temperature to 0 for near-determinism.

**A4 — (b).** **Session 4's lesson again: if you can write the rule correctly, write the rule.** Tax must be exact and auditable.

**A5 — (b).** Text, images, audio in one model.
</details>

## 🎯 Tasks

**Task 1 — The audit.** Find five products you use that added a GenAI feature in the last two years. For each, write what it generates and **whether a predictive model would have done the job better.** At least one of them, the answer will be yes.

**Task 2 — Both halves.** Take a problem from your own life and design a two-part system: a predictive model that decides, and a generative model that explains the decision to a person. **Sketch the inputs and outputs of each.** You will build this in Session 11.

---

# 2. Large Language Models: tokens, parameters, context

An LLM is a very large neural network — **exactly the kind you built in Session 9** — trained to predict the next token.

## Tokens

**Text is not processed as words. It is processed as tokens**, roughly ¾ of a word each.

```python
"Machine learning models predict the next token, repeatedly."
# -> ['Machine', ' learning', ' models', ' predict', ' the',
#     ' next', ' token', ',', ' repeatedly', '.']
# 10 tokens for 8 words
```

**Notice the leading spaces.** ` learning` and `learning` are *different tokens*. This is why models occasionally behave oddly with spacing and punctuation.

> **Tokens are the unit of everything: the context limit, the cost, and the speed.** A rough rule: **1 token ≈ 4 characters ≈ ¾ of a word.** 1,000 words ≈ 1,300 tokens.

## Parameters

The learned weights — the same `W1`, `b1`, `W2`, `b2` from Session 9, at enormous scale.

| Scale | Parameters | Runs on |
|---|---|---|
| Your Session 9 network | 33 | Anything |
| Small open model | ~7 billion | A good laptop |
| Large open model | ~70 billion | A server with several GPUs |
| Frontier models | Hundreds of billions | A data centre |

> ⚠️ **More parameters is not automatically better.** A well-trained 7B model routinely beats a poorly-trained 70B one, and for many tasks a small fast model is the right engineering choice.

## Context window

**How much the model can "see" at once** — your prompt *and* its answer, together.

| Model era | Context | Roughly |
|---|---|---|
| GPT-2 (2019) | 1,024 tokens | 2 pages |
| GPT-3.5 | 4,096 tokens | 6 pages |
| Llama 3 | 8,192 tokens | 12 pages |
| Gemini 1.5 Pro | 1,000,000 tokens | ~1,500 pages |

**Anything beyond the window is simply not there.** This is why long chats start "forgetting" the beginning — a fixed window slides forward.

## Temperature

**How sharply the model samples from its next-token probabilities.** You met the idea at the end of Session 9; here is what it actually does.

Given these six candidate next tokens after *"Tomorrow will be"*:

| Token | T=0.0 | T=0.3 | T=0.7 | T=1.0 | T=1.5 | T=2.5 |
|---|---|---|---|---|---|---|
| sunny | **1.000** | 0.966 | 0.711 | 0.565 | 0.427 | 0.314 |
| cloudy | 0.000 | 0.025 | 0.148 | 0.188 | 0.205 | 0.202 |
| raining | 0.000 | 0.009 | 0.096 | 0.139 | 0.168 | 0.179 |
| cold | 0.000 | 0.000 | 0.027 | 0.057 | 0.092 | 0.125 |
| windy | 0.000 | 0.000 | 0.013 | 0.034 | 0.066 | 0.102 |
| **purple** | 0.000 | 0.000 | 0.005 | 0.017 | 0.041 | **0.077** |

**At temperature 2.5, "Tomorrow will be purple" has a 7.7% chance.** That is what "creative" settings actually mean, and why they produce incoherence.

| Temperature | Use for |
|---|---|
| **0.0 – 0.3** | Extraction, classification, JSON, anything factual |
| **0.7** | General writing. The usual default |
| **1.0+** | Brainstorming, where you want variety |

## 📘 Examples

**Example 1 — count your tokens**

```python
text = "Machine learning models predict the next token, repeatedly."
print(len(text), "characters")
print(len(text) / 4, "tokens (rough estimate)")
```

**Example 2 — temperature in the API**

```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Name one colour.",
    config=types.GenerateContentConfig(temperature=0.0),   # deterministic
)
```

**Example 3 — what a context window costs you**

```python
# A 1,000,000-token context sounds free. It is not:
#   - you pay per input token
#   - the model is slower on long inputs
#   - accuracy on facts buried in the MIDDLE of a long context
#     is measurably worse than at the start or the end
#
# Send what is relevant, not everything you have.
```

## ✏️ Practice

1. Estimate the tokens in a 500-word essay.
2. Build the temperature table above with softmax in NumPy.
3. Sample 12 tokens at T = 0.0, 0.7 and 1.5. Describe the difference.
4. Ask the model the same question at temperature 0 and 1.5, three times each.
5. Why does a long chat start "forgetting" the beginning?

<details><summary>Solutions</summary>

```python
import numpy as np

print("500 words ~", int(500 / 0.75), "tokens")                        # 1

logits = np.array([3.2, 2.1, 1.8, 0.9, 0.4, -0.3])                     # 2
words = ["sunny", "cloudy", "raining", "cold", "windy", "purple"]

def softmax_T(z, T):
    if T <= 0:
        p = np.zeros_like(z); p[np.argmax(z)] = 1.0; return p
    e = np.exp((z - z.max()) / T)
    return e / e.sum()

for T in [0.0, 0.3, 0.7, 1.0, 1.5, 2.5]:
    print(f"T={T}: " + "  ".join(f"{w} {p:.3f}" for w, p in zip(words, softmax_T(logits, T))))
# At T=2.5, "purple" gets 7.7% -- that is what "creative" settings mean.

rng = np.random.default_rng(3)                                         # 3
for T in [0.0, 0.7, 1.5]:
    picks = [words[rng.choice(len(words), p=softmax_T(logits, T))] for _ in range(12)]
    print(f"T={T}: {picks}")
# T=0.0 is identical every time. T=0.7 mostly picks the top token with
# occasional variety. T=1.5 wanders, including into nonsense.

# 5 - The context window is FIXED. Once the conversation exceeds it, the
#     oldest tokens fall out of the window and are simply not there any
#     more. The model is not "forgetting" -- it can no longer see them.
```
</details>

## ❓ MCQs

**Q1.** Roughly how many tokens is 1,000 words?
- (a) 250  (b) 750  (c) 1,300  (d) 4,000

**Q2.** ` learning` (with a leading space) and `learning` are…
- (a) The same token  (b) Different tokens  (c) Not tokens  (d) Always merged

**Q3.** The context window contains…
- (a) Only your prompt  (b) Your prompt and the model's answer together  (c) Only the answer  (d) The training data

**Q4.** For extracting JSON from a document, use temperature…
- (a) 0.0 – 0.3  (b) 0.7  (c) 1.5  (d) 2.5

**Q5.** At temperature 2.5 the token "purple" has 7.7% probability after "Tomorrow will be". This shows…
- (a) The model is broken  (b) High temperature flattens the distribution, making nonsense possible  (c) Purple is a weather type  (d) Temperature has no effect

**Q6.** A 7B model beating a 70B model is…
- (a) Impossible  (b) Possible — training quality matters more than raw size  (c) A measurement error  (d) Only true for code

**Q7.** Long chats "forget" the beginning because…
- (a) Memory leaks  (b) Old tokens fall outside the fixed context window  (c) The model gets tired  (d) The API resets

<details><summary>Answers</summary>

**A1 — (c) ~1,300.** 1 token ≈ ¾ of a word.

**A2 — (b) Different tokens.** Which is why spacing sometimes changes behaviour.

**A3 — (b) Both together.** A long prompt leaves less room for the answer.

**A4 — (a).** Anything factual or structured wants low temperature.

**A5 — (b).** **That is what "creative" settings actually mean.**

**A6 — (b).** **More parameters is not automatically better.**

**A7 — (b).** They are not there to be recalled.
</details>

## 🎯 Tasks

**Task 1 — The temperature study.** Ask one creative question and one factual question at temperatures 0, 0.7 and 1.5, five times each. **Present a table of what changed** and write the one-line rule you would give a teammate.

**Task 2 — The token budget.** Take a real document and estimate its tokens. **Work out what it would cost to send it 1,000 times** at a published API rate, and what you could strip out to halve that. This is a real engineering decision in Session 11.

---

# 3. Leading LLMs, and what they are for

The question is rarely *"which is best?"* — it changes monthly. **The useful questions are: is it open or closed, and what is it built to do?**

| Model | Made by | Open weights? | Built for |
|---|---|---|---|
| **GPT** | OpenAI | No | General-purpose generation and chat |
| **Gemini** | Google | No | General-purpose, multimodal, long context |
| **Claude** | Anthropic | No | General-purpose, long-form reasoning and code |
| **LLaMA** | Meta | **Yes** | Open general-purpose; the base for many fine-tunes |
| **Mistral** | Mistral AI | **Yes** (several) | Efficient open models, strong for their size |
| **Falcon** | TII | **Yes** | Open general-purpose |
| **BERT** | Google | **Yes** | **Understanding text, not generating it** |
| **T5** | Google | **Yes** | Text-to-text: translation, summarisation |

## The distinction that actually matters

> ⚠️ **BERT cannot chat with you, and this is the most common misunderstanding in this topic.**

| Architecture | Reads | Good for | Examples |
|---|---|---|---|
| **Encoder-only** | The whole text at once, both directions | *Understanding*: classification, search, embeddings | BERT |
| **Decoder-only** | Left to right, predicting the next token | *Generating*: chat, writing, code | GPT, Gemini, Claude, LLaMA, Mistral |
| **Encoder-decoder** | Reads all, then writes | *Transforming*: translation, summarisation | T5 |

**BERT is an encoder.** It builds a rich understanding of text you give it — which is exactly what you want for classifying support tickets or powering search. **It was never designed to continue a sentence**, so asking it to write you an email is a category error.

🧠 **Analogy.** BERT is a **reader** — give it a document and it understands it deeply. GPT is a **writer** — give it a start and it continues. T5 is a **translator** — give it something in one form and it produces another.

## Open vs closed: the decision you will actually make

| | Closed (GPT, Gemini, Claude) | Open (LLaMA, Mistral, Falcon) |
|---|---|---|
| Access | API call | Download and run |
| Cost | Per token | Your hardware |
| **Your data** | Leaves your machine | **Stays on your machine** |
| Fine-tuning | Limited | Full control |
| Setup | Minutes | Hours, plus a GPU |
| Best for | Getting started, most apps | Privacy, scale, customisation |

> **For a hospital or a bank, "the data stays on our machine" often decides it outright** — before anyone compares quality. Session 12 covers running open models via Hugging Face.

## Applications

| Application | What the LLM does |
|---|---|
| **Chatbots and assistants** | Answer questions in context |
| **Summarisation** | Long document → key points |
| **Extraction** | Unstructured text → structured JSON |
| **Code** | Generate, explain, review, translate |
| **Classification** | Sentiment, topic, intent — **often with no training data** |
| **RAG** | Answer from *your* documents rather than memory |
| **Translation** | Between languages, or between registers |

**Zero-shot classification is the one that surprises people.** In Session 5 you needed thousands of labelled rows to build a classifier. An LLM will classify sentiment with **zero** training examples — worse than a properly trained model on a narrow task, but instantly, and on a task nobody has data for.

## 📘 Examples

**Example 1 — choosing a model, with reasons**

```python
# A hospital classifying patient notes, data must not leave the building
#   -> an OPEN model (LLaMA / Mistral) running locally
#
# A student project needing a working demo this week
#   -> a CLOSED API (Gemini) - free tier, no GPU, working in minutes
#
# Semantic search over 100,000 documents
#   -> BERT-family EMBEDDINGS, not a chat model
#
# Summarising 300-page reports
#   -> a long-context model (Gemini 1.5 Pro)
```

**Example 2 — zero-shot classification, no training data at all**

```python
reviews = ["The delivery was late and the box was damaged.",
           "Works exactly as described, very happy.",
           "It is fine. Nothing special."]

for r in reviews:
    out = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Classify sentiment as positive, negative or neutral. "
                 f"Reply with one word only.\n\nReview: {r}",
    )
    print(f"{out.text.strip():<10} {r}")
```

**Example 3 — the honest comparison with Session 5**

| | Trained classifier (Session 5) | LLM zero-shot |
|---|---|---|
| Training data needed | Thousands of labelled rows | **None** |
| Setup time | Days | Minutes |
| Cost per prediction | Effectively zero | An API call |
| Accuracy on a narrow, well-defined task | **Usually higher** | Good, not best |
| Handles a task you have no data for | ❌ | ✅ |

> **Neither replaces the other.** If you have 10,000 labelled loan applications, train a Random Forest — it will be faster, cheaper and better. If you need to sort support emails by topic tomorrow and have no labels, use an LLM.

## ✏️ Practice

1. Why can BERT not write you an email?
2. Which architecture for semantic search? For chat? For translation?
3. Classify five reviews zero-shot. How accurate is it against your own judgement?
4. For a bank that cannot let data leave its building, open or closed? Why?
5. When would you still prefer a Session 5 classifier over an LLM?

<details><summary>Solutions</summary>

```python
# api-only: needs a Gemini API key; run this yourself in Colab or locally
# 1 - BERT is an ENCODER. It reads text in both directions to understand
#     it; it was never trained to continue a sequence one token at a time.
#     Asking it to write is a category error, not a limitation to work around.

# 2 - Semantic search -> encoder (BERT-family embeddings)
#     Chat            -> decoder-only (GPT, Gemini, Claude, LLaMA)
#     Translation     -> encoder-decoder (T5) or a modern decoder-only model

import os
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])          # 3
for r in ["Late and damaged.", "Exactly as described, very happy.",
          "It is fine. Nothing special.", "Never buying here again.",
          "Arrived early, great quality."]:
    out = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Classify sentiment as positive, negative or neutral. "
                 f"Reply with one word only.\n\nReview: {r}")
    print(f"{out.text.strip():<10} {r}")

# 4 - OPEN, run locally. "The data stays on our machine" often decides it
#     outright, before anyone compares quality.

# 5 - When you HAVE labelled data and the task is narrow and repeated:
#     a Random Forest on 10,000 loan applications is faster, cheaper,
#     auditable, and usually MORE accurate than an LLM on that task.
```
</details>

## ❓ MCQs

**Q1.** Why can BERT not generate a chat reply?
- (a) It is too small  (b) It is an encoder, built to understand text, not continue it  (c) It is closed-source  (d) It has no context window

**Q2.** Which architecture powers GPT, Gemini and Claude?
- (a) Encoder-only  (b) Decoder-only  (c) Encoder-decoder  (d) Convolutional

**Q3.** A hospital that cannot let patient data leave the building should use…
- (a) A closed API  (b) An open model running locally  (c) Either  (d) No model

**Q4.** T5 is best described as…
- (a) A chat model  (b) An encoder-decoder for transforming text into other text  (c) An image model  (d) An embedding model

**Q5.** Zero-shot classification with an LLM needs…
- (a) Thousands of labelled examples  (b) No training examples at all  (c) A GPU  (d) Fine-tuning

**Q6.** You have 10,000 labelled loan applications. Better choice?
- (a) An LLM zero-shot  (b) A trained classifier — faster, cheaper, auditable and usually more accurate  (c) BERT  (d) T5

**Q7.** "Open weights" means…
- (a) Free to use commercially always  (b) You can download and run the model yourself  (c) The training data is public  (d) It has no licence

<details><summary>Answers</summary>

**A1 — (b).** **The most common misunderstanding in this topic.** BERT reads; it does not continue.

**A2 — (b) Decoder-only.**

**A3 — (b).** Data residency often decides before quality is even discussed.

**A4 — (b).** Text-to-text: translation, summarisation.

**A5 — (b) None.** That is what makes it useful for tasks nobody has data for.

**A6 — (b).** **The LLM is not the answer to every question.**

**A7 — (b).** Note it does *not* automatically mean an unrestricted licence — always check the terms.
</details>

## 🎯 Tasks

**Task 1 — The model selection memo.** For four scenarios — a hospital classifying notes; a student demo due Friday; semantic search over 100,000 documents; summarising 300-page reports — recommend a model **and name the one fact that would change your mind.**

**Task 2 — Zero-shot versus trained, measured.** Take 50 labelled rows from a classification dataset. Classify them zero-shot with an LLM and with your Session 5 model. **Report both accuracies and both costs**, and write which you would deploy and why.

**Task 3 — The architecture map.** Draw the encoder / decoder / encoder-decoder diagram from memory, placing all eight models in the table. **Then explain to a classmate why BERT cannot chat.**

---

# 4. Prompt Engineering Basics

**A prompt is a program written in English.** Vague input gives vague output — and unlike a bug in Python, nothing errors. It just quietly gives you something mediocre.

🧠 **Analogy: briefing an extremely capable new intern who has no context.** They can do almost anything, they have read enormously, and **they know nothing about your situation, your audience or your format.** Everything they need must be in the briefing.

## The five parts

```text
ROLE        You are a ______________________
TASK        ______________________________
CONTEXT     The reader is ______________________
CONSTRAINTS ______________________________
FORMAT      ______________________________
```

**Weak:**

> Tell me about overfitting.

**Strong:**

> **You are a machine learning tutor.** *(role)*
> **Explain overfitting.** *(task)*
> **The reader is a first-year student who has just learned what a decision tree is.** *(context)*
> **Use no mathematical notation and keep it under 120 words. Include one everyday analogy.** *(constraints)*
> **Format: one paragraph, then a two-line summary starting "In short:".** *(format)*

**The second prompt is not longer for the sake of it.** Every added line removes a decision the model would otherwise make for you — badly.

## The habits that matter most

| Habit | Why |
|---|---|
| **Say what you want, not what you don't** | "Write three sentences" beats "don't be too long" |
| **Give the audience** | "for a 10-year-old" changes everything downstream |
| **Specify the format** | Otherwise you get whatever it feels like |
| **Show an example** | One example beats a paragraph of description |
| **Ask for reasoning on hard tasks** | See chain-of-thought below |
| **Iterate** | Your first prompt is a draft, not a finished program |

## 📘 Examples

**Example 1 — weak versus strong, measured by usefulness**

```python
weak = "Tell me about overfitting."

strong = """You are a machine learning tutor.
Explain overfitting.
The reader is a first-year student who has just learned what a decision tree is.
Constraints: no mathematical notation, under 120 words, include one everyday analogy.
Format: one paragraph, then a two-line summary starting "In short:"."""
```

Run both. **The weak prompt gives you a textbook definition; the strong one gives you something you could paste into your own teaching materials.**

**Example 2 — system instructions set persistent behaviour**

```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="What is a p-value?",
    config=types.GenerateContentConfig(
        system_instruction="You are a statistics tutor for beginners. "
                           "Never use notation. Always give one concrete example. "
                           "Keep answers under 100 words.",
        temperature=0.3,
    ),
)
```

**The system instruction applies to every message in the conversation.** Put your role and constraints here; put the actual question in `contents`.

**Example 3 — structured output, which makes an LLM programmable**

```python
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="""Extract from this review: sentiment (positive/negative/neutral),
the product mentioned, and any complaint.

Review: "The headphones arrived two days late but the sound is excellent."
""",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",     # <- guaranteed valid JSON
        temperature=0.0,
    ),
)

import json
data = json.loads(response.text)
print(data["sentiment"])
```

> **`response_mime_type="application/json"` is the single most useful setting for building applications.** It turns a chatty model into a component you can call from code — which is exactly what Session 11 needs.

## ✏️ Practice

1. Run the weak and strong prompts. Compare the outputs.
2. Rewrite three of your own past prompts using all five parts.
3. Use a system instruction to make the model answer only in bullet points.
4. Extract structured JSON from three product reviews.
5. What happens if you ask for JSON **without** setting `response_mime_type`?

<details><summary>Solutions</summary>

```python
# api-only: needs a Gemini API key; run this yourself in Colab or locally
import os, json
from google import genai
from google.genai import types
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for label, p in [("WEAK", "Tell me about overfitting."),               # 1
                 ("STRONG", """You are a machine learning tutor.
Explain overfitting.
The reader is a first-year student who has just learned what a decision tree is.
Constraints: no mathematical notation, under 120 words, include one everyday analogy.
Format: one paragraph, then a two-line summary starting "In short:".""")]:
    print(f"--- {label} ---")
    print(client.models.generate_content(model="gemini-3.5-flash", contents=p).text)

r = client.models.generate_content(                                    # 3
    model="gemini-3.5-flash", contents="Explain cross-validation.",
    config=types.GenerateContentConfig(
        system_instruction="Answer only in bullet points. Never write a paragraph.",
        temperature=0.3))
print(r.text)

for rev in ["The headphones arrived two days late but the sound is excellent.",
            "Perfect fit, shipped fast.",
            "The screen cracked within a week."]:                      # 4
    out = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Extract sentiment (positive/negative/neutral), product, "
                 f"and complaint (or null).\n\nReview: {rev}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0))
    print(json.loads(out.text))

# 5 - You often get valid JSON wrapped in a markdown code fence, or preceded by
#     "Here is the JSON you asked for:". json.loads() then CRASHES.
#     Setting response_mime_type makes the output parseable every time --
#     which is what turns a chat model into a software component.
```
</details>

## ❓ MCQs

**Q1.** What are the five parts of a good prompt?
- (a) Question, answer, example, format, length
- (b) Role, task, context, constraints, format
- (c) Input, output, model, temperature, tokens
- (d) Who, what, when, where, why

**Q2.** "Don't be too long" is a weak constraint because…
- (a) It is rude  (b) It says what you don't want instead of what you do  (c) It is too short  (d) It needs punctuation

**Q3.** A system instruction differs from a normal prompt in that it…
- (a) Is free  (b) Applies to every message in the conversation  (c) Runs faster  (d) Is required

**Q4.** `response_mime_type="application/json"` guarantees…
- (a) A correct answer  (b) Parseable JSON output  (c) Lower cost  (d) Higher temperature

**Q5.** For structured extraction you should set temperature to…
- (a) 0.0  (b) 0.7  (c) 1.5  (d) It does not matter

**Q6.** Asking for JSON without setting the mime type often gives you…
- (a) An error  (b) JSON wrapped in code fences or explanatory text, which breaks `json.loads`  (c) Nothing  (d) XML

<details><summary>Answers</summary>

**A1 — (b).** Role, task, context, constraints, format.

**A2 — (b).** **Say what you want, not what you don't.**

**A3 — (b).** Put your role and standing constraints there.

**A4 — (b).** It does not make the content correct — only the *shape* reliable.

**A5 — (a) 0.0.** You want the same input to give the same structured output.

**A6 — (b).** **The single most common bug when students first build an app.**
</details>

## 🎯 Tasks

**Task 1 — The prompt improvement log.** Take five weak prompts, rewrite each with all five parts, and record both outputs side by side. **Write one sentence per pair on exactly what improved.**

**Task 2 — Build an extractor.** Write a function that takes any product review and returns a Python dict with sentiment, product, and complaint. **Handle the case where the model returns something unexpected** — Session 11 depends on this being robust.

**Task 3 — The system instruction test.** Write one system instruction and ask five different questions under it. **Does the behaviour hold across all five?** Report any question where it broke.

---

# 5. Types of Prompts

| Type | You provide | Use when |
|---|---|---|
| **Zero-shot** | Just the instruction | The task is common and the format is obvious |
| **One-shot** | One example | You need a specific format |
| **Few-shot** | 3–5 examples | Format *and* edge cases matter |
| **Chain-of-thought** | "Think step by step" | Multi-step reasoning, arithmetic, logic |

## Zero-shot

```text
Classify the sentiment of this review as positive, negative or neutral.

Review: "The delivery was late but the product is excellent."
Sentiment:
```

## One-shot — when the format matters

```text
Classify sentiment and give a confidence score.

Review: "Perfect fit, arrived early."
Sentiment: positive | Confidence: high

Review: "The delivery was late but the product is excellent."
Sentiment:
```

**The example does the work a paragraph of description would do badly.** The model now knows the exact separator, the exact vocabulary, the exact shape.

## Few-shot — when edge cases matter

```text
Review: "Perfect fit, arrived early."
Sentiment: positive | Confidence: high

Review: "It broke in a week."
Sentiment: negative | Confidence: high

Review: "It is fine I suppose."
Sentiment: neutral | Confidence: low

Review: "Late delivery, but excellent product."
Sentiment:
```

**The third example is the important one.** It teaches the model that *hedged* language means neutral **and** low confidence — a rule that is genuinely hard to state in words but obvious from one example.

> **Few-shot is the highest-value technique in this session.** Most "the model won't follow my format" problems are solved by three examples.

## Chain-of-thought — when reasoning matters

**Without:**

> A shop sells pens at ₹12. A customer buys 7 and pays with ₹100. How much change?
>
> *Answer:* ₹16

**With `Think step by step.`:**

> 1. Cost = 7 × ₹12 = ₹84
> 2. Paid = ₹100
> 3. Change = ₹100 − ₹84 = ₹16
>
> *Answer:* ₹16

**Why this works:** the model generates one token at a time, and each token can only depend on what came before it. Asking for the intermediate steps gives it **somewhere to do the working** — the partial results become part of the context the final answer is computed from.

🧠 **Analogy: mental arithmetic versus using paper.** Asked for 47 × 23 instantly, you might guess. Given paper, you get it right. **"Think step by step" hands the model the paper.**

> ⚠️ **Chain-of-thought costs tokens** — you pay for all that working. Use it for reasoning tasks; skip it for "classify this as positive or negative".

## 📘 Examples

**Example 1 — the four types, same task**

```python
prompts = {
"zero-shot": """Classify sentiment as positive, negative or neutral.
Review: "Late delivery, but excellent product."
Sentiment:""",

"one-shot": """Classify sentiment and give a confidence score.

Review: "Perfect fit, arrived early."
Sentiment: positive | Confidence: high

Review: "Late delivery, but excellent product."
Sentiment:""",

"few-shot": """Review: "Perfect fit, arrived early."
Sentiment: positive | Confidence: high

Review: "It broke in a week."
Sentiment: negative | Confidence: high

Review: "It is fine I suppose."
Sentiment: neutral | Confidence: low

Review: "Late delivery, but excellent product."
Sentiment:""",
}

for name, p in prompts.items():
    out = client.models.generate_content(model="gemini-3.5-flash", contents=p)
    print(f"{name:<12} {out.text.strip()}")
```

**Example 2 — the classic reasoning test**

```python
puzzle = ("If 5 machines take 5 minutes to make 5 widgets, "
          "how long do 100 machines take to make 100 widgets?")

plain = client.models.generate_content(model="gemini-3.5-flash", contents=puzzle)
cot = client.models.generate_content(
    model="gemini-3.5-flash", contents=puzzle + "\n\nThink step by step.")
```

**The answer is 5 minutes**, not 100. Each machine makes one widget in 5 minutes, so 100 machines make 100 widgets in the same 5 minutes. **Chain-of-thought makes this kind of trap far more reliable to avoid** — and, crucially, it makes the reasoning *visible*, so you can check it.

**Example 3 — choosing, with a rule**

```text
Is the task classification with an obvious format?     -> zero-shot
Do you need a specific output format?                  -> one-shot
Are there edge cases the format alone won't capture?   -> few-shot
Does it need arithmetic, logic or multiple steps?      -> chain-of-thought
Both format AND reasoning?                             -> few-shot + CoT
```

## ✏️ Practice

1. Run all four prompt types on the same review. Which gives the most usable output?
2. Add a fourth few-shot example covering sarcasm. Does it improve?
3. Try the widget puzzle with and without "Think step by step".
4. Try a three-step word problem of your own, both ways.
5. When would chain-of-thought be a waste of tokens?

<details><summary>Solutions</summary>

```python
# api-only: needs a Gemini API key; run this yourself in Colab or locally
import os
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

review = '"Late delivery, but excellent product."'
zero = f'Classify sentiment as positive, negative or neutral.\nReview: {review}\nSentiment:'
one = f'''Classify sentiment and give a confidence score.

Review: "Perfect fit, arrived early."
Sentiment: positive | Confidence: high

Review: {review}
Sentiment:'''
few = f'''Review: "Perfect fit, arrived early."
Sentiment: positive | Confidence: high

Review: "It broke in a week."
Sentiment: negative | Confidence: high

Review: "It is fine I suppose."
Sentiment: neutral | Confidence: low

Review: {review}
Sentiment:'''

for name, p in [("zero-shot", zero), ("one-shot", one), ("few-shot", few)]:   # 1
    print(f"{name:<12}",
          client.models.generate_content(model="gemini-3.5-flash", contents=p).text.strip())
# Few-shot gives the most CONSISTENT format, which is what matters when
# you are parsing the output in code.

puzzle = ("If 5 machines take 5 minutes to make 5 widgets, "                  # 3
          "how long do 100 machines take to make 100 widgets?")
print("\\nplain:", client.models.generate_content(
    model="gemini-3.5-flash", contents=puzzle).text.strip()[:200])
print("\\nCoT  :", client.models.generate_content(
    model="gemini-3.5-flash", contents=puzzle + "\\n\\nThink step by step.").text.strip()[:400])
# The answer is 5 MINUTES, not 100. CoT makes the reasoning VISIBLE so you
# can check it -- which is often more valuable than the accuracy gain.

# 5 - On simple one-step tasks: "classify this as positive or negative"
#     needs no working. You would pay for tokens of reasoning that add
#     nothing, and slow every request down.
```
</details>

## ❓ MCQs

**Q1.** Zero-shot means…
- (a) No model  (b) The instruction only, with no examples  (c) Zero temperature  (d) No context

**Q2.** You need output in an exact format. The cheapest fix is…
- (a) A longer description of the format  (b) One or more examples  (c) Higher temperature  (d) A bigger model

**Q3.** In the few-shot example, why does *"It is fine I suppose."* matter most?
- (a) It is the longest  (b) It teaches that hedged language means neutral **and** low confidence — hard to state, obvious from an example  (c) It is negative  (d) It is a duplicate

**Q4.** Chain-of-thought works because…
- (a) The model thinks harder  (b) Generating intermediate steps gives the final answer something to build on  (c) It uses a bigger model  (d) It lowers temperature

**Q5.** "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets?"
- (a) 100 minutes  (b) 5 minutes  (c) 20 minutes  (d) 1 minute

**Q6.** Chain-of-thought is a waste when…
- (a) The task is arithmetic  (b) The task is a simple one-step classification  (c) The task is logic  (d) Never

**Q7.** For a task needing both an exact format and multi-step reasoning, use…
- (a) Zero-shot  (b) Few-shot **and** chain-of-thought together  (c) One-shot only  (d) Higher temperature

<details><summary>Answers</summary>

**A1 — (b).** Just the instruction.

**A2 — (b).** **One example beats a paragraph of description.**

**A3 — (b).** Edge cases are exactly what examples are for.

**A4 — (b).** **It hands the model the paper.**

**A5 — (b) 5 minutes.** Each machine makes one widget in 5 minutes.

**A6 — (b).** You would pay for reasoning tokens that add nothing.

**A7 — (b).** They combine freely.
</details>

## 🎯 Tasks

**Task 1 — The four-way comparison.** Run all four prompt types on the same ten reviews. **Report format consistency, not just correctness** — count how many outputs you could parse with code without special-casing.

**Task 2 — Build a few-shot classifier.** Choose a classification task with a genuine edge case, write a four-example few-shot prompt, and test it on 20 items. **Report accuracy and note every item where it failed** — then add an example covering that failure and re-test.

**Task 3 — Chain-of-thought, costed.** Take five reasoning problems and run each with and without CoT. **Record accuracy and output length for both.** Then state the rule you would give a team about when the extra tokens are worth paying for.

---

# ✅ Before you move on

- [ ] I can say what makes a system generative rather than predictive
- [ ] I know roughly how many tokens my text is, and why that is the unit of cost
- [ ] I can predict what temperature 0, 0.7 and 1.5 will do
- [ ] I can explain why BERT cannot chat with me
- [ ] I know when open weights matter more than model quality
- [ ] I write prompts with role, task, context, constraints and format
- [ ] I use `response_mime_type="application/json"` when I need parseable output
- [ ] I choose zero-, one-, few-shot or CoT deliberately, with a reason
- [ ] I know that few-shot fixes format and CoT fixes reasoning
- [ ] **I know an LLM is not the right answer to every question**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-10-genai-llms.ipynb) | Every example above, runnable |
| [Prompt library](../prompts.md) | Every prompt here, ready to copy and paste |
| [Google AI Studio](https://aistudio.google.com/) | Test prompts in the browser, no code |
| [Setup guide](../setup-guide.md) | Getting your API key working |
