# Session 11 — AI-Powered Applications

**AI-powered Application Concepts · Retrieval & RAG · Building GenAI Applications using Streamlit · Integrating Machine Learning with Generative AI**

| | |
|---|---|
| **Notebook** | [session-11-ai-apps.ipynb](../notebooks/session-11-ai-apps.ipynb) |
| **Previous** | [Session 10 — GenAI & LLMs](session-10-genai-llms.md) |
| **Streamlit** | [simple](../tutorials/apps/streamlit-app-simple.md) · [advanced](../tutorials/apps/streamlit-app-advanced.md) · [loan app](../tutorials/apps/loan-app.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Everything in this course has been building to this session.** Session 5 gave you a model that decides. Session 10 gave you a model that writes. Here you join them into something a real person can use — which is also the shape of your capstone project.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Name the five common AI application patterns and pick the right one
2. Explain hallucination and why grounding is the fix
3. Build a retrieval step, and explain why keyword search is not enough
4. Describe the RAG pipeline end to end
5. Build a Streamlit chat interface with memory
6. **Combine an ML model and an LLM into one application**
7. Design an application that fails safely when the model is wrong

---

## The four topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Application concepts](#1-ai-powered-application-concepts) | The LLM is a component, not the application |
| 2 | [Retrieval & RAG](#2-retrieval-and-rag) | Ground the answer or it will invent one |
| 3 | [GenAI with Streamlit](#3-building-genai-applications-with-streamlit) | Chat is `session_state` plus a loop |
| 4 | [ML + GenAI](#4-integrating-machine-learning-with-generative-ai) | **The model decides. The LLM explains.** |

---

# 1. AI-powered Application Concepts

**An AI-powered application is not "a chatbot with a wrapper".** It is ordinary software in which one component happens to be a model.

🧠 **Analogy: a database-backed web app.** The database is essential, but nobody calls it "a database application" — it is a shop, or a booking system, that *uses* a database. **Treat your model the same way.** Your app still needs input validation, error handling, a sensible interface, and a plan for when the component fails.

## The five patterns

| Pattern | What it does | Example |
|---|---|---|
| **Assistant** | Answers questions conversationally | A support chatbot |
| **Transformer** | Text in one form → another form | Summariser, translator, rewriter |
| **Extractor** | Unstructured text → structured data | CV → JSON of skills |
| **Generator** | A brief → new content | Product descriptions from specs |
| **Augmenter** | Adds a layer to an existing system | **Your Session 5 model, explained in English** |

> **The last one is the most valuable and the least demonstrated.** Most real business value is in augmenting a system that already works.

## The architecture that actually ships

```text
┌─────────────────────────────────────────────────────┐
│  1. INPUT       validate before you spend a token   │
│  2. RETRIEVE    fetch relevant context (Topic 2)    │
│  3. PROMPT      build it from a template            │
│  4. MODEL       call the LLM / the ML model         │
│  5. PARSE       JSON, with a fallback that works    │
│  6. VALIDATE    is the output actually usable?      │
│  7. DISPLAY     with an honest confidence signal    │
│  8. LOG         what was asked, what came back      │
└─────────────────────────────────────────────────────┘
```

**Steps 1, 5, 6 and 8 are what separate a demo from an application.** The model call is the easy part.

## The four failure modes you must design for

| Failure | What it looks like | Your defence |
|---|---|---|
| **Hallucination** | A confident, fluent, wrong answer | Ground it (Topic 2); cite sources |
| **Bad format** | JSON that will not parse | `response_mime_type`, plus a try/except |
| **Latency** | Ten seconds of blank screen | Stream the output; show a spinner |
| **Cost** | A large bill from a loop | Cache; cap tokens; validate input first |

> ⚠️ **An LLM never says "I do not know" unless you ask it to.** Its job is to produce plausible text, and it will do that whether or not it knows the answer. **Designing for this is not optional.**

## 📘 Examples

**Example 1 — validate before you spend**

```python
def handle(question: str) -> str:
    q = question.strip()
    if len(q) < 5:
        return "Please ask a fuller question."          # no API call at all
    if len(q) > 2000:
        return "That is too long — please shorten it."  # no runaway cost
    return call_model(q)
```

**Two lines of validation prevent the two most common production incidents:** empty prompts burning quota, and a pasted document costing more than expected.

**Example 2 — a prompt template, not string-mashing**

```python
TEMPLATE = """You are a support assistant for {company}.
Answer using ONLY the context below.
If the context does not contain the answer, say exactly: "I don't have that information."

Context:
{context}

Question: {question}
"""

prompt = TEMPLATE.format(company="Acme", context=ctx, question=q)
```

**The "say exactly" line is doing real work.** Without it the model will invent an answer rather than admit the gap.

**Example 3 — parsing that survives contact with reality**

```python
import json

FENCE = chr(96) * 3        # the three-backtick marker, written without typing it

def parse_json(text, fallback=None):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Sometimes the model wraps it in a code fence despite instructions
        cleaned = (text.strip()
                   .removeprefix(FENCE + "json").removeprefix(FENCE).removesuffix(FENCE))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback
```

**Write this once and reuse it everywhere.** Even with `response_mime_type` set, a defensive parse costs nothing and saves an outage.

## ✏️ Practice

1. Classify five AI products you use into the five patterns.
2. Write a validation function rejecting empty, too-short and too-long input.
3. Write a prompt template with placeholders for context and question.
4. Write a `parse_json` with a fallback and test it on three malformed strings.
5. For each of the four failure modes, name your defence in one sentence.

<details><summary>Solutions</summary>

```python
import json

def validate(q):                                                       # 2
    q = (q or "").strip()
    if not q:            return None, "Please type a question."
    if len(q) < 5:       return None, "Please ask a fuller question."
    if len(q) > 2000:    return None, "That is too long - please shorten it."
    return q, None

for t in ["", "hi", "x" * 3000, "What is the refund policy?"]:
    print(validate(t)[1] or "OK")

TEMPLATE = """You are a support assistant for {company}.                # 3
Answer using ONLY the context below.
If the context does not contain the answer, say exactly: "I don't have that information."

Context:
{context}

Question: {question}
"""

FENCE = chr(96) * 3        # the three-backtick marker, written without typing it

def parse_json(text, fallback=None):                                   # 4
    for candidate in (text, text.strip().removeprefix(FENCE + "json")
                                 .removeprefix(FENCE).removesuffix(FENCE)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, AttributeError):
            continue
    return fallback

print(parse_json('{"a": 1}'))
print(parse_json(FENCE + 'json\n{"a": 1}\n' + FENCE))
print(parse_json('Sorry, I cannot do that.', fallback={"error": True}))

# 5 - Hallucination -> ground the answer in retrieved context and cite it.
#     Bad format     -> response_mime_type, plus a defensive parse.
#     Latency        -> stream the output; never show a blank screen.
#     Cost           -> validate first, cache repeats, cap max tokens.
```
</details>

## ❓ MCQs

**Q1.** In a well-built AI application, the LLM is…
- (a) The whole application  (b) One component among many  (c) Optional  (d) The user interface

**Q2.** Which pattern describes adding plain-English explanations to your Session 5 classifier?
- (a) Assistant  (b) Extractor  (c) Augmenter  (d) Generator

**Q3.** Why validate input *before* calling the model?
- (a) It is polite  (b) To avoid wasted cost and runaway requests  (c) The API requires it  (d) To improve accuracy

**Q4.** An LLM asked something outside its knowledge will usually…
- (a) Say "I don't know"  (b) Produce a confident, plausible, wrong answer  (c) Return an error  (d) Return empty text

**Q5.** Which step most distinguishes an application from a demo?
- (a) The model call  (b) Validation, parsing, and logging around the model call  (c) The colour scheme  (d) The model choice

**Q6.** The instruction *"If the context does not contain the answer, say exactly: I don't have that information"* exists because…
- (a) It is polite  (b) Without it the model invents an answer rather than admitting the gap  (c) It saves tokens  (d) The API needs it

<details><summary>Answers</summary>

**A1 — (b).** Like a database in a web app: essential, but not the application.

**A2 — (c) Augmenter.** **The most valuable pattern and the least demonstrated.**

**A3 — (b).** Empty prompts and pasted documents are the two most common cost incidents.

**A4 — (b).** **Its job is plausible text, not true text.** Design for this.

**A5 — (b).** The model call is the easy part.

**A6 — (b).** You have to give it explicit permission to not know.
</details>

## 🎯 Tasks

**Task 1 — Architecture on paper.** Design an AI application for a problem at your college. **Draw all eight steps** and write one line per step. Mark which steps you would build first for a demo, and which you would need before letting a real user near it.

**Task 2 — The failure catalogue.** For an app idea of your own, write out the four failure modes with a concrete example of each **going wrong in your specific app** — not in general. Then write your defence for each.

---

# 2. Retrieval and RAG

**A model knows what was in its training data. It does not know your documents, your prices, or what happened last week.** Ask anyway and it will produce something confident and wrong.

🧠 **Analogy: a brilliant graduate with no access to your files.** They can reason superbly about anything you show them — and they have never seen your company handbook. **Asking them your refund policy from memory is unfair to them and dangerous for you. Hand them the handbook first.**

**RAG — Retrieval-Augmented Generation — is exactly that:**

```text
1. RETRIEVE   find the documents relevant to the question
2. AUGMENT    paste them into the prompt as context
3. GENERATE   ask the model to answer USING ONLY that context
```

## Why keyword search is not enough

The obvious first attempt is keyword matching (TF-IDF). **It breaks immediately, and seeing how it breaks is the point of this topic.**

Measured on a seven-document support knowledge base:

**Questions that share words with the documents:**

| Question | Best match score |
|---|---|
| "how do I reset my password?" | **0.707** ✅ |
| "how long does shipping take?" | 0.305 ✅ |
| "what is the **refund** policy?" | **0.000** ❌ |

**That third row should stop you.** The document says *"**Refunds** are available within 30 days"*. The question says *"refund"*. **TF-IDF treats `refund` and `refunds` as unrelated strings** and scores zero.

**Questions that mean the same thing in different words:**

| Question | Best match score |
|---|---|
| "how long do I have to **return** something?" | **0.000** ❌ |
| "when are you **open**?" | **0.000** ❌ |
| "my package arrived **broken**" | **0.000** ❌ |

**Every one fails.** The documents cover refunds, office hours and damaged items — the answers are all there. But *return* ≠ *refund*, *open* ≠ *office hours*, *broken* ≠ *damaged*.

> **This is why RAG uses embeddings rather than keywords.** An embedding maps text to a vector where *meaning* determines position, so "broken" lands near "damaged" even though they share no letters. **Keyword search matches strings; embeddings match meaning.**

## The full pipeline

```text
BUILD (once)
  documents -> split into chunks -> embed each chunk -> store the vectors

ANSWER (per question)
  question -> embed -> find the nearest chunks -> paste into the prompt
           -> "answer using ONLY this context" -> cite the sources
```

| Decision | Typical choice | Why it matters |
|---|---|---|
| Chunk size | 200–500 words | Too big wastes context; too small loses meaning |
| Chunks retrieved | 3–5 | Too many buries the answer (Session 10's middle-of-context problem) |
| "Only use context" | **Always** | The difference between grounded and invented |
| Cite sources | **Always** | Lets a human check you |

## 📘 Examples

**Example 1 — retrieval, with the failure visible**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DOCS = [
    "Refunds are available within 30 days of purchase with a valid receipt.",
    "Our office hours are Monday to Friday, 9am to 6pm IST.",
    "Damaged items must be reported within 48 hours of delivery with photographs.",
]
vec = TfidfVectorizer(stop_words="english")
M = vec.fit_transform(DOCS)

def retrieve(q, k=2):
    sims = cosine_similarity(vec.transform([q]), M).ravel()
    return [(sims[i], DOCS[i]) for i in np.argsort(-sims)[:k]]

print(retrieve("my package arrived broken"))    # every score 0.000
```

**Example 2 — embeddings, which fix it**

```python
# Not executed here - needs an API key
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["my package arrived broken",
              "Damaged items must be reported within 48 hours."],
)
# The two vectors sit CLOSE TOGETHER despite sharing no words.
# That closeness is what retrieval actually needs.
```

**Example 3 — the grounded prompt**

```python
RAG_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say exactly:
"I don't have that information."
After your answer, cite which context items you used.

Context:
{context}

Question: {question}
"""

context = "\n".join(f"[{i+1}] {d}" for i, (_, d) in enumerate(retrieve(q)))
prompt = RAG_PROMPT.format(context=context, question=q)
```

**Numbering the context items is what makes citation possible.** Without numbers the model cannot point at anything, and neither can your user.

## ✏️ Practice

1. Build the TF-IDF retriever over seven support documents.
2. Query it with three questions that share words. What scores do you get?
3. Query it with three paraphrases. What scores do you get?
4. Explain why "what is the refund policy?" scores 0.000.
5. Write the grounded prompt template with numbered citations.

<details><summary>Solutions</summary>

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = [
    "Refunds are available within 30 days of purchase with a valid receipt.",
    "Our office hours are Monday to Friday, 9am to 6pm IST.",
    "Shipping within India takes 3-5 working days. International orders take 10-14 days.",
    "To reset your password, click 'Forgot password' on the login page.",
    "Warranty covers manufacturing defects for 12 months from the date of purchase.",
    "We accept UPI, credit cards, debit cards and net banking.",
    "Damaged items must be reported within 48 hours of delivery with photographs.",
]
vec = TfidfVectorizer(stop_words="english")                            # 1
M = vec.fit_transform(DOCS)

def retrieve(q, k=1):
    s = cosine_similarity(vec.transform([q]), M).ravel()
    return [(round(float(s[i]), 3), DOCS[i]) for i in np.argsort(-s)[:k]]

print("SHARES WORDS:")                                                 # 2
for q in ["how do I reset my password?", "how long does shipping take?",
          "what is the refund policy?"]:
    print(f"  {retrieve(q)[0][0]:.3f}  {q}")

print("\\nSAME MEANING, DIFFERENT WORDS:")                             # 3
for q in ["how long do I have to return something?", "when are you open?",
          "my package arrived broken"]:
    print(f"  {retrieve(q)[0][0]:.3f}  {q}")
# EVERY ONE scores 0.000. The answers are all in the documents.

# 4 - The document says "Refunds"; the question says "refund". TF-IDF
#     treats them as two unrelated strings. Even a plural breaks it.
#     THIS is why RAG uses embeddings: they map text to vectors where
#     MEANING determines position, so "broken" lands near "damaged".

RAG_PROMPT = """Answer the question using ONLY the context below.       # 5
If the context does not contain the answer, say exactly:
"I don't have that information."
After your answer, cite which context items you used.

Context:
{context}

Question: {question}
"""
q = "my package arrived broken"
ctx = "\\n".join(f"[{i+1}] {d}" for i, (_, d) in enumerate(retrieve(q, k=3)))
print("\\n" + RAG_PROMPT.format(context=ctx, question=q))
# Numbering the context is what makes citation possible.
```
</details>

## ❓ MCQs

**Q1.** What does RAG stand for, and what does it do?
- (a) Rapid AI Generation — speeds up the model
- (b) Retrieval-Augmented Generation — fetches relevant documents and puts them in the prompt
- (c) Recursive Answer Grounding — checks answers twice
- (d) Random Access Generation — samples randomly

**Q2.** "what is the **refund** policy?" scored 0.000 against a document saying "**Refunds** are available…". Why?
- (a) The document is wrong  (b) TF-IDF matches strings, and `refund` ≠ `refunds`  (c) Stop words removed it  (d) The document was too long

**Q3.** Embeddings beat keyword search because…
- (a) They are faster  (b) They place text by *meaning*, so "broken" lands near "damaged"  (c) They use less memory  (d) They need no model

**Q4.** Why instruct the model to use *only* the provided context?
- (a) To save tokens  (b) So it grounds the answer instead of inventing one  (c) To speed it up  (d) It is required by the API

**Q5.** Retrieving 50 chunks instead of 4 is a bad idea because…
- (a) It costs more **and** buries the answer in the middle of a long context  (b) The API rejects it  (c) It is always better  (d) Embeddings break

**Q6.** Why number your context items?
- (a) Style  (b) So the model can cite them and a human can verify  (c) The API requires it  (d) To sort them

<details><summary>Answers</summary>

**A1 — (b).** Retrieve, augment the prompt, then generate.

**A2 — (b).** **Even a plural breaks keyword matching.**

**A3 — (b).** Meaning determines position in the vector space.

**A4 — (b).** It is the difference between grounded and invented.

**A5 — (a).** Session 10's middle-of-context problem, now costing you money as well.

**A6 — (b).** Citation is what makes the answer checkable.
</details>

## 🎯 Tasks

**Task 1 — Your own knowledge base.** Build a 15-document knowledge base about something you know well. Build the TF-IDF retriever and **write ten questions: five sharing words, five paraphrased.** Report the scores and **count how many paraphrases failed.**

**Task 2 — The full RAG prompt.** Take your best-scoring retrieval and construct the complete grounded prompt with numbered citations. **Then ask a question your documents genuinely do not answer** and confirm the model says "I don't have that information" rather than inventing one. **If it invents one, strengthen the instruction and try again** — that iteration is the real exercise.

**Task 3 — Chunking.** Take one long document and split it three ways: 100 words, 300 words, 800 words. **Retrieve against the same question for each** and report which chunk size gave the most useful context, and why.

---

# 3. Building GenAI applications with Streamlit

You built a prediction app in Session 5. **A chat app needs one extra idea: memory.**

🧠 **Recall Streamlit's core behaviour from Session 5:** the whole script reruns on every interaction. **For a chat app that means your conversation would vanish on every message** — unless you keep it in `st.session_state`.

## The minimum chat app

```python
# chat_app.py
import streamlit as st
import os
from google import genai

st.title("Study Assistant")

@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

# 1. MEMORY - survives the rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. REPLAY the whole conversation on every rerun
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 3. TAKE new input
if prompt := st.chat_input("Ask me anything about the course"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = client.models.generate_content(
                model="gemini-3.5-flash", contents=prompt).text
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
```

```bash
streamlit run chat_app.py
```

**That is a complete chat application in about 30 lines.** The three numbered steps are the entire pattern: remember, replay, append.

## Giving it real memory

The app above sends only the *latest* message, so the model has no idea what you discussed a moment ago. **The fix is a chat session:**

```python
@st.cache_resource
def get_chat():
    return get_client().chats.create(model="gemini-3.5-flash")

reply = get_chat().send_message(prompt).text     # the history travels with it
```

> ⚠️ **`@st.cache_resource` on the chat means every visitor shares one conversation.** Fine for a demo; wrong for anything public. For multiple users, keep the history in `st.session_state` and send it explicitly.

## The three settings that make it feel professional

| Setting | Effect |
|---|---|
| `st.spinner("Thinking...")` | Never show a blank screen |
| Streaming the response | Text appears as it is generated |
| `st.chat_input` (not `text_input`) | Pinned to the bottom, like a real chat |

## 📘 Examples

**Example 1 — streaming, which changes the whole feel**

```python
with st.chat_message("assistant"):
    box = st.empty()
    full = ""
    for chunk in client.models.generate_content_stream(
            model="gemini-3.5-flash", contents=prompt):
        full += chunk.text or ""
        box.markdown(full + "▌")     # a blinking cursor
    box.markdown(full)
```

**A ten-second wait feels broken. Ten seconds of text appearing feels fast.** Same duration, completely different experience.

**Example 2 — the secrets file**

```toml
# .streamlit/secrets.toml   <- add this to .gitignore
GEMINI_API_KEY = "your-key-here"
```

```python
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
```

> ⚠️ **Never commit a key.** Automated scanners find keys pushed to GitHub within minutes.

**Example 3 — a clear-chat button and a token guard**

```python
with st.sidebar:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{len(st.session_state.messages)} messages")

    # A long conversation eventually exceeds the context window
    if len(st.session_state.messages) > 40:
        st.warning("Long conversation — consider clearing it.")
```

## ✏️ Practice

1. Build and run the minimum chat app.
2. Remove `st.session_state`. What happens to the conversation?
3. Add streaming. Does it feel faster?
4. Add a system instruction making it a course tutor that refuses off-topic questions.
5. Add a clear-conversation button and a message counter.

<details><summary>Solutions</summary>

```python
# streamlit-only: run with `streamlit run app.py`, not as a plain script

# 2 - Every message wipes the conversation. Streamlit reruns the WHOLE
#     script on each interaction, so a plain Python list is recreated
#     empty every time. Only st.session_state survives a rerun.

# 3 - Yes, dramatically. The total time is identical; the EXPERIENCE is
#     completely different. A blank ten seconds feels broken; ten seconds
#     of appearing text feels fast.

# 4
from google.genai import types
SYSTEM = ("You are a tutor for an ML and GenAI internship course. "
          "Answer only questions about machine learning, data science and "
          "generative AI. For anything else, reply: 'That is outside this "
          "course - please ask me about the course material.'")

reply = client.models.generate_content(
    model="gemini-3.5-flash", contents=prompt,
    config=types.GenerateContentConfig(system_instruction=SYSTEM,
                                       temperature=0.3)).text

# 5
with st.sidebar:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{len(st.session_state.messages)} messages")
```
</details>

## ❓ MCQs

**Q1.** Why must chat history live in `st.session_state`?
- (a) It is faster  (b) Streamlit reruns the whole script, so anything else is recreated empty  (c) The API requires it  (d) To save memory

**Q2.** Which widget gives a proper chat input pinned to the bottom?
- (a) `st.text_input`  (b) `st.chat_input`  (c) `st.text_area`  (d) `st.input`

**Q3.** Streaming the response changes…
- (a) The total time  (b) The perceived speed — the same duration feels far faster  (c) The cost  (d) The accuracy

**Q4.** Where should your API key live in a Streamlit app?
- (a) At the top of the script  (b) `.streamlit/secrets.toml`, gitignored  (c) In the URL  (d) In session state

**Q5.** `@st.cache_resource` on a chat session means…
- (a) Faster responses  (b) **Every visitor shares one conversation** — fine for a demo, wrong in public  (c) More memory  (d) Nothing

**Q6.** The minimum chat pattern is…
- (a) Fetch, sort, display  (b) Remember, replay, append  (c) Train, test, deploy  (d) Load, clean, plot

<details><summary>Answers</summary>

**A1 — (b).** Session 5's core Streamlit lesson, now with consequences.

**A2 — (b) `st.chat_input`.**

**A3 — (b).** Identical duration, completely different experience.

**A4 — (b).** Scanners find committed keys within minutes.

**A5 — (b).** A real multi-user app keeps history per session.

**A6 — (b) Remember, replay, append.**
</details>

## 🎯 Tasks

**Task 1 — Ship a chat app.** Build a chat assistant for a subject you know, with a system instruction, streaming, a clear button and a message counter. **Include a visible note saying what it should not be trusted for.**

**Task 2 — Break it deliberately.** Ask your app three questions it should refuse, three it cannot know, and one 3,000-word input. **Report what happened in each case and what you changed to handle it.**

---

# 4. Integrating Machine Learning with Generative AI

**This is the most valuable pattern in the course, and the one that best distinguishes a capstone project.**

Your Session 5 model produces `0.87`. **A person cannot act on `0.87`.** They need to know what it means, what drove it, and what they could do about it.

🧠 **Analogy: a blood test and a doctor.** The lab returns numbers — precise, objective, and meaningless to you. The doctor reads them and says *"your iron is low, here is what to do."* **The lab does not guess and the doctor does not measure.** Each does the part it is good at.

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  ML MODEL   │ --> │   THE NUMBER │ --> │     LLM     │ --> a person
│  decides    │     │   0.87       │     │  explains   │
└─────────────┘     └──────────────┘     └─────────────┘
   objective          the evidence          in English
   auditable                                 actionable
```

> ⚠️ **The LLM must never make the decision.** It explains a decision that was already made by a model you have measured, cross-validated and bootstrapped. **Ask the LLM to decide and you throw away every guarantee Sessions 5–8 gave you.**

## Why this division of labour

| | ML model | LLM |
|---|---|---|
| Decides | ✅ | ❌ **never** |
| Auditable | ✅ | ❌ |
| Consistent for the same input | ✅ | ❌ |
| Explains in English | ❌ | ✅ |
| Handles an unexpected question | ❌ | ✅ |

## 📘 Examples

**Example 1 — the pattern, end to end**

```python
# STEP 1 - the ML model decides. Objective, measured, reproducible.
proba = pipeline.predict_proba(applicant_row)[0][1]
decision = "approved" if proba > 0.5 else "declined"

# STEP 2 - gather the evidence, from the model itself
importances = dict(zip(feature_names, model.feature_importances_))
top3 = sorted(importances.items(), key=lambda kv: -kv[1])[:3]

# STEP 3 - the LLM explains, using ONLY what it was given
prompt = f"""You are a loan officer writing to an applicant.

DECISION: {decision} (confidence {proba:.0%})
The model weighted these factors most heavily: {top3}
Applicant's values: {applicant_row.to_dict('records')[0]}

Write 3-4 sentences explaining this decision kindly and clearly.
Do NOT change the decision. Do NOT invent factors not listed above.
End with one specific, actionable suggestion.
"""
```

**The two "Do NOT" lines are the whole safety design.** Without them the model will soften a rejection into an approval, or invent a reason that sounds plausible and is not in your data.

**Example 2 — where each part of the value comes from**

| Piece | Comes from | Why it must |
|---|---|---|
| The decision | Random Forest, ROC-AUC 0.96 | Auditable, consistent, measured |
| The confidence | `predict_proba` | A real number, not a vibe |
| The factors | `feature_importances_` | Grounded in the actual model |
| The explanation | The LLM | The only part it is uniquely good at |

**Example 3 — the whole application**

```python
# app.py
import streamlit as st, joblib, pandas as pd
from google import genai

st.title("Loan Decision Assistant")

@st.cache_resource
def load(): return joblib.load("loan_pipeline.joblib"), genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"])
pipeline, client = load()

income = st.number_input("Annual income", 0, 1_000_000, 50_000, step=1_000)
amount = st.number_input("Loan amount", 0, 500_000, 10_000, step=1_000)
score  = st.slider("Credit score", 300, 850, 650)

if st.button("Assess"):
    row = pd.DataFrame([{"person_income": income, "loan_amnt": amount,
                         "credit_score": score}])
    proba = pipeline.predict_proba(row)[0][1]          # ML DECIDES
    decision = "approved" if proba > .5 else "declined"

    st.metric("Decision", decision.title(), f"{proba:.0%} confidence")

    with st.spinner("Writing explanation..."):         # LLM EXPLAINS
        explanation = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=f"""Explain this loan decision in 3-4 kind sentences.
DECISION: {decision} (confidence {proba:.0%})
Income {income}, loan {amount}, credit score {score}.
Do NOT change the decision. Do NOT invent factors.
End with one actionable suggestion.""").text
    st.write(explanation)

    st.caption("Educational demo. Decisions are made by a statistical model "
               "and should be reviewed by a person.")
```

## ✏️ Practice

1. Load a Session 5 pipeline and get `predict_proba` for one row.
2. Extract the top three feature importances.
3. Build the explanation prompt from those two pieces.
4. Why must the prompt say "Do NOT change the decision"?
5. What would go wrong if you asked the LLM to make the decision instead?

<details><summary>Solutions</summary>

```python
import pandas as pd, joblib, tempfile, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

L = pd.read_csv(BASE + "loan_data_10k.csv").dropna().reset_index(drop=True)
for c in L.select_dtypes(include="object").columns:
    L[c] = LabelEncoder().fit_transform(L[c])
X, y = L.drop(columns=["loan_status"]), L["loan_status"]
a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(a, c)

row = b.iloc[[0]]                                                      # 1
proba = model.predict_proba(row)[0][1]
decision = "approved" if proba > .5 else "declined"
print(f"{decision} ({proba:.0%} confidence)")

imp = pd.Series(model.feature_importances_, index=X.columns)           # 2
top3 = imp.nlargest(3)
print(top3.round(4).to_dict())

prompt = f"""You are a loan officer writing to an applicant.           # 3

DECISION: {decision} (confidence {proba:.0%})
The model weighted these factors most heavily: {top3.round(3).to_dict()}
Applicant's values: {row.to_dict('records')[0]}

Write 3-4 sentences explaining this decision kindly and clearly.
Do NOT change the decision. Do NOT invent factors not listed above.
End with one specific, actionable suggestion.
"""
print(prompt)

# 4 - Language models are trained to be agreeable and helpful. Given a
#     rejection to explain, one will readily soften it into "you may still
#     qualify" or reverse it outright. The instruction is what keeps the
#     text consistent with the decision your model actually made.

# 5 - You would throw away EVERYTHING Sessions 5-8 gave you: the
#     cross-validated score, the confidence interval, the auditability,
#     and consistency (the same applicant could get different answers on
#     different days). The ML model decides; the LLM explains. Never swap them.
```
</details>

## ❓ MCQs

**Q1.** In the ML + GenAI pattern, who makes the decision?
- (a) The LLM  (b) The ML model  (c) Both vote  (d) The user

**Q2.** Why must the LLM never decide?
- (a) It is slower  (b) You lose auditability, consistency and every measured guarantee  (c) It costs more  (d) It cannot read numbers

**Q3.** *"Do NOT change the decision"* is needed because…
- (a) It saves tokens  (b) Models are trained to be agreeable and will soften or reverse a rejection  (c) The API requires it  (d) It improves grammar

**Q4.** Where should the "top factors" in the explanation come from?
- (a) The LLM's own guess  (b) The trained model's `feature_importances_`  (c) The user  (d) A fixed list

**Q5.** The blood test analogy maps how?
- (a) Lab = LLM, doctor = ML model  (b) Lab = ML model, doctor = LLM  (c) Both are the LLM  (d) Neither

**Q6.** The same applicant asks twice. With the LLM deciding, you might get…
- (a) The same answer always  (b) Different answers, because it samples  (c) An error  (d) A refusal

<details><summary>Answers</summary>

**A1 — (b) The ML model.** Always.

**A2 — (b).** **Sessions 5–8 exist to earn those guarantees. Do not discard them at the last step.**

**A3 — (b).** Agreeableness is a real and documented failure mode here.

**A4 — (b).** Grounded in the actual model — the Topic 2 lesson applied to numbers.

**A5 — (b).** The lab measures; the doctor explains. Neither does the other's job.

**A6 — (b).** Inconsistency alone disqualifies it for a decision that affects someone.
</details>

## 🎯 Tasks

**Task 1 — Build the integrated app.** Take your Session 5 saved pipeline and build a full Streamlit app: inputs, an ML decision with confidence, an LLM explanation grounded in feature importances, and an honest disclaimer. **This is a complete capstone-grade deliverable.**

**Task 2 — Try to break the guardrails.** Feed your app a borderline case (confidence 0.51) and read the explanation carefully. **Does the language match the decision, or does it hedge into the opposite?** Strengthen the prompt until it holds, and record what you had to add.

**Task 3 — The design document.** Write two pages on your app: the architecture diagram, which component does what and why, the four failure modes with your defences, and **three situations where the app should not be trusted.** Every deployed AI system needs this document, and yours is now capable of having one.

---

# ✅ Before you move on

- [ ] I can name the five application patterns and pick between them
- [ ] I know the LLM is a component, not the application
- [ ] I validate input **before** spending a token
- [ ] I can explain hallucination and why grounding fixes it
- [ ] I have seen keyword search fail on paraphrases, and know why embeddings fix it
- [ ] I can describe the RAG pipeline end to end
- [ ] I can build a Streamlit chat app with memory, streaming and a clear button
- [ ] I never commit an API key
- [ ] **I know the ML model decides and the LLM explains — never the reverse**
- [ ] I can design an application that fails safely

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-11-ai-apps.ipynb) | Every example above, runnable |
| [AI-powered apps](../tutorials/concepts/ai-powered-apps.md) | Concepts in more depth |
| [ML + GenAI tutorial](../tutorials/apps/ml_gen_ai.md) | The integration pattern, step by step |
| [Streamlit apps collection](../tutorials/apps/streamlit-apps-collection.md) | More app examples |
| [Prompt library](../prompts.md) | Ready-to-paste prompts |
