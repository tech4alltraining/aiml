# Session 11 — AI-Powered Applications

**Application concepts · Six working Streamlit apps · Three pure GenAI, three ML + GenAI**

| | |
|---|---|
| **Notebook** | [session-11-ai-apps.ipynb](../notebooks/session-11-ai-apps.ipynb) |
| **Previous** | [Session 10 — Generative AI & LLMs](session-10-genai-llms.md) |
| **Next** | [Session 12 — Open Source, Hugging Face & Responsible AI](session-12-opensource-ethics.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **[Session 5C](session-05c-deployment.md) taught you to serve a model. [Session 10](session-10-genai-llms.md) taught you to call an LLM.**
>
> **This session puts them in the same application** — six of them, each complete enough to run.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Name the **five patterns** almost every AI application is built from
2. Draw the **architecture that actually ships**, and say what each layer is for
3. Name the **four failure modes** and the design that handles each
4. Build a **GenAI Streamlit app** end to end
5. Get **structured, parseable output** from an LLM inside an app
6. Build a **multi-turn chat** that survives Streamlit's reruns
7. Explain **why you would combine an ML model with an LLM** — and which does which job
8. Build a **hybrid app**: a model decides, an LLM explains
9. **Measure and control hallucination** in your own application
10. Say what each of your apps **must not be used for**

---

## How this session is organised

| Part | What it covers |
|---|---|
| **A — [Application concepts](#part-a--ai-powered-application-concepts)** | **Concepts only, no code.** Patterns, architecture, failure modes |
| **B — [GenAI apps with Streamlit](#part-b--building-genai-applications-with-streamlit)** | **Apps 1–3.** The LLM does all the work |
| **C — [Integrating ML with GenAI](#part-c--integrating-machine-learning-with-generative-ai)** | **Apps 4–6.** A trained model decides; the LLM explains |

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [What an AI application is](#1-what-an-ai-powered-application-is) | | 8 | [App 1 — Text summariser](#8-app-1--text-summariser) |
| 2 | [The five patterns](#2-the-five-patterns) | | 9 | [App 2 — Ticket triage](#9-app-2--support-ticket-triage) |
| 3 | [The architecture that ships](#3-the-architecture-that-ships) | | 10 | [App 3 — Chat assistant](#10-app-3--chat-assistant-with-memory) |
| 4 | [The four failure modes](#4-the-four-failure-modes) | | 11 | [Why combine ML and GenAI](#11-why-combine-ml-and-genai) |
| 5 | [Designing for non-determinism](#5-designing-for-non-determinism) | | 12 | [App 4 — Loan decision + explanation](#12-app-4--loan-decision--explanation) |
| 6 | [Cost, latency and keys](#6-cost-latency-and-keys) | | 13 | [App 5 — Diabetes screening + report](#13-app-5--diabetes-screening--patient-report) |
| 7 | [The shared app skeleton](#7-the-shared-app-skeleton) | | 14 | [App 6 — Car valuation + listing](#14-app-6--car-valuation--listing-writer) |
| | | | 15 | [The invention problem, measured](#15-the-invention-problem-measured) |

**The [20 MCQs](#-session-11--20-mcqs) and [tasks](#-session-11--tasks) are at the end.**

---

# Part A — AI-powered application concepts

**Concepts only. No code in this part.**

---

# 1. What an AI-powered application is

> **An AI-powered application is ordinary software with a model somewhere inside it.**

**That sentence is deliberately unglamorous, and it is the most useful thing in this part.**

🧠 **Analogy: a restaurant with a very good chef.** **The chef is not the restaurant.** **You still need a door, a menu, tables, someone to take the order, a way to handle "the kitchen is closed", and a bill at the end.**
>
> **The model is the chef. Everything else is the application** — and everything else is where projects actually fail.

## What the application has to do that the model does not

| The model does | **The application must do** |
|---|---|
| Produce an output | **Collect valid input** |
| | **Handle the model being slow, wrong, or unavailable** |
| | **Show uncertainty honestly** |
| | **Keep the API key secret** |
| | **Not cost more than it earns** |
| | **Say what it must not be used for** |

> **In [Session 5C](session-05c-deployment.md) the model was 30 lines and the app was 60.** **That ratio is normal, and it gets worse — in production, the model is usually the smallest part of the system.**

---

# 2. The five patterns

**Almost every AI feature you will be asked to build is one of these five, or two of them joined together.**

| # | Pattern | The shape | Example |
|---|---|---|---|
| **1** | **Transform** | **Text in → text out** | Summarise, translate, rewrite |
| **2** | **Extract** | **Unstructured in → structured out** | An email → `{name, date, amount}` |
| **3** | **Classify** | **Input → one of N labels** | Route a ticket; flag a risk |
| **4** | **Converse** | **A sequence of turns, with memory** | A support assistant |
| **5** | **Augment** | **A model's output → explained, formatted, or acted on** | **A classifier decides, an LLM explains** |

## Which tool for which pattern

| Pattern | **Predictive model** | **LLM** | Notes |
|---|---|---|---|
| Transform | ✗ | **✓** | Only an LLM does this |
| Extract | ✗ | **✓** | Structured output — [§9](#9-app-2--support-ticket-triage) |
| **Classify** | **✓** | **✓** | **Both work. The choice is a real decision** |
| Converse | ✗ | **✓** | |
| **Augment** | **✓ and ✓** | | **This is [Part C](#part-c--integrating-machine-learning-with-generative-ai)** |

> **Row 3 is the interesting one.** **[Session 10](session-10-genai-llms.md#22-few-shot-prompting) built a classifier from three examples in two minutes; [Session 5B](session-05b-classification.md) built one from 10,000 labelled rows in an afternoon.**
>
> **Use the LLM when you have no labels and low volume. Use the trained model when you have labels, high volume, or need to defend the decision.**

## Pattern 5 is the one worth learning

> **Almost every genuinely useful business application is pattern 5.**
>
> **The model produces a number nobody can read. The LLM turns it into a sentence somebody can act on.** **Neither half is sufficient alone**, and Apps 4, 5 and 6 are all this shape.

---

# 3. The architecture that ships

```text
┌─────────────────────────────────────────────────────────┐
│  1. INTERFACE          Streamlit: widgets, results       │
├─────────────────────────────────────────────────────────┤
│  2. VALIDATION         is the input sane? reject early    │
├─────────────────────────────────────────────────────────┤
│  3. LOGIC              the ML model, the prompt, or both  │
├─────────────────────────────────────────────────────────┤
│  4. GUARDRAILS         parse, check, cap, fall back       │
├─────────────────────────────────────────────────────────┤
│  5. PRESENTATION       the answer AND its uncertainty     │
└─────────────────────────────────────────────────────────┘
```

| Layer | Its job | Skipping it means |
|---|---|---|
| **1. Interface** | Collect input, show output | — |
| **2. Validation** | **Reject bad input before spending anything** | You pay for API calls on empty text |
| **3. Logic** | The actual model call | — |
| **4. Guardrails** | **Parse the output; handle the failure** | **Your app crashes on a real user's first unusual request** |
| **5. Presentation** | **Show the answer and how much to trust it** | Users treat a guess as a fact |

> **Layers 2 and 4 are the ones beginners skip, and they are the ones that decide whether the app survives contact with a user.**

## 🧠 Analogy: the two doors

> **Validation is the bouncer at the entrance** — it stops nonsense getting in.
>
> **Guardrails are quality control at the exit** — they stop nonsense getting out.
>
> **A model sits between two doors, and it does not guard either one.**

---

# 4. The four failure modes

**Every AI application fails in one of four ways. Design for all four.**

## 1. Hallucination — confidently wrong

| | |
|---|---|
| **What it looks like** | A fluent, well-formatted, invented answer |
| **Why it happens** | The model predicts plausible text; plausible is not true |
| **Design for it** | **Supply the facts in the prompt. Ban invention explicitly. Verify in code. Show the source** |

> **[§15](#15-the-invention-problem-measured) measures this in App 6 and shows the fix working.**

## 2. Malformed output — right answer, wrong shape

| | |
|---|---|
| **What it looks like** | You asked for JSON; you got JSON inside a markdown fence, and `json.loads` raised |
| **Why it happens** | The model was trained to be conversational |
| **Design for it** | **Constrain the API, not just the prompt.** Then parse defensively anyway |

## 3. Latency — the blank screen

| | |
|---|---|
| **What it looks like** | Three seconds of nothing, and the user clicks again |
| **Why it happens** | A network round trip plus generation, token by token |
| **Design for it** | **A spinner, always. Stream the output. Cache repeated inputs. Disable the button while it runs** |

## 4. Cost — the invisible bill

| | |
|---|---|
| **What it looks like** | A demo costs pennies; a launch costs hundreds |
| **Why it happens** | **You pay per token — including [Session 10](session-10-genai-llms.md#15-what-the-machine-sees--the-raw-json)'s hidden thinking tokens** |
| **Design for it** | **Validate first. Cap `max_output_tokens`. Cache. Turn thinking off where it buys nothing** |

> **Remember the measurement from Session 10: `"Hi"` cost 196 tokens, 185 of them invisible.** **Multiply that by ten thousand users.**

---

# 5. Designing for non-determinism

**This is the shift that catches every developer moving from ML to GenAI.**

| | **A trained model** | **An LLM** |
|---|---|---|
| Same input twice | **Same output** | **Different output** |
| Testable with | `assert predict(x) == 1` | **Not that** |
| Debuggable by | Rerunning it | **Rerunning it may not reproduce the bug** |

## The three rules

**1. Set `temperature=0` for anything structured.**

> **Classification, extraction, routing.** **[Session 10](session-10-genai-llms.md#22-few-shot-prompting) measured a ticket flipping between `[SALES]` and `[BILLING]` at default temperature, and 5/5 identical at zero.**

**2. Never assert on exact strings — even at temperature 0.**

> **Session 10 measured 2 distinct outputs in 5 runs at `temperature=0.0`.** **Assert on the parsed structure, on a field, on a range.**

**3. Log the inputs and outputs.**

> **If you cannot reproduce a bug by rerunning, the log is the only evidence you will ever have.**

---

# 6. Cost, latency and keys

## Keys

> ⚠️ **The rule from [Session 10](session-10-genai-llms.md#12-setup--install-and-api-key) is absolute and it matters more here, because an app gets deployed.**
>
> **The key goes in an environment variable or a Streamlit secret. Never in the code. Never in the repository.**

| Where you deploy | Where the key goes |
|---|---|
| Local | **`.env`, listed in `.gitignore`** |
| **Streamlit Community Cloud** | **App settings → Secrets** |
| Anywhere else | The platform's environment variables |

## Cost control, in order of effectiveness

| # | Technique | Typical saving |
|---|---|---|
| **1** | **Validate before calling** — reject empty or over-long input | **100% of wasted calls** |
| **2** | **Turn thinking off** where reasoning is not needed | **Session 10 measured 510 → 39 tokens** |
| **3** | **Cache repeated inputs** | Depends on repeat rate |
| **4** | **Cap `max_output_tokens`** | Bounds the worst case |
| **5** | Use a smaller model | Often large |

## Latency

| Technique | Effect |
|---|---|
| **`st.spinner`** | **The user knows it is working.** The cheapest fix there is |
| **Streaming** | First words appear in under a second |
| **Cache** | A repeat is instant |
| **Do the ML first** | A trained model answers in microseconds; show that while the LLM works |

---

# Part B — Building GenAI applications with Streamlit

**Three apps. The LLM does all the work in every one.**

| App | Pattern | New thing it teaches |
|---|---|---|
| **[1 — Text summariser](#8-app-1--text-summariser)** | Transform | **The skeleton, and validation before you spend** |
| **[2 — Ticket triage](#9-app-2--support-ticket-triage)** | Extract + Classify | **Structured output you can parse** |
| **[3 — Chat assistant](#10-app-3--chat-assistant-with-memory)** | Converse | **Memory that survives Streamlit's reruns** |

---

# 7. The shared app skeleton

**Every app in this session starts from these fifteen lines. Write them once.**

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# genai_helper.py  -  shared by every app in this session
import os
import streamlit as st
from google import genai
from google.genai import types

MODEL_ID = "gemini-3.5-flash"

@st.cache_resource
def get_client():
    """One client, created once, shared by every rerun."""
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not key:
        st.error("GEMINI_API_KEY is not set. Add it to .env or to Streamlit secrets.")
        st.stop()
    return genai.Client(api_key=key)

def ask(prompt, temperature=0.3, max_tokens=800, thinking=False, json_out=False):
    """One call, with every guardrail from Part A applied."""
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0) if not thinking else None,
        response_mime_type="application/json" if json_out else None,
    )
    response = get_client().models.generate_content(
        model=MODEL_ID, contents=prompt, config=config)

    # GUARDRAIL: response.text can be None when the budget runs out
    if response.candidates[0].finish_reason.name != "STOP":
        st.warning(f"Response was cut short ({response.candidates[0].finish_reason.name}).")
    return response.text or "", response.usage_metadata.total_token_count
```

## Why each line is there

| Line | Reason |
|---|---|
| `@st.cache_resource` | **[Session 5C](session-05c-deployment.md#10-the-three-rules-that-fix-most-streamlit-bugs)'s rule** — the script reruns on every interaction; the client should not be rebuilt |
| `os.environ … or st.secrets` | **Works locally and on Streamlit Cloud with no code change** |
| `st.stop()` | **Fail loudly and early** rather than crashing three lines later |
| `thinking_budget=0` **by default** | **Session 10 measured 510 → 39 tokens.** Turn it on deliberately, not by accident |
| `max_output_tokens` | **Bounds the worst-case bill** |
| `finish_reason` check | **`response.text` can be `None`** — measured in Session 10 |
| **returns the token count** | **You cannot control a cost you do not display** |

> **Every app below imports `ask` from this file.** **The guardrails are written once and cannot be forgotten.**

---

# 8. App 1 — Text summariser

**The simplest useful GenAI app: text in, shorter text out.**

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app1_summariser.py
import streamlit as st
from genai_helper import ask

st.set_page_config(page_title="Summariser", page_icon="📝")
st.title("📝 Text Summariser")

text = st.text_area("Paste the text to summarise", height=250)

col1, col2 = st.columns(2)
style = col1.selectbox("Summary style",
                       ["three bullet points", "a single sentence",
                        "a paragraph for a non-technical reader"])
temperature = col2.slider("Creativity", 0.0, 1.0, 0.3, 0.1,
                          help="0.0 = same summary every time")

# --- LAYER 2: VALIDATION - reject before spending anything
MIN_CHARS, MAX_CHARS = 200, 20_000
too_short = len(text.strip()) < MIN_CHARS
too_long = len(text) > MAX_CHARS

if text and too_short:
    st.info(f"Add a little more text — {len(text.strip())} of {MIN_CHARS} characters minimum.")
if too_long:
    st.error(f"That is {len(text):,} characters. The limit is {MAX_CHARS:,}.")

if st.button("Summarise", type="primary", disabled=too_short or too_long):
    prompt = (f"Summarise the text below in {style}. "
              f"Output only the summary — no preamble, no heading.\n\n{text}")

    with st.spinner("Summarising..."):                      # LAYER 4: latency
        summary, tokens = ask(prompt, temperature=temperature)

    st.subheader("Summary")
    st.write(summary)

    c1, c2, c3 = st.columns(3)
    c1.metric("Input characters", f"{len(text):,}")
    c2.metric("Summary words", len(summary.split()))
    c3.metric("Tokens used", f"{tokens:,}")

    st.download_button("Download summary", summary, file_name="summary.txt")

st.caption("The summary is generated by a language model. It can omit or misstate "
           "details. Read the original before relying on it.")
```

## Measured

**Run on a four-sentence paragraph about cross-validation:**

| Style | Tokens used |
|---|---|
| `three bullet points` | **212** |
| `a single sentence` | **132** |

**The single-sentence result:**

```text
Cross-validation provides a reliable performance estimate, especially on small
datasets where single data splits are highly unstable, by averaging the scores
of a model trained and tested across multiple data folds.
```

> **Accurate, and it kept the point about instability.** **Which is worth checking, because a summariser that drops the caveat is worse than no summariser.**

## The three things that make this an application

| | Why it matters |
|---|---|
| **The `disabled=` on the button** | **You cannot spend money on an empty box.** Validation, enforced by the interface |
| **`st.spinner`** | **Three seconds of blank screen makes users click twice** — and pay twice |
| **The token metric** | **The user sees the cost.** So do you, while developing |

---

# 9. App 2 — Support ticket triage

**Extract *and* classify, with output your code can actually use.**

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app2_triage.py
import json
import pandas as pd
import streamlit as st
from genai_helper import ask

st.set_page_config(page_title="Ticket Triage", page_icon="🎫", layout="wide")
st.title("🎫 Support Ticket Triage")

CATEGORIES = ["BILLING", "TECH_ISSUE", "SALES", "OTHER"]

PROMPT = """You are a support triage system.

Classify the ticket into exactly one category: {categories}.
Also rate urgency as LOW, MEDIUM or HIGH, and write a one-line summary.

Return JSON with exactly these keys: category, urgency, summary.

Ticket: "{ticket}"
"""

tickets_text = st.text_area(
    "One ticket per line",
    "I was charged twice this month and nobody has replied to my three emails.\n"
    "Do you offer a student discount?\n"
    "The app crashes every time I open the reports tab.",
    height=150)

if st.button("Triage all tickets", type="primary"):
    tickets = [t.strip() for t in tickets_text.split("\n") if t.strip()]
    rows, total_tokens = [], 0

    progress = st.progress(0.0)
    for i, ticket in enumerate(tickets, start=1):
        raw, tokens = ask(
            PROMPT.format(categories=", ".join(CATEGORIES), ticket=ticket),
            temperature=0.0,          # deterministic - it is a classification
            json_out=True,            # structural, not a polite request
        )
        total_tokens += tokens

        # --- LAYER 4: GUARDRAIL. Parse defensively, always.
        try:
            data = json.loads(raw)
            row = {
                "ticket": ticket[:60],
                "category": data.get("category", "OTHER"),
                "urgency": data.get("urgency", "UNKNOWN"),
                "summary": data.get("summary", ""),
            }
            if row["category"] not in CATEGORIES:      # it invented a category
                row["category"] = "OTHER"
        except json.JSONDecodeError:
            row = {"ticket": ticket[:60], "category": "PARSE_FAILED",
                   "urgency": "UNKNOWN", "summary": raw[:80]}

        rows.append(row)
        progress.progress(i / len(tickets))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tickets", len(df))
    c2.metric("High urgency", int((df["urgency"] == "HIGH").sum()))
    c3.metric("Tokens used", f"{total_tokens:,}")

    st.download_button("Download CSV", df.to_csv(index=False), "triage.csv")

st.caption("Categories are assigned by a language model at temperature 0. "
           "Ambiguous tickets should still be reviewed by a person.")
```

## Measured

**The three example tickets, at `temperature=0.0` with `response_mime_type="application/json"`:**

| Ticket | Category | Urgency | Tokens |
|---|---|---|---|
| *"charged twice… nobody has replied"* | **BILLING** | **HIGH** | 112 |
| *"Do you offer a student discount?"* | **SALES** | LOW | 99 |
| *"The app crashes… reports tab"* | **TECH_ISSUE** | **HIGH** | 105 |

> **All three parsed on the first attempt.** **About 105 tokens per ticket — so 10,000 tickets is roughly a million tokens, which is a number you can put in a budget.**

## The three guardrails, and why each exists

| Guardrail | The failure it prevents |
|---|---|
| **`json_out=True`** | **[Session 10](session-10-genai-llms.md#20-zero-shot-prompting) measured the model wrapping JSON in a ` ```json ` fence, which makes `json.loads` raise.** This constrains the API rather than asking politely |
| **`try/except json.JSONDecodeError`** | **The belt to that braces.** A parse failure becomes a row, not a crash |
| **`if row["category"] not in CATEGORIES`** | **The model can invent a category that was never on your list.** Anything unexpected becomes `OTHER` |

> **Notice that the second and third guardrails assume the first one failed.** **That is the correct posture: the model is not a function, and you do not control it.**

---

# 10. App 3 — Chat assistant with memory

**The pattern everyone wants to build, and the one Streamlit's rerun model makes counter-intuitive.**

## ⚠️ The problem first

> **[Session 5C](session-05c-deployment.md#10-the-three-rules-that-fix-most-streamlit-bugs)'s Rule 1: the script reruns top to bottom on every interaction.**
>
> **So an ordinary Python list of messages is destroyed and recreated on every single message.** **The conversation would be one turn long, forever.**
>
> **`st.session_state` is the only thing that survives a rerun.**

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app3_chat.py
import streamlit as st
from genai_helper import get_client, MODEL_ID
from google.genai import types

st.set_page_config(page_title="Chat Assistant", page_icon="💬")
st.title("💬 Course Assistant")

SYSTEM = ("You are a teaching assistant for a machine learning course. "
          "Answer in at most 120 words. If you are not sure, say so. "
          "Never invent a citation, a dataset or a result.")

# --- 1. MEMORY that survives the rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.metric("Turns", len(st.session_state.messages) // 2)

# --- 2. REPLAY the whole conversation on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. TAKE new input
if question := st.chat_input("Ask about the course..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # --- 4. SEND THE WHOLE HISTORY - this is what makes it a conversation
    history = [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part(text=m["content"])])
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = get_client().models.generate_content_stream(
                model=MODEL_ID,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    temperature=temperature,
                    max_output_tokens=600,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            answer = st.write_stream(chunk.text for chunk in stream if chunk.text)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("This assistant can be confidently wrong. Check anything that matters "
           "against the session guides.")
```

## The four steps, and why the order matters

| Step | What breaks without it |
|---|---|
| **1. `st.session_state.messages`** | **The conversation resets on every message** |
| **2. Replay the history** | **The screen goes blank except for the newest message** |
| **3. `st.chat_input`** | — |
| **4. Send the *whole* history** | ⚠️ **The model has no memory.** Send only the last message and it cannot answer *"and why is that?"* |

> **Step 4 is the one that surprises people.** **The model is stateless.** **"Memory" in a chat application is entirely your list, resent in full on every turn.**
>
> **Which is also a cost warning: turn 20 sends all 20 previous turns.** **A long conversation gets quadratically expensive, and capping the history at the last N turns is a normal thing to do.**

## `st.write_stream` — the one-line latency fix

> **`generate_content_stream` returns chunks as they are generated, and `st.write_stream` renders them as they arrive.**
>
> **The first words appear in well under a second instead of the whole answer appearing after four.** **The total time is identical. The experience is not.**

## ✏️ Practice — the GenAI apps

1. Build App 1. **Paste in a session guide and summarise it three ways.** Compare the token counts.
2. **Remove the `disabled=` from App 1's button and click it on an empty box.** What did that cost?
3. Build App 2 and feed it ten tickets of your own. **Include one that is genuinely ambiguous** and see what it does.
4. **Remove `json_out=True` from App 2.** Report the exact error, and what `raw` contained.
5. Build App 3. **Ask a follow-up question that only makes sense with memory** — *"and why is that?"*
6. **Break App 3's memory:** send only the newest message instead of the whole history. **Describe what the conversation feels like.**

<details><summary>Answers</summary>

**2.** **An API call on an empty prompt.** It costs tokens, returns something useless, and does it again every time the user clicks. **Validation before the call is the cheapest optimisation in this session.**

**4.** **`json.JSONDecodeError`** — the model returns the JSON wrapped in a ` ```json ` fence, so `raw` starts with the three backticks and `json.loads` fails on the first character. **Session 10 measured this.** The `try/except` catches it and the row becomes `PARSE_FAILED`.

**6.** **Every answer is a first answer.** *"And why is that?"* produces a request for clarification, because from the model's point of view nothing came before it. **The model is stateless; the conversation lives entirely in your list.**
</details>

---

# Part C — Integrating Machine Learning with Generative AI

**Three apps. A trained model makes the decision; the LLM makes it understandable.**

| App | The model decides | The LLM produces |
|---|---|---|
| **[4 — Loan](#12-app-4--loan-decision--explanation)** | Approve or decline, with a probability | **A letter the applicant can act on** |
| **[5 — Diabetes](#13-app-5--diabetes-screening--patient-report)** | A risk percentage | **A patient-safe explanation** |
| **[6 — Car](#14-app-6--car-valuation--listing-writer)** | A price and an error range | **A sales listing** |

---

# 11. Why combine ML and GenAI

**Each is bad at what the other is good at.**

| | **Trained model** | **LLM** |
|---|---|---|
| Consistent | **✓ identical every time** | ✗ |
| Auditable | **✓ you can inspect it** | ✗ |
| Cheap at volume | **✓ microseconds, no API** | ✗ |
| Calibrated probability | **✓** | ✗ |
| **Explains itself in English** | ✗ | **✓** |
| **Adapts tone to the reader** | ✗ | **✓** |
| **Handles an unanticipated question** | ✗ | **✓** |

## The division of labour

```text
                 ┌──────────────────┐
   user input ──►│  TRAINED MODEL   │──► a number:  0.14
                 │  decides         │    a label:   DECLINED
                 └──────────────────┘
                          │
                          ▼  the number, and the facts behind it
                 ┌──────────────────┐
                 │  LLM             │──► "Your application was declined.
                 │  explains        │     Two factors drove this: ..."
                 └──────────────────┘
```

> **The rule: the LLM never makes the decision.**
>
> **It is handed a decision and asked to put it into words.** **That keeps the decision consistent, auditable and cheap, and it keeps the explanation readable.**

## ⚠️ Why the LLM must not decide

| If the LLM decided | Consequence |
|---|---|
| Same applicant, different answer on Tuesday | **[Session 10](session-10-genai-llms.md#22-few-shot-prompting) measured a classification flipping between runs** |
| No probability you can threshold | You cannot tune the precision/recall trade-off |
| No audit trail | **In lending and healthcare, that is often illegal** |
| A cost per decision | Microseconds become hundreds of milliseconds and a bill |

---

# 12. App 4 — Loan decision + explanation

**[Session 5C](session-05c-deployment.md#12-app-2--loan-approval)'s loan model, plus an LLM that writes the letter.**

## Step 1 — `train.py` is unchanged

**Use Session 5C's loan `train.py` exactly as it is.** **It saves `models/loan_model.joblib` with 0.8865 test accuracy.**

## Step 2 — `app.py`

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app4_loan.py
import joblib
import pandas as pd
import streamlit as st
from genai_helper import ask

st.set_page_config(page_title="Loan Decision", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/loan_model.joblib")

bundle = load_model()
pipeline, options = bundle["pipeline"], bundle["options"]

st.title("🏦 Loan Decision & Explanation")

with st.sidebar:
    st.header("Applicant")
    age = st.slider("Age", 20, 80, 24)
    income = st.number_input("Annual income", 8_000, 500_000, 45_000, step=1_000)
    emp_exp = st.slider("Years employed", 0, 58, 2)
    education = st.selectbox("Education", options["person_education"])
    home = st.selectbox("Home ownership", options["person_home_ownership"])

    st.header("Loan")
    amount = st.number_input("Loan amount", 1_000, 100_000, 20_000, step=500)
    intent = st.selectbox("Purpose", options["loan_intent"])
    rate = st.slider("Interest rate (%)", 5.0, 20.0, 15.5, 0.1)
    hist = st.slider("Credit history (years)", 2, 30, 3)
    score = st.slider("Credit score", 418, 768, 590)
    defaults = st.radio("Previous defaults on file",
                        options["previous_loan_defaults_on_file"], horizontal=True)

applicant = pd.DataFrame([{
    "person_age": age, "person_income": income, "person_emp_exp": emp_exp,
    "loan_amnt": amount, "loan_int_rate": rate,
    "loan_percent_income": round(amount / income, 4),
    "cb_person_cred_hist_length": hist, "credit_score": score,
    "person_gender": options["person_gender"][0],       # not shown - see the warning
    "person_education": education, "person_home_ownership": home,
    "loan_intent": intent, "previous_loan_defaults_on_file": defaults,
}])

if st.button("Assess application", type="primary"):
    # ---------- STEP 1: the MODEL decides ----------
    approved = int(pipeline.predict(applicant)[0])
    probability = float(pipeline.predict_proba(applicant)[0][1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", "Approved" if approved else "Declined")
    c2.metric("Approval probability", f"{probability:.1%}")
    c3.metric("Loan-to-income", f"{amount / income:.2f}")

    if 0.40 < probability < 0.60:
        st.warning("Borderline case — this should go to a human reviewer.")

    # ---------- STEP 2: the LLM EXPLAINS the decision ----------
    prompt = f"""You are a loan officer writing to an applicant. Be clear, kind and factual.

A machine learning model reviewed this application. Its decision is given below and you
must not contradict it or suggest it may change.

DECISION: {"APPROVED" if approved else "DECLINED"}
Model confidence that it would be approved: {probability:.0%}
Model accuracy on held-out data: {bundle["accuracy"]:.0%}

The applicant's details:
- Age {age}, income {income:,}/year, {emp_exp} years employed
- Requesting {amount:,} at {rate}% — that is {amount / income:.0%} of annual income
- Credit score {score}, {hist} years of credit history
- Previous defaults on file: {defaults}

Write three short paragraphs: the decision, the two factors that most likely drove it,
and one concrete thing they could change.

RULES:
- Use ONLY the figures above. Do not invent policy, criteria, thresholds or amounts.
- Do not add a currency symbol — the figures are unitless.
- Do not promise a different outcome.
"""

    with st.spinner("Writing the explanation..."):
        letter, tokens = ask(prompt, temperature=0.3, max_tokens=600)

    st.subheader("Explanation")
    st.write(letter)
    st.caption(f"{tokens:,} tokens")

st.caption(
    f"Decision by a Random Forest ({bundle['accuracy']:.1%} test accuracy). "
    "Explanation written by a language model from the model's output. "
    "**A teaching demonstration — not a lending decision.**")
```

## Measured

**For a 24-year-old requesting 44% of their annual income, credit score 590, with previous defaults on file:**

```text
MODEL:  decision = DECLINED    P(approve) = 0.140
LLM:    455 tokens
```

**The letter correctly identified both drivers — the credit profile and the 44% loan-to-income ratio — and suggested improving the credit score.**

## ⚠️ Two problems this app has, and you must know about both

**1. The LLM invented a currency.**

> **The measured output said `"$20,000"`.** **The dataset has no currency.** **The model filled a gap that looked like it needed filling** — which is exactly [§4](#4-the-four-failure-modes)'s failure mode 1.
>
> **The `Do not add a currency symbol` rule in the prompt above was added because of that measurement.** **[§15](#15-the-invention-problem-measured) shows how much difference an explicit rule makes.**

**2. `person_gender` is a model input.**

> **[Session 5C](session-05c-deployment.md#12-app-2--loan-approval) flagged this and it is worth repeating.** **The model was trained on historic approvals, so it learned whatever bias was in them, and gender is one of its features.**
>
> **This app hides the control and passes a fixed value — which does not fix the model, it only hides the problem.** **The real fix is to retrain without the column and measure what that costs.** **[Session 12](session-12-opensource-ethics.md) covers this properly.**

---

# 13. App 5 — Diabetes screening + patient report

**A harder problem, because the reader is a patient and the model is deliberately over-cautious.**

**Recall the numbers from [Session 5C](session-05c-deployment.md#13-app-3--diabetes-screening): recall 0.88, precision 0.43. It finds 88% of true cases and fewer than half the people it flags actually have diabetes.**

> **An app that prints "DIABETES DETECTED" at 43% precision would be actively harmful.** **The whole design problem here is communicating an uncertain result responsibly.**

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app5_diabetes.py
import joblib
import pandas as pd
import streamlit as st
from genai_helper import ask

st.set_page_config(page_title="Diabetes Screening", page_icon="🩺")

@st.cache_resource
def load_model():
    return joblib.load("models/diabetes_model.joblib")

bundle = load_model()
pipeline, options = bundle["pipeline"], bundle["options"]

st.title("🩺 Diabetes Risk Screening")
st.info("This is a **screening** tool. It flags people who should be tested. "
        "It does not diagnose anybody.")

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 1, 100, 52)
        gender = st.selectbox("Gender", options["gender"])
        bmi = st.slider("BMI", 10.0, 60.0, 31.4, 0.1)
        smoking = st.selectbox("Smoking history", options["smoking_history"])
    with c2:
        hba1c = st.slider("HbA1c level", 3.5, 9.0, 6.4, 0.1)
        glucose = st.slider("Blood glucose", 80, 300, 168, 5)
        hypertension = st.toggle("Diagnosed with hypertension", value=True)
        heart_disease = st.toggle("Diagnosed with heart disease")
    submitted = st.form_submit_button("Assess risk", type="primary")

threshold = st.slider("Flagging threshold", 0.10, 0.90, 0.50, 0.05)

if submitted:
    patient = pd.DataFrame([{
        "age": age, "bmi": bmi, "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": int(hypertension), "heart_disease": int(heart_disease),
        "gender": gender, "smoking_history": smoking}])

    # ---------- STEP 1: the MODEL scores the risk ----------
    risk = float(pipeline.predict_proba(patient)[0][1])
    flagged = risk >= threshold

    st.metric("Estimated risk", f"{risk:.1%}")
    st.progress(min(risk, 1.0))

    # ---------- STEP 2: the LLM writes it for a patient ----------
    prompt = f"""You are writing to a patient about a SCREENING result.
You are not a doctor and this is not a diagnosis.

Screening tool estimated risk: {risk:.0%}
Flagging threshold used: {threshold:.0%}
Result: {"ABOVE the threshold" if flagged else "below the threshold"}
About the tool: it finds about 88% of true cases, but fewer than half of the
people it flags actually have diabetes.

The patient's readings: age {age}, BMI {bmi}, HbA1c {hba1c},
blood glucose {glucose}, hypertension: {"yes" if hypertension else "no"},
smoking history: {smoking}.

Write at most 120 words covering: what the result means, what it does NOT mean,
and the single next step.

RULES:
- Do not diagnose. Do not prescribe. Do not name any medication.
- Do not state a probability that the patient has diabetes.
- Do not tell the patient not to worry, and do not alarm them.
"""

    with st.spinner("Preparing your result..."):
        report, tokens = ask(prompt, temperature=0.3, max_tokens=400)

    (st.warning if flagged else st.success)(report)
    st.caption(f"{tokens:,} tokens")

st.caption(
    "Screening only. This tool finds about 88% of true cases (recall) and fewer "
    "than half of the people it flags actually have diabetes (precision 0.43). "
    "It is tuned to over-flag on purpose. **Not a medical device. Not a diagnosis.**")
```

## Measured

**For a 52-year-old with BMI 31.4, HbA1c 6.4, glucose 168 and hypertension:**

```text
MODEL risk = 83.8%
```

**The generated report, in full:**

```text
Your recent screening estimated your risk at 84%, which is above our 50%
flagging threshold.

This is a screening result, not a diagnosis. It does not mean you have
diabetes. While the tool is good at flagging potential cases, fewer than half
of the people it flags actually have the condition. Your readings simply
indicate that further evaluation is needed.

Your single next step is to schedule an appointment with a primary care doctor.
Only a healthcare provider can perform the diagnostic tests needed to determine
your actual health status.
```

> **Read what it did *not* do.** **It did not diagnose. It did not say "you probably have diabetes". It repeated the precision caveat in plain words, and it gave exactly one action.**
>
> **That is the prompt's rules doing their job** — and it is why the rules are enumerated rather than summarised as "be careful".

## ⚠️ The design principle here

> **The riskier the domain, the more the prompt should say what is *forbidden* rather than what is wanted.**
>
> **"Write a helpful message" is an invitation to invent.** **"Do not diagnose, do not prescribe, do not name a medication, do not state a probability of having the condition" is a specification.**

---

# 14. App 6 — Car valuation + listing writer

**Regression this time: the model produces a number and an error range, and the LLM writes the advert.**

## Step 1 — `train.py`

```python
# train.py  -  python train.py
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

url = ("https://raw.githubusercontent.com/tech4alltraining/aiml/"
       "refs/heads/main/datasets/regression/cardekho_preprocessed.csv")
cars = pd.read_csv(url)

# Session 3's step 4 - impossible values out
cars = cars[(cars["seats"] > 0) & (cars["km_driven"] <= 1_000_000)].reset_index(drop=True)

FEATURES = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
X, y = cars[FEATURES], cars["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=12,
                              random_state=42, n_jobs=-1).fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R2 {r2:.4f}   MAE {mae:,.0f}")

pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES, "r2": r2, "mae": mae},
            "models/car_model.joblib")
print("saved -> models/car_model.joblib")
```

**Output:**

```text
R2 0.7290   MAE 113,788
saved -> models/car_model.joblib
```

> ⚠️ **MAE of 113,788 against a median price of 559,000 is about 20%.** **That is a large error, and the app has to say so** — which is why the listing states a range.

## Step 2 — `app.py`

```python
# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app6_car.py
import joblib
import pandas as pd
import streamlit as st
from genai_helper import ask

st.set_page_config(page_title="Car Valuation", page_icon="🚗")

@st.cache_resource
def load_model():
    return joblib.load("models/car_model.joblib")

bundle = load_model()
model, mae = bundle["model"], bundle["mae"]

st.title("🚗 Car Valuation & Listing Writer")

c1, c2 = st.columns(2)
with c1:
    vehicle_age = st.slider("Age (years)", 0, 25, 4)
    km_driven = st.number_input("Kilometres driven", 100, 500_000, 38_000, step=1_000)
    mileage = st.slider("Mileage (kmpl)", 4.0, 34.0, 21.4, 0.1)
with c2:
    engine = st.number_input("Engine (cc)", 700, 6600, 1197, step=50)
    max_power = st.slider("Max power (bhp)", 38.0, 400.0, 88.5, 0.5)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=2)

car = pd.DataFrame([{"vehicle_age": vehicle_age, "km_driven": km_driven,
                     "mileage": mileage, "engine": engine,
                     "max_power": max_power, "seats": seats}])

if st.button("Value this car", type="primary"):
    # ---------- STEP 1: the MODEL predicts the price ----------
    price = float(model.predict(car)[0])
    low, high = price - mae, price + mae

    m1, m2 = st.columns(2)
    m1.metric("Estimated value", f"{price:,.0f}")
    m2.metric("Typical error (±)", f"{mae:,.0f}")
    st.info(f"**A realistic range is {low:,.0f} to {high:,.0f}.** "
            "The single figure is the middle of that range, not a promise.")

    # ---------- STEP 2: the LLM writes the listing ----------
    prompt = f"""Write a short used-car listing (max 60 words) for this vehicle.

Estimated market value: {price:,.0f} (typical error ± {mae:,.0f})
Age: {vehicle_age} years | Driven: {km_driven:,} km
Mileage: {mileage} kmpl | Engine: {engine} cc | Power: {max_power} bhp | Seats: {seats}

RULES — these are absolute:
- Use ONLY the facts listed above. Every number and every adjective must be
  traceable to them.
- Do NOT mention condition, ownership, servicing, accidents, brand, model,
  body type or colour.
- Do NOT use evaluative words such as "excellent", "perfect", "pristine"
  or "well-maintained".
- State the price as a range, not a single figure.
"""

    with st.spinner("Writing the listing..."):
        listing, tokens = ask(prompt, temperature=0.7, max_tokens=300)

    st.subheader("Draft listing")
    st.write(listing)
    st.caption(f"{tokens:,} tokens")

    # ---------- LAYER 4: GUARDRAIL - verify the rules were obeyed ----------
    BANNED = ["excellent", "perfect", "pristine", "well-maintained", "well maintained",
              "single owner", "one owner", "serviced", "accident", "hatchback",
              "sedan", "suv", "showroom", "immaculate"]
    found = sorted({w for w in BANNED if w in listing.lower()})
    if found:
        st.error(f"⚠️ The draft contains unverifiable claims: {', '.join(found)}. "
                 "Remove them before publishing.")

st.caption(f"Price from a Random Forest (R² {bundle['r2']:.2f}, "
           f"typical error ±{mae:,.0f} on a median price of about 559,000). "
           "Listing text is generated and must be checked before publication.")
```

## Measured

```text
MODEL: price = 732,568   ± 113,788

LISTING (221 tokens):
For sale: 4-year-old, 5-seater vehicle. Powered by a 1197 cc engine delivering
88.5 bhp. Driven 38,000 km with a mileage of 21.4 kmpl.

Estimated price range: 618,780 to 846,356 rupees.
```

> **Every claim in that listing traces back to a number the model was given.** **No condition, no brand, no invented history.**
>
> **That did not happen by accident.** **[§15](#15-the-invention-problem-measured) is the measurement that produced those rules.**

---

# 15. The invention problem, measured

**App 6's first version had a reasonable-sounding instruction:**

```text
Do not invent a brand, model, colour or service history.
```

**It did not work. Here is the measurement — the same car, the same temperature, three runs of each prompt, with the output checked in code against a list of banned terms.**

| Prompt | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| **Weak** — *"do not invent…"* | `perfect`, `well-maintained` | `hatchback`, `perfect`, `well-maintained` | `perfect`, `well-maintained` |
| **Strong** — the enumerated rules | **none** | **none** | **none** |

**The weak version's actual output:**

```text
For sale: A well-maintained 5-seater, just 4 years old with only 38,000 km
driven. ... Perfect for city driving.
```

> **"Well-maintained" is a claim about a car nobody has inspected.** **"Perfect for city driving" is a recommendation the model invented.** **Both appeared in every single run.**
>
> **And notice: the weak instruction *did* work for the things it named.** No brand, no colour. **It failed on everything it did not name.**

## The three lessons

**1. A negative instruction only covers what it enumerates.**

> **"Do not invent" is not a rule — it is a sentiment.** **"Do not use evaluative words such as *excellent*, *perfect*, *pristine* or *well-maintained*" is a rule.**
>
> **This is [Session 10](session-10-genai-llms.md#19-the-principles)'s principle 4 — *say what to do, not only what to avoid* — with a measurement attached.**

**2. Verify in code. Do not trust the instruction.**

```python
# illustrative: a syntax reference, not runnable as written.
BANNED = ["excellent", "perfect", "pristine", "well-maintained", ...]
found = sorted({w for w in BANNED if w in listing.lower()})
if found:
    st.error(f"Draft contains unverifiable claims: {', '.join(found)}")
```

> **Twenty lines of `if` statements caught what a paragraph of polite instruction could not.** **The strong prompt reduced the failure rate to zero in three runs — the check catches the fourth.**

**3. This is how you find your own guardrails.**

> **Run your prompt ten times. Write down everything the model produced that was not in your input. That list is your `BANNED` list.**
>
> **You cannot design guardrails from first principles.** **You measure the failures and then guard against those.**

## ✏️ Practice — the hybrid apps

1. Build App 4 with Session 5C's loan model. **Find an applicant the model declines with probability under 0.10 and read the explanation.** Is it accurate?
2. **Remove the "do not add a currency symbol" rule from App 4.** Run it three times. Does a currency appear?
3. Build App 5. **Move the threshold from 0.9 down to 0.1 for one patient** and watch both the flag and the wording change.
4. **Delete the three "do not" rules from App 5's prompt.** Run it three times. **Report anything it said that a screening tool must not say.**
5. Build App 6 and run the weak-versus-strong comparison yourself. **Report your own hit counts.**
6. **Build a `BANNED` list for App 4 or App 5** by running the prompt ten times and writing down every invented claim.

<details><summary>Answers</summary>

**2.** **A currency symbol appears** — it was `$` in the measured runs, on a dataset with no currency at all. **The model fills gaps that look like they need filling.** That is failure mode 1 from [§4](#4-the-four-failure-modes), and the fix is to name the gap in the prompt.

**4.** Expect at least one of: **a diagnosis** (*"you have pre-diabetes"*), **a probability of having the condition**, **a reassurance** (*"there is no need to worry"*), or **a lifestyle prescription**. **The rules exist because each of those appeared during development** — which is exactly the process described in lesson 3 above.

**5.** Your counts will differ from the guide's — a different model version, a different day. **What should reproduce is the *direction*: the enumerated prompt produces fewer invented claims than the vague one.** **If it does not, that is a finding worth writing down.**
</details>

---

# ❓ Session 11 — 20 MCQs

**Answer from memory first, then check.**

### Application concepts

**Q1.** "An AI-powered application is ordinary software with a model somewhere inside it." The point of that sentence is…
- (a) AI is overrated  (b) **Everything around the model — validation, guardrails, error handling — is where projects actually fail**  (c) Models are simple  (d) Streamlit is enough

**Q2.** Which pattern describes "a classifier decides and an LLM explains the decision"?
- (a) Transform  (b) Extract  (c) Classify  (d) **Augment**

**Q3.** The two architecture layers beginners skip are…
- (a) Interface and logic  (b) **Validation and guardrails**  (c) Logic and presentation  (d) Interface and presentation

**Q4.** Validation belongs *before* the model call because…
- (a) It is faster  (b) **You pay for an API call on empty input, and you pay again every time the user clicks**  (c) Streamlit requires it  (d) It improves accuracy

**Q5.** The four failure modes of an AI application are…
- (a) Bugs, crashes, typos, downtime  (b) **Hallucination, malformed output, latency, cost**  (c) Overfitting, underfitting, leakage, bias  (d) Speed, memory, disk, network

**Q6.** You cannot reproduce a bug in a GenAI feature by rerunning it. The reason is…
- (a) A caching problem  (b) **The model is non-deterministic — the same input gives a different output**  (c) The network  (d) Streamlit reruns the script

**Q7.** For a classification step inside an app you should set…
- (a) `temperature=1.0`  (b) **`temperature=0.0`**  (c) `temperature=2.0`  (d) It makes no difference

**Q8.** Even at `temperature=0`, your test should not assert on an exact string because…
- (a) Strings are slow  (b) **Session 10 measured 2 distinct outputs in 5 runs at zero** — it is near-deterministic, not deterministic  (c) Encoding issues  (d) It is bad style

### The GenAI apps

**Q9.** `@st.cache_resource` on `get_client()` matters because…
- (a) It speeds up the model  (b) **Streamlit reruns the whole script on every interaction, so the client would be rebuilt on every click**  (c) It caches responses  (d) It stores the key

**Q10.** The shared helper sets `thinking_budget=0` by default because…
- (a) Thinking is broken  (b) **Session 10 measured a one-sentence question costing 510 tokens with thinking and 39 without** — it should be turned on deliberately  (c) It is faster to type  (d) Reasoning is never useful

**Q11.** `response_mime_type="application/json"` is better than asking for JSON in the prompt because…
- (a) It is shorter  (b) **It constrains the API rather than relying on the model to obey an instruction**  (c) It is cheaper  (d) It is required

**Q12.** App 2 still wraps the parse in `try/except` even with `json_out=True` because…
- (a) Style  (b) **The guardrail assumes the first guardrail failed — the model is not a function you control**  (c) JSON is slow  (d) Streamlit requires it

**Q13.** App 2 checks `if row["category"] not in CATEGORIES`. This catches…
- (a) Empty tickets  (b) **A category the model invented that was never on your list**  (c) Parse errors  (d) Rate limits

**Q14.** In App 3, the conversation must live in `st.session_state` because…
- (a) It is faster  (b) **An ordinary Python variable is destroyed and recreated on every rerun**  (c) The model requires it  (d) To save memory

**Q15.** App 3 sends the whole message history on every turn because…
- (a) It is cheaper  (b) **The model is stateless — "memory" is entirely your list, resent in full**  (c) Streamlit requires it  (d) It improves accuracy

**Q16.** Sending the whole history on every turn has one important cost…
- (a) None  (b) **Turn 20 sends all 20 previous turns, so a long conversation gets quadratically expensive**  (c) It is slower to type  (d) It breaks streaming

### The hybrid apps

**Q17.** In Apps 4–6, the LLM must never make the decision because…
- (a) It is slower  (b) **The decision would stop being consistent, auditable and cheap — and in lending or healthcare that is often illegal**  (c) It is less accurate  (d) It cannot classify

**Q18.** App 4's prompt says "do not add a currency symbol" because…
- (a) Style  (b) **A measured run invented `$` on a dataset that has no currency** — the model fills gaps that look like they need filling  (c) The dataset is in rupees  (d) Symbols break Streamlit

**Q19.** The weak instruction "do not invent a brand, model, colour or service history" produced invented claims in 3 of 3 runs. The lesson is…
- (a) The model is broken  (b) **A negative instruction only covers what it enumerates — it worked for brand and colour, and failed on everything it did not name**  (c) Use a bigger model  (d) Lower the temperature

**Q20.** App 6 checks the generated listing against a `BANNED` word list because…
- (a) Speed  (b) **You verify in code rather than trusting the instruction — and you build the list by measuring what the model actually invents**  (c) It is required by Streamlit  (d) To reduce tokens

<details><summary>Answers</summary>

**A1 — (b) Everything around the model.** **In Session 5C the model was 30 lines and the app was 60.** In production the model is usually the smallest part.

**A2 — (d) Augment.** **Almost every genuinely useful business application is this pattern**, and Apps 4, 5 and 6 are all this shape.

**A3 — (b) Validation and guardrails.** **The bouncer at the entrance and quality control at the exit.** A model guards neither door.

**A4 — (b) You pay for empty input.** **Removing the `disabled=` from App 1's button is the practice that demonstrates it.**

**A5 — (b) Hallucination, malformed output, latency, cost.** **Design for all four; you will meet all four.**

**A6 — (b) Non-determinism.** **Which is why logging inputs and outputs is not optional** — the log is the only evidence you will have.

**A7 — (b) `temperature=0.0`.** **Session 10 measured a ticket flipping between `[SALES]` and `[BILLING]` at default temperature, and 5/5 identical at zero.**

**A8 — (b) Near-deterministic, not deterministic.** **Two variants separated by one comma.** Assert on the parsed structure instead.

**A9 — (b) The script reruns on every interaction.** **Session 5C's Rule 1, with a cost attached.**

**A10 — (b) 510 tokens against 39.** **Hidden reasoning should be a deliberate choice, not a default you never noticed.**

**A11 — (b) It constrains the API.** **An instruction is a request the model may ignore; a config field is not.** Structure beats discipline.

**A12 — (b) The guardrail assumes the first one failed.** **That is the correct posture towards anything you do not control.**

**A13 — (b) An invented category.** **Real users produce inputs you did not anticipate, and the model will happily invent a label for them.**

**A14 — (b) Ordinary variables do not survive a rerun.** **The conversation would be one turn long, forever.**

**A15 — (b) The model is stateless.** **Send only the last message and it cannot answer "and why is that?"**

**A16 — (b) Quadratic cost.** **Capping the history at the last N turns is a normal thing to do.**

**A17 — (b) Consistency, auditability and cost.** **The LLM is handed a decision and asked to put it into words. It never makes one.**

**A18 — (b) A measured run invented `$`.** **The rule was added because of the measurement, not the other way round.**

**A19 — (b) It only covers what it enumerates.** **No brand and no colour appeared — but `perfect` and `well-maintained` appeared in every run.**

**A20 — (b) Verify in code.** **You cannot design guardrails from first principles. Run the prompt ten times, write down every invented claim, and guard against those.**
</details>

---

# 🎯 Session 11 — Tasks

**These tasks are about building GenAI applications. Every one produces something you can run.**

## Concepts

**Task 1 — Pattern-match five products.** Take five AI features you have used. **Identify which of the five patterns each is, and whether an LLM or a trained model is doing the work.**

**Task 2 — Draw your own architecture.** Pick a feature you want to build. **Draw the five layers and write one sentence per layer**, including what specifically you would validate and what you would guard against.

**Task 3 — The failure-mode audit.** For any AI product you use, find evidence of all four failure modes. **Screenshot or transcribe each one.**

## The GenAI apps

**Task 4 — App 1, built and measured.** Build the summariser. **Summarise the same text at three styles and three temperatures — nine runs — and report the token counts in a table.**

**Task 5 — Cost the summariser.** From your token counts, **estimate the monthly bill for 1,000 summaries a day** at the current price of your model.

**Task 6 — App 2, ten tickets.** Build the triage app and run it on ten tickets of your own, **including two you consider genuinely ambiguous.** Run each twice at temperature 0 and confirm they are stable.

**Task 7 — Break the parser.** Remove `json_out=True` from App 2. **Report the exact exception, what `raw` contained, and which guardrail caught it.**

**Task 8 — Make it invent a category.** Feed App 2 a ticket that fits none of your four categories. **Does it return `OTHER`, or does it invent something? Report exactly what happened.**

**Task 9 — App 3, built.** Build the chat assistant. **Have a five-turn conversation where each turn depends on the previous one**, and paste the transcript.

**Task 10 — Cap the history.** Modify App 3 to send only the last 6 messages. **Have a long conversation and describe the exact moment the assistant forgets something.**

**Task 11 — Add a system instruction.** Give App 3 a persona and a hard constraint of your own. **Try to make it break the constraint, and report whether you succeeded.**

## The hybrid apps

**Task 12 — App 4, built.** Build the loan app on Session 5C's model. **Produce one approval and one decline, and paste both letters.**

**Task 13 — Check the explanation against the model.** For App 4, **compare the two factors the LLM named against the model's actual feature importances.** Do they agree?

**Task 14 — App 5, built.** Build the diabetes app. **Run one patient at thresholds 0.2, 0.5 and 0.8** and paste all three reports.

**Task 15 — Strip the safety rules.** Delete the "do not" rules from App 5's prompt and run it five times. **Report every sentence a screening tool should not have produced.**

**Task 16 — App 6, built.** Build the car app including the `BANNED` check. **Find an input where the check fires.**

**Task 17 — Reproduce the invention measurement.** Run App 6's weak and strong prompts three times each. **Report your hit counts in the same table format as §15.**

**Task 18 — Build your own guardrail list.** For any app above, **run the prompt ten times, list every claim not present in your input, and turn that list into a code check.**

## Your own application

**Task 19 — Design it.** Choose a problem from your own domain that needs both a trained model and an LLM. **Write a one-page design: the five patterns it uses, the five layers, the four failure modes and your mitigation for each.**

**Task 20 — Build and ship it.** Build that application. **Deploy it** ([Session 5C](session-05c-deployment.md#15-deploying-to-streamlit-community-cloud)). **Then write the caption that says what it must not be used for** — and go through Session 5C's pre-deployment checklist row by row, reporting anything it fails.

---

## ✅ Session 11 checklist

- [ ] I can name the **five patterns** and identify them in real products
- [ ] I can draw the **five architecture layers** and say what each is for
- [ ] I **validate before I call**, so bad input costs nothing
- [ ] I **parse defensively**, even when I constrained the API
- [ ] I know the **four failure modes** and design for all four
- [ ] I set **`temperature=0` for anything structured**
- [ ] I **never assert on exact strings**, even at temperature 0
- [ ] I use `@st.cache_resource` for the client and `st.session_state` for the conversation
- [ ] I know the model is **stateless** — memory is my list, resent in full
- [ ] I turn **thinking off** unless the task needs reasoning
- [ ] I **show the token count**, to the user and to myself
- [ ] **The trained model decides; the LLM explains.** Never the other way round
- [ ] I write prompt rules by **enumerating what is forbidden**, not by asking nicely
- [ ] I **verify the output in code** against a list I built by measuring
- [ ] **Every app says what it must not be used for**

---

| | |
|---|---|
| **Previous** | [Session 10 — Generative AI & LLMs](session-10-genai-llms.md) |
| **Next** | [Session 12 — Open Source, Hugging Face & Responsible AI](session-12-opensource-ethics.md) |
| **Notebook** | [session-11-ai-apps.ipynb](../notebooks/session-11-ai-apps.ipynb) |
| **More practice** | [Exercises & assignments](../exercises-assignments.md) |
