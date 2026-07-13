# Generative AI and Agentic AI

## Student Hands-on Handbook

**Beginner-friendly material for a three-hour practical session**  
**Concepts | Prompting | Python | Tools | Agents | Streamlit | Responsible AI**  
**Version: July 2026**

---

## How to use this handbook

This handbook is intended for students with no previous knowledge of Generative AI or Agentic AI. Follow the sections in order during the training.

For every practical example:

1. Read the explanation before running the code.
2. Copy the code into the specified `.py` file when using VS Code.
3. Colab users can paste the same Python code into a code cell after completing the Colab setup.
4. Run the program once without changing it.
5. Change one input or prompt and run it again.
6. Compare the outputs.
7. Verify important facts before using the generated content.

> **Important:** AI-generated text can sound confident and still be incomplete, biased, or incorrect. Human verification remains necessary.

---

## Table of contents

1. [Learning outcomes](#learning-outcomes)
2. [Three-hour session map](#three-hour-session-map)
3. [Software and accounts required](#software-and-accounts-required)
4. [Option A: Google Colab setup](#option-a-google-colab-setup)
5. [Option B: Local setup with VS Code](#option-b-local-setup-with-vs-code)
6. [Example Python filenames](#example-python-filenames)
7. [Basic concepts](#basic-concepts)
8. [Real-life Generative AI applications](#real-life-generative-ai-applications)
9. [Generative AI coding labs](#generative-ai-coding-labs)
10. [Prompt engineering](#prompt-engineering)
11. [Local Streamlit Generative AI app](#local-streamlit-generative-ai-app)
12. [Understanding Agentic AI](#understanding-agentic-ai)
13. [Real-life Agentic AI applications](#real-life-agentic-ai-applications)
14. [Agentic AI foundation labs](#agentic-ai-foundation-labs)
15. [Academic Agent: Study Assistant](#academic-agent-study-assistant)
16. [Local Streamlit Academic Agent](#local-streamlit-academic-agent)
17. [Professional Agent: Customer Support](#professional-agent-customer-support)
18. [Local Streamlit Professional Agent](#local-streamlit-professional-agent)
19. [Responsible and safe AI](#responsible-and-safe-ai)
20. [Troubleshooting](#troubleshooting)
21. [Review questions](#review-questions)
22. [Glossary](#glossary)
23. [Official references](#official-references)

---

# Learning outcomes

By the end of the session, students should be able to:

1. Explain Artificial Intelligence, Machine Learning, Generative AI, Large Language Models, and Agentic AI.
2. Distinguish a predictive system from a generative system.
3. Explain prompts, tokens, context, responses, hallucination, grounding, tools, and memory.
4. Write structured prompts using role, task, context, constraints, and output format.
5. Call a Generative AI model from a Python program.
6. Build a quiz generator, summarizer, professional email assistant, and conversational example.
7. Run a simple Generative AI interface locally using Streamlit.
8. Explain the difference between a chatbot, workflow, and agent.
9. Define Python functions that can be exposed as agent tools.
10. Build an academic Study Assistant Agent.
11. Build a professional Customer Support Agent using simulated data.
12. Apply safeguards such as tool allow-lists, input validation, step limits, and human approval.

---

# Three-hour session map

| Time | Topic | Main activity |
|---:|---|---|
| 0-15 min | Setup and first interaction | Configure the environment and run the first prompt |
| 15-35 min | Basic concepts | AI, ML, GenAI, LLM, prompt, token, context, hallucination |
| 35-60 min | Prompt engineering | Compare weak and strong prompts |
| 60-90 min | GenAI coding | Quiz, summarizer, email, and conversation memory |
| 90-100 min | Break | Discussion: chatbot or agent? |
| 100-120 min | Agentic AI concepts | Goal, tools, memory, actions, observations, guardrails |
| 120-145 min | Agent foundations | Rule-based routing and function calling |
| 145-165 min | Academic agent | Study Assistant Agent and local Streamlit interface |
| 165-175 min | Professional agent | Customer Support Agent demonstration |
| 175-180 min | Review | Safety checklist and exit questions |

The instructor may demonstrate the Streamlit applications while students complete them after the session if classroom time is limited.

---

# Software and accounts required

Students need:

- A Google account for Google AI Studio and optionally Google Colab
- A Gemini API key
- An internet connection
- For local work: Python, VS Code, and the VS Code Python extension

The examples use the official Google GenAI Python SDK and the Gemini Interactions API.

> API access can have rate limits and usage charges depending on the account and model. Do not repeatedly run large prompts without a purpose.

---

# Option A: Google Colab setup

Use Colab when the class needs the simplest browser-based setup.

## Step 1: Install the SDK

```python
!pip install -q -U google-genai
```

## Step 2: Enter the API key safely

```python
import os
from getpass import getpass

os.environ["GEMINI_API_KEY"] = getpass("Enter your Gemini API key: ")
```

## Step 3: Create the client

```python
from google import genai

client = genai.Client()
MODEL = "gemini-3.5-flash"
```

## Step 4: Test the connection

```python
interaction = client.interactions.create(
    model=MODEL,
    input="Explain Generative AI in two simple sentences.",
)

print(interaction.output_text)
```

For later examples, Colab users can paste the core program logic into separate code cells. Streamlit instructions in this handbook are only for local-machine use.

---

# Option B: Local setup with VS Code

VS Code students will use normal `.py` files. A local Jupyter installation is not required.

## Step 1: Install software

Install:

- Python 3
- Visual Studio Code
- The **Python** extension in VS Code

## Step 2: Create a working folder

Open a terminal:

```bash
mkdir genai-agentic-ai
cd genai-agentic-ai
code .
```

When the `code` command is unavailable, open VS Code manually and choose **File > Open Folder**.

## Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
.venv\Scriptsctivate.bat
```

### macOS or Linux

```bash
source .venv/bin/activate
```

After activation, the terminal normally begins with `(.venv)`.

## Step 4: Select the interpreter in VS Code

1. Press `Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS.
2. Search for **Python: Select Interpreter**.
3. Choose the interpreter inside `.venv`.

## Step 5: Install packages

```bash
python -m pip install --upgrade pip
pip install -U google-genai python-dotenv streamlit
```

Create `requirements.txt`:

```text
google-genai
python-dotenv
streamlit
```

Later, the environment can be recreated using:

```bash
pip install -r requirements.txt
```

## Step 6: Create the environment file

Create `.env`:

```text
GEMINI_API_KEY=replace_with_your_actual_key
GEMINI_MODEL=gemini-3.5-flash
```

Create `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
```

> Never upload `.env` to GitHub. It contains a secret API key.

## Step 7: Run a Python file

```bash
python first_genai.py
```

## Step 8: Run a Streamlit file locally

```bash
streamlit run streamlit_genai_app.py
```

Streamlit starts a local server and normally opens the application in the default browser. Stop the server using `Ctrl+C` in the terminal.

---

# Example Python filenames

Create the following files as the training progresses. Use these filenames so that the examples remain easy to identify and run.

| Order | Filename | Purpose |
|---:|---|---|
| 1 | `first_genai.py` | First Gemini API request |
| 2 | `prompt_comparison.py` | Compare weak and structured prompts |
| 3 | `quiz_generator.py` | Generate a beginner quiz |
| 4 | `text_summarizer.py` | Summarize a meeting note |
| 5 | `professional_email.py` | Draft a controlled customer email |
| 6 | `conversation_memory.py` | Continue an interaction using prior context |
| 7 | `streamlit_genai_app.py` | Local GenAI graphical interface |
| 8 | `rule_based_agent.py` | Understand basic tool routing |
| 9 | `calculator_agent.py` | Learn function calling |
| 10 | `study_assistant_agent.py` | Academic multi-tool agent |
| 11 | `streamlit_study_agent_app.py` | Local academic agent interface |
| 12 | `customer_support_agent.py` | Professional multi-tool agent |
| 13 | `streamlit_customer_support_app.py` | Local professional agent interface |

---

# Basic concepts


## Quick analogy bank for beginners

Use these analogies whenever a concept feels abstract.

| Concept | Simple analogy | Meaning |
|---|---|---|
| AI | **Umbrella** | AI is the broad field; many techniques come under it. |
| Machine Learning | **Student learning from solved examples** | The system learns patterns from examples instead of fixed rules. |
| Predictive AI | **Weather forecast** | It estimates what is likely to happen or which label fits. |
| Generative AI | **Creative assistant or chef** | It creates a new output from ingredients given in the prompt. |
| LLM | **Powerful autocomplete** | It generates likely text based on the prompt and context. |
| Prompt | **Order given to a restaurant waiter** | Clearer instructions produce a more suitable result. |
| Token | **Lego block** | Text is split into small pieces the model can process. |
| Context | **Open notebook during an exam** | The model can use only the information available to it. |
| Hallucination | **Confident classmate guessing** | The answer may sound correct even when it is wrong. |
| Grounding | **Open-book answer from an approved textbook** | The answer is based on trusted supplied information. |
| Tool | **Calculator or office machine** | A controlled function performs a specific reliable action. |
| Memory/state | **Sticky notes** | The system remembers useful details across steps. |
| Agent | **Office assistant with approved tools** | It chooses safe tools and actions to complete a goal. |
| Guardrail | **Road safety barrier** | It prevents unsafe or unauthorized actions. |

### Mini activity: Match the analogy

Match each item with the correct concept.

| Analogy | Concept |
|---|---|
| A calculator used by the AI system | __________ |
| A student learning from solved examples | __________ |
| A confident friend who guesses an answer | __________ |
| A waiter taking detailed food instructions | __________ |
| An assistant checking order status and creating a ticket | __________ |

Expected answers: tool, Machine Learning, hallucination, prompt, agent.

---

## 1. Artificial Intelligence

Artificial Intelligence is the broad field of making computer systems perform tasks associated with human intelligence.

### Everyday examples

- A phone recognizes a face before unlocking.
- An email service detects spam.
- A map recommends a route.
- A bank flags an unusual transaction.
- A shopping site recommends products.

### Simple analogy

AI is an umbrella. Machine Learning, Generative AI, computer vision, speech recognition, robotics, and agents are areas under that umbrella.

### Rule-based example

```python
temperature = 18

if temperature < 20:
    print("Turn on the heater")
else:
    print("The room is comfortable")
```

This program makes a decision, but it is not Machine Learning because it does not learn from data.

---

## 2. Machine Learning

Machine Learning allows a computer to learn patterns from examples instead of receiving a separate rule for every possible situation.

### Real-life example: spam detection

```text
Training emails + Correct labels -> Learned model
New email -> Learned model -> Spam probability
```

### Simple analogy

A child learns to distinguish apples and oranges after seeing many examples. A Machine Learning system similarly learns patterns from data.

### Traditional programming versus Machine Learning

```text
Traditional programming:
Rules + Input -> Output

Machine Learning training:
Examples + Correct outputs -> Learned model

Machine Learning prediction:
Learned model + New input -> Prediction
```

---

## 3. Predictive AI

Predictive AI estimates a category, score, or numerical value.

### Simple analogy

Predictive AI is like a weather forecast. It does not create a new story; it estimates what is likely to happen based on patterns.

Examples:

- Fraud or not fraud
- Expected house price
- Probability of equipment failure
- Expected customer demand

The main output is a prediction rather than newly written content.

---

## 4. Generative AI

Generative AI creates new content based on patterns learned from data.

### Simple analogy

Generative AI is like a chef. The prompt provides ingredients and instructions, and the model prepares a new dish-like output. A better recipe usually gives a better dish.

It can generate:

- Text
- Source code
- Images
- Audio
- Video
- Summaries
- Questions
- Explanations

### Example

Input:

```text
Write a polite reminder that invoice INV-104 is due tomorrow.
```

Possible output:

```text
This is a friendly reminder that invoice INV-104 is due tomorrow.
Please contact us if you need any clarification.
```

### Predictive versus generative systems

| Predictive system | Generative system |
|---|---|
| Predicts spam or not spam | Drafts a response to the email |
| Predicts a house price | Writes a property description |
| Detects a product defect | Produces an inspection summary |
| Predicts customer churn | Creates a retention-message draft |

---

## 5. Large Language Model

A Large Language Model, or LLM, is trained on a large amount of text. It learns relationships between tokens and generates a likely continuation based on the prompt and context.

### Analogy: powerful autocomplete

A normal autocomplete completes a word or short phrase. An LLM can continue paragraphs, follow instructions, summarize, classify, change tone, and generate code.

### Limitation

An LLM produces likely language. It does not automatically verify that every claim is true.

---

## 6. Prompt

A prompt is the instruction or input supplied to a model.

Weak prompt:

```text
Write an email.
```

Improved prompt:

```text
Write a polite email to a customer explaining that delivery will be delayed
by two days. Include the revised date, use fewer than 100 words, and do not
promise compensation.
```

A prompt resembles instructions given to a new employee. Clear instructions usually produce a more useful result.

---

## 7. Token

A token is like a Lego block. A sentence is broken into small pieces before the model processes it.

A token is a small unit of text processed by a language model. It may be a word, part of a word, punctuation mark, or number.

More input and output tokens generally mean more processing and potentially more cost. Beginners do not need to count tokens manually for these exercises.

---

## 8. Context

Context is like the open notebook available during an exam. The model can use what is present in the current interaction, but it should not guess missing private facts.

Context is the information available to the model while it generates a response.

Without useful context:

```text
What should I study next?
```

With useful context:

```text
I understand Python variables and if statements, but I have not learned
loops. What should I study next? Include one small exercise.
```

Good context reduces ambiguity.

---

## 9. Response

A response is the content generated by the model. A response can be:

- Correct and useful
- Correct but too complex
- Fluent but incorrect
- Incomplete
- In the wrong format
- Unsupported by the supplied information

Evaluate factual quality and instruction following separately.

---

## 10. Hallucination

A hallucination is like a confident classmate who gives an answer without checking the textbook. The answer may sound correct, but it still needs verification.

A hallucination occurs when a model produces incorrect or unsupported information while sounding confident.

Example:

```text
What time does a small local shop close today?
```

Without current verified data, the model may invent a time.

Better practice:

- Use approved documents or current databases.
- Ask for missing information.
- Use calculators for arithmetic.
- Verify claims using authoritative sources.
- Permit the system to say that information is unavailable.

---

## 11. Grounding

Grounding connects the model to reliable information such as an approved document, database, API, or current search result.

Analogy:

- Ungrounded answer: responding from memory
- Grounded answer: checking an approved reference before responding

---

## 12. Tool

A tool is like a calculator, attendance register, or office machine given to an assistant. The assistant can request it, but the system controls how it is used.

A tool is a controlled function, calculator, database query, file reader, or API available to an AI system.

Example request:

```text
What is 256 multiplied by 47?
```

A calculator tool should perform the arithmetic because it is deterministic and verifiable.

---

## 13. Memory or state

Memory or state is like sticky notes used while completing a task. It helps the system remember useful details without asking again.

Memory or state stores information needed across turns or steps.

First message:

```text
My order number is ORD-1002.
```

Second message:

```text
Has it been delivered?
```

A system with state can connect the second message to the order number in the first.

---

## 14. Chatbot

A chatbot primarily conducts a conversation. It may answer questions and maintain chat history, but it does not necessarily use tools or take actions.

---

## 15. Workflow

A workflow follows a predefined sequence.

```text
Collect form -> Validate fields -> Save data -> Send confirmation
```

The steps are known in advance.

---

## 16. Agent

An agent works toward a goal by selecting actions or tools, observing results, and deciding what to do next.

Analogy:

- Chatbot: an employee who answers questions
- Workflow: an employee who follows a fixed checklist
- Agent: an employee who selects approved tools according to the situation

---

# Real-life Generative AI applications

## Customer communication

- Draft delivery-delay messages
- Rewrite a response in a friendlier tone
- Prepare appointment reminders
- Generate FAQ drafts

Human staff must verify names, dates, policies, promises, and private data.

## Marketing

- Product-description drafts
- Campaign ideas
- Social-media captions
- Advertisement variations
- Website headings

Review unsupported claims, brand tone, copyright concerns, and legal requirements.

## Meetings

- Summarize notes
- Extract decisions
- Identify action items
- Create a follow-up draft

The model should not invent an owner or deadline that was not recorded.

## Software development

- Explain source code
- Draft a function
- Generate unit-test ideas
- Produce documentation
- Suggest debugging steps

Generated code requires review, testing, and security checks.

## Human resources

- Draft job descriptions
- Rewrite internal announcements
- Summarize policy text
- Suggest interview questions

AI should not make unreviewed employment decisions.

## Healthcare administration

- Reformat administrative notes
- Draft appointment instructions
- Summarize public guidance

Clinical diagnosis and treatment decisions require qualified professionals and approved systems.

## Finance and operations

- Draft invoice reminders
- Summarize operational reports
- Explain a variance using supplied figures
- Rewrite a procedure

The model must not invent figures or approve transactions.

---

# Generative AI coding labs

## Lab 1: First Generative AI program

Create `first_genai.py`:

```python
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

interaction = client.interactions.create(
    model=model,
    input="Explain Generative AI in three simple sentences.",
)

print(interaction.output_text)
```

Run:

```bash
python first_genai.py
```

### Student experiments

Change the prompt to:

```text
Explain Generative AI to a 10-year-old using a drawing assistant analogy.
```

Then try:

```text
Explain Generative AI using exactly four bullet points.
```

Observe how audience and output format alter the response.

---

# Prompt engineering

Prompt engineering means writing clear instructions that guide a model toward a useful output.

## A five-part prompt structure

```text
Role
Task
Context
Constraints
Output format
```

Example:

```text
Role:
You are a customer-service writing assistant.

Task:
Draft a response about a delayed package.

Context:
The expected delivery date moved from Monday to Wednesday.

Constraints:
Use fewer than 120 words. Do not blame another company. Do not promise compensation.

Output format:
Subject line followed by the email body.
```

## Why the parts matter

| Part | Question answered |
|---|---|
| Role | What perspective should the model use? |
| Task | What exactly should it do? |
| Context | What facts and audience details matter? |
| Constraints | What rules or limits apply? |
| Output format | How should the answer be organized? |

## Lab 2: Compare prompts

Create `prompt_comparison.py`:

```python
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

prompts = [
    "Explain machine learning.",
    "Explain machine learning to a 12-year-old.",
    """
    Explain machine learning to a first-year engineering student.
    Use one everyday analogy and exactly four bullet points.
    Avoid equations and advanced terminology.
    End with one question that checks understanding.
    """,
]

for number, prompt in enumerate(prompts, start=1):
    interaction = client.interactions.create(model=model, input=prompt)
    print(f"\n--- RESPONSE {number} ---")
    print(interaction.output_text)
```

Run:

```bash
python prompt_comparison.py
```

### Student task

Write a structured prompt that explains phishing to office employees. Include:

- A role
- The target audience
- Two constraints
- One real-life example
- A required output format

---

## Lab 3: Quiz generator

Create `quiz_generator.py`:

```python
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

topic = input("Enter a quiz topic: ").strip()
difficulty = input("Enter difficulty (beginner/intermediate): ").strip()
number_of_questions = int(input("Number of questions: "))

prompt = f"""
You are a teacher preparing a short quiz.

Create {number_of_questions} multiple-choice questions about {topic}.
Difficulty: {difficulty}

Requirements:
- Provide four options labelled A, B, C, and D.
- Only one option must be correct.
- Put all answers after the final question.
- Give a one-sentence explanation for each answer.
- Do not include facts that you are uncertain about.
"""

interaction = client.interactions.create(model=model, input=prompt)
print(interaction.output_text)
```

Run:

```bash
python quiz_generator.py
```

### Experiment

Generate quizzes for the same topic at two difficulty levels. Check whether the questions actually change in difficulty.

---

## Lab 4: Text summarizer

Create `text_summarizer.py`:

```python
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

meeting_note = """
The operations team reviewed delayed customer orders. Most delays were
caused by incomplete addresses and inventory mismatches. The team decided
to confirm addresses before dispatch and reconcile inventory every evening.
Arun will update the order form by Friday. Meera will prepare the inventory
report template, but no deadline was assigned to her task.
"""

prompt = f"""
Summarize the meeting note below.

Return exactly these sections:
1. Main issue
2. Decisions
3. Action items with owner and stated deadline

Rules:
- Do not add information that is absent from the note.
- Write "Not specified" when a deadline is missing.

Meeting note:
{meeting_note}
"""

interaction = client.interactions.create(model=model, input=prompt)
print(interaction.output_text)
```

Run:

```bash
python text_summarizer.py
```

### Verification task

Check whether the response assigns a deadline to Meera. The source states that her deadline was not specified. Any invented deadline is a hallucination.

---

## Lab 5: Professional email assistant

Create `professional_email.py`:

```python
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

customer_name = input("Customer name: ").strip()
original_date = input("Original delivery date: ").strip()
revised_date = input("Revised delivery date: ").strip()

prompt = f"""
You are a professional customer-service writing assistant.

Draft a delivery-delay email to {customer_name}.
The original delivery date was {original_date}.
The revised delivery date is {revised_date}.

Requirements:
- Include a concise subject line.
- Use fewer than 120 words.
- Apologize once.
- Do not blame the courier.
- Do not promise compensation or a refund.
- Preserve the dates exactly as supplied.
"""

interaction = client.interactions.create(model=model, input=prompt)
print(interaction.output_text)
```

Run:

```bash
python professional_email.py
```

### Experiment

Run the program using formal, friendly, and concise wording in the prompt. Verify that the delivery dates remain unchanged.

---

## Lab 6: Conversation memory

Create `conversation_memory.py`:

```python
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

first = client.interactions.create(
    model=model,
    input=(
        "My name is Anu. I understand Python variables and if statements, "
        "but I have not learned loops. Remember this for the next response."
    ),
)

print("FIRST RESPONSE")
print(first.output_text)

second = client.interactions.create(
    model=model,
    previous_interaction_id=first.id,
    input=(
        "Recommend the next topic I should learn and give me one small "
        "practice exercise."
    ),
)

print("\nSECOND RESPONSE")
print(second.output_text)
```

Run:

```bash
python conversation_memory.py
```

The second request uses `previous_interaction_id` to continue from the first interaction.

> Conversation memory alone does not make a system agentic. An agent normally has a goal, tools, decisions, observations, and a controlled action loop.

---

# Local Streamlit Generative AI app

Streamlit converts a Python script into a local browser interface. This section is for local-machine use only.

Create `streamlit_genai_app.py`:

```python
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to the .env file.")
    st.stop()

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

st.set_page_config(page_title="Beginner GenAI App", page_icon="✨")
st.title("✨ Beginner Generative AI App")
st.caption("Choose a task, provide input, and review the generated response.")

task = st.selectbox(
    "Choose a task",
    ["Explain a topic", "Summarize text", "Draft a professional email"],
)

if task == "Explain a topic":
    topic = st.text_input("Topic", "Generative AI")
    audience = st.selectbox(
        "Audience", ["school student", "beginner", "engineering student"]
    )
    user_input = f"""
Explain {topic} to a {audience}.
Use one analogy, four bullet points, and one review question.
"""
elif task == "Summarize text":
    source_text = st.text_area(
        "Text to summarize",
        height=180,
        placeholder="Paste a paragraph or meeting note here.",
    )
    user_input = f"""
Summarize the text below using three bullet points.
Do not add information that is absent from the source.

Source text:
{source_text}
"""
else:
    purpose = st.text_input("Purpose", "Inform a customer about a delay")
    facts = st.text_area(
        "Facts that must be preserved",
        "Original date: 15 July\nRevised date: 17 July",
        height=120,
    )
    user_input = f"""
Draft a concise professional email for this purpose: {purpose}
Use only the supplied facts. Do not promise compensation.
Include a subject line.

Facts:
{facts}
"""

if st.button("Generate response", type="primary"):
    if not user_input.strip():
        st.warning("Enter the required information first.")
    else:
        try:
            with st.spinner("Generating..."):
                interaction = client.interactions.create(
                    model=model,
                    input=user_input,
                )
            st.subheader("Generated response")
            st.markdown(interaction.output_text)
            st.info("Verify important facts before using the response.")
        except Exception as error:
            st.error(f"The request failed: {error}")
```

Run:

```bash
streamlit run streamlit_genai_app.py
```

The app provides three Generative AI tasks:

1. Explain a topic
2. Summarize text
3. Draft a professional email

### Student experiments

- Add a new task called **Generate quiz**.
- Add a slider for response length.
- Add a checkbox named **Use simple language**.
- Add a warning whenever the user leaves the input blank.

### What Streamlit adds

- Text inputs
- Drop-down menus
- Buttons
- Browser-based output
- Error and warning messages

It does not change the model itself; it provides a user interface around the Python logic.

---

# Understanding Agentic AI

## Definition

Agentic AI refers to systems that work toward a goal by deciding what action to take, selecting tools, observing results, and continuing until the goal is completed or the system must stop.

## Chatbot versus agent

| Chatbot | Agent |
|---|---|
| Primarily produces conversational responses | Attempts to complete a goal |
| Often responds once | May perform several steps |
| May use only prompt context | Can use tools and external data |
| Does not necessarily take actions | Can call approved functions or APIs |
| May store chat history | Often stores task state and tool results |

## Core components

### Goal

The outcome the agent should achieve.

### Model

The LLM interprets the request and decides which tool may help.

### Tools

Approved capabilities such as:

- Calculator
- Course-note lookup
- Deadline lookup
- Order-status database
- Ticket creation

### State

Information stored during the task, such as a subject, order number, prior tool result, or current step.

### Action

A selected tool call with structured arguments.

### Observation

The result returned by the tool.

### Agent loop

```text
Receive goal
    -> Decide next action
    -> Select a tool
    -> Execute the tool
    -> Observe the result
    -> Continue or finish
```

### Guardrails

Rules that reduce risk:

- Allow only approved tools
- Validate arguments
- Limit agent rounds
- Require confirmation for consequential actions
- Log important tool calls
- Avoid exposing secrets

## Function calling

Function calling allows the model to request a declared function and generate structured arguments.

The model does not directly execute a local Python function. The application must:

1. Inspect the requested function name.
2. Check that it is allowed.
3. Validate its arguments.
4. Execute the Python function.
5. Return the result to the model.
6. Request a final user-facing answer.

---

# Real-life Agentic AI applications

## Academic support

An agent can retrieve approved course notes, check deadlines, perform calculations, and prepare a study response.

## Customer support

An agent can identify an order number, check order status, evaluate a return rule, and create a support ticket.

## IT service desk

An agent can classify an incident, retrieve an approved solution, collect device details, and open a ticket.

## Sales support

An agent can extract requirements, check product-fit rules, record a lead, and draft a follow-up response.

## Meeting assistant

An agent can check available time slots, suggest times, prepare an agenda, and create a draft invitation. Sending the invitation should require confirmation.

## Invoice processing

An agent can extract invoice fields, compare them with purchase-order data, identify exceptions, and route the case to finance staff.

## Inventory management

An agent can check stock, identify low-stock products, calculate a suggested reorder, and create a draft purchase request. Actual purchasing should follow approval limits.

---

# Agentic AI foundation labs

## Lab 7: Rule-based agent without an AI API

Create `rule_based_agent.py`:

```python
def get_college_information(topic: str) -> str:
    information = {
        "library": "The library is open from 9:00 AM to 6:00 PM.",
        "laboratory": "The computer laboratory is in Block B.",
        "examination": "Students must carry their identity card.",
    }
    return information.get(topic.lower(), "Information is not available.")


def simple_agent(user_request: str) -> str:
    request = user_request.lower()

    if "library" in request:
        return get_college_information("library")
    if "laboratory" in request or "lab" in request:
        return get_college_information("laboratory")
    if "exam" in request:
        return get_college_information("examination")

    return "I could not select a suitable tool."


if __name__ == "__main__":
    print(simple_agent("When does the library close?"))
    print(simple_agent("Where is the computer lab?"))
```

Run:

```bash
python rule_based_agent.py
```

### Why it is limited

- It depends on keywords.
- It cannot understand many alternative expressions.
- Tool selection is written manually.
- It does not perform multi-step reasoning.

It is still useful because it demonstrates routing between user requests and functions.

---

## Lab 8: Calculator agent with function calling

Create `calculator_agent.py`:

```python
import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def calculate(number1: float, number2: float, operation: str) -> dict:
    if operation == "add":
        answer = number1 + number2
    elif operation == "subtract":
        answer = number1 - number2
    elif operation == "multiply":
        answer = number1 * number2
    elif operation == "divide":
        if number2 == 0:
            return {"error": "Division by zero is not allowed."}
        answer = number1 / number2
    else:
        return {"error": f"Unsupported operation: {operation}"}

    return {"operation": operation, "answer": answer}


calculator_tool = {
    "type": "function",
    "name": "calculate",
    "description": "Performs one basic arithmetic operation on two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "number1": {"type": "number"},
            "number2": {"type": "number"},
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
        },
        "required": ["number1", "number2", "operation"],
    },
}


def run_calculator_agent(user_request: str) -> str:
    interaction = client.interactions.create(
        model=model,
        input=user_request,
        tools=[calculator_tool],
    )

    function_call = next(
        (step for step in interaction.steps if step.type == "function_call"),
        None,
    )

    if function_call is None:
        return interaction.output_text

    if function_call.name != "calculate":
        raise ValueError(f"Tool not allowed: {function_call.name}")

    result = calculate(**function_call.arguments)

    final_interaction = client.interactions.create(
        model=model,
        previous_interaction_id=interaction.id,
        tools=[calculator_tool],
        input=[
            {
                "type": "function_result",
                "name": function_call.name,
                "call_id": function_call.id,
                "result": [
                    {"type": "text", "text": json.dumps(result)}
                ],
            }
        ],
    )

    return final_interaction.output_text


if __name__ == "__main__":
    request = (
        "A company received 18 orders in the morning and 24 orders "
        "in the evening. What is the total?"
    )
    print(run_calculator_agent(request))
```

Run:

```bash
python calculator_agent.py
```

### Observe

The model should select the `calculate` tool and produce structured arguments. Python performs the arithmetic and returns the result to the model.

### Student experiments

- Change the problem to subtraction.
- Test division by zero.
- Ask a non-mathematical question and observe whether the tool is selected.

---

# Academic Agent: Study Assistant

## Problem

A student may ask combined questions such as:

```text
Explain Generative AI using approved notes, add my morning and evening study
time, and tell me the assignment deadline.
```

A normal one-shot chatbot may answer from model memory. The Study Assistant Agent has controlled tools for:

1. Approved course notes
2. Stored assignment deadlines
3. Arithmetic

## Create the academic agent

Create `study_assistant_agent.py`:

```python
import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

COURSE_NOTES = {
    "python": (
        "Python is a high-level programming language with readable syntax. "
        "Beginner topics include variables, conditions, loops, functions, "
        "and lists."
    ),
    "generative ai": (
        "Generative AI creates new content such as text, images, audio, "
        "and code by learning patterns from data. Important limitations "
        "include hallucination, bias, and privacy risk."
    ),
    "agentic ai": (
        "Agentic AI systems pursue goals by selecting tools, observing "
        "results, maintaining state, and deciding the next action."
    ),
}

ASSIGNMENT_DEADLINES = {
    "python": "Friday at 5:00 PM",
    "generative ai": "Wednesday at 4:00 PM",
    "agentic ai": "Monday at 10:00 AM",
}


def get_course_notes(topic: str) -> dict:
    notes = COURSE_NOTES.get(topic.lower())
    if notes is None:
        return {"found": False, "topic": topic}
    return {"found": True, "topic": topic, "notes": notes}


def get_assignment_deadline(subject: str) -> dict:
    deadline = ASSIGNMENT_DEADLINES.get(subject.lower())
    if deadline is None:
        return {"found": False, "subject": subject}
    return {"found": True, "subject": subject, "deadline": deadline}


def calculate(number1: float, number2: float, operation: str) -> dict:
    if operation == "add":
        answer = number1 + number2
    elif operation == "subtract":
        answer = number1 - number2
    elif operation == "multiply":
        answer = number1 * number2
    elif operation == "divide":
        if number2 == 0:
            return {"error": "Division by zero is not allowed."}
        answer = number1 / number2
    else:
        return {"error": f"Unsupported operation: {operation}"}
    return {"operation": operation, "answer": answer}


notes_tool = {
    "type": "function",
    "name": "get_course_notes",
    "description": "Retrieves approved introductory notes for a subject.",
    "parameters": {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
}

deadline_tool = {
    "type": "function",
    "name": "get_assignment_deadline",
    "description": "Retrieves the stored assignment deadline for a subject.",
    "parameters": {
        "type": "object",
        "properties": {"subject": {"type": "string"}},
        "required": ["subject"],
    },
}

calculator_tool = {
    "type": "function",
    "name": "calculate",
    "description": "Performs one basic arithmetic operation on two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "number1": {"type": "number"},
            "number2": {"type": "number"},
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
        },
        "required": ["number1", "number2", "operation"],
    },
}

AVAILABLE_FUNCTIONS = {
    "get_course_notes": get_course_notes,
    "get_assignment_deadline": get_assignment_deadline,
    "calculate": calculate,
}

TOOLS = [notes_tool, deadline_tool, calculator_tool]


def run_study_agent(user_request: str, max_rounds: int = 5) -> str:
    current_input = f"""
You are a beginner-friendly academic study assistant.
Use the available tools whenever the student asks for stored notes,
an assignment deadline, or arithmetic.

Rules:
- Do not invent course notes or deadlines.
- Explain the final response using simple language.
- Mention when requested information is unavailable.
- Do not perform more than {max_rounds} tool-use rounds.

Student request:
{user_request}
"""
    previous_interaction_id = None

    for _ in range(max_rounds):
        request = {"model": model, "input": current_input, "tools": TOOLS}
        if previous_interaction_id is not None:
            request["previous_interaction_id"] = previous_interaction_id

        interaction = client.interactions.create(**request)
        function_calls = [
            step for step in interaction.steps if step.type == "function_call"
        ]

        if not function_calls:
            return interaction.output_text

        function_results = []
        for function_call in function_calls:
            function_name = function_call.name

            if function_name not in AVAILABLE_FUNCTIONS:
                result = {"error": f"Tool not allowed: {function_name}"}
            else:
                try:
                    result = AVAILABLE_FUNCTIONS[function_name](
                        **function_call.arguments
                    )
                except (TypeError, ValueError) as error:
                    result = {
                        "error": "Invalid tool arguments.",
                        "details": str(error),
                    }

            function_results.append(
                {
                    "type": "function_result",
                    "name": function_name,
                    "call_id": function_call.id,
                    "result": [
                        {"type": "text", "text": json.dumps(result)}
                    ],
                }
            )

        current_input = function_results
        previous_interaction_id = interaction.id

    return "The study agent stopped after reaching its safety step limit."


if __name__ == "__main__":
    request = (
        "Explain Generative AI using the approved course notes. "
        "I studied for 45 minutes in the morning and 35 minutes in the "
        "evening; calculate my total study time. Also tell me the "
        "Generative AI assignment deadline."
    )
    print(run_study_agent(request))
```

Run:

```bash
python study_assistant_agent.py
```

## What makes it agentic?

The system can:

1. Interpret a natural-language goal.
2. Decide which tools are required.
3. Call more than one tool.
4. Observe tool results.
5. Continue until it can prepare the final answer.
6. Stop after a maximum number of rounds.

## Academic safeguards

- Notes come only from `COURSE_NOTES`.
- Deadlines come only from `ASSIGNMENT_DEADLINES`.
- Arithmetic is performed by a calculator function.
- Unavailable information is reported rather than invented.
- The tool allow-list blocks unknown functions.
- The loop has a maximum number of rounds.

## Student extension ideas

Add one tool:

- `get_exam_room(subject)`
- `get_book_reference(topic)`
- `recommend_study_duration(difficulty)`
- `record_completed_topic(topic)`

For each extension:

1. Write the Python function.
2. Test it directly.
3. Write the tool declaration.
4. Add it to `AVAILABLE_FUNCTIONS`.
5. Add it to `TOOLS`.
6. Test it through a natural-language request.

---

# Local Streamlit Academic Agent

The Streamlit interface imports and uses the academic agent from `study_assistant_agent.py`. Keep both files in the same folder.

Create `streamlit_study_agent_app.py`:

```python
import streamlit as st

from study_assistant_agent import run_study_agent

st.set_page_config(page_title="Study Assistant Agent", page_icon="🎓")
st.title("🎓 Academic Study Assistant Agent")
st.caption(
    "The agent can use approved notes, assignment deadlines, and a calculator."
)

with st.sidebar:
    st.subheader("Try these requests")
    st.code("Explain Generative AI using the approved notes.")
    st.code("What is the Agentic AI assignment deadline?")
    st.code("Add 45 and 35 minutes and explain the total study time.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask the study agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Selecting tools and preparing the answer..."):
                response = run_study_agent(prompt)
            st.markdown(response)
    except Exception as error:
        response = f"The agent could not complete the request: {error}"
        st.error(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
```

Run:

```bash
streamlit run streamlit_study_agent_app.py
```

### Try these requests

```text
Explain Generative AI using the approved course notes.
```

```text
What is the Agentic AI assignment deadline?
```

```text
I studied 45 minutes in the morning and 35 minutes in the evening. Calculate the total.
```

```text
Explain Python, calculate 25 + 40, and tell me the Python assignment deadline.
```

### Important distinction

`st.session_state` stores the visible chat history. The agent itself uses Gemini interaction state while completing each tool-use request.

---

# Professional Agent: Customer Support

## Business problem

A small online store repeatedly receives questions such as:

- Where is my order?
- Is my delivered item within the return window?
- Can a support ticket be created?

The demonstration agent uses three simulated tools:

1. `get_order_status`
2. `check_return_eligibility`
3. `create_support_ticket`

The agent does not issue refunds. It reports rule-based eligibility and marks that human approval is required.

## Create the professional agent

Create `customer_support_agent.py`:

```python
import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

ORDERS = {
    "ORD-1001": {
        "status": "Packed",
        "days_since_purchase": 2,
        "delivered": False,
    },
    "ORD-1002": {
        "status": "Shipped; expected delivery tomorrow",
        "days_since_purchase": 5,
        "delivered": False,
    },
    "ORD-1003": {
        "status": "Delivered",
        "days_since_purchase": 10,
        "delivered": True,
    },
    "ORD-1004": {
        "status": "Delivered",
        "days_since_purchase": 45,
        "delivered": True,
    },
}

SUPPORT_TICKETS = []


def get_order_status(order_id: str) -> dict:
    order_id = order_id.upper()
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id, **order}


def check_return_eligibility(order_id: str, reason: str) -> dict:
    order_id = order_id.upper()
    order = ORDERS.get(order_id)

    if order is None:
        return {"eligible": False, "reason": "Order not found."}
    if not order["delivered"]:
        return {
            "eligible": False,
            "reason": "The order has not been delivered yet.",
        }

    return {
        "eligible": order["days_since_purchase"] <= 30,
        "order_id": order_id,
        "customer_reason": reason,
        "policy": "Returns are considered within 30 days of purchase.",
        "requires_human_approval": True,
    }


def create_support_ticket(order_id: str, issue: str, priority: str) -> dict:
    order_id = order_id.upper()
    priority = priority.lower()

    if order_id not in ORDERS:
        return {"created": False, "message": "Unknown order."}
    if priority not in {"low", "medium", "high"}:
        return {"created": False, "message": "Invalid priority."}

    ticket = {
        "ticket_id": f"TKT-{len(SUPPORT_TICKETS) + 1:04d}",
        "order_id": order_id,
        "issue": issue,
        "priority": priority,
        "status": "Open",
    }
    SUPPORT_TICKETS.append(ticket)
    return {"created": True, **ticket}


order_status_tool = {
    "type": "function",
    "name": "get_order_status",
    "description": "Looks up the current status of a customer order.",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}

return_tool = {
    "type": "function",
    "name": "check_return_eligibility",
    "description": (
        "Checks whether an order is within the return window. "
        "It does not approve or issue a refund."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["order_id", "reason"],
    },
}

ticket_tool = {
    "type": "function",
    "name": "create_support_ticket",
    "description": "Creates a simulated customer-support ticket.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "issue": {"type": "string"},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": ["order_id", "issue", "priority"],
    },
}

AVAILABLE_FUNCTIONS = {
    "get_order_status": get_order_status,
    "check_return_eligibility": check_return_eligibility,
    "create_support_ticket": create_support_ticket,
}

TOOLS = [order_status_tool, return_tool, ticket_tool]


def run_customer_support_agent(
    user_request: str, max_rounds: int = 5
) -> str:
    current_input = f"""
You are a customer-support assistant for a small online store.
Use the available tools when needed.

Rules:
- Never invent order information.
- Never say that a refund was approved or issued.
- Return eligibility requires human approval.
- Create a support ticket only when follow-up is requested or needed.
- Keep the final response polite, concise, and factual.

Customer request:
{user_request}
"""
    previous_interaction_id = None

    for _ in range(max_rounds):
        request = {"model": model, "input": current_input, "tools": TOOLS}
        if previous_interaction_id is not None:
            request["previous_interaction_id"] = previous_interaction_id

        interaction = client.interactions.create(**request)
        function_calls = [
            step for step in interaction.steps if step.type == "function_call"
        ]

        if not function_calls:
            return interaction.output_text

        function_results = []
        for function_call in function_calls:
            function_name = function_call.name

            if function_name not in AVAILABLE_FUNCTIONS:
                result = {"error": f"Tool not allowed: {function_name}"}
            else:
                try:
                    result = AVAILABLE_FUNCTIONS[function_name](
                        **function_call.arguments
                    )
                except (TypeError, ValueError) as error:
                    result = {
                        "error": "Invalid tool arguments.",
                        "details": str(error),
                    }

            function_results.append(
                {
                    "type": "function_result",
                    "name": function_name,
                    "call_id": function_call.id,
                    "result": [
                        {"type": "text", "text": json.dumps(result)}
                    ],
                }
            )

        current_input = function_results
        previous_interaction_id = interaction.id

    return "The support agent stopped after reaching its safety step limit."


if __name__ == "__main__":
    request = (
        "Order ORD-1003 arrived damaged. Check whether it is within the "
        "return window and create a high-priority ticket for follow-up."
    )
    print(run_customer_support_agent(request))
```

Run:

```bash
python customer_support_agent.py
```

## Test requests

### Order status

```text
Please check the status of order ORD-1002.
```

### Return check and ticket creation

```text
Order ORD-1003 arrived damaged. Check whether it is within the return window
and create a high-priority support ticket.
```

### Outside the return window

```text
Can I return order ORD-1004 because I no longer need it?
```

## Professional safeguards

- Only functions in `AVAILABLE_FUNCTIONS` can run.
- Order IDs are checked against simulated known data.
- Priority is restricted to low, medium, or high.
- The agent has a maximum number of rounds.
- It cannot approve or issue a refund.
- Return eligibility explicitly requires human approval.

## Why human approval matters

Consequential actions may affect money, legal obligations, customer rights, accounts, or reputation. An agent may prepare information or a draft action, but final authorization should follow business policy.

---

# Local Streamlit Professional Agent

Keep `streamlit_customer_support_app.py` and `customer_support_agent.py` in the same folder.

Create `streamlit_customer_support_app.py`:

```python
import streamlit as st

from customer_support_agent import SUPPORT_TICKETS, run_customer_support_agent

st.set_page_config(page_title="Customer Support Agent", page_icon="📦")
st.title("📦 Customer Support Agent")
st.caption("Demonstration with simulated order data—no real business action occurs.")

with st.sidebar:
    st.subheader("Sample order IDs")
    st.write("ORD-1001, ORD-1002, ORD-1003, ORD-1004")
    st.subheader("Example request")
    st.code(
        "Order ORD-1003 arrived damaged. Check return eligibility and "
        "create a high-priority ticket."
    )
    st.subheader("Created tickets")
    st.json(SUPPORT_TICKETS)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe the customer issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Checking tools and business rules..."):
                response = run_customer_support_agent(prompt)
            st.markdown(response)
            st.warning(
                "Refunds and final return decisions require human approval."
            )
    except Exception as error:
        response = f"The agent could not complete the request: {error}"
        st.error(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
```

Run:

```bash
streamlit run streamlit_customer_support_app.py
```

The interface shows:

- A chat input
- Sample order IDs
- An example request
- Simulated tickets created during the demonstration
- A human-approval warning

> The data is simulated. Restarting the Python process resets the in-memory orders and tickets. A production system would use authenticated databases, access controls, audit logs, monitoring, and transaction safeguards.

---

# Responsible and safe AI

## Hallucination

The model may create unsupported facts. Ground important answers in reliable sources and verify them.

## Privacy

Do not submit:

- Passwords
- API keys
- Government identifiers
- Private medical information
- Confidential institutional data
- Customer records without authorization

## Bias

Generated content may reproduce unfair patterns from data or instructions. Review decisions that affect people.

## Over-reliance

Students and professionals should understand, verify, and take responsibility for submitted work.

## Prompt injection

A document, website, or user message may contain instructions intended to manipulate an agent. Treat external content as data unless it is explicitly trusted as an instruction source.

## Unsafe tool execution

An agent should not automatically:

- Send money
- Approve refunds
- Delete files
- Publish content
- Change marks
- Send external emails
- Execute unrestricted operating-system commands

## Minimum agent safeguards

```text
Use an allow-list of tools.
Validate tool arguments.
Limit the number of rounds.
Require approval for consequential actions.
Log important tool calls and results.
Do not expose secrets.
Handle errors safely.
Use the least privilege necessary.
```

---

# Troubleshooting

## `ModuleNotFoundError: No module named 'google'`

Activate the correct virtual environment and install the SDK:

```bash
pip install -U google-genai
```

## `ModuleNotFoundError: No module named 'dotenv'`

```bash
pip install -U python-dotenv
```

## `ModuleNotFoundError: No module named 'streamlit'`

```bash
pip install -U streamlit
```

## API key missing

Check that `.env` contains:

```text
GEMINI_API_KEY=your_actual_key
```

Run the script from the folder containing `.env`.

## Wrong Python interpreter in VS Code

Use **Python: Select Interpreter** and select the interpreter inside `.venv`.

## PowerShell prevents activation

Use Command Prompt activation:

```bat
.venv\Scriptsctivate.bat
```

Alternatively, follow the institution's approved PowerShell execution-policy guidance.

## Streamlit command not found

Ensure the virtual environment is active. Then run:

```bash
python -m streamlit run streamlit_genai_app.py
```

## Rate-limit or quota error

- Wait before retrying.
- Reduce repeated requests.
- Use shorter prompts.
- Check the account's API usage and limits.

## Model name error

Available model names can change. Check the official Gemini model documentation and update `GEMINI_MODEL` in `.env`.

## Agent reaches its step limit

Possible causes:

- The request is ambiguous.
- A required tool is missing.
- A tool returns insufficient information.
- The model repeatedly selects tools.

Improve the tool descriptions, add validation, or simplify the request. Do not simply remove the safety limit.

## Streamlit app shows old content

- Save the `.py` file.
- Refresh the browser.
- Stop and rerun the Streamlit command when necessary.

---

# Review questions

1. What is the difference between Artificial Intelligence and Machine Learning?
2. How is Generative AI different from predictive AI?
3. What is an LLM?
4. What is a prompt?
5. Why is context important?
6. What is hallucination?
7. What is grounding?
8. How is a chatbot different from an agent?
9. What is a tool in Agentic AI?
10. Who executes the Python function requested by the model?
11. Why should an agent use a tool allow-list?
12. Why should an agent have a maximum number of rounds?
13. Why do refund decisions require human approval?
14. What is the role of Streamlit?
15. Why must `.env` be excluded from GitHub?

---

# Practical exit challenge

Choose one task:

## Challenge A: Extend the GenAI Streamlit app

Add a **Quiz generator** option with:

- Topic input
- Difficulty selector
- Number-of-questions slider
- Generated questions and answers

## Challenge B: Extend the academic agent

Add a tool that retrieves an examination room from approved data.

## Challenge C: Extend the professional agent

Add a tool that retrieves a store policy. The model must not invent policy text when the requested policy is absent.

Before demonstrating the extension, answer:

1. What data does the tool use?
2. What arguments does it accept?
3. What errors can occur?
4. What action requires human approval?
5. What prevents the agent from calling an unknown function?

---

# Glossary

| Term | Simple meaning |
|---|---|
| AI | Broad field of computer systems performing intelligent tasks |
| Machine Learning | Learning patterns from data |
| Predictive AI | Producing a category, score, or numerical estimate |
| Generative AI | Creating new content |
| LLM | Language model trained on a large amount of text |
| Prompt | Instruction supplied to a model |
| Token | Small unit of text processed by a model |
| Context | Information available during generation |
| Response | Content produced by the model |
| Hallucination | Incorrect or unsupported generated information |
| Grounding | Connecting a model to reliable external information |
| Tool | Approved function, API, database query, or capability |
| Function calling | Model requests a declared function with structured arguments |
| State | Information retained during a conversation or task |
| Workflow | Predefined sequence of steps |
| Agent | System that selects actions or tools to pursue a goal |
| Guardrail | Rule or control that limits unsafe behaviour |
| Streamlit | Python framework for creating interactive browser applications |

---

# Official references

- [Gemini API documentation](https://ai.google.dev/gemini-api/docs)
- [Gemini Interactions API getting started](https://ai.google.dev/gemini-api/docs/get-started)
- [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Gemini model documentation](https://ai.google.dev/gemini-api/docs/models)
- [Google GenAI SDK libraries](https://ai.google.dev/gemini-api/docs/libraries)
- [Streamlit installation](https://docs.streamlit.io/get-started/installation/command-line)
- [Run a Streamlit application](https://docs.streamlit.io/develop/concepts/architecture/run-your-app)
- [Streamlit chat elements](https://docs.streamlit.io/develop/api-reference/chat)

---

## Final summary

```text
Generative AI creates content.

Agentic AI uses a model, approved tools, state, observations, and actions
to work toward a goal.

Streamlit provides a simple browser interface around Python applications.

Reliable AI requires clear prompts, grounded information, controlled tools,
validated outputs, safety limits, and human oversight.
```
