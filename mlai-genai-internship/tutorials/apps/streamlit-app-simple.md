# Streamlit GenAI App Development using Gemini API

A **Streamlit GenAI app** is a web application built with Python where users can enter text, upload content, or chat with an AI model. The app sends the user input to a **Large Language Model**, such as **Gemini**, receives the generated response, and displays it in the browser.

In simple words:

> **Streamlit provides the user interface. Gemini API provides the intelligence.**

Google currently recommends the **Google GenAI SDK** for Gemini API development, and it can be installed in Python using `pip install google-genai`. The Gemini API supports content generation tasks such as text, code, image, audio, and tool-based generation. ([Google AI for Developers][1])

---

# 1. Basic Architecture of a Streamlit GenAI App

```text
User
 ↓
Streamlit Interface
 ↓
Prompt / User Input
 ↓
Gemini API / LLM
 ↓
Generated Response
 ↓
Streamlit Output
```

Example:

```text
Student enters:
"Explain machine learning in simple words."

Gemini responds:
"Machine learning is a method where computers learn patterns from data..."
```

---

# 2. Required Installation

Install the required libraries:

```bash
pip install streamlit google-genai
```

---

# 3. Store Gemini API Key Securely

Do **not** write your API key directly inside the Python file.

Create this folder and file:

```text
.streamlit/secrets.toml
```

Inside `secrets.toml`, write:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Streamlit provides `st.secrets` for reading secrets from a `secrets.toml` file, and it works like a dictionary-style interface for sensitive values such as API keys. ([Streamlit Docs][2])

---

# Example App 1: AI Text Generator App

## Purpose

This app takes a topic from the user and generates a short explanation using Gemini.

Example input:

```text
Artificial Intelligence
```

Example output:

```text
Artificial Intelligence is a branch of computer science that enables machines to perform tasks that normally require human intelligence...
```

---

## File Name

Save as:

```text
gemini_text_generator.py
```

## Code

```python
import streamlit as st
from google import genai

# -----------------------------------------
# Streamlit page settings
# -----------------------------------------

st.set_page_config(
    page_title="Gemini Text Generator",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Gemini AI Text Generator")

st.write(
    "Enter a topic and Gemini will generate a simple explanation."
)

# -----------------------------------------
# Gemini client
# -----------------------------------------

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# -----------------------------------------
# User input
# -----------------------------------------

topic = st.text_input("Enter a topic", placeholder="Example: Artificial Intelligence")

style = st.selectbox(
    "Choose explanation style",
    ["Simple", "Detailed", "For School Students", "For College Students"]
)

word_limit = st.slider(
    "Approximate word limit",
    min_value=50,
    max_value=300,
    value=120,
    step=50
)

# -----------------------------------------
# Generate response
# -----------------------------------------

if st.button("Generate Explanation"):
    if topic.strip() == "":
        st.warning("Please enter a topic.")
    else:
        prompt = f"""
        Explain the topic: {topic}

        Style: {style}
        Word limit: approximately {word_limit} words.

        Make the explanation clear and suitable for students.
        """

        with st.spinner("Generating response..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.subheader("Generated Explanation")
        st.write(response.text)
```

---

## Run the App

```bash
streamlit run gemini_text_generator.py
```

---

## Explanation for Students

| Part                 | Meaning                                  |
| -------------------- | ---------------------------------------- |
| `st.title()`         | Displays the app title                   |
| `st.text_input()`    | Takes topic from user                    |
| `st.selectbox()`     | Lets user choose explanation style       |
| `st.slider()`        | Lets user choose approximate word limit  |
| `genai.Client()`     | Connects the app to Gemini API           |
| `generate_content()` | Sends prompt to Gemini and gets response |
| `st.write()`         | Displays generated answer                |

---

## Classroom Practice

Ask students to test the app with topics such as:

```text
Machine Learning
Responsible AI
Data Preprocessing
Feature Engineering
Neural Networks
```

Then ask:

```text
Does changing the explanation style change the output?
Does increasing the word limit make the answer more detailed?
Is the answer always correct?
Should we verify important AI-generated content?
```

---

# Example App 2: Gemini Chatbot App

## Purpose

This app creates a simple chatbot where users can ask questions and Gemini replies.

Example:

```text
User: What is feature scaling?
AI: Feature scaling is a preprocessing technique used to bring numerical features into a common range...
```

---

## File Name

Save as:

```text
gemini_chatbot.py
```

## Code

```python
import streamlit as st
from google import genai

# -----------------------------------------
# Streamlit page settings
# -----------------------------------------

st.set_page_config(
    page_title="Gemini Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Gemini AI Chatbot")

st.write(
    "Ask any question. The chatbot uses Gemini API to generate responses."
)

# -----------------------------------------
# Gemini client
# -----------------------------------------

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# -----------------------------------------
# Initialize chat history
# -----------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------
# Display previous chat messages
# -----------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------------------
# User input
# -----------------------------------------

user_prompt = st.chat_input("Type your question here...")

if user_prompt:
    # Store and display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.write(user_prompt)

    # Build conversation context
    conversation = ""

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        conversation += f"{role}: {content}\n"

    prompt = f"""
    You are a helpful AI tutor.
    Answer clearly and simply.

    Conversation so far:
    {conversation}

    Assistant:
    """

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            assistant_reply = response.text
            st.write(assistant_reply)

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

# -----------------------------------------
# Clear chat button
# -----------------------------------------

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
```

---

## Run the App

```bash
streamlit run gemini_chatbot.py
```

---

## Explanation for Students

| Component            | Purpose                              |
| -------------------- | ------------------------------------ |
| `st.chat_input()`    | Takes chat message from user         |
| `st.chat_message()`  | Displays user and assistant messages |
| `st.session_state`   | Stores chat history                  |
| `generate_content()` | Sends conversation to Gemini         |
| `Clear Chat` button  | Resets the conversation              |

---

# 4. Difference Between the Two Apps

| App            | Type                  | Input                    | Output                   |
| -------------- | --------------------- | ------------------------ | ------------------------ |
| Text Generator | Single-turn GenAI app | Topic, style, word limit | Generated explanation    |
| Chatbot        | Multi-turn GenAI app  | User questions           | Conversational responses |

The **Text Generator App** is useful for:

```text
Generating explanations
Writing summaries
Creating teaching notes
Generating short content
```

The **Chatbot App** is useful for:

```text
Question answering
Student doubt clearing
Interactive tutoring
General conversation
```

---

# 5. Important GenAI Concepts Used

## Prompt

A **prompt** is the instruction given to the LLM.

Example:

```text
Explain artificial intelligence for school students in 100 words.
```

## LLM

An **LLM**, or Large Language Model, is an AI model trained to understand and generate human language.

Examples:

```text
Gemini
GPT
Llama
Mistral
Claude
Qwen
```

## API

An **API** allows the Streamlit app to communicate with Gemini.

The app sends:

```text
User prompt
```

Gemini returns:

```text
Generated response
```

## Session State

`st.session_state` stores information while the app is running.

In the chatbot app, it stores previous messages so the conversation does not disappear after every input.

---

# 6. Suggested Classroom Activity

## Activity 1: Modify the Text Generator

Ask students to add:

```text
Language selection
Tone selection
Bullet-point output
Example generation
Summary mode
```

Example prompt modification:

```text
Explain the topic in Malayalam.
Explain the topic using three examples.
Explain the topic in bullet points.
```

---

## Activity 2: Modify the Chatbot

Ask students to create different chatbot roles:

```text
AI Tutor
Python Teacher
Career Guide
Interview Coach
Medical Information Assistant
```

Example system instruction:

```text
You are a Python tutor. Explain concepts with simple examples.
```

---

# 7. Responsible AI Points to Tell Students

When developing GenAI apps:

```text
Do not expose API keys
Do not enter sensitive personal data
Verify important answers
Mention that AI can make mistakes
Use human supervision for medical, legal, and financial advice
Check model output before using it professionally
```

For example, a Gemini chatbot can explain symptoms, but it should not replace a qualified doctor.

---

# 8. Final Summary

A Streamlit GenAI app has two main parts:

```text
Streamlit → User Interface
Gemini API → AI Intelligence
```

The basic development flow is:

```text
Create Streamlit UI
        ↓
Collect user prompt
        ↓
Send prompt to Gemini API
        ↓
Receive LLM response
        ↓
Display response in Streamlit
```

The two simple example apps are:

```text
1. Gemini AI Text Generator
2. Gemini AI Chatbot
```

These examples help students understand how modern AI applications combine a web interface with a powerful LLM through an API.

[1]: https://ai.google.dev/gemini-api/docs/libraries?utm_source=chatgpt.com "Gemini API libraries - Google AI for Developers"
[2]: https://docs.streamlit.io/develop/api-reference/connections/st.secrets?utm_source=chatgpt.com "st.secrets - Streamlit Docs"
