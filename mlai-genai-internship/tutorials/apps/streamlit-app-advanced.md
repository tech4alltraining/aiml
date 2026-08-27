Below are **two advanced Streamlit GenAI app examples using Google Gemini API**:

1. **YouTube Video Summarizer using Gemini**
2. **Diagnostic Helper with Image Input using Gemini Vision**

Gemini supports video understanding, including public YouTube URLs, and can process images as multimodal input through the `generateContent` API. Google notes that YouTube URL support is currently a preview feature, with public-video and rate-limit restrictions. ([Google AI for Developers][1]) Gemini image input can be passed inline for smaller files or through the File API for larger/reusable files. ([Google AI for Developers][2])

---

# Common Setup

## Install libraries

```bash
pip install streamlit google-genai pillow
```

## Create `.streamlit/secrets.toml`

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

# Example 1: YouTube Video Summarization App

## Purpose

This app takes a **public YouTube video URL**, sends it to Gemini, and generates:

```text
Short summary
Detailed summary
Key points
Timestamps if available
Quiz questions
```

## App file name

```text
youtube_summarizer_app.py
```

## Code

```python
import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 YouTube Video Summarizer using Gemini")
st.write(
    "Enter a public YouTube video URL. Gemini will analyze the video and generate a structured summary."
)

# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "gemini-3.5-flash"

# --------------------------------------------------
# User inputs
# --------------------------------------------------

youtube_url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

summary_type = st.selectbox(
    "Choose output type",
    [
        "Short Summary",
        "Detailed Summary",
        "Key Points",
        "Timestamp-wise Summary",
        "Generate Quiz Questions",
        "Teaching Notes"
    ]
)

audience = st.selectbox(
    "Target audience",
    [
        "School Students",
        "College Students",
        "Beginners",
        "Technical Audience",
        "General Audience"
    ]
)

# --------------------------------------------------
# Prompt builder
# --------------------------------------------------

def build_prompt(summary_type, audience):
    return f"""
    You are an expert educational video summarizer.

    Analyze the YouTube video and produce the output for: {audience}.

    Required output type: {summary_type}

    Follow these rules:
    1. Use clear and simple language.
    2. Do not invent details that are not present in the video.
    3. If the video content is unclear, mention that clearly.
    4. Organize the answer with headings and bullet points.
    5. If timestamps are available or inferable, include them only when useful.

    Output format:
    - Title or topic of the video
    - Main summary
    - Important points
    - Practical learning outcome
    - If quiz mode is selected, generate 5 questions with answers
    """

# --------------------------------------------------
# Generate summary
# --------------------------------------------------

if st.button("Summarize Video"):
    if not youtube_url.strip():
        st.warning("Please enter a YouTube video URL.")
    else:
        prompt = build_prompt(summary_type, audience)

        with st.spinner("Analyzing video with Gemini..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=types.Content(
                        parts=[
                            types.Part(
                                file_data=types.FileData(
                                    file_uri=youtube_url
                                )
                            ),
                            types.Part(text=prompt)
                        ]
                    )
                )

                st.subheader("Generated Output")
                st.write(response.text)

            except Exception as e:
                st.error("An error occurred while processing the video.")
                st.write(str(e))

# --------------------------------------------------
# Notes for students
# --------------------------------------------------

st.divider()

st.subheader("Important Notes")
st.write(
    """
    - The YouTube video must be public.
    - Private or unlisted videos may not work.
    - Very long videos may take more time.
    - AI-generated summaries should be verified before academic or professional use.
    """
)
```

## Run the app

```bash
streamlit run youtube_summarizer_app.py
```

---

## How to Explain This App to Students

This app follows this workflow:

```text
User enters YouTube URL
        ↓
Streamlit collects URL
        ↓
Prompt is created
        ↓
Gemini receives video URL + instruction
        ↓
Gemini analyzes video content
        ↓
Summary is displayed in Streamlit
```

## Classroom Discussion

Ask students:

```text
What happens if the video is private?
Can Gemini summarize both audio and visual content?
Should we trust the summary without checking the video?
How can this app help teachers and students?
```

## Student Practice Tasks

Ask students to modify the app to add:

```text
1. A language selection option
2. A "Generate MCQ" option
3. A "Create lecture notes" option
4. A "Summarize in 100 words" option
5. A "Extract important technical terms" option
```

---

# Example 2: Diagnostic Helper with Image Input

## Purpose

This app allows the user to upload a medical-related image, such as:

```text
Skin image
X-ray image
Wound image
Eye image
General medical report screenshot
```

Gemini analyzes the image and gives **general observations**.

Important: This app must be presented as a **diagnostic helper**, not a doctor replacement.

---

## Safety Statement for Students

Tell students clearly:

> This app does not provide a confirmed diagnosis. It only gives educational observations from the image. A qualified doctor must make the final medical decision.

---

## App file name

```text
diagnostic_helper_app.py
```

## Code

```python
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Diagnostic Helper",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Diagnostic Helper using Gemini Vision")

st.warning(
    """
    This tool is for educational and assistive purposes only.
    It does not provide a confirmed medical diagnosis.
    Always consult a qualified healthcare professional.
    """
)

# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "gemini-3.5-flash"

# --------------------------------------------------
# User inputs
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a medical-related image",
    type=["jpg", "jpeg", "png", "webp"]
)

analysis_type = st.selectbox(
    "Choose analysis type",
    [
        "General Observation",
        "Possible Visual Findings",
        "Patient-Friendly Explanation",
        "Doctor-Review Checklist",
        "Urgency / Red Flag Screening"
    ]
)

additional_context = st.text_area(
    "Optional: Add patient context without personal identity",
    placeholder=(
        "Example: 45-year-old patient, skin rash for 3 days, mild itching. "
        "Do not enter name, phone number, address, or other personal details."
    )
)

# --------------------------------------------------
# Prompt builder
# --------------------------------------------------

def build_medical_prompt(analysis_type, additional_context):
    return f"""
    You are an AI medical image assistant.

    Important safety rules:
    - Do not provide a final diagnosis.
    - Do not prescribe medicines.
    - Do not suggest dosage.
    - Do not replace a doctor.
    - Clearly state that the output is only an AI-assisted observation.
    - Recommend consulting a qualified healthcare professional.
    - If there may be urgent red flags, advise immediate medical attention.

    Analysis type requested: {analysis_type}

    Additional context from user:
    {additional_context}

    Analyze the uploaded image and provide:

    1. Image quality note
       - Is the image clear enough?
       - Are there limitations?

    2. Visible observations
       - Describe only what is visible.
       - Avoid unsupported assumptions.

    3. Possible concerns
       - Mention broad possibilities, not a confirmed diagnosis.

    4. Questions for the patient/doctor
       - List useful questions a clinician may ask.

    5. Red flags
       - Mention signs that require urgent medical attention.

    6. Recommended next step
       - Suggest consulting a qualified doctor or specialist.

    Use simple and careful language.
    """

# --------------------------------------------------
# Image display and analysis
# --------------------------------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze Image"):
        image_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type

        prompt = build_medical_prompt(analysis_type, additional_context)

        with st.spinner("Analyzing image with Gemini..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        ),
                        prompt
                    ]
                )

                st.subheader("AI-Assisted Observation")
                st.write(response.text)

                st.error(
                    "Reminder: This is not a medical diagnosis. Please consult a qualified doctor."
                )

            except Exception as e:
                st.error("An error occurred while analyzing the image.")
                st.write(str(e))

else:
    st.info("Please upload an image to begin.")

# --------------------------------------------------
# Responsible AI section
# --------------------------------------------------

st.divider()

st.subheader("Responsible AI Notes")
st.write(
    """
    - Do not upload personally identifiable medical records.
    - Do not rely only on AI for medical decisions.
    - AI may miss important findings.
    - Image quality can affect the result.
    - Final decisions must be made by qualified healthcare professionals.
    """
)
```

## Run the app

```bash
streamlit run diagnostic_helper_app.py
```

---

## How to Explain This App to Students

This app uses **multimodal AI**, meaning the model can process more than one type of input.

Here the inputs are:

```text
Image + Text instruction + Optional patient context
```

The workflow is:

```text
User uploads image
        ↓
Streamlit reads image bytes
        ↓
Image + prompt sent to Gemini
        ↓
Gemini analyzes visual content
        ↓
AI-assisted observations displayed
```

## Why This Is More Advanced

Compared to a normal chatbot, this app uses:

```text
Image input
Medical safety prompt
User context
Structured output
Responsible AI warning
```

---

# Difference Between the Two Advanced Apps

| App                | Input                    | Gemini Capability Used | Output                           |
| ------------------ | ------------------------ | ---------------------- | -------------------------------- |
| YouTube Summarizer | YouTube URL + prompt     | Video understanding    | Summary, notes, quiz             |
| Diagnostic Helper  | Image + prompt + context | Image understanding    | Visual observations and guidance |

---

# Important Responsible AI Points

For the **YouTube summarizer**:

```text
The summary may miss details.
The app should not claim to replace watching the full video.
Copyright and educational fair-use policies should be respected.
```

For the **Diagnostic helper**:

```text
The app should not diagnose.
The app should not prescribe medicine.
The app should not provide dosage.
The app should clearly recommend doctor consultation.
The app should avoid storing sensitive patient data.
```

---

# Suggested Classroom Activity

## Activity 1: Improve YouTube Summarizer

Ask students to add:

```text
1. Summary length selection
2. Output language selection
3. MCQ generation
4. Timestamp-based notes
5. Export summary as text
```

## Activity 2: Improve Diagnostic Helper

Ask students to add:

```text
1. Symptom input field
2. Age group selection
3. Emergency warning section
4. Doctor checklist output
5. Privacy warning before upload
```

---

# Final Teaching Summary

These two apps demonstrate how Streamlit and Gemini can be combined to build advanced GenAI applications:

```text
Streamlit = Web interface
Gemini API = AI reasoning and multimodal understanding
```

The key learning outcomes for students are:

```text
How to connect Streamlit with Gemini API
How to send YouTube URLs to Gemini
How to send image input to Gemini
How to create structured prompts
How to design responsible GenAI applications
How to handle sensitive domains carefully
```

[1]: https://ai.google.dev/gemini-api/docs/video-understanding "Gemini API  |  Google AI for Developers"
[2]: https://ai.google.dev/gemini-api/docs/image-understanding "Gemini API  |  Google AI for Developers"
