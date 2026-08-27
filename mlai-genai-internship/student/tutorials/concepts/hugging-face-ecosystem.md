# Hugging Face Ecosystem

**Hugging Face** is a platform and set of open-source tools used to find, use, train, fine-tune, evaluate, and deploy machine learning and Generative AI models.

In simple classroom language:

> **Hugging Face is like GitHub for AI models, datasets, and AI demo applications.**

It provides:

```text
Models
Datasets
Tokenizers
Training tools
Evaluation tools
Fine-tuning tools
Deployment tools
Demo app hosting
```

The Hugging Face Hub hosts model repositories, datasets, and Spaces, and the official documentation describes it as a platform for discovering, uploading, and collaborating on models, datasets, and AI apps. ([Hugging Face][1])

---

# 0. Before you start

**Module 5 · Session 5.1.** All commands assume the `genai` conda environment from the [Student Handbook](../../setup-guide.md).

```bash
conda activate genai
pip install transformers torch datasets evaluate gradio
```

> ⚠️ **This is a 2-3 GB download.** Install it the **evening before** the session, not during it. Twenty students downloading PyTorch simultaneously on classroom wifi will not end well.

Create a free account at [huggingface.co/join](https://huggingface.co/join). You do not need a token for the models in this tutorial, but you will need the account to use Spaces.

## Verify PyTorch works before the session

Run this. It takes two seconds and saves an hour of confusion:

```bash
conda activate genai
python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

> ⚠️ **Apple Silicon Mac users:** if this prints `torch 2.2.2` (or older), your conda environment is an Intel/Rosetta build. PyTorch stopped publishing Intel macOS wheels after 2.2.2, and that version cannot work with NumPy 2 — every example below will fail with `RuntimeError: Numpy is not available`. The fix is in the [Student Handbook troubleshooting section](../../troubleshooting.md#environment-problems).

**The first time you run each example**, the model is downloaded (a few hundred MB) and cached in `~/.cache/huggingface`. Later runs start immediately.

---

# 1. Major Components of the Hugging Face Ecosystem

## A. Hugging Face Hub

The **Hub** is the central platform where users can find and share:

```text
Pretrained models
Datasets
Demo applications
Model cards
Dataset cards
Fine-tuned models
```

Examples of models available on the Hub:

```text
BERT
RoBERTa
DistilBERT
Llama
Mistral
Qwen
Whisper
Stable Diffusion
```

The Hub works with Git-based repositories, which means models and datasets can be version-controlled and shared like software projects. ([Hugging Face][1])

---

## B. Transformers Library

The **Transformers** library is used to run and fine-tune pretrained transformer models.

It supports tasks such as:

```text
Text classification
Question answering
Summarization
Translation
Text generation
Token classification
Speech processing
Vision-language tasks
```

The official Transformers documentation says that the library provides access to more than 1M model checkpoints on the Hugging Face Hub. ([Hugging Face][2])

Simple explanation for students:

> Transformers allows us to use powerful pretrained AI models with only a few lines of Python code.

---

## C. Datasets Library

The **Datasets** library helps users easily load and process datasets.

It supports datasets for:

```text
Natural Language Processing
Computer Vision
Audio
Multimodal AI
```

The official Hugging Face Datasets documentation describes it as a library for accessing and sharing AI datasets for audio, computer vision, and NLP tasks. ([Hugging Face][3])

Example datasets:

```text
IMDb movie reviews
SQuAD question answering dataset
GLUE benchmark
Common Voice speech dataset
Image classification datasets
```

---

## D. Tokenizers

A **tokenizer** converts text into smaller units called tokens.

Example:

```text
Input sentence:
I love machine learning.

Tokens:
["I", "love", "machine", "learning", "."]
```

Why tokenization is needed:

```text
AI models do not directly understand raw text.
Text must first be converted into numerical token IDs.
```

In NLP, tokenization is one of the first steps before giving text to a transformer model.

---

## E. Pipelines

The **pipeline** API is one of the easiest ways to use Hugging Face models.

It hides many technical steps and allows students to quickly try AI tasks.

Example tasks:

```text
Sentiment analysis
Text generation
Question answering
Summarization
Translation
Image classification
Automatic speech recognition
```

For beginners, `pipeline()` is the best starting point.

---

## F. Diffusers Library

The **Diffusers** library is used for generative models such as diffusion models.

It supports tasks such as:

```text
Text-to-image generation
Image-to-image generation
Image inpainting
Audio generation
3D molecular structure generation
```

The official Diffusers documentation describes it as a modular toolbox for pretrained diffusion models for generating images, audio, and even 3D molecular structures. ([Hugging Face][4])

Examples:

```text
Stable Diffusion
SDXL
Flux-based models
Image editing models
```

---

## G. Evaluate Library

The **Evaluate** library is used to calculate model performance metrics.

Examples:

```text
Accuracy
Precision
Recall
F1-score
BLEU
ROUGE
WER
```

This is useful when comparing model performance after training or fine-tuning.

---

## H. PEFT

**PEFT** means **Parameter-Efficient Fine-Tuning**.

It allows users to fine-tune large models by updating only a small number of additional parameters instead of retraining the full model.

The Hugging Face PEFT documentation explains that PEFT adapts large pretrained models to downstream tasks while reducing computational and storage costs. ([Hugging Face][5])

Simple explanation:

> Instead of modifying the entire large model, PEFT modifies only a small trainable part.

Common PEFT method:

```text
LoRA — Low-Rank Adaptation
```

---

## I. AutoTrain

**AutoTrain** is a no-code or low-code tool for training models.

It can be used for:

```text
Text classification
LLM fine-tuning
Image classification
Token classification
Question answering
Summarization
Translation
Tabular classification
Tabular regression
```

The Hugging Face AutoTrain documentation describes it as a no-code platform for training state-of-the-art models across NLP, computer vision, and tabular tasks. ([Hugging Face][6])

This is useful for students who are new to deep learning.

---

## J. Spaces

**Hugging Face Spaces** allows users to host AI demo applications.

Students can create a simple web app using:

```text
Gradio
Streamlit
Static HTML
Docker
```

The Hugging Face documentation describes Spaces as a way to host ML demo apps directly on a profile or organization page. ([Hugging Face][7])

Example Spaces:

```text
Sentiment analysis app
AI chatbot
Image classifier
Text summarizer
Speech-to-text demo
```

---

## K. Inference Providers and Endpoints

Hugging Face also provides ways to run models without setting up your own GPU server.

**Inference Providers** give access to machine learning models through serverless inference. ([Hugging Face][8])

**Inference Endpoints** are managed deployment services for production use, with features such as autoscaling and observability. ([Hugging Face Endpoints][9])

Simple explanation:

> These services help users deploy and use AI models through APIs.

---

# 2. Hugging Face Ecosystem Summary Table

| Component | Purpose | Student-Friendly Meaning |
| ------------------- | ---------------------------------------- | ------------------------- |
| Hub | Store and share models/datasets/apps | GitHub for AI |
| Transformers | Use pretrained NLP and multimodal models | Run models easily |
| Datasets | Load datasets | Get data for ML tasks |
| Tokenizers | Convert text into tokens | Prepare text for AI |
| Pipelines | Simple model usage | AI in a few lines |
| Diffusers | Generate images/audio | Text-to-image AI |
| Evaluate | Calculate metrics | Check model performance |
| PEFT | Efficient fine-tuning | Train small adapters |
| AutoTrain | No-code training | Train without deep coding |
| Spaces | Host demo apps | Create AI web apps |
| Inference Endpoints | Deploy models | Use models through APIs |

---

# 3. Hands-on Activity 1: Explore Hugging Face Hub

## Objective

Students will understand how to search for models, datasets, and demo apps.

## Steps

Ask students to visit the Hugging Face website and explore:

```text
Models
Datasets
Spaces
Tasks
```

## Student Tasks

Students should find:

```text
1 text classification model
1 text generation model
1 image generation model
1 speech recognition model
1 dataset
1 Space demo
```

## Discussion Questions

Ask students:

```text
What is the model name?
Who created it?
What task does it perform?
What license does it use?
Does it have a model card?
Can it be used commercially?
```

## Learning Outcome

Students learn that AI models should not be used blindly. They should check:

```text
Model description
License
Limitations
Intended use
Training data information
Bias or risk warnings
```

---

# 4. Hands-on Activity 2: Sentiment Analysis Using Pipeline

## Objective

Students will use a pretrained model to classify text sentiment.

## Concept

Sentiment analysis means identifying whether a sentence is positive, negative, or neutral.

Example:

```text
"I love this phone." → Positive
"This product is very bad." → Negative
```

## Code

```python
from transformers import pipeline

# Always name the model explicitly. If you write pipeline("sentiment-analysis")
# with no model, you get whatever default the library currently ships - and
# that default changes between versions, so your results stop being reproducible.
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

sentences = [
    "I love machine learning.",
    "This movie was very boring.",
    "The product quality is excellent.",
    "I am not happy with this service."
]

results = sentiment_model(sentences)

for sentence, result in zip(sentences, results):
    print(sentence)
    print(result)
    print()
```

## Explanation

The pipeline automatically performs:

```text
Tokenization
Model loading
Prediction
Output formatting
```

## Student Practice

Ask students to test these sentences:

```text
The food was delicious.
The delivery was very late.
The class was useful and interesting.
The phone battery is terrible.
```

## Learning Outcome

Students understand how pretrained models can perform NLP tasks without training from scratch.

---

# 5. Hands-on Activity 3: Text Generation

## Objective

Students will generate text using a pretrained language model.

## Concept

Text generation models predict the next words based on the input prompt.

## Code

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

prompt = "Artificial Intelligence is useful because"

output = generator(
    prompt,
    max_new_tokens=40,        # how many NEW tokens to add after the prompt
    num_return_sequences=1,
    do_sample=True,           # sample instead of always taking the top token
    temperature=0.8,          # same parameter you met in Activity 4.7
)

print(output[0]["generated_text"])
```

## Explanation

The model continues the input sentence by predicting likely next words.

## Student Practice

Try prompts such as:

```text
Machine learning helps doctors by
A smart classroom of the future will
Responsible AI is important because
```

## Learning Outcome

Students understand how generative AI produces new text from prompts.

---

# 6. Hands-on Activity 4: Question Answering

## Objective

Students will use AI to answer questions from a given paragraph.

## Concept

Question answering models extract answers from a context passage.

## Code

```python
from transformers import pipeline

qa_model = pipeline("question-answering",
                    model="distilbert-base-cased-distilled-squad")

context = """
Artificial Intelligence is a branch of computer science that enables machines
to perform tasks that normally require human intelligence. These tasks include
learning, reasoning, problem-solving, understanding language, and recognizing images.
"""

question = "What is Artificial Intelligence?"

answer = qa_model(
    question=question,
    context=context
)

print(answer)
```

## Explanation

The model does not search the internet. It answers from the given paragraph.

## Student Practice

Ask:

```text
What tasks require human intelligence?
What field does AI belong to?
What can AI recognize?
```

## Learning Outcome

Students understand extractive question answering.

---

# 7. Hands-on Activity 5: Load Dataset from Hugging Face

## Objective

Students will learn how to load datasets directly from Hugging Face.

## Code

```python
from datasets import load_dataset

dataset = load_dataset("imdb")

print(dataset)
print(dataset["train"][0])
```

## Explanation

The IMDb dataset contains movie reviews labeled as positive or negative.

## Student Tasks

Ask students to check:

```text
Number of training samples
Number of test samples
Column names
First review
First label
```

## Learning Outcome

Students understand how datasets can be loaded easily for model training and evaluation.

---

# 8. Hands-on Activity 6: Evaluate Model Output

## Objective

Students will understand model evaluation using accuracy.

## Concept

A classification model must be evaluated by comparing predicted labels with actual labels.

## Example

```text
Actual:    Positive, Negative, Positive, Negative
Predicted: Positive, Negative, Negative, Negative
```

Here, 3 out of 4 predictions are correct.

Accuracy:

```text
3 / 4 = 0.75
```

## Code

```python
import evaluate

accuracy = evaluate.load("accuracy")

predictions = [1, 0, 0, 0]
references = [1, 0, 1, 0]

result = accuracy.compute(
    predictions=predictions,
    references=references
)

print(result)
```

## Learning Outcome

Students learn that AI output must be measured, not just observed.

---

# 9. Hands-on Activity 7: Create a Simple Gradio AI App

## Objective

Students will create a simple web interface for a model.

## Install

```bash
conda activate genai
pip install gradio transformers
```

## Code

```python
import gradio as gr
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return f"Label: {result['label']}, Score: {result['score']:.4f}"

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis App",
    description="Enter a sentence and the model will predict its sentiment."
)

demo.launch()
```

## Explanation

This creates a simple browser-based AI app.

Students can type a sentence and see the model prediction.

## Extension

Ask students to deploy the app on Hugging Face Spaces.

Spaces support Gradio apps and make it possible to host ML demos directly on Hugging Face. ([Hugging Face][10])

---

# 10. Hands-on Activity 8: Compare Two Models

## Objective

Students will understand that different models may produce different outputs.

## Task

Run the same sentences through two sentiment models trained on **different kinds of text**, and compare.

```text
distilbert-base-uncased-finetuned-sst-2-english   <- trained on movie reviews
cardiffnlp/twitter-roberta-base-sentiment-latest  <- trained on tweets
```

## Code

```python
from transformers import pipeline

model_a = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english")
model_b = pipeline("sentiment-analysis",
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentences = [
    "The lab session was great.",
    "The lab session was fine I guess.",
    "Oh brilliant, another three-hour lab.",     # sarcasm
    "not bad at all",                            # double negative
    "The food was ok but the service was terrible.",
]

for sentence in sentences:
    a = model_a(sentence)[0]
    b = model_b(sentence)[0]
    print(f"\n{sentence}")
    print(f"  Model A (movie reviews): {a['label']:<10} {a['score']:.3f}")
    print(f"  Model B (tweets)      : {b['label']:<10} {b['score']:.3f}")
```

## Student Questions

```text
Do both models agree on every sentence? Where do they disagree?
Sentence 3 is sarcasm. Did either model catch it?
Sentence 4 is a double negative. Which handled it better?
Sentence 5 is genuinely both positive and negative. What does the
   confidence score tell you about the model struggling?
Model B was trained on tweets, Model A on film reviews.
   Which would you deploy for student feedback, and why?
```

> **The lesson:** "which model is best?" is the wrong question. **"Best for what text?"** is the right one. Always check what a model was trained on before trusting it on your data.

## Learning Outcome

Students learn that model choice matters.

---

# 11. Hands-on Activity 9: Responsible AI with Model Cards

## Objective

Students learn responsible model usage.

## Activity

Ask students to open a Hugging Face model page and read the model card.

They should identify:

```text
Model name
Developer
Task
Training data
Intended use
Limitations
Bias warnings
License
```

## Discussion

Ask:

```text
Can we use this model in a medical system?
Can we use it for loan approval?
Can we use it commercially?
What are the risks?
```

## Learning Outcome

Students understand that model selection is not only technical; it also involves ethics, license, privacy, and safety.

---

# 12. Suggested Hands-on Sequence

Work through the activities in this order. Each one builds on the last.

| Step | Topic | Activity |
| ---: | ----- | -------- |
| 1 | Introduction to Hugging Face | Explain the Hub and the ecosystem |
| 2 | Explore the Hub | Students search models and datasets |
| 3 | Transformers `pipeline()` | Sentiment analysis |
| 4 | Text generation | Prompt-based generation |
| 5 | Question answering | Answer from context |
| 6 | Datasets library | Load the IMDb dataset |
| 7 | Evaluation | Accuracy calculation |
| 8 | Gradio app | Build a simple AI app |
| 9 | Responsible AI | Read model cards |

---

# 13. In summary

Hugging Face is an important ecosystem for modern AI because it brings together models, datasets, tools, and deployment options in one place.

A student can use Hugging Face to:

```text
Find pretrained models
Load datasets
Run AI models
Fine-tune models
Evaluate performance
Create AI demo apps
Deploy models
Learn responsible AI practices
```

The best beginner path is:

```text
Explore Hub
   ↓
Use pipeline()
   ↓
Load datasets
   ↓
Evaluate models
   ↓
Create Gradio app
   ↓
Deploy on Spaces
```

This gives students a complete practical introduction to using modern Generative AI tools.

[1]: https://huggingface.co/docs/hub/index "Hugging Face Hub documentation"
[2]: https://huggingface.co/docs/transformers/en/index "Transformers"
[3]: https://huggingface.co/docs/datasets/en/index "Datasets"
[4]: https://huggingface.co/docs/diffusers/v0.30.2/en/index "Diffusers"
[5]: https://huggingface.co/docs/peft/index "PEFT"
[6]: https://huggingface.co/docs/autotrain/index "AutoTrain"
[7]: https://huggingface.co/docs/hub/en/spaces "Spaces"
[8]: https://huggingface.co/docs/inference-providers/en/index "Inference Providers"
[9]: https://endpoints.huggingface.co/ "Inference Endpoints by Hugging Face"
[10]: https://huggingface.co/docs/hub/spaces-overview "Spaces Overview"
