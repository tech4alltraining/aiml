# **ML/AI & GenAI Internship Program**

*Course Delivery Plan — the canonical topic and resource list*

| Also see | For |
|---|---|
| [**Student Handbook**](../student/student-handbook.md) | Students — concepts, analogies, runnable code, all 38 activities |
| [**Session Plan**](session-plan.md) | Trainers — the same content, time-blocked minute by minute |
| [**Activities**](activities.md) | Facilitation notes for instructor-led activities |
| [**Prompt Library**](../student/prompts.md) | Every demo prompt, ready to copy |
| [**Notebooks**](../student/notebooks) | Six Colab/Jupyter notebooks, outputs included |
| [**Streamlit Apps**](../student/tutorials/apps/streamlit-apps-collection.md) | 15 runnable apps, simple to advanced |
| [**Exercises**](../student/exercises-assignments.md) | Practice per concept, plus graded assignments |
| [**Setup Guide**](../student/setup-guide.md) | Windows, Ubuntu, macOS · venv or conda |
| [**Index**](../student/README.md) | Everything in this folder |

> **All code in this programme runs in an environment named `genai`.** Setup for Windows, Ubuntu and macOS: [Setup Guide](../student/setup-guide.md).

### Week 0 || Python Pre-work *(optional, self-paced)*
---
#### **For students with no Python background**
##### Python Foundations Part A: printing, variables, data types, operators, strings
- [Notebook 00a](../student/notebooks/00a_python_foundations.ipynb)
##### Python Foundations Part B: collections, conditions, loops, functions
- [Notebook 00b](../student/notebooks/00b_python_foundations_2.ipynb)
##### Scenario Worksheets: ten real-world problems
- [Notebook 00c](../student/notebooks/00c_python_scenarios.ipynb)
- Scenario 10 (ML Data Prep) is the direct bridge into Module 1
##### Additional drills
- [Python exercises](../../python-internship/Python_Exercise1.md) · [Scenario worksheet](../../python-internship/Python_Exercise2.ipynb)

---
### Week 1 || Module 1
---

#### **Session 1.1: *Machine Learning, Python Refresher***

##### Introduction to Machine Learning & Artificial Intelligence
##### Types of Machine Learning
##### Real-world applications of ML & GenAI
- [Demo 1.1 - Teachable Machine](https://teachablemachine.withgoogle.com/) — *Handbook Activity 1.3*
- [Dataset 1.1: Fruits](https://github.com/tech4alltraining/aiml/blob/main/datasets/cv/image-classification.zip)
- *Handbook Activity 1.1 — Sort the mail · Activity 1.2 — Number or category?*
##### Google Colab Tutorial
##### Python Refresher: Variables, Data Types, Operators, Conditional Statements & Loops, Functions and Collections
- [Tutorial 1.1: Python Basics](https://www.w3schools.com/python/)
- [Notebook 1.1: Python Refresher](https://colab.research.google.com/drive/1MVi0l_AcxpJQvCoKSBgoLJMgR_thg7ax?usp=sharing)
- [Exercise 1.1: Python Problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/python-exercises.ipynb)
- *Handbook Activity 1.4 — Fix the broken program*

#### **Session 1.2: *Python Libraries for Data Processing***
##### Introduction to Python Libraries
##### NumPy Fundamentals
- [Numpy Tutorial](https://www.w3schools.com/python/numpy/default.asp)
- [Exercise 1.2: NumPy Problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/numpy-exercises.ipynb) · [Colab (Trainer)](https://colab.research.google.com/drive/1nKs3UcYWgZhIHDwFrFFv1rsY7WQvbYN4?usp=sharing)
- *Handbook Activity 1.5 — The marks calculator*
##### Pandas for Data Handling
- [Pandas Tutorial](https://www.w3schools.com/python/pandas/default.asp)
- [Notebook 1.2: Pandas Practice](https://colab.research.google.com/drive/1qZDpguw8vC8z7gsaagCGOhecHNC6EbZP?usp=sharing)
- *Handbook Activity 1.6 — Twelve rows you can see (`pre_data.csv`)*
##### Data Loading & Exploration
##### Exploratory Data Analysis
- **Practice:** [Dataset 1.2](https://github.com/tech4alltraining/aiml/blob/main/datasets/regression/cardekho_dataset.csv) **-** [Notebook 1.3](https://colab.research.google.com/drive/1dPXOKWHW2GlQ35-uoi5AlHIMtqtlwftj?usp=sharing)
- *Handbook Activity 1.7 — Data detective*
##### NumPy & Pandas Practice
- [Exercise 1.3: Pandas Problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/pandas-exercises.ipynb)


---
### Week 1 || Module 2
---
#### **Session 2.1: *Data Visualization, Supervised Learning***
##### Data Visualization (Matplotlib & Seaborn)
- [Matplotlib Tutorial](https://www.w3schools.com/python/matplotlib_intro.asp)
- [Seaborn Tutorial - GeeksforGeeks](https://www.geeksforgeeks.org/python-seaborn-tutorial/)
- [Seaborn Tutorial - DataCamp](https://www.datacamp.com/tutorial/seaborn-python-tutorial)
- [Exercise 2.1: Matplotlib & Seaborn Problems](https://github.com/tech4alltraining/aiml/blob/main/mlai-internship/matplotlib-seaborn-exercises.ipynb)
- *Handbook Activity 2.1 — The chart chooser · Activity 2.2 — Draw it before you code it*
##### Data Preprocessing
- **Practice:** [Dataset 2.1](https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/prepreprocessing/pre_data.csv) **-** [Notebook 2.1](https://colab.research.google.com/drive/1MsdDPnB3WE3qZZUcZU7QRkJcl0kddz6e?usp=sharing)
- *Handbook Activity 2.3 — Preprocess twelve rows by hand*
- ⚠️ **Data leakage:** always split before you scale. This is the most common serious mistake in student projects.
##### Supervised Machine Learning Concepts
##### Regression Techniques
- **Practice:** [Dataset 2.2](https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/regression/salary_data.csv) **-** [Notebook 2.2 - Salary Data](https://colab.research.google.com/drive/14E7_EF2Eo4Bc-iz6mb0du-66TIwxj3ru?usp=sharing) 
- **Exercise:** [Exercise 2.2: Used car price prediction](https://colab.research.google.com/drive/1RC0GU70efQL-9dabTVIwv-HmGYOz7u3L?usp=sharing)
- *Handbook Activity 2.4 — Beat the model*
##### Classification Techniques
- **Practice:** [Dataset 2.3 - Loan Dataset](https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/loan_data_10k.csv) **-** [Notebook 2.3 - Loan Approval](https://colab.research.google.com/drive/1mJQBMddfoi4P8EdgnPZVLo0tFPlAyBsH?usp=sharing)
- *Handbook Activity 2.5 — What does a mistake cost? · Activity 2.6 — Build the useless 99% model*

#### **Session 2.2: *Machine Learning Workflow, Model Evaluation***
##### Machine Learning Workflow / Life Cycle / Pipeline
##### Scikit-Learn API (preprocessing, modeling, evaluation)
##### Model Evaluation Metrics (regression & classification)
##### Model Training & Validation 
##### Hands-on ML Activity
- [**Exercise**](https://github.com/tech4alltraining/aiml/blob/main/assessments/ml_ai_practice.md) 

---
### Week 1 || Module 3
---
#### **Session: *Feature Engineering, Model Optimization, Model Improvement***
##### Data Augmentation
- **Practice:** [Notebook 3.1 - Data Augmentation](https://colab.research.google.com/drive/1bvfMkPtrSTILCFbxGvaUPr5iGwk65mTe?usp=sharing)
##### Feature Engineering
- **Practice:** [Notebook 3.2 - Feature Engineering](https://colab.research.google.com/drive/1xKoiS5WaH9_kNHjg1dNP-xuAgaTmZ0Wr?usp=sharing)
##### Feature Reduction Techniques
- **Practice:** [Notebook 3.3 - Feature Reduction](https://colab.research.google.com/drive/1wKr-AwnHXF3HgPvb-6K89SjVxUxxHsEK?usp=sharing)

##### Overfitting & Underfitting
- **Practice:** [Notebook 3.4 - Overfitting & Underfitting](https://colab.research.google.com/drive/1NJSLQ3slItQTyvYYQkkzP36h9vKwKT2z?usp=sharing)
##### Model Improvement Strategies
##### Model Evaluation Techniques
##### K-Fold Cross Validation
- **Practice:** [Notebook 3.5 - K-Fold Cross Validation](https://colab.research.google.com/drive/1_MHdhg7Y1x5BnelDgPn9KB3d4NpwYwpf?usp=sharing)
##### Hyperparameter Tuning
- **Practice:** [Notebook 3.6 - Hyperparameter Tuning](https://colab.research.google.com/drive/1-eyZakV1mK4C-_9nvLzV66lqsJBLS0JO?usp=sharing)
- *Handbook Activity 3.1 — Legal or illegal augmentation? · 3.2 — Invent three features · 3.3 — Watch a model overfit · 3.4 — Prove one split is unreliable*


---
### Week 1 || Module 4
---
#### **Session 4.1: *Deep Learning & Unsupervised Learning***

##### Introduction to Deep Learning
##### AI Ethics & Responsible AI
##### Introduction to Unsupervised Learning
##### Clustering Algorithms
- **Practice:** [Dataset 4.1](https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv) **-** [Notebook 4.1](https://colab.research.google.com/drive/1jXCcXXQbSmYnxjulhgtgJB9ueQ76u6Pj?usp=sharing)
- *Handbook Activity 4.1 — Cluster the classroom · Activity 4.2 — Choose k, then defend it*


#### **Session 4.2: *Generative AI & Large Language Models***

##### Introduction to Generative AI
 * [Activity 4.1] : Predict the next word — *Handbook Activity 4.3, "Be the language model"*
##### Large Language Models (LLMs)
##### Prompt Engineering Basics
 * [Demo 4.1] - Prompting Demo (Paris trip example)
 * [Demo 4.1b] - Without Prompt Engineering (smartphone) — see [prompts.md §2](../student/prompts.md#2-weak-vs-strong-prompts)
 > *Prompt:* Generate a product description for a smartphone.
 >
 > *Sample Output:*
 > This smartphone has a high resolution display, powerful processor, and a longlasting battery life.
 >
 > *Prompt:* Create a product descriptions for a budget friendly smartphone perfect for the young professionals highlights, it's affordable, sleek and packed with a top-notch camera features
 * [Activity 4.10] : The Prompting Tournament — [facilitation notes](activities.md#activity-410---the-prompting-tournament)
 > Get the AI to output exactly the phrase "***The eagle flies at midnight***" without using any of those words in the prompt itself.
##### Types of prompts (zero-shot, one-shot, few-shot, chain-of-thought)
 * [Demo 4.2] - Zero-shot prompting 
 * *Example A: The Formatting Challenge* 
 > Extract the name, occupation, and city from the following sentence and output it as JSON:
 >
 > "My name is Sarah, I work as a mechanical engineer, and I just moved to Seattle."
 * *Example B: Creative Constraint*
 > Write a two-sentence horror story about a smartphone. The first sentence must be exactly 5 words. The second sentence must be exactly 3 words.

 * [Demo 4.3] - One-shot prompting
 * *Example A: Tone Translation*
 > Convert the corporate jargon into plain English.
 >
 > Corporate: "Let's synergize our bandwidth to touch base on the deliverables."
 >
 > Plain English: "Let's work together and meet to discuss the project."
 >
 > Corporate: "We need to boil the ocean to shift a paradigm in this vertical."
 >
 > Plain English:
 * *Example B: Strict Data Extraction*
 > Extract the flight details into a pipe-separated format.
 >
 > Input: "I'm flying on Delta flight 402 from JFK to LAX on Tuesday."
 >
 > Output: Delta | 402 | JFK | LAX | Tuesday
 >
 > Input: "Book me on United 88 departing from ORD and arriving at SFO tomorrow."
 >
 > Output:

 * [Demo 4.4] - Few-shot prompting
 * *Example A: Custom Routing Logic*
 > Classify the customer support ticket into one of three categories: [BILLING], [TECH_ISSUE], or [SALES].
 >
 > Ticket: "My screen is cracked and the touch sensor won't work."
 >
 > Category: [TECH_ISSUE]
 >
 > Ticket: "Do you offer enterprise discounts for teams of 50 or more?"
 >
 > Category: [SALES]
 >
 > Ticket: "I was double-charged on my credit card this month."
 >
 > Category: [BILLING]
 >
 > Ticket: "How do I upgrade my account from basic to premium?"
 >
 > Category:
 * *Example B: Grocery Categorization* 
 > Categorize the grocery items into the correct department: [PRODUCE], [DAIRY], [BAKERY], or [MEAT].
 >
 > Item: Granny Smith Apples
 >
 > Department: [PRODUCE]
 >
 > Item: Whole Milk
 >
 > Department: [DAIRY]
 >
 > Item: Sourdough Loaf
 >
 > Department: [BAKERY]
 >
 > Item: Ground Beef
 >
 > Department: [MEAT]
 >
 > Item: Organic Carrots
 >
 > Department:
 * [Demo 4.5] - Chain-of-thought prompting
 * *Example A: The Logic Puzzle*
 > Solve the following logic puzzle. Before giving the final answer, break down your reasoning step-by-step.
 > Puzzle: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?

 * *Example B: Schedule Resolution*
 > Let's find a meeting time. Think through the constraints step-by-step before giving the final answer.
 > Constraints:
 > * Alice is available from 9:00 AM to 11:30 AM.
 > * Bob is available from 10:00 AM to 12:00 PM.
 > * Charlie is available from 10:30 AM to 2:00 PM.
 > * The meeting needs to last exactly 45 minutes.
 > 
 > What is the earliest possible time they can all meet?
 
##### Exploring Generative AI Tools
##### Hands-on GenAI Activities
 * [Practice 4.1] - GenAI API demo — [Notebook: Gemini API](https://colab.research.google.com/drive/1PLrhYse2JtZfKHN6i0mnlVswwl5oNOL0?usp=sharing) · *Handbook 4.5*
 * [Activity 4.7] - The Temperature Dial (`temperature`) — [notes](activities.md#activity-47---the-temperature-dial)
 * [Activity 4.8] - Top-p and Top-k (`top_p`, `top_k`) — [notes](activities.md#activity-48---top-p-and-top-k)
 * [Activity 4.13] - The JSON Treasure Hunt — [notes](activities.md#activity-413---the-json-treasure-hunt)
 * [Activity 4.9] - Prove the model has no memory — *Handbook 4.5*
 * [Apps: Text Generator, Chatbot](../student/tutorials/apps/streamlit-app-simple.md)
 * [Apps: YouTube Summarizer, Diagnostic Helper](../student/tutorials/apps/streamlit-app-advanced.md)

##### Ethical considerations and responsible AI
* [Activity 4.11]: The AI Fact-Checker — hallucination & bias — [notes](activities.md#activity-411---the-ai-fact-checker)
* [Activity 4.12]: The Red Team Challenge (Jailbreak & Guardrails) — [notes](activities.md#activity-412---the-red-team-challenge-jailbreak--guardrails)



---
### Week 1 || Module 5
---

#### **Session 5.1: *Open Source GenAI Models & Hugging Face Ecosystem***
##### Recap: Types of Prompts (from Module 4)
##### Integrating Machine Learning with Generative AI
 - *[ML Model Deployment - Streamlit - [Loan Approval Prediction](https://colab.research.google.com/drive/1CDg6A7P7EL9eaLQiqKglFk1oOy6Su52c?usp=sharing)]* - [App Code](https://github.com/tech4alltraining/aiml/blob/main/mlai-genai-internship/tutorials/loan-app.md)
 - *[ML + GenAI Deployment - Streamlit [GenAU](https://github.com/tech4alltraining/aiml/blob/main/mlai-genai-internship/tutorials/ml_gen_ai.md)]*
 - ML + GenAI integration patterns, Using GenAI to explain ML predictions, Automating insights from ML outputs, Best practices in ML & GenAI development
##### Open Source Generative AI Models
- [Tutorial: Open-Source GenAI Models](../student/tutorials/concepts/open-source-gen-ai.md)
##### Hugging Face Ecosystem
- [Tutorial: Hugging Face Ecosystem](../student/tutorials/concepts/hugging-face-ecosystem.md)
- *Handbook Activity 5.1 — Two models, one sentence · Activity 5.2 — Read a model card*
##### AI-powered Application Concepts
- [Tutorial: AI-Powered Application Concepts](../student/tutorials/concepts/ai-powered-apps.md)


#### **Session 5.2: *Capstone Project & GenAI Application Development***
##### Streamlit App Development
- [Tutorial: Loan Approval App (ML model + Streamlit)](../student/tutorials/apps/loan-app.md) — *Handbook Activity 5.3*
- [Tutorial: ML + GenAI (prediction + explanation)](../student/tutorials/apps/ml_gen_ai.md) — *Handbook Activity 5.4*
##### Project Grouping & Topic Selection
##### Capstone Project Planning
- [Capstone guide, deliverables and marking](../student/student-handbook.md#capstone-project-guide)
##### Internship Guidelines & Mentoring Plan

---

### Reviews & Milestones

| Milestone | When | Deliverable |
|---|---|---|
| Review Update 01 | Week 2 (online) | Problem statement, dataset, EDA findings |
| Review Update 02 | Week 3 (online) | Baseline model and evaluation metrics |
| Review Update 03 | Week 4 (online) | Improved model and app demo |
| Project submission | End of Week 4 | Repository and report |
| Final presentation | End of Week 4 | 8–10 minutes plus questions |


### **References & Video Tutorials**

- **Introduction to Machine Learning Life Cycle**
 - [https://medium.com/@sabihaali1/machine-learning-lifecycle-80c4a265e589](https://medium.com/@sabihaali1/machine-learning-lifecycle-80c4a265e589)
- **Data Preprocessing**
 - Part 1 [https://medium.com/womenintechnology/data-preprocessing-steps-for-machine-learning-in-phyton-part-1-18009c6f1153](https://medium.com/womenintechnology/data-preprocessing-steps-for-machine-learning-in-phyton-part-1-18009c6f1153)
 - Part 2 [https://medium.com/womenintechnology/data-preprocessing-steps-for-machine-learning-in-phyton-part-2-7cbf5856c757](https://medium.com/womenintechnology/data-preprocessing-steps-for-machine-learning-in-phyton-part-2-7cbf5856c757)
- **Supervised & Unsupervised Learning**
 - [https://www.analyticsvidhya.com/blog/2020/04/supervised-learning-unsupervised-learning/](https://www.analyticsvidhya.com/blog/2020/04/supervised-learning-unsupervised-learning/)
 - [https://www.analyticsvidhya.com/blog/2021/04/understanding-supervised-and-unsupervised-learning/](https://www.analyticsvidhya.com/blog/2021/04/understanding-supervised-and-unsupervised-learning/)
 - [https://www.datacamp.com/blog/introduction-to-unsupervised-learning/](https://www.datacamp.com/blog/introduction-to-unsupervised-learning/)
- **Linear Regression:**
 - [https://www.analyticsvidhya.com/blog/2021/06/linear-regression-in-machine-learning/](https://www.analyticsvidhya.com/blog/2021/06/linear-regression-in-machine-learning/)
 - [https://www.analyticsvidhya.com/blog/2021/10/everything-you-need-to-know-about-linear-regression/](https://www.analyticsvidhya.com/blog/2021/10/everything-you-need-to-know-about-linear-regression/)
 - [https://medium.com/analytics-vidhya/understanding-the-linear-regression-808c1f6941c0](https://medium.com/analytics-vidhya/understanding-the-linear-regression-808c1f6941c0)
- **Regression Evaluation Metrics:**
 - [https://www.analyticsvidhya.com/blog/2021/05/know-the-best-evaluation-metrics-for-your-regression-model/](https://www.analyticsvidhya.com/blog/2021/05/know-the-best-evaluation-metrics-for-your-regression-model/)
- **Classification in Machine Learning:**
 - [https://www.analyticsvidhya.com/blog/2021/06/classification-in-machine-learning/](https://www.analyticsvidhya.com/blog/2021/06/classification-in-machine-learning/)
 - [https://iaviral.medium.com/all-classification-models-explained-b03b9b6a4f71](https://iaviral.medium.com/all-classification-models-explained-b03b9b6a4f71)
- **Evaluation Metrics for Classification:**
 - [https://medium.com/analytics-vidhya/complete-guide-to-machine-learning-evaluation-metrics-615c2864d916](https://medium.com/analytics-vidhya/complete-guide-to-machine-learning-evaluation-metrics-615c2864d916)
 - [https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/](https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/)
 - [https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-models-e2f0d8009d69](https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-models-e2f0d8009d69)

- **Introduction Clustering**
 - [https://www.geeksforgeeks.org/clustering-in-machine-learning/](https://www.geeksforgeeks.org/clustering-in-machine-learning/)
- **kMeans Clsutering**
 - [https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)
- **Agglomerative Clustering**
 - [https://medium.com/@khalidassalafy/agglomerative-hierarchical-clustering-a-study-and-implementation-in-python-fddfdb6a7a64](https://medium.com/@khalidassalafy/agglomerative-hierarchical-clustering-a-study-and-implementation-in-python-fddfdb6a7a64)
- **Artificial Neural Networks**
 - [https://medium.com/data-science/an-introduction-to-artificial-neural-networks-5d2e108ff2c3](https://medium.com/data-science/an-introduction-to-artificial-neural-networks-5d2e108ff2c3)


- **Machine Learning Tutorials**
 - (Video) [Machine Learning Full Course - 12 Hours - Machine Learning Roadmap [2024]-Edureka](https://www.youtube.com/watch?v=N5fSpaaxoZc)
 - (Video) [Machine Learning Engineer Full Course - 10 Hours - Machine Learning Roadmap [2024] - Edureka](https://www.youtube.com/watch?v=kx7JCsRdMGQ)
 - (Video) [Machine Learning Full Course 2024 - Learn it LIVE - Machine Learning Tutorial - Simplilearn](https://www.youtube.com/watch?v=fTmR-br9Mjw)
 - (Video) [Machine Learning Course curriculum - Machine Learning - Roadmap](https://www.youtube.com/watch?v=bY__YW-xknU&list=PLfFghEzKVmjsNtIRwErklMAN8nJmebB0I)

- **Mathematics for Machine Learning**
 - (Video) [Mathematics for Machine Learning [Full Course] - Essential Math for Machine Learning - Edureka](https://www.youtube.com/watch?v=1VSZtNYMntM)
 - (Video) [Mathematics for Machine Learning - Introduction - Machine Learning Course
](https://www.youtube.com/watch?v=VCF8kiLtBzU&list=PLfFghEzKVmjtZb9G6jvO9PLKvwUvK5avI)

