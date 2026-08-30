# Session 4 — Introduction to Artificial Intelligence & Machine Learning

**Human vs Artificial Intelligence · Applications of AI & GenAI · What Machine Learning is · Types of ML · Types of Data · Mathematical Foundations · Building AI · The ML Workflow · ML & AI APIs**

| | |
|---|---|
| **Notebook** | [session-04-intro-ml-ai.ipynb](../notebooks/session-04-intro-ml-ai.ipynb) |
| **Previous** | [Session 3 — EDA & Data Preprocessing](session-03-eda-preprocessing.md) |
| **Next** | [Session 5 — Supervised Learning](session-05-regression.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Sessions 1–3 taught you to handle data. This session is where the Machine Learning starts.**
>
> It is mostly concepts, and it needs no maths beyond arithmetic. **Read it properly — everything from Session 5 onwards assumes the vocabulary you build here.**

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Say what human intelligence actually does, and which parts a machine can copy
2. Define Artificial Intelligence in one sentence, and place ML and GenAI inside it
3. Name real applications of AI and GenAI, and say which is which
4. Explain what a machine, learning, and Machine Learning each mean
5. Train an image classifier with no code, and explain what it did
6. Tell supervised, unsupervised and reinforcement learning apart
7. Recognise structured, unstructured and semi-structured data
8. Say what maths a beginner actually needs, and what can wait
9. Choose tools, platforms and datasets to build with
10. Walk the seven-stage ML lifecycle on a real problem
11. Choose between scikit-learn, TensorFlow, PyTorch and Keras

---

## The nineteen topics

**Part A — Introduction to Artificial Intelligence**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [Human Intelligence](#1-human-intelligence) | | 4 | [Applications of AI in Real Life](#4-applications-of-ai-in-real-life) |
| 2 | [What is Artificial Intelligence?](#2-what-is-artificial-intelligence) | | 5 | [Generative AI & its Applications](#5-generative-ai--its-applications) |
| 3 | [Human Intelligence vs AI](#3-human-intelligence-vs-artificial-intelligence) | | | |

**Part B — What is Machine Learning?**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 6 | [What is a Machine?](#6-what-is-a-machine) | | 8 | [What is Machine Learning?](#8-what-is-machine-learning) |
| 7 | [What is Learning?](#7-what-is-learning) | | 9 | [Examples of Machine Learning](#9-examples-of-machine-learning) |

**Part C — Types of Machine Learning**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 10 | [Supervised Learning](#10-supervised-learning) | | 12 | [Reinforcement Learning](#12-reinforcement-learning) |
| 11 | [Unsupervised Learning](#11-unsupervised-learning) | | | |

**Part D — Types of Data · Part E — Maths · Part F — Building AI**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 13 | [Types of Data in ML](#13-types-of-data-in-machine-learning) | | 15 | [AI Programming Languages & Ecosystem](#15-ai-programming-languages--ecosystem) |
| 14 | [Mathematical Foundations](#14-mathematical-foundations-for-machine-learning) | | 16 | [Local, Cloud, Edge & No-Code](#16-local-cloud-edge--no-code-platforms) |
| | | | 17 | [Dataset Repositories](#17-aiml-dataset-repositories) |

**Part G — Workflow · Part H — APIs**

| # | Topic |
|---|---|
| 18 | [The Machine Learning Workflow](#18-the-machine-learning-workflow) |
| 19 | [ML & AI APIs](#19-ml--ai-apis) |

**Five checkpoint problems** sit between the topics:

| After topic | Problem |
|---|---|
| 5 | [⭐ The AI audit](#-checkpoint-problem-1--the-ai-audit) |
| 9 | [⭐ Rules versus learning](#-checkpoint-problem-2--rules-versus-learning) |
| 12 | [⭐ Name that learning type](#-checkpoint-problem-3--name-that-learning-type) |
| 13 | [⭐ Data type sorter](#-checkpoint-problem-4--data-type-sorter) |
| 18 | [⭐ Walk the workflow](#-checkpoint-problem-5--walk-the-workflow) |

---

# Part A — Introduction to Artificial Intelligence

# 1. Human Intelligence

**Before asking what artificial intelligence is, it is worth asking what the real thing does.**

## 🎯 Activity — Intelligence in Daily Life

**Before reading on, write down everything you did in the last hour.** Then mark which ones needed thinking.

A typical morning:

| What you did | What your intelligence did |
|---|---|
| Woke to an alarm and worked out the day | **Recalled** — memory |
| Decided what to wear given the weather | **Reasoned** from evidence |
| Recognised your friend from far away | **Perceived** — pattern recognition |
| Understood a joke | **Understood language and context** |
| Chose the faster route to college | **Planned and optimised** |
| Realised a friend was upset before they said so | **Read emotion** |
| Learned a new shortcut on your phone | **Learned from experience** |

**You did all of that before lunch, without noticing.** That is the bar.

## What human intelligence is made of

| Ability | Meaning | Everyday example |
|---|---|---|
| **Learning** | Getting better with experience | Riding a bicycle |
| **Reasoning** | Drawing conclusions from facts | *It is cloudy, so I will take an umbrella* |
| **Problem solving** | Finding a route to a goal | Fitting your revision into one week |
| **Perception** | Making sense of what you sense | Recognising a face in a crowd |
| **Language** | Understanding and producing meaning | Reading this sentence |
| **Creativity** | Making something new | Writing a song |
| **Emotional intelligence** | Reading and responding to feelings | Comforting a friend |
| **Common sense** | Knowing what goes without saying | A person cannot be in two places at once |

> **The last two are where machines are furthest behind, and it is not close.** A model can pass a professional exam and still not know that you cannot fit an elephant in a matchbox.

## 📘 Examples

**Example 1 — recognising a friend**

You see someone 50 metres away, from behind, in poor light, wearing a coat you have never seen. **You know instantly who it is.** You used posture, height, walk, and the fact that they are usually here at this time — **and you could not explain how you did it.**

**Example 2 — understanding "it"**

> *"The trophy did not fit in the suitcase because it was too big."*

**What was too big?** The trophy. Change one word:

> *"The trophy did not fit in the suitcase because it was too small."*

**Now "it" is the suitcase.** You resolved that instantly using knowledge about the physical world. **No grammar rule tells you this.**

**Example 3 — learning from one example**

Show a child one photograph of a zebra and they will recognise zebras for life — in a cartoon, in a photograph, from behind, half-hidden behind a tree. **A machine typically needs thousands of images.**

**Example 4 — common sense a machine lacks**

```text
"I put the trophy in my bag and walked home."   -> Where is the trophy?
"The ice cream was in the sun for three hours." -> Is it still solid?
"I dropped the glass."                          -> What probably happened?
```

**You answered all three without thinking.** None of that is written down anywhere; you know it because you live in the world.

## 🌍 Scenarios

**Scenario 1 — a doctor's diagnosis**

A doctor sees a patient and combines **perception** (they look pale), **memory** (a similar case last year), **reasoning** (these symptoms plus that test result), **language** (the patient's description) and **emotional intelligence** (they are frightened, so explain gently). **Five kinds of intelligence in one consultation.**

**Scenario 2 — crossing a busy road**

You judge the speed of several vehicles at once, predict where they will be in three seconds, notice a cyclist in the corner of your eye, and decide. **You do this without arithmetic.**

**Scenario 3 — the thing you find easy and a machine finds hard**

| Task | You | A machine |
|---|---|---|
| Multiply 47,381 × 8,294 | Slow, error-prone | **Instant, exact** |
| Recognise your mother's voice on a bad phone line | **Instant** | Hard |
| Play chess | Good with years of practice | **Superhuman** |
| Pick up an unfamiliar object without dropping it | **Trivial** | Genuinely hard |

> **This is Moravec's paradox:** the things we find hard are easy for machines, and the things a toddler does effortlessly are the hardest of all.

## ✏️ Tasks

1. List ten things you did today and label the kind of intelligence each needed.
2. Find three things you do effortlessly that you could not explain to someone step by step.
3. Write three sentences where a pronoun's meaning depends on knowing about the world.
4. Name two tasks where a machine already beats you, and two where it does not.
5. Explain Moravec's paradox in your own words, with an example of your own.

<details><summary>Solutions</summary>

```text
1  Perception, memory, reasoning, planning, language, motor skill,
   emotional intelligence, creativity, common sense - most activities
   use several at once.

2  Recognising a face; balancing on a bicycle; knowing a sentence
   "sounds wrong" grammatically. You do all three instantly and cannot
   give the rule you used - which is exactly why these needed Machine
   Learning rather than hand-written rules.

3  "I poured water from the jug into the cup until it was full."
       -> the CUP
   "I poured water from the jug into the cup until it was empty."
       -> the JUG
   "The council refused the marchers a permit because they feared
    violence."  -> the COUNCIL
   Nothing in the grammar decides these. Knowledge of the world does.

4  Machine wins : arithmetic, chess, searching a million records,
                  never getting tired.
   You win      : picking up an unfamiliar object, understanding a joke,
                  knowing when a friend is upset, learning from one example.

5  Things that took humans millions of years to evolve - seeing, walking,
   grasping - feel effortless and are extremely hard to program. Things
   humans invented recently - arithmetic, chess, logic - feel hard to us
   and are easy for machines.
   Example: a robot that beats a grandmaster at chess still cannot
   reliably pick up the pieces.
```
</details>

## ❓ MCQs

**Q1.** Which ability are machines furthest behind on?
- (a) Arithmetic  (b) Common sense and emotional intelligence  (c) Memory  (d) Searching

**Q2.** A child recognises zebras after seeing one picture. A model typically needs…
- (a) One picture too  (b) Thousands of pictures  (c) No pictures  (d) Ten pictures

**Q3.** *"The trophy did not fit in the suitcase because it was too big."* What does "it" refer to, and how do you know?
- (a) The suitcase; grammar tells you  (b) The trophy; knowledge about the physical world tells you  (c) It is ambiguous  (d) Neither

**Q4.** Moravec's paradox says…
- (a) Machines are better at everything  (b) What we find hard is easy for machines, and what a toddler does easily is hardest  (c) Machines cannot learn  (d) Intelligence is one skill

**Q5.** Recognising a friend from behind at 50 metres is an example of…
- (a) Reasoning  (b) Perception and pattern recognition  (c) Arithmetic  (d) Language

<details><summary>Answers</summary>

**A1 — (b).** A model can pass a professional exam and not know an elephant does not fit in a matchbox.

**A2 — (b) Thousands.** Learning from one example remains a hard research problem.

**A3 — (b) The trophy** — and you know from physics, not grammar. Swap "big" for "small" and the answer flips.

**A4 — (b).** A robot that beats a grandmaster still cannot reliably pick up the pieces.

**A5 — (b) Perception.** And you could not explain how you did it — which is exactly why it needs ML.
</details>

---

# 2. What is Artificial Intelligence?

> **Artificial Intelligence means teaching computers to think, learn and solve problems the way humans do.**

That is the whole definition. Everything else is detail about *how*.

🧠 **Analogy: teaching a child versus writing a recipe.** A recipe is a fixed list of steps — follow it exactly and you get the dish, but it cannot cope with anything unexpected. **Teaching a child is different: you show examples, correct mistakes, and eventually they handle situations you never showed them.** AI is the attempt to build the second kind of thing.

## What counts as AI

**Anything that makes a machine appear intelligent.** That includes hand-written rules.

| System | Is it AI? | Why |
|---|---|---|
| A calculator | ❌ | Exact arithmetic; no judgement |
| A thermostat with a fixed rule | Arguably | Simple, but it decides |
| A chess engine following programmed strategy | ✅ **Yes, and it is not ML** | It appears intelligent |
| A spam filter trained on examples | ✅ | And it *is* ML |
| ChatGPT | ✅ | AI, ML, deep learning and generative |

> **The surprise for most beginners: a rule-based chess engine is AI but not Machine Learning.** AI is the older and wider word — it dates from 1956. ML is one way of achieving it, and today the dominant one.

## The four things an AI system does

```text
1. PERCEIVE    take in data - text, images, numbers, sound
2. REASON      process it against what it knows
3. LEARN       improve from experience or examples
4. ACT         produce an output - a decision, a label, a sentence
```

## The nesting

```text
┌───────────────────────────────────────────────┐
│ ARTIFICIAL INTELLIGENCE   (1956)              │
│ any machine that appears intelligent          │
│  ┌─────────────────────────────────────────┐  │
│  │ MACHINE LEARNING                        │  │
│  │ learns the rules from data              │  │
│  │  ┌───────────────────────────────────┐  │  │
│  │  │ DEEP LEARNING                     │  │  │
│  │  │ ML using many-layered networks    │  │  │
│  │  │  ┌─────────────────────────────┐  │  │  │
│  │  │  │ GENERATIVE AI               │  │  │  │
│  │  │  │ deep learning that CREATES  │  │  │  │
│  │  │  └─────────────────────────────┘  │  │  │
│  │  └───────────────────────────────────┘  │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

## Narrow AI and General AI

| | Narrow AI | General AI |
|---|---|---|
| Does | **One task**, often superbly | Anything a human can |
| Examples | Every system in this course | None — it does not exist |
| Status | **Everywhere, today** | A research goal |

> **Everything you will ever build, and everything currently deployed anywhere, is Narrow AI.** A model that plays chess at superhuman level cannot make a cup of tea, and does not know it exists.

## 📘 Examples

**Example 1 — AI without any learning**

```python
# illustrative: a syntax reference, not runnable as written.
def thermostat(temperature):
    """A rule-based system. It decides - but it never learns."""
    if temperature < 18:
        return "heating ON"
    if temperature > 24:
        return "cooling ON"
    return "off"
```

**It makes a decision, so many would call it AI. It has learned nothing.**

**Example 2 — the same job, learned**

```python
# illustrative: a syntax reference, not runnable as written.
# Given a year of readings and what the household actually did,
# a model works out ITS OWN thresholds - which may differ per room,
# per season, and per time of day.
model.fit(past_readings, past_decisions)
model.predict(todays_reading)
```

**Nobody wrote 18 or 24.** The numbers came from the data.

**Example 3 — placing real systems**

| System | Layer |
|---|---|
| Traffic light on a fixed timer | Rules (AI, arguably) |
| Chess engine with programmed strategy | AI, **not** ML |
| Netflix recommendations | ML |
| Face unlock | Deep Learning |
| ChatGPT writing an email | Generative AI |

**Example 4 — the four steps, in one system**

```text
A self-driving car:
  PERCEIVE  cameras, radar and lidar read the road
  REASON    is that a pedestrian? will they step out?
  LEARN     millions of driven miles improve the model
  ACT       brake, steer, accelerate
```

## 🌍 Scenarios

**Scenario 1 — your phone unlocking**

```text
PERCEIVE  the camera captures your face
REASON    compare against the stored pattern
LEARN     it adapts slowly as you age or grow a beard
ACT       unlock, or ask for the PIN
```

**Scenario 2 — a bank flagging a transaction**

```text
PERCEIVE  amount, location, time, merchant, your history
REASON    how unusual is this for THIS customer?
LEARN     every confirmed fraud improves the model
ACT       allow, flag for review, or block
```

**Scenario 3 — where AI is the wrong answer**

```text
Computing income tax from published bands.

The rules are KNOWN, EXACT and legally binding. A model would be
slower, occasionally wrong, and impossible to audit.

WRITE THE RULE. Session 1's lesson, still true: if you can write the
rule correctly, write the rule.
```

## ✏️ Tasks

1. Define AI in one sentence without using the word "intelligent".
2. Name three systems that are AI but not Machine Learning.
3. For a system you use daily, write out its perceive / reason / learn / act steps.
4. Explain the difference between Narrow and General AI, with an example of each — or say why you cannot.
5. Name three tasks where a rule beats a model, and say why.

<details><summary>Solutions</summary>

```text
1  Artificial Intelligence is teaching computers to think, learn and
   solve problems the way humans do.

2  A rule-based chess engine; a thermostat; a spell-checker that
   matches against a fixed dictionary; a traffic light on a timer.
   All of them DECIDE. None of them LEARN.

3  Google Maps route planning:
     PERCEIVE  your location, the destination, live traffic
     REASON    which route is fastest given current conditions
     LEARN     actual journey times feed back into the estimates
     ACT       show the route and speak the directions

4  NARROW AI does ONE task, often superbly - every system in this
   course, and everything deployed anywhere today.
   GENERAL AI would do anything a human can. There is no example,
   because it does not exist.

5  Income tax from published bands - the rules are exact and legally
     binding, and must be auditable.
   Checking a password - it either matches or it does not.
   Calculating VAT - arithmetic, not judgement.
   In all three, a model would be slower, occasionally wrong, and
   impossible to explain to a regulator.
```
</details>

## ❓ MCQs

**Q1.** In one sentence, what is Artificial Intelligence?
- (a) Any computer program  (b) Teaching computers to think, learn and solve problems like humans  (c) Only neural networks  (d) Robots

**Q2.** A chess engine following programmed strategy is…
- (a) Machine Learning  (b) AI, but not ML  (c) Deep Learning  (d) Not AI at all

**Q3.** Which is the widest and oldest of these terms?
- (a) Generative AI  (b) Deep Learning  (c) Machine Learning  (d) Artificial Intelligence

**Q4.** Everything deployed in the world today is…
- (a) General AI  (b) Narrow AI  (c) Both  (d) Neither

**Q5.** For computing income tax from published bands you should use…
- (a) Machine Learning  (b) A written rule  (c) Deep Learning  (d) GenAI

<details><summary>Answers</summary>

**A1 — (b).** Everything else is detail about how.

**A2 — (b).** **AI is the older, wider word.** It appears intelligent, but nothing was learned from data.

**A3 — (d) Artificial Intelligence**, from 1956.

**A4 — (b) Narrow AI.** A superhuman chess engine cannot make tea.

**A5 — (b) A written rule.** Exact, auditable, and legally binding.
</details>

---

# 3. Human Intelligence vs Artificial Intelligence

**The useful comparison is not "which is better" — it is "which parts map onto each other, and where does the mapping break".**

## The mapping

| Human | Machine | Honest verdict |
|---|---|---|
| Eyes, ears, skin | Cameras, microphones, sensors | **Machine wins on range** — infrared, ultrasound, 360° |
| Memory | Storage | **Machine wins decisively** — perfect and unlimited |
| Recall speed | Database lookup | **Machine wins** |
| Learning from experience | Training on data | **Human wins on efficiency** — one example versus thousands |
| Reasoning from few facts | Statistical inference | **Human wins** on novel situations |
| Arithmetic | Computation | **Machine wins overwhelmingly** |
| Transfer to a new task | Retraining needed | **Human wins decisively** |
| Common sense | — | **Human wins; machines have almost none** |
| Creativity | Recombination of patterns | Contested — machines produce novel *combinations* |
| Emotion | Simulated at best | **Human**, and it is not close |
| Getting tired | Never | **Machine** |
| Explaining a decision | Often opaque both ways | Draw |

## The three that matter most in practice

| | |
|---|---|
| **1. Data efficiency** | You learn a zebra from one picture. A model needs thousands. |
| **2. Transfer** | You learned to drive a car; a van takes an afternoon. A model trained on cars knows nothing about vans. |
| **3. Common sense** | You know a person cannot be in two cities at once. A model must be shown. |

> **This is why AI systems are built as tools for people rather than replacements.** The machine brings tireless accuracy at scale; the person brings context, judgement and responsibility. **Session 11 builds exactly that pairing.**

## 📘 Examples

**Example 1 — the machine wins**

```text
Search 10 million medical records for one pattern.
  Human  : weeks, with mistakes
  Machine: seconds, exactly
```

**Example 2 — the human wins**

```text
"The restaurant was so good we went back three times... to complain."
  Human  : catches the sarcasm instantly
  Machine: often reads "so good" and "went back" as positive
```

**Example 3 — data efficiency, side by side**

| | Human child | Image model |
|---|---|---|
| Examples of a zebra needed | **1** | ~1,000+ |
| Recognises a cartoon zebra | ✅ | Often ❌ |
| Recognises a zebra behind a tree | ✅ | Sometimes ❌ |
| Knows a zebra is an animal that eats | ✅ | ❌ — no such concept |

**Example 4 — the pairing that works**

```text
Radiology:
  MACHINE  scans 1,000 images overnight and flags 40 as suspicious
  HUMAN    reviews those 40 with full context and decides

Neither could do the other's job. Together they are better than either.
```

## 🌍 Scenarios

**Scenario 1 — hiring**

```text
MACHINE  reads 5,000 CVs in a minute and ranks them by stated skills
HUMAN    interviews the shortlist, judges fit and potential, decides

DANGER   if the machine also DECIDES, it inherits whatever bias is in
         the historical hiring data. Session 12 measures exactly this.
```

**Scenario 2 — driving**

```text
MACHINE  never tired, never distracted, reacts in milliseconds,
         sees in every direction at once
HUMAN    handles the situation nobody anticipated - a flooded road,
         a police officer waving you the wrong way down a street
```

**Scenario 3 — the honest summary for a report**

```text
The model processes more cases than any team could, consistently and
without fatigue.

It has no understanding of what it is doing, no common sense, and no
way to recognise a situation outside its training data.

It is therefore used to PRIORITISE, and a person DECIDES.
```

## ✏️ Tasks

1. Build the comparison table for five abilities of your choice and give a verdict on each.
2. Find three tasks where a machine clearly wins, and three where you clearly do.
3. Write a sentence a human understands instantly that a model would probably misread.
4. Describe a job where a human-machine pairing beats either alone.
5. Explain data efficiency with an example that is not the zebra.

<details><summary>Solutions</summary>

```text
1  Memory        machine (perfect, unlimited)
   Arithmetic    machine (overwhelmingly)
   Common sense  human (machines have almost none)
   Transfer      human (a model must be retrained; you adapt)
   Tiredness     machine (it never tires)

2  Machine: counting objects in 10,000 photographs; searching a
     database; playing chess.
   Human: understanding a joke; picking up an unfamiliar object;
     knowing when a rule should be broken.

3  "Great, another meeting that could have been an email."
   Every individual word is positive or neutral. The meaning is not.

4  Air traffic control: the system tracks every aircraft continuously
   and warns of conflicts; the controller handles the emergency, the
   unusual request, and the judgement call.

5  You learn what a new coin looks like from one glance and will
   recognise it worn, dirty, at an angle, or in poor light. A model
   needs many photographs of that coin in many conditions.
```
</details>

## ❓ MCQs

**Q1.** Where do humans most clearly beat machines?
- (a) Arithmetic  (b) Common sense and transfer to new tasks  (c) Memory  (d) Speed

**Q2.** "Data efficiency" refers to…
- (a) Storage size  (b) How many examples are needed to learn something  (c) Processing speed  (d) File format

**Q3.** A model trained on cars is asked about vans. It…
- (a) Transfers easily  (b) Generally needs retraining — transfer is a human strength  (c) Refuses  (d) Works better

**Q4.** In a human-machine pairing for radiology, the machine should…
- (a) Decide  (b) Prioritise, and let a person decide  (c) Replace the radiologist  (d) Do nothing

**Q5.** Why is sarcasm hard for a model?
- (a) It is rare  (b) The meaning contradicts the individual words  (c) It is ungrammatical  (d) It is not hard

<details><summary>Answers</summary>

**A1 — (b).** Machines have almost no common sense, and cannot transfer.

**A2 — (b).** One zebra picture versus a thousand.

**A3 — (b).** You learned a van in an afternoon; the model learned nothing.

**A4 — (b).** **The machine brings scale; the person brings judgement and responsibility.**

**A5 — (b).** Every word can be positive while the sentence is not.
</details>

---

# 4. Applications of AI in Real Life

**AI is not coming. It is already in almost everything you touched today.**

## Twenty you have used or been affected by

| # | Application | What the AI does | Kind |
|---|---|---|---|
| 1 | **Face recognition** | Matches your face against a stored pattern | Deep Learning |
| 2 | **Self-driving cars** | Reads the road, predicts, steers | Deep Learning |
| 3 | **Virtual assistants** | Understands speech and answers | DL + GenAI |
| 4 | **Healthcare diagnostics** | Flags tumours on scans | Deep Learning |
| 5 | **Fraud detection** | Spots unusual transactions | Classical ML |
| 6 | **Recommendation systems** | Predicts what you will watch or buy | Classical ML |
| 7 | **Google Maps** | Predicts travel time and reroutes | ML + optimisation |
| 8 | **Smart home devices** | Learns your routine | Classical ML |
| 9 | **Language translation** | Converts between languages | GenAI |
| 10 | **Social media moderation** | Detects harmful content | DL + GenAI |
| 11 | **Predictive maintenance** | Predicts a machine failure before it happens | Classical ML |
| 12 | **Financial trading** | Predicts price movements, executes | Classical ML |
| 13 | **Drug discovery** | Predicts which molecules might work | DL |
| 14 | **Personalised marketing** | Chooses which advert you see | Classical ML |
| 15 | **VR and gaming** | Opponents that adapt to you | ML + RL |
| 16 | **Cybersecurity** | Detects intrusions from unusual patterns | Classical ML |
| 17 | **Climate modelling** | Predicts weather and long-term change | ML + simulation |
| 18 | **Supply chain optimisation** | Forecasts demand, plans routes | ML + optimisation |
| 19 | **Education and e-learning** | Adapts difficulty to the learner | Classical ML |
| 20 | **Precision agriculture** | Spots disease from drone images, targets water | DL |

> **Notice how many say "Classical ML".** Most working AI in the world is not deep learning and not generative — it is the kind you will build in Session 5, on tables of numbers.

## The pattern behind all twenty

```text
Every one of them is:
  a PREDICTION problem   (what will happen? what is this?)
  learned from HISTORY   (what happened before?)
  applied at a SCALE     no team of people could match
```

## 📘 Examples

**Example 1 — your morning, annotated**

```text
06:30  Phone unlocks with your face          -> face recognition
06:45  Maps says "leave now, traffic"        -> travel-time prediction
07:10  Music app suggests a song             -> recommendation
08:00  Bank texts about an odd payment       -> fraud detection
08:30  Email sorts a message into spam       -> classification
09:00  Autocomplete finishes your sentence   -> generative
```

**Six AI systems before nine o'clock**, and you noticed none of them.

**Example 2 — the same technique, different industries**

```text
"Predict a rare event from historical patterns" is:
  fraud detection        in banking
  predictive maintenance in manufacturing
  intrusion detection    in cybersecurity
  disease screening      in healthcare

SAME PROBLEM SHAPE. Same imbalanced-data challenge you will meet
in Session 5, and the same reason accuracy is the wrong metric.
```

**Example 3 — where AI failed publicly, and why it matters**

```text
A recruiting tool trained on ten years of hiring data learned to
downgrade CVs mentioning women's colleges - because the historical
data reflected historical hiring.

The model was working exactly as designed. THE DATA WAS THE PROBLEM.
Session 12 measures this on a real dataset.
```

**Example 4 — classical ML still runs the world**

```text
Applications 5, 6, 8, 11, 12, 14, 16, 18, 19 in the table above are
CLASSICAL machine learning on tabular data.

That is the majority - and it is exactly what Sessions 5 to 8 teach.
```

## 🌍 Scenarios

**Scenario 1 — a hospital**

```text
Diagnostics    flag suspicious regions on a scan for a radiologist
Triage         predict which arrivals are most urgent
Readmission    predict who is likely to return within 30 days
Scheduling     predict no-shows and overbook accordingly
Drug discovery narrow millions of molecules to a few hundred
```

**Scenario 2 — a small business**

```text
Demand forecasting  how much to stock next week
Customer churn      who is about to stop buying
Pricing             what will this item sell for
Support             an assistant that answers common questions
```

**All four are within reach of what you will know by Session 11.**

**Scenario 3 — a farm**

```text
PERCEIVE  a drone photographs the field
REASON    a model spots diseased plants from the images
ACT       spray only those areas
RESULT    less chemical, lower cost, better yield
```

## ✏️ Tasks

1. List ten AI systems you used in the last week and name what each predicts.
2. Pick three from the table of twenty and say whether each is classical ML, deep learning or generative — and why.
3. Find two applications from different industries that are the same problem shape.
4. Research one publicised AI failure and write three sentences on what went wrong.
5. Choose an organisation you know and list four AI applications it could realistically use.

<details><summary>Solutions</summary>

```text
1  Face unlock (is this the owner?), Maps (how long will this take?),
   Spotify (what will they play next?), spam filter (is this spam?),
   autocomplete (what word comes next?), bank alerts (is this fraud?),
   YouTube (what will they watch?), keyboard swipe (what word was that?),
   photo search (what is in this picture?), translation (what does this
   mean in English?).

2  Fraud detection      -> classical ML, tabular transaction data
   Face recognition     -> deep learning, learns from raw pixels
   Language translation -> generative, produces new text

3  Fraud detection and predictive maintenance: both predict a RARE
   event from historical patterns, both have badly imbalanced data,
   and for both, accuracy is the wrong metric.

4  Answers will vary. A good answer names the system, what it got
   wrong, and WHY - usually biased training data, a distribution shift
   after deployment, or a metric that did not match the real goal.

5  A college: predict dropout risk; forecast canteen demand; route
   library book purchasing by predicted demand; an assistant that
   answers questions from the student handbook (Session 11's RAG).
```
</details>

## ❓ MCQs

**Q1.** Most working AI systems in the world today are…
- (a) Generative AI  (b) Classical ML on tabular data  (c) Deep learning  (d) Rule-based

**Q2.** Fraud detection and predictive maintenance share…
- (a) Nothing  (b) The same problem shape — predicting a rare event from history  (c) The same industry  (d) The same data

**Q3.** A recruiting model downgraded CVs from women's colleges because…
- (a) The algorithm was biased by design  (b) The historical training data reflected historical hiring  (c) A coding error  (d) Too little data

**Q4.** What do all twenty applications have in common?
- (a) They use deep learning  (b) They are prediction problems learned from history, applied at scale  (c) They are generative  (d) They need a GPU

**Q5.** Which is generative rather than predictive?
- (a) Fraud detection  (b) Language translation  (c) Recommendation  (d) Predictive maintenance

<details><summary>Answers</summary>

**A1 — (b) Classical ML.** Exactly what Sessions 5 to 8 teach.

**A2 — (b).** Same shape, same imbalance problem, same wrong metric to avoid.

**A3 — (b).** **The model worked as designed. The data was the problem.**

**A4 — (b).** Prediction, from history, at a scale no team could match.

**A5 — (b) Translation.** It produces new text rather than choosing a label.
</details>

---

# 5. Generative AI & its Applications

**Every application in Topic 4 that *chose* something was predictive. Generative AI *creates* something that did not exist before.**

🧠 **Analogy: a music critic and a musician.** The critic listens and classifies — *this is jazz, this is three stars*. Useful, and bounded. **The musician writes a piece that has never been played.** Same domain, entirely different job.

| | Predictive AI | Generative AI |
|---|---|---|
| Output | A label or a number | New text, images, audio, video, code |
| Question | *Which one?* | *Make me one* |
| Answer space | Fixed and known | Effectively unlimited |

## Thirteen applications

| # | Application | What it creates |
|---|---|---|
| 1 | **Text generation** | Articles, emails, summaries, answers |
| 2 | **Image generation** | Pictures from a description |
| 3 | **Music generation** | Melodies, backing tracks, sound effects |
| 4 | **Video generation** | Short clips from a description |
| 5 | **Code generation** | Functions, tests, explanations, translations between languages |
| 6 | **Data augmentation** | Extra training examples where real data is scarce |
| 7 | **Virtual assistants and chatbots** | Conversational answers in context |
| 8 | **Personalised content** | Copy written for one reader |
| 9 | **Gaming and entertainment** | Dialogue, levels, characters |
| 10 | **Healthcare and drug discovery** | Candidate molecules with desired properties |
| 11 | **Education and e-learning** | Practice questions, explanations at the right level |
| 12 | **Marketing and advertising** | Product descriptions, campaign variants |
| 13 | **Design and creativity** | Logos, layouts, first drafts |

> **Number 6 deserves attention.** Generative models are used to *create training data* for predictive models — which is Session 6's topic, and a genuine loop between the two families.

## How it works, in one line

**A generative model predicts the next piece, then feeds its own output back in and predicts again.**

```text
"The capital of France is"        -> "Paris"
"The capital of France is Paris"  -> "."
```

**Generation is prediction, run in a loop.** Session 9 builds up to exactly this, and Session 10 uses it.

## 📘 Examples

**Example 1 — the same problem, both ways**

```python
# illustrative: a syntax reference, not runnable as written.
# PREDICTIVE (Session 5): choose from a fixed set
model.predict(loan_application)      # -> 0 or 1

# GENERATIVE (Session 10): produce something new
llm("Explain in two kind sentences why this loan was declined.")
# -> a paragraph nobody wrote in advance
```

**Example 2 — sorting real tasks**

| Task | Which |
|---|---|
| Will this customer churn? | Predictive |
| Write the email asking them to stay | Generative |
| Is this transaction fraud? | Predictive |
| Explain to the customer why their card was blocked | Generative |
| Compute tax from published bands | **Neither — write the rule** |

**Example 3 — the two working together**

```text
A loan application arrives.

  PREDICTIVE MODEL  decides: declined, 0.87 confidence
                    (auditable, consistent, measured)
  GENERATIVE MODEL  explains it in three kind sentences
                    (readable, actionable)

The model DECIDES. The LLM EXPLAINS. Never the other way round -
Session 11 builds this and explains why the order matters.
```

**Example 4 — where GenAI is the wrong tool**

```text
- Anything needing a guaranteed-correct answer (tax, interest, payroll)
- Anything needing the SAME answer every time (a generative model samples)
- Anything you must audit line by line
- Any task where you already have 10,000 labelled rows and a clear target
  -> train a classifier; it will be faster, cheaper and more accurate
```

## 🌍 Scenarios

**Scenario 1 — a college using GenAI honestly**

```text
Good uses    generating practice questions from lecture notes
             summarising long documents for students
             explaining a concept at three different levels
             first-draft feedback a teacher then reviews

Bad uses     grading final exams without review
             generating "facts" for a syllabus without checking
             anything where a confident wrong answer causes harm
```

**Scenario 2 — a designer's workflow**

```text
BEFORE  sketch 5 concepts by hand over two days, client picks one
AFTER   generate 50 concepts in an hour, client picks a direction,
        designer then does the real work properly

The generative step replaced the SKETCHING, not the designing.
```

**Scenario 3 — data augmentation, the loop between the families**

```text
Problem  only 200 photographs of a rare crop disease
Step 1   a generative model creates realistic variations
Step 2   a PREDICTIVE model trains on the enlarged set
Result   a better classifier than 200 images alone would allow

Session 6 covers this - and its risks.
```

## ✏️ Tasks

1. List five generative tools you have used and say what each creates.
2. Classify eight tasks from your own life as predictive, generative, or "write a rule".
3. Describe a system that uses both a predictive and a generative model, and say which decides.
4. Name three tasks where GenAI would be the wrong choice, with a reason each.
5. Explain next-token prediction to someone in two sentences.

<details><summary>Solutions</summary>

```text
1  ChatGPT/Gemini (text), Copilot (code), DALL-E/Imagen (images),
   ElevenLabs (speech), Google Translate (text in another language).

2  Predictive: will it rain, is this spam, is this my face.
   Generative: write my email, summarise this chapter, make an image.
   Write a rule: compute my grade from marks, check a password,
                 calculate interest.

3  A loan assistant: a Random Forest DECIDES (auditable, consistent,
   measured across Sessions 5-8), and an LLM EXPLAINS the decision in
   plain English. The MODEL decides. Never the LLM.

4  Computing tax - must be exact and auditable.
   Anything needing identical output every time - a generative model
     samples, so it varies.
   A task with 10,000 labelled rows and a clear target - a classifier
     will be faster, cheaper and more accurate.

5  The model is given some text and predicts which piece comes next.
   It then adds that piece to the text and predicts again, so
   generation is just prediction run in a loop.
```
</details>

## ❓ MCQs

**Q1.** What makes a system *generative*?
- (a) It uses a neural network  (b) It produces new content rather than choosing from fixed options  (c) It is large  (d) It runs on a GPU

**Q2.** Generation works by…
- (a) Looking up answers  (b) Predicting the next piece and feeding the output back in, repeatedly  (c) Searching the web  (d) Copying training data

**Q3.** In a system with both models, which should make the decision?
- (a) The generative model  (b) The predictive model  (c) Both vote  (d) Either

**Q4.** "Data augmentation" as a GenAI application means…
- (a) Compressing data  (b) Creating extra training examples where real data is scarce  (c) Cleaning data  (d) Labelling data

**Q5.** You have 10,000 labelled rows and a clear target. You should…
- (a) Use an LLM  (b) Train a classifier — faster, cheaper and more accurate  (c) Use GenAI to relabel  (d) Do nothing

<details><summary>Answers</summary>

**A1 — (b).** The answer space is effectively unlimited rather than a fixed list.

**A2 — (b).** **Generation is prediction, run in a loop.**

**A3 — (b) The predictive model.** Auditable, consistent, measured — Session 11 explains why the order matters.

**A4 — (b).** A genuine loop between the two families; Session 6 covers it.

**A5 — (b).** The LLM is not the answer to every question.
</details>

---

## ⭐ Checkpoint Problem 1 — The AI audit

> **Uses:** Topics 1–5.

**The problem.** Audit one day of your own life. Find at least **twelve** points where an AI system touched it, and for each record: what it did, what it predicted or generated, and which category it falls into.

<details><summary>Solution</summary>

```python
audit = [
    # time,  system,               what it does,                    category
    ("06:30", "Face unlock",        "is this the owner?",            "Deep Learning"),
    ("06:35", "Keyboard autocomplete","what word comes next?",       "Generative"),
    ("06:45", "Maps travel time",   "how long will this take?",      "Classical ML"),
    ("07:10", "Music recommendation","what will they play next?",    "Classical ML"),
    ("07:30", "Spam filter",        "is this email spam?",           "Classical ML"),
    ("08:00", "Bank fraud alert",   "is this transaction unusual?",  "Classical ML"),
    ("09:00", "Search ranking",     "which result is most relevant?","Classical ML"),
    ("11:00", "Photo search",       "what is in this picture?",      "Deep Learning"),
    ("13:00", "Translation",        "what does this mean?",          "Generative"),
    ("15:00", "Shopping suggestion","what else might they buy?",     "Classical ML"),
    ("18:00", "Video recommendation","what will keep them watching?","Classical ML"),
    ("21:00", "Voice assistant",    "what did they say, and reply",  "DL + Generative"),
]

print(f"{'time':<7}{'system':<24}{'what it predicts/creates':<34}{'category'}")
print("-" * 92)
for t, s, w, c in audit:
    print(f"{t:<7}{s:<24}{w:<34}{c}")

from collections import Counter
print("\nBY CATEGORY")
for cat, n in Counter(c for *_, c in audit).most_common():
    print(f"  {n:>2}  {cat}")

print(f"""
WHAT THE COUNT SHOWS

  Classical ML dominates - {Counter(c for *_, c in audit)['Classical ML']} of {len(audit)}.
  Most working AI is not deep learning and not generative. It is
  prediction from tabular history, which is exactly what Sessions 5
  to 8 teach you to build.

  Every single one is NARROW AI: each does one task and knows nothing
  outside it.
""")
```

**The point of the exercise is the final count.** Students arrive expecting AI to mean ChatGPT. **The audit shows that most of the AI in their day is the unglamorous kind they are about to learn to build.**
</details>

**Make it harder:**

1. For each entry, say what data it must have been trained on.
2. Mark which ones would still work with no internet connection.
3. Pick the one that would cause most harm if it were wrong, and say why.

---

# Part B — What is Machine Learning?

# 6. What is a Machine?

**Before "Machine Learning", it is worth being clear what each word means on its own.**

**A machine is something that takes an input, does something to it, and produces an output.** It follows instructions exactly, every time, without getting bored.

🧠 **Analogy: a vending machine.** Coins and a button code go in; a specific snack comes out. **Press B4 a thousand times and you get the same thing a thousand times.** That reliability is the entire point of a machine.

🧠 **Analogy: a calculator.** Type `7 × 8`, get `56`. It will never be tired, never be distracted, and never give you `57`. **But it will also never notice that you meant to type `7 × 9`.**

```text
INPUT  ->  [ fixed instructions ]  ->  OUTPUT
```

## What a machine is good at

| Strength | Consequence |
|---|---|
| **Exactness** | The same input always gives the same output |
| **Speed** | Millions of operations a second |
| **Tirelessness** | The billionth run is as good as the first |
| **Scale** | Ten million records is not harder than ten |

## What a plain machine cannot do

**It cannot handle anything its instructions did not anticipate.**

```text
A vending machine given a torn note: fails.
A calculator given "seven times eight": fails.
A program given a spelling it was not told about: fails.
```

> **That last one is why Machine Learning exists.** You cannot write instructions for every spelling of every spam word — so instead you build a machine that works the rules out for itself.

## 📘 Examples

**Example 1 — a machine in code**

```python
def vending_machine(code):
    """Fixed instructions. Same input, same output, forever."""
    menu = {"A1": "crisps", "A2": "chocolate", "B4": "biscuits"}
    return menu.get(code, "invalid code")

print(vending_machine("B4"))     # biscuits
print(vending_machine("B4"))     # biscuits - identical, every time
print(vending_machine("Z9"))     # invalid code
```

**Example 2 — exactness is the strength**

```python
def add_tax(amount, rate=18):
    return amount * (1 + rate / 100)

for _ in range(3):
    print(f"{add_tax(1000):.2f}")     # 1180.00, three times
```

**Example 3 — and the limitation**

```python
def is_yes(answer):
    """Instructions written for exactly two spellings."""
    return answer in ("yes", "Yes")

for a in ["yes", "Yes", "YES", "yeah", "y", " yes"]:
    print(f"{a!r:>8} -> {is_yes(a)}")
```

**Four of those six are obviously "yes" to a person, and the machine says no.** You could add each one — and then someone types `Yep`.

**Example 4 — the machine has no idea what it is doing**

```python
def classify(n):
    return "big" if n > 100 else "small"

print(classify(500))        # big
print(classify(-999999))    # small
print(classify(101))        # big
# It has no concept of size, quantity or meaning. It compares a number.
```

## 🌍 Scenarios

**Scenario 1 — where a machine is exactly right**

```text
Payroll. Salary, tax band and deductions are KNOWN and EXACT.
You want the same answer every time, and you must be able to audit it.

A plain machine following written rules is the correct tool.
Machine Learning here would be slower, occasionally wrong, and
impossible to defend to an auditor.
```

**Scenario 2 — where it breaks down**

```text
Reading handwritten addresses on envelopes.

Every person writes differently. There is no finite set of rules that
covers every hand. The instructions cannot be written.
```

**Scenario 3 — the honest boundary**

```text
Ask: CAN I WRITE THE RULE CORRECTLY AND COMPLETELY?

  Yes -> write it. It will be faster, exact and auditable.
  No  -> you need Machine Learning.

That single question decides more projects than any algorithm choice.
```

## ✏️ Tasks

1. Write a function that acts as a machine: fixed input, fixed output. Call it three times and show the output is identical.
2. Write a rule-based classifier and find three inputs a person would handle that it gets wrong.
3. Name three jobs where exactness matters more than flexibility.
4. Name three jobs where the rules genuinely cannot be written.
5. For a task of your choice, answer the question "can I write the rule correctly?" and justify it.

<details><summary>Solutions</summary>

```python
def grade(marks):                                                      # 1
    """A machine: fixed instructions, identical output every time."""
    if marks >= 90: return "A"
    if marks >= 75: return "B"
    if marks >= 40: return "C"
    return "F"

for _ in range(3):
    print(grade(82), end=" ")     # B B B
print()

def is_yes(answer):                                                    # 2
    return answer in ("yes", "Yes")

for a in ["YES", "yeah", "y", "Yep", " yes"]:
    print(f"{a!r:>8} -> {is_yes(a)}")      # all False, all obviously yes

# 3 - Payroll, tax calculation, banking transfers, medication dosage
#     arithmetic. Exact, auditable, legally binding.

# 4 - Reading handwriting, recognising a face, detecting sarcasm,
#     identifying a plant disease from a photograph. No finite rule set
#     covers them.

# 5 - "Can I write the rule correctly AND completely?"
#     Yes -> write it: faster, exact, auditable.
#     No  -> you need Machine Learning.
#     That single question decides more projects than any algorithm choice.
```
</details>

## ❓ MCQs

**Q1.** What defines a machine, in this sense?
- (a) It has moving parts  (b) Input, fixed instructions, output — the same every time  (c) It learns  (d) It is electronic

**Q2.** A machine's greatest strength is…
- (a) Creativity  (b) Exactness and tirelessness  (c) Judgement  (d) Common sense

**Q3.** A rule-based `is_yes` returns `False` for `"YES"` and `"yeah"`. This shows…
- (a) A bug  (b) You cannot write instructions for every case a person would handle  (c) The machine is broken  (d) Nothing

**Q4.** When should you write a rule rather than train a model?
- (a) Never  (b) When you can write the rule correctly and completely  (c) Always  (d) Only for numbers

**Q5.** `classify(-999999)` returns `"small"`. What does the machine understand about size?
- (a) A great deal  (b) Nothing — it compared a number  (c) Enough  (d) It learned it

<details><summary>Answers</summary>

**A1 — (b).** Reliability is the point.

**A2 — (b).** The billionth run is as good as the first.

**A3 — (b).** **And this is precisely why Machine Learning exists.**

**A4 — (b).** Faster, exact, and auditable. It decides more projects than any algorithm choice.

**A5 — (b) Nothing.** It has no concept of quantity or meaning.
</details>

---

# 7. What is Learning?

**Learning is getting better at something through experience, without being given the rule.**

🧠 **Analogy: learning to ride a bicycle.** Nobody hands you the physics of balance. You wobble, you correct, you fall, you adjust — and one day you can ride. **You still cannot write down the rule you learned.**

🧠 **Analogy: learning to recognise ripe fruit.** After a few market trips you can pick a good mango. Colour, smell, give under the thumb — **you weight all three without ever having been told the weights.**

## The three ingredients

```text
1. EXPERIENCE   examples, attempts, or observations
2. FEEDBACK     did that work? was that right?
3. ADJUSTMENT   change what you do next time
```

**Remove any one and learning stops.** Practise with no feedback and you get better at your mistakes.

## Learning versus memorising

| | Memorising | Learning |
|---|---|---|
| The past paper | Perfect | Good |
| A new question | **Lost** | **Fine** |
| What was acquired | Specific answers | The underlying pattern |

> **This distinction becomes *overfitting* in Session 8**, where you will watch a model score 100% on data it has seen and do worse than a simpler model on data it has not. **It is the same failure, with numbers attached.**

## 📘 Examples

**Example 1 — the three ingredients, in a child**

```text
EXPERIENCE   the child touches a hot cup
FEEDBACK     it hurts
ADJUSTMENT   they do not touch it again

One example was enough, because the feedback was strong.
```

**Example 2 — learning without being told the rule**

```text
Nobody gives you a formula for "this sentence sounds wrong".
You read thousands of sentences and absorbed the pattern.

"The cat sat on the mat."   sounds right
"The cat sitted on mat the." sounds wrong

You cannot state the rule you applied - and you applied it instantly.
```

**Example 3 — the difference between practice and feedback**

```text
Throwing darts BLINDFOLDED for a year: no improvement.
Throwing darts and SEEING where they land: steady improvement.

Same experience. Only one has FEEDBACK. Only one is learning.
```

**Example 4 — memorising, and where it fails**

```text
Student A memorises the answers to 50 past-paper questions.
Student B works out the method behind them.

On those 50 questions      : A scores higher
On 50 NEW questions        : A collapses, B is fine

That gap is exactly what a test set measures.
```

## 🌍 Scenarios

**Scenario 1 — learning to cook**

```text
EXPERIENCE   you make the dish twenty times
FEEDBACK     it is too salty; it is undercooked; that one was good
ADJUSTMENT   less salt, longer on the heat

After twenty attempts you cook it well - and you can now adapt it to
a different pan, a different stove, and half the quantity.
That last part is TRANSFER, and it is what memorising cannot do.
```

**Scenario 2 — a shopkeeper learning demand**

```text
Over a year the shopkeeper notices:
  more cold drinks when it is hot
  more umbrellas before the monsoon
  more chocolate near festivals

NOBODY TOLD THEM THESE RULES. They emerged from experience.

That is exactly what a demand-forecasting model does - the same
learning, from the same evidence, at a scale one person cannot hold.
```

**Scenario 3 — feedback that arrives too late**

```text
A doctor prescribes a treatment and learns the outcome in six months.
A chess player learns the outcome in an hour.
A typist learns the outcome instantly.

The FASTER and CLEARER the feedback, the faster the learning.
This is why supervised learning - where every example comes with its
answer attached - is the easiest kind, and Session 5 starts there.
```

## ✏️ Tasks

1. Describe something you learned and identify the experience, the feedback and the adjustment.
2. Give an example of practice with no feedback, and say what happens.
3. Explain the difference between memorising and learning, with an example of your own.
4. Describe a skill you have that you cannot explain step by step.
5. Rank three learning situations by how fast the feedback arrives, and say what that implies.

<details><summary>Solutions</summary>

```text
1  Learning to cycle:
     EXPERIENCE  repeated attempts
     FEEDBACK    falling over, or staying upright
     ADJUSTMENT  small corrections to balance and steering
   Nobody supplied the physics.

2  Throwing darts blindfolded, or practising an instrument without
   ever hearing yourself. You get better at your mistakes - the
   experience is there but the feedback is not, so no learning happens.

3  Memorising is storing specific answers; learning is acquiring the
   pattern that produced them. A student who memorises 50 past answers
   scores well on those 50 and collapses on 50 new ones.
   THAT GAP IS WHAT A TEST SET MEASURES - and Session 8 calls it
   overfitting.

4  Recognising a friend's voice; knowing a sentence sounds wrong;
   catching a ball. All instant, all inexplicable.

5  Typing (instant) > chess (an hour) > medical treatment (six months).
   Faster, clearer feedback means faster learning. This is why
   SUPERVISED learning, where every example carries its answer, is the
   easiest kind - and where Session 5 starts.
```
</details>

## ❓ MCQs

**Q1.** What are the three ingredients of learning?
- (a) Memory, speed, storage  (b) Experience, feedback, adjustment  (c) Data, model, output  (d) Input, process, output

**Q2.** Practising with no feedback produces…
- (a) Fast learning  (b) No learning — you get better at your mistakes  (c) Perfect skill  (d) Memorisation

**Q3.** A student memorises 50 past answers and fails on 50 new questions. In ML this is called…
- (a) Underfitting  (b) Overfitting  (c) Leakage  (d) Bias

**Q4.** Why is supervised learning the easiest kind?
- (a) It uses less data  (b) Every example arrives with its correct answer attached  (c) It is faster  (d) It needs no maths

**Q5.** The shopkeeper who notices cold drinks sell in hot weather has…
- (a) Been told a rule  (b) Learned a pattern from experience  (c) Memorised  (d) Guessed

<details><summary>Answers</summary>

**A1 — (b).** Remove any one and learning stops.

**A2 — (b).** Darts blindfolded for a year.

**A3 — (b) Overfitting** — and Session 8 shows it with numbers.

**A4 — (b).** The feedback is immediate and unambiguous.

**A5 — (b).** Nobody told them; it emerged from a year of evidence.
</details>

---

# 8. What is Machine Learning?

**Put the two words together.**

> **Machine Learning is a subset of AI that lets computers learn from data and improve their performance without being explicitly programmed.**
>
> Put practically: **it creates predictive models by finding relationships in data.**

## The one diagram that matters

```text
Traditional programming:   DATA + RULES    ->  ANSWERS
Machine Learning:          DATA + ANSWERS  ->  RULES
```

**Read those two lines until the swap is obvious.** That reversal is the whole idea, and everything in Sessions 5 to 9 is a way of carrying it out.

**The rules that come out are called a *model*.**

🧠 **Analogy: the recipe and the chef.** A **recipe** is exact instructions — 200g onion, fry 8 minutes. If today's onions are unusually sweet, the recipe cannot adapt. **A chef** cooks a thousand curries, tastes each one, and works out for themselves what makes a good one. **Nobody wrote those rules down.**

🧠 **Analogy: teaching a child to recognise a cat.** You do not list *four legs, whiskers, retractable claws, triangular ears*. You point at cats and say "cat", point at dogs and say "dog", and correct the mistakes. **Eventually they recognise a breed you never showed them.**

🧠 **Analogy: a tailor who has measured a thousand customers.** Give them a height and a build, and they can predict the measurements closely — not from a formula, but from a thousand fittings. **That is a predictive model, built from data, held in a person.**

## Why "without being explicitly programmed" matters

```python
# EXPLICITLY PROGRAMMED - you write the rule
def is_spam(text):
    return "lottery" in text.lower()

print(is_spam("You have WON a lottery prize!"))      # True
print(is_spam("Cheap m3dicine, no prescription"))    # False - and it IS spam
```

**The spammer wrote `m3dicine` to dodge your keyword.** You cannot write a rule for every spelling — but a model trained on examples catches it, because the *shape* of the message is still spam-like.

## What "finding relationships in data" means

```text
Data                              Relationship the model finds
------------------------------    -----------------------------------
hours studied -> exam mark        each extra hour is worth ~8 marks
income, loan size -> approved?    a big loan on a small income is risky
pixels -> "cat" or "dog"          these edge patterns mean whiskers
```

**In every case the model was never told the relationship. It measured it.**

## 📘 Examples

**Example 1 — the swap, in code**

```python
# TRADITIONAL: you supply the rule
def predict_mark(hours):
    return 8 * hours + 30          # YOU worked out 8 and 30

# MACHINE LEARNING: the model works the rule out
# model.fit(hours, marks)          -> it discovers 8 and 30 itself
```

**Example 2 — the model recovering a rule nobody gave it**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

hours = np.array([[1], [2], [3], [4], [5], [6]])
marks = np.array([38, 46, 54, 62, 70, 78])

model = LinearRegression().fit(hours, marks)
print(f"learned: mark = {model.coef_[0]:.1f} * hours + {model.intercept_:.1f}")
```

**It was never told the formula. It recovered it from six examples.**

**Example 3 — learning from examples where no rule exists**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = ["win a free prize now", "claim your lottery money",
          "meeting moved to 3pm", "notes from the lecture attached",
          "cheap m3dicine no prescription", "fee payment reminder"]
labels = [1, 1, 0, 0, 1, 0]        # 1 = spam

vec = CountVectorizer()
model = MultinomialNB().fit(vec.fit_transform(emails), labels)

print(model.predict(vec.transform(["free money claim now",
                                   "lecture notes for tomorrow"])))
```

**Nobody wrote a rule. Six examples were enough to separate them.**

**Example 4 — ML sits inside AI**

```text
AI  = making machines appear intelligent   (includes written rules)
ML  = one way of doing it: learn from data (today, the dominant way)

Every ML system is AI.
Not every AI system is ML - the chess engine from Topic 2 is not.
```

## 🌍 Scenarios

**Scenario 1 — a bank moving from rules to learning**

```text
BEFORE  a committee writes: "decline if credit score < 600"
        Simple, auditable - and it ignores everything else about
        the applicant.

AFTER   a model learns from 100,000 past decisions and their outcomes.
        It finds that a low score with a small loan and a long
        employment history is often FINE - a nuance no committee
        wrote down.

COST    the model is harder to explain. Session 11 solves that by
        pairing it with an LLM; Session 12 audits it for fairness.
```

**Scenario 2 — where the relationship was genuinely unknown**

```text
Hospitals found that a model predicting 30-day readmission weighted
"number of previous admissions" far more heavily than clinicians
expected, and lab values less.

The model did not know medicine. It measured what actually predicted
the outcome in that data - which is both its strength and exactly why
it needs a clinician to review.
```

**Scenario 3 — the honest limit**

```text
ML finds RELATIONSHIPS. It does not find CAUSES.

A model may learn that patients seen by Doctor X recover faster.
That could mean Doctor X is excellent - or that Doctor X is assigned
the easier cases.

The model cannot tell the difference. YOU have to ask.
```

## ✏️ Tasks

1. Write the traditional-versus-ML swap in your own words, without looking.
2. Write a rule-based classifier, then find inputs that defeat it.
3. Fit a `LinearRegression` on six points and show it recovered the underlying rule.
4. Describe a task where the rule is genuinely unwriteable, and say why.
5. Give an example where a model finds a relationship that is real but not causal.

<details><summary>Solutions</summary>

```python
# 1 - Traditional programming: you give the computer DATA and RULES,
#     and it produces ANSWERS.
#     Machine Learning: you give it DATA and ANSWERS, and it produces
#     the RULES. Those rules are the MODEL.

def is_spam(text):                                                     # 2
    return "lottery" in text.lower()

for t in ["Free m0ney!!!", "Cl41m your prize", "WIN NOW"]:
    print(f"{t!r:>22} -> {is_spam(t)}")     # all False, all spam
# You cannot write a rule for every spelling.

import numpy as np                                                     # 3
from sklearn.linear_model import LinearRegression
hours = np.array([[1], [2], [3], [4], [5], [6]])
marks = np.array([38, 46, 54, 62, 70, 78])
m = LinearRegression().fit(hours, marks)
print(f"mark = {m.coef_[0]:.1f} * hours + {m.intercept_:.1f}")
# It recovered 8 and 30 without ever being told them.

# 4 - Recognising handwriting, detecting sarcasm, identifying a plant
#     disease from a photo. Every person writes, jokes and photographs
#     differently; no finite rule set covers them.

# 5 - Ice-cream sales predict drownings. Both are caused by hot weather.
#     The RELATIONSHIP is real; the CAUSE is not what it looks like.
#     ML finds relationships. It does not find causes.
```
</details>

## ❓ MCQs

**Q1.** What is the core swap that defines Machine Learning?
- (a) Faster computers  (b) Traditional gives data + rules → answers; ML gives data + answers → rules  (c) More data  (d) Neural networks

**Q2.** What is a "model"?
- (a) The dataset  (b) The rules the algorithm worked out from the examples  (c) The library  (d) The accuracy

**Q3.** "Without being explicitly programmed" means…
- (a) No code is written  (b) Nobody wrote the decision rule — the model derived it from data  (c) It runs itself  (d) It has no parameters

**Q4.** Is every AI system a Machine Learning system?
- (a) Yes  (b) No — a rule-based chess engine is AI but not ML  (c) Only modern ones  (d) Only deep ones

**Q5.** A model finds that patients seen by Doctor X recover faster. This proves…
- (a) Doctor X is better  (b) Nothing about cause — Doctor X may get the easier cases  (c) The data is wrong  (d) Causation

<details><summary>Answers</summary>

**A1 — (b).** That reversal is the whole idea.

**A2 — (b).** The thing you save to a file and reuse.

**A3 — (b).** You still write plenty of code — you just do not write the rule.

**A4 — (b) No.** **AI is the wider, older word.**

**A5 — (b).** **ML finds relationships, not causes.** You have to ask.
</details>

---

# 9. Examples of Machine Learning

**Concrete cases, so the definition stops being abstract.**

## Five classic problems

| Problem | Input | Output | Type |
|---|---|---|---|
| **Apple vs orange** | An image | Which fruit | Classification |
| **House price prediction** | Size, location, rooms | A price | Regression |
| **Spam email detection** | The message text | Spam or not | Classification |
| **Stock price prediction** | Past prices, volume, news | A future price | Regression |
| **Customer churn prediction** | Usage, tenure, complaints | Will they leave? | Classification |

> **Notice there are only two shapes here.** The output is either a **number** (regression) or a **category** (classification). **Deciding which you have is the single most important modelling decision** — Topic 10 covers it, and Session 5 builds both.

## 🎯 Demo — Google Teachable Machine

**Train a real image classifier in your browser, with no code, in about ten minutes.**

**Dataset:** [`datasets/cv/image-classification.zip`](../../../datasets/cv/) — **apple, banana and orange**, 15 training images each plus 3 test images.

```text
1. Unzip image-classification.zip
2. Go to  https://teachablemachine.withgoogle.com/
3. Choose "Image Project" -> "Standard image model"
4. Create three classes and name them: apple, banana, orange
5. Upload the 15 images from training-images/apple into the apple class,
   and likewise for banana and orange
6. Click "Train Model" and wait
7. Test it with the three images in test-images/
```

### What to notice while it trains

| What you did | What it corresponds to |
|---|---|
| Sorted images into named folders | **Labelling** — creating `y` |
| Uploaded them | **The training set** — `X_train`, `y_train` |
| Clicked "Train Model" | **`model.fit()`** — Session 5 |
| Tested on unseen images | **`model.predict()`** on a test set |
| Saw confidence percentages | **`predict_proba()`** — Session 5 |

**You just did the entire Session 5 workflow with a mouse.** The rest of this course is doing it in code, so that you control every step.

### Questions to answer after the demo

1. What happens if you upload only **3** images per class instead of 15?
2. What happens if you show it a **pear**?
3. What happens if all your apple photos have a white background and you then test on an apple on a table?

> **That third question is the important one.** If every apple image shares a white background, the model may be learning *"white background = apple"*. It will score perfectly on your test images and fail completely in a real kitchen. **This is Session 8's overfitting, discovered with your own hands.**

## 📘 Examples

**Example 1 — apple vs orange, as a table**

```python
import pandas as pd

fruit = pd.DataFrame({
    "weight_g":   [150, 170, 140, 130, 180, 160],
    "texture":    ["smooth", "smooth", "bumpy", "bumpy", "smooth", "bumpy"],
    "label":      ["apple", "apple", "orange", "orange", "apple", "orange"],
})
print(fruit)
print("\nA model learns: smooth + heavier -> apple, bumpy -> orange.")
print("Nobody wrote that rule. It is measurable in the six rows.")
```

**Example 2 — house prices, as regression**

```python
houses = pd.DataFrame({
    "size_sqft": [800, 1200, 1500, 2000, 950],
    "bedrooms":  [2, 3, 3, 4, 2],
    "price_lakh":[45, 68, 82, 110, 52],
})
print(houses)
print("\nThe TARGET is a NUMBER, so this is REGRESSION.")
```

**Example 3 — churn, as classification**

```python
customers = pd.DataFrame({
    "months_active": [24, 3, 36, 2, 18],
    "complaints":    [0, 3, 1, 4, 0],
    "churned":       [0, 1, 0, 1, 0],
})
print(customers)
print("\nThe TARGET is a CATEGORY (0/1), so this is CLASSIFICATION.")
print("Notice: few months + many complaints -> churn. The model measures it.")
```

**Example 4 — sorting problems by their shape**

```python
problems = [
    ("How much will this house sell for?",     "number  -> REGRESSION"),
    ("Is this email spam?",                    "category-> CLASSIFICATION"),
    ("What will this stock be worth tomorrow?","number  -> REGRESSION"),
    ("Will this customer leave?",              "category-> CLASSIFICATION"),
    ("Is this an apple or an orange?",         "category-> CLASSIFICATION"),
    ("How many visitors tomorrow?",            "number  -> REGRESSION"),
]
for q, a in problems:
    print(f"{q:<44}{a}")
```

## 🌍 Scenarios

**Scenario 1 — spam detection, and why it needed ML**

```text
1990s   hand-written keyword rules. Spammers changed spelling.
        Rules were added. Spammers changed again. Endless.

2000s   models trained on millions of labelled emails.
        They learn the SHAPE of spam - unusual character mixes,
        odd link patterns, sender reputation - which survives
        respelling.

RESULT  a problem nobody could solve with rules, solved with data.
```

**Scenario 2 — house prices, and the honest caveat**

```text
A model trained on Kochi house prices will predict Kochi prices well.

Ask it about Delhi and it will answer CONFIDENTLY and be WRONG,
because it never saw Delhi's price levels.

A model knows only what was in its training data. It cannot tell you
when it is outside that range - which is why Session 12 insists on a
LIMITATIONS section.
```

**Scenario 3 — stock prediction, and the reality check**

```text
Stock prediction appears in every list of ML examples, including this
one. It is also the hardest, and mostly does not work:

  - the pattern changes as soon as people trade on it
  - the signal is tiny and the noise is enormous
  - a model that is right 52% of the time is considered excellent

If a beginner tutorial shows a model predicting stock prices at 95%
accuracy, it has leaked the future into the training data.
Session 3's lesson - and Session 8 explains how to catch it.
```

## ✏️ Tasks

1. Do the Teachable Machine demo with the apple/banana/orange dataset and record its confidence on the three test images.
2. Repeat it with only 3 images per class. What happens to the confidence?
3. Show the model a pear. What does it say, and why is that answer inevitable?
4. For the five classic problems, state the input, the output, and whether it is regression or classification.
5. Write down three ML problems your own college could pose, with input, output and type.

<details><summary>Solutions</summary>

```text
1  With 15 images per class the model usually classifies all three test
   images correctly, often above 90% confidence.

2  With 3 images per class it still often gets them right, but with LOWER
   and less stable confidence - and it will be far more sensitive to
   background and lighting. Less data means a less reliable model, and
   you can feel it.

3  It will confidently say apple, banana or orange - because those are
   the ONLY THREE ANSWERS IT HAS. A classifier cannot output "none of
   these" unless you train it with a "none" class.
   This is one of the most important practical lessons in the course.

4  apple vs orange : image        -> fruit name  : CLASSIFICATION
   house prices    : size, rooms  -> a price     : REGRESSION
   spam detection  : message text -> spam or not : CLASSIFICATION
   stock price     : past prices  -> a price     : REGRESSION
   customer churn  : usage, tenure-> leave or not: CLASSIFICATION

5  Predict dropout risk (attendance, marks -> yes/no : CLASSIFICATION)
   Predict canteen demand (day, weather -> meals    : REGRESSION)
   Predict library demand (past loans -> copies     : REGRESSION)
```
</details>

## ❓ MCQs

**Q1.** In the Teachable Machine demo, clicking "Train Model" corresponds to…
- (a) `model.predict()`  (b) `model.fit()`  (c) `train_test_split()`  (d) Loading data

**Q2.** You show the fruit classifier a pear. It will…
- (a) Say "unknown"  (b) Confidently pick one of the three classes it knows  (c) Crash  (d) Return nothing

**Q3.** All your apple photos have a white background. The risk is that the model learns…
- (a) Nothing  (b) "White background = apple", and fails on a real table  (c) The apple's shape  (d) Colour only

**Q4.** House price prediction is…
- (a) Classification  (b) Regression  (c) Clustering  (d) Generation

**Q5.** A beginner tutorial shows 95% accuracy on stock prediction. The most likely explanation is…
- (a) A very good model  (b) Future information leaked into the training data  (c) A new algorithm  (d) Lucky data

<details><summary>Answers</summary>

**A1 — (b) `model.fit()`.** You did the whole Session 5 workflow with a mouse.

**A2 — (b).** **A classifier can only answer with the classes it was given.** You must train a "none" class to get "none".

**A3 — (b).** Perfect on your test images, useless in a kitchen. **Session 8's overfitting, found with your own hands.**

**A4 — (b) Regression.** A price is a number.

**A5 — (b) Leakage.** A model right 52% of the time is considered excellent in this domain.
</details>

---

## ⭐ Checkpoint Problem 2 — Rules versus learning

> **Uses:** Topics 6–9.

**The problem.** For twelve real systems, decide whether each should use a written rule or Machine Learning, and justify each in one sentence. Then take one rule-based system and explain what would have to change for ML to become the better choice.

<details><summary>Solution</summary>

```python
systems = [
    ("Bank OTP check",              "RULE", "the code either matches or it does not"),
    ("Netflix recommendations",     "ML",   "nobody can write the rule for taste"),
    ("Spell-check against a dictionary", "RULE", "a fixed word list, exactly matched"),
    ("Autocorrect for typos",       "ML",   "which typo means which word is learned"),
    ("Traffic light on a timer",    "RULE", "fixed timing, no judgement needed"),
    ("Adaptive traffic lights",     "ML",   "timing learned from measured flow"),
    ("Face unlock",                 "ML",   "no rule describes a face"),
    ("Income tax from bands",       "RULE", "exact, legally binding, must be auditable"),
    ("Spam filter",                 "ML",   "spammers adapt faster than rules can"),
    ("Chess engine (classic)",      "RULE", "the game's rules are complete and known"),
    ("Google Translate",            "ML",   "language is not a finite rule set"),
    ("ATM PIN verification",        "RULE", "exact match, and it must never be wrong"),
]

print(f"{'system':<36}{'choice':<8}why")
print("-" * 96)
for name, choice, why in systems:
    print(f"{name:<36}{choice:<8}{why}")

from collections import Counter
c = Counter(ch for _, ch, _ in systems)
print(f"\nRULE {c['RULE']}   ML {c['ML']}")

print("""
THE TEST THAT DECIDES EVERY ROW

  Can I write the rule CORRECTLY and COMPLETELY?
     Yes -> write it. Faster, exact, auditable.
     No  -> you need Machine Learning.

NOTICE THE TWO TRAFFIC LIGHT ROWS

  A fixed timer is a rule. Lights that adapt to measured traffic flow
  are ML. SAME PROBLEM, and the choice depends entirely on whether you
  are willing to fix the timing in advance.

WHAT WOULD MOVE A RULE TO ML

  Spell-check is a rule today because it matches a fixed dictionary.
  It becomes ML the moment you want it to handle names it has never
  seen, new slang, or to guess which correction the user MEANT -
  because at that point the rule can no longer be written completely.
""")
```

**The two traffic-light rows are the exercise.** Most systems are not inherently one or the other — **the choice depends on how much variation you are willing to hand-write.**
</details>

**Make it harder:**

1. Add three systems of your own and classify them.
2. Find a system that started rule-based and became ML, and say what forced the change.
3. Find a case where ML was used and a rule would have been better, and explain the cost.

---

# Part C — Types of Machine Learning

# 10. Supervised Learning

**You give the model inputs *and* the correct answers. It learns the mapping between them.**

🧠 **Analogy: past exam papers with the answer key.** You study a hundred solved problems. Nobody explained the underlying theory — you inferred it from worked examples. **Then you sit a new paper.**

🧠 **Analogy: a child learning fruit with a parent.** You point and say "apple". You point and say "orange". You correct the mistakes. **The label is supplied every time — that is the "supervision".**

```text
INPUT (X)          ANSWER (y)
size, rooms   ->   price
message text  ->   spam / not spam
image pixels  ->   apple / orange
```

## The two kinds

| | Regression | Classification |
|---|---|---|
| Answer is | A **number** | A **category** |
| Question | *How much? How many?* | *Which one? Yes or no?* |
| Examples | House price, temperature, sales | Spam, churn, disease, fruit |
| A wrong answer is | Off by an amount | Simply wrong |
| Typical metric | RMSE, R² | Accuracy, F1, ROC-AUC |

> **Decide whether your target is a number or a category and you have made the single most important modelling decision.** It fixes which models you may use, which metrics you must report, and how you evaluate everything.

## The trap

**Some numbers are really categories.**

```text
age 30 + age 30 = 60        -> meaningful  -> age is a NUMBER
pincode + pincode           -> nonsense    -> pincode is a CATEGORY

If the ARITHMETIC is meaningless, it is a category - however it is stored.
```

## 📘 Examples

**Example 1 — the same four lines, either way**

```python
# illustrative: a syntax reference, not runnable as written.
from sklearn.linear_model import LinearRegression, LogisticRegression

LinearRegression().fit(X_train, y_price)       # y = 450000, 320000, ...
LogisticRegression().fit(X_train, y_approved)  # y = 1, 0, 1, ...
```

**Example 2 — deciding from the target column**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

ads = pd.read_csv(BASE + "regression/advertising.csv")
loans = pd.read_csv(BASE + "loan_data_10k.csv")

print(f"Sales       : {ads['Sales'].nunique()} distinct values -> REGRESSION")
print(f"loan_status : {loans['loan_status'].nunique()} distinct values -> CLASSIFICATION")
```

**Example 3 — sorting real questions**

```python
questions = [
    ("How much will this house sell for?",          "Regression"),
    ("Will this customer churn?",                   "Classification"),
    ("How many visitors tomorrow?",                 "Regression"),
    ("Which of 5 plans will they choose?",          "Classification"),
    ("What temperature will it be at noon?",        "Regression"),
    ("Is this transaction fraudulent?",             "Classification"),
]
for q, a in questions:
    print(f"{q:<46}{a}")
```

**Example 4 — the genuinely ambiguous case**

```text
"What rating out of 5 will this user give?"

  As REGRESSION      : the answer is a number 1-5, and 4 really is
                       closer to 5 than 1 is
  As CLASSIFICATION  : only five values exist, and you cannot
                       meaningfully predict 3.7 stars

BOTH ARE DEFENSIBLE. Real teams argue about exactly this. Build both,
compare, and say which you would ship and why - that reasoning is the
deliverable, not the answer.
```

## 🌍 Scenarios

**Scenario 1 — a hospital, both kinds**

```text
REGRESSION      how many days will this patient stay?
CLASSIFICATION  will this patient be readmitted within 30 days?

SAME PATIENT, SAME DATA, DIFFERENT TARGETS - and therefore different
models, different metrics, and different definitions of "good".
```

**Scenario 2 — where the labels came from, and what it costs**

```text
Supervised learning needs LABELS, and somebody has to make them.

  Spam         : users clicking "report spam" - free, and noisy
  Medical scans: a radiologist labelling each one - accurate, and
                 extremely expensive
  Loan outcomes: they arrive on their own, months later

THE COST AND QUALITY OF LABELS IS OFTEN THE REAL CONSTRAINT ON A
PROJECT - more than the choice of algorithm.
```

**Scenario 3 — the label you cannot get**

```text
"Which of our customers WOULD have bought if we had called them?"

You only observe what happened to the people you DID call. The label
for everyone else does not exist and never will.

Supervised learning cannot answer this. It needs a different design -
an experiment - and recognising that is worth more than any model.
```

## ✏️ Tasks

1. For six questions of your own, decide regression or classification and say why.
2. Take a dataset and identify which columns could serve as a regression target and which as a classification target.
3. Find a numeric column that is really a category, and explain the test you applied.
4. Argue both sides of the 1–5 rating question.
5. For a supervised problem you care about, say where the labels would come from and what they would cost.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

# 1 - number -> regression; category -> classification.

for col in df.columns:                                                 # 2
    n = df[col].nunique()
    kind = ("classification target" if n <= 10 else
            "regression target" if pd.api.types.is_numeric_dtype(df[col]) else
            "feature only")
    print(f"{col:<30}{n:>6} distinct   {kind}")

# 3 - The test: IS THE ARITHMETIC MEANINGFUL?
#     age 30 + age 30 = 60 makes sense, so age is a number.
#     pincode + pincode is nonsense, so pincode is a category.
#     Also: phone numbers, roll numbers, any ID, and year-as-a-label.

# 4 - REGRESSION: ratings are ORDERED and 4 is genuinely closer to 5
#       than 1 is, so the model should know that.
#     CLASSIFICATION: only five values exist, you cannot ship "3.7 stars",
#       and the gap between 4 and 5 may not equal the gap between 1 and 2.
#     Build both, compare, and defend your choice.

# 5 - Spam: free from user reports, but noisy.
#     Medical images: a specialist labels each one - accurate, expensive,
#       and often the binding constraint on the whole project.
#     Loan outcomes: arrive by themselves, but months later.
```
</details>

## ❓ MCQs

**Q1.** What makes learning "supervised"?
- (a) A human watches it train  (b) The training data includes the correct answers  (c) It uses a GPU  (d) The data is clean

**Q2.** Predicting a house price is…
- (a) Classification  (b) Regression  (c) Clustering  (d) Generation

**Q3.** A `pincode` column of numbers should be treated as…
- (a) Numeric — it is stored as a number  (b) A category, because arithmetic on it is meaningless  (c) The target  (d) Dropped

**Q4.** Why does regression need different metrics from classification?
- (a) Different libraries  (b) A regression prediction can be *nearly* right; a classification one is right or wrong  (c) Classification is harder  (d) They do not

**Q5.** On many real projects, the binding constraint is…
- (a) The algorithm  (b) The cost and quality of getting labels  (c) CPU speed  (d) The programming language

<details><summary>Answers</summary>

**A1 — (b).** You supply the answer key; the model learns the mapping.

**A2 — (b) Regression.** A price is a number.

**A3 — (b).** **If the arithmetic is meaningless, it is a category.**

**A4 — (b).** "Off by ₹4,000" is meaningful; "40% wrong on this row" is not.

**A5 — (b) Labels.** More often than the choice of algorithm.
</details>

---

# 11. Unsupervised Learning

**No answers. You give the model data and it finds structure on its own.**

🧠 **Analogy: a box of loose photographs.** Supervised learning is a box already sorted into labelled envelopes — *family, holidays, work* — and you learn to file new photos correctly. **Unsupervised learning is an unsorted box: you decide the piles yourself.** Two people would make different piles, and neither would be wrong.

```text
SUPERVISED    X and y  ->  learn X -> y
UNSUPERVISED  X only   ->  find structure in X
```

> **There is no answer key, so there is no accuracy score.** That single fact changes how you judge everything in this family.

## The three kinds

| Kind | Question it answers | Example |
|---|---|---|
| **Clustering** | *What natural groups exist?* | Customer segments |
| **Dimensionality reduction** | *Can I describe this with fewer columns?* | Compressing 50 columns to 2 for a plot |
| **Association rule mining** | *What goes with what?* | "Customers who bought X also bought Y" |

## Clustering

```python
# illustrative: a syntax reference, not runnable as written.
from sklearn.cluster import KMeans

groups = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(X)
# Notice: NO y anywhere. That missing y is the whole difference.
```

**You must tell k-Means how many groups to look for.** It cannot decide that for you — Session 7 shows how to choose with evidence.

## Dimensionality reduction

🧠 **Analogy: the shadow of a teapot.** A teapot is three-dimensional; its shadow is flat. **A well-chosen angle casts a shadow you still recognise as a teapot; a bad angle casts a blob.** Reduction is choosing the angle.

**You cannot plot thirteen columns. You can plot two.**

## Association rule mining

🧠 **Analogy: watching a supermarket's trolleys all day.** Bread and butter keep appearing together — a real pattern. **But so do bread and shopping bags, because almost every trolley has a bag.** Telling those apart is the whole skill, and Session 7 gives you the number that does it (*lift*).

## 📘 Examples

**Example 1 — the missing `y`**

```python
# illustrative: a syntax reference, not runnable as written.
clf = LogisticRegression().fit(X, y)     # supervised   - needs y
km  = KMeans(n_clusters=5).fit(X)        # unsupervised - NO y anywhere
```

**Example 2 — customer segmentation**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

mall = pd.read_csv(BASE + "clustering/Mall_Customers.csv")
print(mall.head())
print("\nNobody labelled these customers.")
print("The segments do not exist until you find them.")
```

**Example 3 — the three kinds, on one business**

```text
A shop's data:

  CLUSTERING     which kinds of customer do we have?
                 -> five segments, which marketing can act on

  REDUCTION      our 40 survey questions really measure 3 things
                 -> plot every customer on one chart

  ASSOCIATION    what is bought together?
                 -> put those two products on the same shelf
```

**Example 4 — why there is no accuracy score**

```text
Two analysts cluster the same customers.
  Analyst A finds 4 segments.
  Analyst B finds 6.

NEITHER IS WRONG. There is no correct answer to compare against.

You judge a clustering by whether the groups are SEPARATED (a
silhouette score) and whether anyone can ACT on them - and the second
matters more.
```

## 🌍 Scenarios

**Scenario 1 — segmentation that a business can use**

```text
The clustering produces five groups. The DELIVERABLE is not the
silhouette score - it is this table:

  Careful    high income, low spending   -> persuade, they can afford more
  Target     high income, high spending  -> protect, your best customers
  Standard   mid, mid                    -> the bulk, steady offers
  Careless   low income, high spending   -> watch credit risk
  Sensible   low, low                    -> low priority

A CLUSTER YOU CANNOT NAME IS A CLUSTER YOU CANNOT USE.
```

**Scenario 2 — reduction for seeing**

```text
A survey has 40 questions. Nobody can look at 40 dimensions.

Reduce to 2, plot every respondent, and the groups become visible in
one glance - along with the outliers who answered unlike anyone else.

ALWAYS report how much information the 2-D picture kept. At 90% it is
a fair summary; at 40% it can mislead badly.
```

**Scenario 3 — association rules, and the trap**

```text
"90% of people who buy bread also buy shopping bags."

Sounds like a finding. But if 90% of ALL baskets contain a bag, you
have discovered nothing at all.

The number that catches this is LIFT, and Session 7 covers it. High
confidence with lift near 1 is the classic beginner's trap.
```

## ✏️ Tasks

1. Name the three kinds of unsupervised learning and give a business question for each.
2. Load the mall customer data and say which two columns you would cluster on, and why.
3. Explain why unsupervised learning has no accuracy score.
4. Describe a case where reducing to two dimensions would help you see something.
5. Write an association rule that sounds impressive but means nothing, and say why.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

# 1 - Clustering  : "which customer segments should get which offer?"
#     Reduction   : "can I plot my 40-column survey to spot outliers?"
#     Association : "which products should sit next to each other?"

mall = pd.read_csv(BASE + "clustering/Mall_Customers.csv")             # 2
print(mall.columns.tolist())
# Income and Spending Score - they are the two the business can ACT on,
# because they map directly onto "who should get which offer".

# 3 - There is no correct answer to compare against. Accuracy needs an
#     answer key, and unsupervised learning does not have one. Two
#     analysts can produce different clusterings and both be defensible.

# 4 - A 40-question survey: reduce to 2, plot every respondent, and the
#     groups and the odd-one-out respondents become visible at a glance.
#     ALWAYS report how much information the 2-D view kept.

# 5 - "90% of people who buy bread also buy shopping bags."
#     If 90% of ALL baskets contain a bag, the bread told you nothing.
#     The number that catches this is LIFT - Session 7.
```
</details>

## ❓ MCQs

**Q1.** What distinguishes unsupervised from supervised learning?
- (a) It is faster  (b) There is no target column  (c) It uses neural networks  (d) It needs more data

**Q2.** Why can you not report accuracy for a clustering?
- (a) It is too slow  (b) There is no correct answer to compare against  (c) sklearn lacks it  (d) You can

**Q3.** Which family answers *"what products are bought together?"*
- (a) Clustering  (b) Association rule mining  (c) Dimensionality reduction  (d) Regression

**Q4.** Two analysts cluster the same data and get different groupings. This means…
- (a) One is wrong  (b) It can be perfectly legitimate — there is no single right answer  (c) The data is bad  (d) A bug

**Q5.** The real deliverable of a customer segmentation is…
- (a) The silhouette score  (b) A table of named segments someone can act on  (c) The number of clusters  (d) The code

<details><summary>Answers</summary>

**A1 — (b).** No `y` anywhere.

**A2 — (b).** Accuracy needs an answer key.

**A3 — (b) Association rule mining.**

**A4 — (b).** Different features, scalings and k values all give defensible answers.

**A5 — (b).** **A cluster you cannot name is a cluster you cannot use.**
</details>

---

# 12. Reinforcement Learning

**No answer key, and no fixed dataset. The model learns by *doing* and being rewarded or punished.**

🧠 **Analogy: training a dog.** You do not hand the dog a labelled dataset of sit-versus-stand. It tries something, and gets a treat or nothing. **Over many attempts it works out which actions earn treats.**

🧠 **Analogy: learning a video game with no instructions.** You press buttons. Sometimes the score goes up, sometimes you die. **Nobody told you the rules — you inferred them from consequences.**

```text
        +-------------+   action    +---------------+
        |    AGENT    | ----------> |  ENVIRONMENT  |
        |  (learner)  | <---------- |   (world)     |
        +-------------+  reward     +---------------+
                        + new state
```

## The vocabulary

| Term | Means | In a game |
|---|---|---|
| **Agent** | The learner | The player |
| **Environment** | The world it acts in | The game |
| **State** | The current situation | The screen right now |
| **Action** | What it can do | Move, jump, fire |
| **Reward** | The feedback signal | Points gained or lost |
| **Policy** | Its strategy | What to do in each situation |

## How it differs from the other two

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Needs labels | ✅ | ❌ | ❌ |
| Needs a fixed dataset | ✅ | ✅ | **❌ — it generates its own** |
| Feedback | The right answer | None | **A reward, often delayed** |
| Learns from | Examples | Structure | **Consequences** |

> **The hard part is delayed reward.** In chess you learn you lost at move 60 — **which of your 60 moves was the mistake?** Assigning credit across a long sequence is the central problem of the field.

## Where it is genuinely used

| Domain | Application |
|---|---|
| **Games** | Chess, Go, StarCraft — superhuman |
| **Robotics** | Learning to walk, grasp, balance |
| **Recommendations** | What to show next to keep someone engaged |
| **Data centres** | Cooling control that cuts energy use |
| **Autonomous vehicles** | Decision-making in simulation |
| **LLM alignment** | RLHF — tuning a model on human preference |

> **That last row matters for Session 10.** The step that turns a raw language model into a helpful assistant is reinforcement learning from human feedback.

## ⚠️ Why this course does not build one

**Reinforcement learning needs an environment the agent can act in, millions of times.** A game has one. A loan dataset does not — you cannot approve a loan, see the outcome, and try again.

> **RL is the right tool for sequential decisions with feedback.** For "predict this from that", which is almost every business problem, supervised learning is the answer. **This course covers RL as a concept and builds the other two.**

## 📘 Examples

**Example 1 — the loop, in words**

```text
STATE   the maze position is (3, 4)
ACTION  move right
REWARD  -1 (a step costs time)
STATE   now at (3, 5)
...
ACTION  reach the exit
REWARD  +100

After thousands of runs the agent has a POLICY: from any position,
which move leads to the exit fastest.
```

**Example 2 — the reward defines the behaviour, for better or worse**

```text
Reward a cleaning robot for "amount of dirt collected".
It learns to tip the bin out and clean it up again.

The robot did EXACTLY what you rewarded. You rewarded the wrong thing.
THE REWARD IS THE SPECIFICATION - and getting it wrong is the most
common RL failure.
```

**Example 3 — the three types on one problem**

```text
A recommendation system:

  SUPERVISED     predict the rating this user would give this film
  UNSUPERVISED   group users with similar taste
  REINFORCEMENT  choose what to show NEXT to maximise long-term
                 engagement, accounting for the fact that today's
                 choice changes what they watch tomorrow

Only the third one models the CONSEQUENCES of its own choices.
```

**Example 4 — delayed reward, made concrete**

```text
Chess: 60 moves, then you lose.

  Which move lost it? Move 12 may have been the real error, and moves
  13-59 merely played out a lost position.

  The agent must spread the blame backwards across the whole sequence.
  This is the CREDIT ASSIGNMENT problem, and it is why RL needs so
  many episodes to learn anything.
```

## 🌍 Scenarios

**Scenario 1 — a traffic light system**

```text
AGENT        the signal controller
ENVIRONMENT  the junction
STATE        queue lengths on each approach
ACTION       which direction gets green, and for how long
REWARD       negative total waiting time

Over months it learns timings no engineer wrote - and adapts when a
new road opens.
```

**Scenario 2 — where RL is the wrong tool**

```text
"Predict whether this loan will default."

You have 10,000 past decisions with known outcomes. That is a LABELLED
DATASET, and supervised learning answers it directly.

RL would require APPROVING loans to see what happens - experimenting
on real people with real money. Wrong tool, and unethical.
```

**Scenario 3 — RLHF, and why you will meet it again**

```text
A raw language model predicts the next word from internet text. It is
not yet helpful, and it has no idea what a good answer looks like.

  1. People rank pairs of model answers, best first
  2. A reward model learns to predict those human preferences
  3. The language model is tuned by RL to score well on that reward

THIS IS THE STEP THAT MAKES AN ASSISTANT USEFUL, and it is
reinforcement learning. Session 10 returns to it.
```

## ✏️ Tasks

1. Define agent, environment, state, action and reward for a game you know.
2. Explain the delayed-reward problem with an example of your own.
3. Design a reward for a robot vacuum, then find a way it could be exploited.
4. For one problem, describe how supervised, unsupervised and reinforcement approaches would each tackle it.
5. Explain why a loan-default problem should not use reinforcement learning.

<details><summary>Solutions</summary>

```text
1  A racing game:
     AGENT        the car being driven
     ENVIRONMENT  the track and the other cars
     STATE        position, speed, what is ahead
     ACTION       steer, accelerate, brake
     REWARD       + for progress and finishing, - for crashing

2  A student studies all term and gets one mark at the end. WHICH
   study session helped? The feedback is a single number covering
   months of decisions - the same problem chess poses over 60 moves.

3  Reward = "dirt collected". The robot tips the bin out and collects
   it again. Or reward = "time spent cleaning", and it cleans a clean
   floor forever.
   THE REWARD IS THE SPECIFICATION. Getting it wrong is the most
   common RL failure.

4  Recommending films:
     SUPERVISED     predict the rating a user would give
     UNSUPERVISED   group users with similar taste
     REINFORCEMENT  choose what to show next to maximise long-term
                    engagement, knowing today's choice changes
                    tomorrow's options

5  You already HAVE 10,000 labelled outcomes, so supervised learning
   answers it directly. RL would mean approving loans to see what
   happens - experimenting on real people with real money. Wrong tool,
   and unethical.
```
</details>

## ❓ MCQs

**Q1.** Reinforcement learning learns from…
- (a) Labelled examples  (b) Rewards and consequences of its own actions  (c) Structure in data  (d) Nothing

**Q2.** What is the "credit assignment" problem?
- (a) Deciding who owns the model  (b) Working out which of many earlier actions caused a delayed outcome  (c) Splitting the data  (d) Choosing a reward

**Q3.** A cleaning robot rewarded for "dirt collected" tips out the bin. This shows…
- (a) A bug  (b) The reward is the specification, and it was specified wrongly  (c) RL does not work  (d) Too little training

**Q4.** Why does this course not build an RL system?
- (a) It is too hard  (b) RL needs an environment to act in millions of times, which a fixed dataset is not  (c) It is obsolete  (d) No libraries exist

**Q5.** Which step turns a raw language model into a helpful assistant?
- (a) Supervised classification  (b) Reinforcement learning from human feedback  (c) Clustering  (d) Dimensionality reduction

<details><summary>Answers</summary>

**A1 — (b).** No labels, no fixed dataset — it generates its own experience.

**A2 — (b).** Which of 60 chess moves lost the game?

**A3 — (b).** **The robot did exactly what you rewarded.**

**A4 — (b).** You cannot approve a loan, see the outcome, and try again.

**A5 — (b) RLHF.** Session 10 returns to it.
</details>

---

## ⭐ Checkpoint Problem 3 — Name that learning type

> **Uses:** Topics 10–12.

**The problem.** For fifteen real problems, name the learning type — and where it is supervised, say whether it is regression or classification. **Then find the two that are genuinely ambiguous and argue both sides.**

<details><summary>Solution</summary>

```python
problems = [
    ("Predict tomorrow's temperature",              "Supervised - Regression"),
    ("Is this email spam?",                         "Supervised - Classification"),
    ("Group customers by buying habits",            "Unsupervised - Clustering"),
    ("Teach a robot arm to grasp a cup",            "Reinforcement"),
    ("Predict a house price",                       "Supervised - Regression"),
    ("Which products are bought together?",         "Unsupervised - Association"),
    ("Plot a 40-column survey in 2-D",              "Unsupervised - Reduction"),
    ("Will this patient be readmitted?",            "Supervised - Classification"),
    ("Play chess at superhuman level",              "Reinforcement"),
    ("Detect an unusual credit card transaction",   "Either - see below"),
    ("How many units will we sell next month?",     "Supervised - Regression"),
    ("Which of five plans will they choose?",       "Supervised - Classification"),
    ("Control data-centre cooling to cut energy",   "Reinforcement"),
    ("What rating out of 5 will they give?",        "Either - see below"),
    ("Find natural topics in 10,000 documents",     "Unsupervised - Clustering"),
]

print(f"{'problem':<46}type")
print("-" * 78)
for p, t in problems:
    print(f"{p:<46}{t}")

from collections import Counter
print("\nCOUNT BY FAMILY")
for fam, n in Counter(t.split(" - ")[0] for _, t in problems).most_common():
    print(f"  {n:>2}  {fam}")

print("""
THE TWO AMBIGUOUS ONES

1. DETECTING AN UNUSUAL TRANSACTION

   SUPERVISED if you have confirmed fraud labels: it is a
     classification problem, and a badly imbalanced one.
   UNSUPERVISED if you do not: it becomes anomaly detection - find
     what does not look like the rest, with no labels at all.

   The SAME BUSINESS PROBLEM, and the family depends entirely on
   whether anyone labelled the past cases.

2. PREDICTING A 1-5 RATING

   REGRESSION: the values are ordered, and 4 really is closer to 5
     than 1 is. A model should know that.
   CLASSIFICATION: only five values exist, you cannot ship "3.7
     stars", and the gap between 4 and 5 may not equal the gap
     between 1 and 2.

   Build both, compare, and defend the choice. The reasoning is the
   deliverable, not the answer.

WHAT THE COUNT SHOWS

  Supervised dominates, and that is true of industry too. It is why
  Sessions 5, 6 and 8 are all supervised, Session 7 is unsupervised,
  and reinforcement learning stays a concept in this course.
""")
```

**The exercise is the ambiguous pair.** Beginners want every problem to have one right label. **The fraud row shows that the family can depend on something outside the data science entirely — whether anyone did the labelling work.**
</details>

**Make it harder:**

1. Add five problems from your own field and classify them.
2. For the fraud row, describe what you would do if you had labels for only 200 of 2 million transactions.
3. For each of the three families, name the metric you would report and why.

---

# Part D — Types of Data in Machine Learning

# 13. Types of Data in Machine Learning

**From the real world to numbers.** A model does arithmetic — so everything it learns from must eventually become numbers. **How much work that takes depends on what shape the data arrives in.**

🧠 **Analogy: a filing cabinet, a shoebox and a labelled envelope.** A **filing cabinet** has drawers, folders and a fixed layout — you find anything instantly. A **shoebox of photos, letters and receipts** has everything in it and no structure at all. A **labelled envelope** sits between: not a cabinet, but at least the outside tells you what is inside.

## The three types

| | Structured | Unstructured | Semi-structured |
|---|---|---|---|
| Shape | Rows and columns | No fixed shape | Tagged, but flexible |
| Examples | CSV, Excel, SQL tables | Text, images, audio, video | JSON, XML, HTML, logs |
| Share of the world's data | ~20% | **~80%** | The rest |
| Ready for a model? | **Almost** | **No — needs heavy work** | Nearly — needs flattening |
| Tools | Pandas, SQL | Deep learning | Parsers, then Pandas |
| This course | **Sessions 2–8** | Sessions 9–12 | Session 11 |

> **Roughly 80% of the world's data is unstructured**, and almost all of the *easy value* is in the 20% that is structured. **That is why this course spends Sessions 2 to 8 on tables.**

## Structured data

```text
person_age  person_income  loan_intent  loan_status
        29          39704      MEDICAL            1
        28          36889     PERSONAL            0
```

**Fixed columns, one type each, one row per thing.** This is what `pd.read_csv` gives you, and what every model in Sessions 5 to 8 expects.

## Unstructured data

```text
TEXT    "The delivery was late but the product is excellent."
IMAGE   a 1920 x 1080 grid of pixel values
AUDIO   44,100 amplitude measurements per second
VIDEO   30 images per second, plus audio
```

**None of it is rows and columns.** To use it you must first turn it into numbers — pixels into arrays, words into token IDs, sound into waveforms. **That conversion is what deep learning is for**, and Session 9 explains how.

## Semi-structured data

```json
{
  "customer_id": 4471,
  "name": "Priya Sharma",
  "orders": [
    {"item": "tea", "qty": 2, "price": 15},
    {"item": "samosa", "qty": 3, "price": 20}
  ]
}
```

**It has structure — keys, nesting, tags — but not a fixed table shape.** One customer has two orders; another has none. **You flatten it into a table before modelling.**

## 📘 Examples

**Example 1 — structured, straight into Pandas**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv")
print(df.shape)
print(df.dtypes.head())
print("\nRows and columns, one type per column. Ready for Session 5.")
```

**Example 2 — unstructured text, turned into numbers**

```python
from sklearn.feature_extraction.text import CountVectorizer

reviews = ["the delivery was late",
           "excellent product, fast delivery",
           "late and damaged"]

vec = CountVectorizer()
counts = vec.fit_transform(reviews)

print("vocabulary:", vec.get_feature_names_out())
print("as numbers:\n", counts.toarray())
print("\nThree sentences became a table of counts. NOW a model can use it.")
```

**Example 3 — semi-structured JSON, flattened**

```python
import pandas as pd

data = [
    {"customer_id": 1, "name": "Priya",
     "orders": [{"item": "tea", "qty": 2}, {"item": "samosa", "qty": 3}]},
    {"customer_id": 2, "name": "Arun", "orders": [{"item": "coffee", "qty": 1}]},
]

flat = pd.json_normalize(data, record_path="orders",
                         meta=["customer_id", "name"])
print(flat)
print("\nNested JSON -> a flat table. THAT is the work semi-structured needs.")
```

**Example 4 — an image is just numbers too**

```python
import numpy as np

image = np.array([[0, 128, 255],
                  [64, 192, 32],
                  [255, 0, 128]], dtype=np.uint8)

print("shape:", image.shape, " -> a 3x3 greyscale image")
print("flattened for a model:", image.reshape(-1))
print("\nA colour photo is the same idea: height x width x 3 colour channels.")
```

## 🌍 Scenarios

**Scenario 1 — one company, all three types**

```text
STRUCTURED       the sales database - orders, amounts, dates
UNSTRUCTURED     customer support emails, product photos, call recordings
SEMI-STRUCTURED  the website's server logs, and the API responses

The structured data answers "how much did we sell?" TODAY.
The unstructured data answers "why are customers unhappy?" - and needs
far more work to get at.
```

**Scenario 2 — the same question, three ways**

```text
"Are our customers satisfied?"

  STRUCTURED       average star rating from the ratings table
                   -> one line of Pandas
  SEMI-STRUCTURED  parse review JSON from the API, then aggregate
                   -> flatten first, then one line of Pandas
  UNSTRUCTURED     read 50,000 free-text reviews and judge sentiment
                   -> needs an NLP model (Session 10 or 12)

Same question. Wildly different amounts of work.
```

**Scenario 3 — why beginners should start with structured data**

```text
A beginner project on IMAGES needs: a GPU, thousands of labelled
examples, augmentation, a deep learning framework, and hours of
training - before you learn anything about modelling.

The SAME beginner on a CSV can load, explore, clean, train, evaluate
and deploy in an afternoon - and learns every core idea properly.

Learn the ideas on tables. Apply them to images later.
```

## ✏️ Tasks

1. Classify ten data sources you use as structured, unstructured or semi-structured.
2. Turn three sentences into a numeric table with `CountVectorizer`.
3. Flatten a nested JSON structure into a Pandas table.
4. Represent a small greyscale image as a NumPy array and flatten it.
5. Take one business question and describe how you would answer it from each of the three data types.

<details><summary>Solutions</summary>

```python
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# 1 - Structured      : a CSV of marks, a bank statement, an SQL table
#     Unstructured    : WhatsApp messages, photos, voice notes, videos
#     Semi-structured : an API's JSON response, an HTML page, server logs

reviews = ["the delivery was late",                                    # 2
           "excellent product, fast delivery",
           "late and damaged"]
vec = CountVectorizer()
counts = vec.fit_transform(reviews)          # fit FIRST
print(vec.get_feature_names_out())           # then the vocabulary exists
print(counts.toarray())

data = [{"customer_id": 1, "name": "Priya",                            # 3
         "orders": [{"item": "tea", "qty": 2}, {"item": "samosa", "qty": 3}]},
        {"customer_id": 2, "name": "Arun",
         "orders": [{"item": "coffee", "qty": 1}]}]
print(pd.json_normalize(data, record_path="orders", meta=["customer_id", "name"]))

img = np.array([[0, 128, 255], [64, 192, 32], [255, 0, 128]], dtype=np.uint8)  # 4
print(img.shape, "->", img.reshape(-1))

# 5 - "Are customers satisfied?"
#     Structured      : average the rating column - one line
#     Semi-structured : parse the review JSON, then aggregate
#     Unstructured    : run sentiment analysis over 50,000 free-text
#                       reviews - needs an NLP model
#     Same question, wildly different amounts of work.
```
</details>

## ❓ MCQs

**Q1.** Roughly what share of the world's data is unstructured?
- (a) 20%  (b) 50%  (c) 80%  (d) 5%

**Q2.** JSON and XML are…
- (a) Structured  (b) Unstructured  (c) Semi-structured  (d) Not data

**Q3.** Which type is nearly ready for a model as it arrives?
- (a) Structured  (b) Unstructured  (c) Semi-structured  (d) None

**Q4.** Why does this course spend Sessions 2–8 on tables?
- (a) Tables are more important  (b) Structured data lets you learn every core idea without a GPU or thousands of labels  (c) Images are impossible  (d) Pandas is easier

**Q5.** To use text in a model you must first…
- (a) Nothing  (b) Convert it into numbers  (c) Translate it  (d) Sort it

<details><summary>Answers</summary>

**A1 — (c) ~80%.** And almost all the *easy* value is in the structured 20%.

**A2 — (c) Semi-structured.** Tagged and nested, but not a fixed table.

**A3 — (a) Structured.** `pd.read_csv` and you are almost there.

**A4 — (b).** **Learn the ideas on tables; apply them to images later.**

**A5 — (b).** Pixels into arrays, words into token IDs — that conversion is what deep learning is for.
</details>

---

## ⭐ Checkpoint Problem 4 — Data type sorter

> **Uses:** Topic 13.

**The problem.** Take fifteen real data sources, classify each of the three types, and for each say what work is needed before a model could use it.

<details><summary>Solution</summary>

```python
sources = [
    ("Student marks spreadsheet",     "Structured",      "encode text columns, scale"),
    ("Customer support emails",       "Unstructured",    "tokenise, embed, or use an LLM"),
    ("Server access logs",            "Semi-structured", "parse lines, then aggregate"),
    ("Bank transaction table",        "Structured",      "almost ready; encode + scale"),
    ("Product photographs",           "Unstructured",    "resize, normalise pixels, augment"),
    ("Weather API JSON response",     "Semi-structured", "json_normalize into a table"),
    ("Call centre recordings",        "Unstructured",    "speech-to-text, then NLP"),
    ("SQL sales database",            "Structured",      "query into a DataFrame"),
    ("Scraped HTML product pages",    "Semi-structured", "parse tags, extract fields"),
    ("CCTV video",                    "Unstructured",    "frame extraction, then vision model"),
    ("Sensor readings CSV",           "Structured",      "handle gaps, resample by time"),
    ("Tweets about a brand",          "Unstructured",    "clean text, then sentiment model"),
    ("Config files (XML)",            "Semi-structured", "parse into key-value pairs"),
    ("Hospital admissions table",     "Structured",      "encode, handle missing values"),
    ("Handwritten forms (scanned)",   "Unstructured",    "OCR, then treat as text"),
]

print(f"{'source':<30}{'type':<18}work needed before modelling")
print("-" * 92)
for s, t, w in sources:
    print(f"{s:<30}{t:<18}{w}")

from collections import Counter
counts = Counter(t for _, t, _ in sources)
print("\nCOUNT")
for t, n in counts.most_common():
    print(f"  {n:>2}  {t}")

print(f"""
WHAT TO NOTICE

  The STRUCTURED rows need encoding and scaling - which you can already
  do, from Session 3. They are a day's work.

  The UNSTRUCTURED rows need a whole extra stage before modelling even
  begins: OCR, speech-to-text, tokenisation, or pixel normalisation.
  Sessions 9 to 12 cover those.

  The SEMI-STRUCTURED rows are the middle case: parse and flatten, and
  then they become structured rows. That is usually an afternoon of
  fiddly but straightforward work.

  {counts['Structured']} of {len(sources)} sources are ready quickly. That ratio is why
  a first project should almost always start with a table.
""")
```

**The final paragraph is the lesson.** Students choose image and audio projects because they sound impressive, and then spend the whole project on data preparation. **The classification tells you what you are signing up for.**
</details>

**Make it harder:**

1. For three unstructured sources, name the specific library or model you would use.
2. Take one semi-structured source and write the code that flattens it.
3. Rank all fifteen by how long you think preparation would take, and justify the top and bottom.

---

# Part E — Mathematical Foundations

# 14. Mathematical Foundations for Machine Learning

**You need less maths than you fear, and you need it later than you think.**

🧠 **Analogy: driving a car.** You can drive safely and usefully without knowing how an engine works. **You need the engine when something breaks, when you want to tune it, or when you want to build one.** Maths is the engine.

> **You can complete Sessions 1 to 8 of this course with school arithmetic.** The maths below is what lets you *understand* rather than *use* — and understanding is what separates someone who can apply a library from someone who can diagnose a problem.

## The four areas, and what each buys you

| Area | The idea in one line | Where it shows up | Needed for |
|---|---|---|---|
| **Statistics** | Summarising and comparing data | Mean, median, spread, correlation | **Everything — start here** |
| **Linear algebra** | Doing arithmetic on whole grids at once | Vectors, matrices, dot products | Understanding NumPy and neural networks |
| **Calculus** | How a change in one thing changes another | Slopes, gradients | Understanding *how models train* |
| **Probability** | Reasoning about uncertainty | Distributions, conditional probability | Classification, Naive Bayes, evaluation |

## What you genuinely need, in order

### 1. Statistics — needed from day one

| Concept | Why |
|---|---|
| Mean and median | Comparing them detects skew — Session 3 |
| Standard deviation | The spread; `StandardScaler` uses it |
| Quartiles and IQR | Outlier detection — Session 3 |
| Correlation | Which columns move together — Session 2 |
| Distributions | Is this column bell-shaped or skewed? |

**You have already used all five.**

### 2. Linear algebra — needed to understand, not to use

```text
A VECTOR   is a list of numbers        [78, 92, 45]        - one student
A MATRIX   is a grid of numbers        a whole class       - your X
A DOT PRODUCT multiplies and sums      78*0.3 + 92*0.5 + 45*0.2
```

**Every row of your DataFrame is a vector; the whole table is a matrix.** When Session 9 writes `X @ W1 + b1`, that is one matrix multiplication doing thousands of dot products at once.

### 3. Calculus — needed only to understand training

**One idea does almost all the work: a *gradient* is a slope, and it tells you which way is downhill.**

```text
Loss is high        -> which way should I change this weight to reduce it?
The gradient answers exactly that. Step that way. Repeat.
```

**That is gradient descent, and it is how every neural network learns.** Session 9 builds it in NumPy — **and you will not need to differentiate anything by hand.**

### 4. Probability — needed for classification

| Concept | Where |
|---|---|
| Probability of an event | `predict_proba` — Session 5 |
| Conditional probability | Naive Bayes — Session 5 |
| Distributions | Understanding what "normal" means for a column |
| Confidence intervals | Honest reporting — Session 8 |

## The honest priority order

```text
1. STATISTICS      learn properly - you use it every single day
2. PROBABILITY     learn the basics - classification depends on it
3. LINEAR ALGEBRA  learn the vocabulary - vectors, matrices, dot product
4. CALCULUS        understand gradients conceptually; the rest can wait

DO NOT delay starting ML until you have "finished the maths".
Nobody finishes. You learn the maths as you meet the need for it.
```

## 📘 Examples

**Example 1 — the statistics you already use**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

col = df["person_income"]
print(f"mean   {col.mean():>12,.0f}")
print(f"median {col.median():>12,.0f}")
print(f"std    {col.std():>12,.0f}")
q1, q3 = col.quantile([0.25, 0.75])
print(f"IQR    {q3 - q1:>12,.0f}")
print("\nMean far above median -> SKEWED. That single comparison is")
print("statistics doing real work, and you have used it since Session 2.")
```

**Example 2 — a vector and a dot product, by hand**

```python
import numpy as np

marks = np.array([78, 92, 45])            # a VECTOR - one student
weights = np.array([0.3, 0.5, 0.2])       # subject weights

print("dot product:", marks @ weights)     # 78*0.3 + 92*0.5 + 45*0.2
print("by hand    :", 78*0.3 + 92*0.5 + 45*0.2)
print("\nThat one operation is what a neural network does thousands of")
print("times per layer. Session 9 writes exactly this.")
```

**Example 3 — a gradient, without calculus**

```python
def loss(w):
    """A simple bowl-shaped loss. The lowest point is at w = 3."""
    return (w - 3) ** 2

w = 10.0
for step in range(6):
    # The gradient measured numerically: does the loss rise or fall
    # if I nudge w slightly upward?
    nudge = 0.001
    gradient = (loss(w + nudge) - loss(w)) / nudge
    w -= 0.3 * gradient            # step DOWNHILL
    print(f"step {step + 1}: w = {w:6.3f}   loss = {loss(w):7.4f}")

print("\nIt walked to w = 3 without anyone differentiating anything.")
print("That is gradient descent, and Session 9 uses exactly this idea.")
```

**Example 4 — probability, as you will meet it**

```python
# illustrative: a syntax reference, not runnable as written.
model.predict(X_test)          # -> 1          the decision
model.predict_proba(X_test)    # -> [0.13, 0.87]  the CONFIDENCE

# The second is a probability, and it is far more useful:
# it lets you set your own threshold instead of always using 0.5.
```

## 🌍 Scenarios

**Scenario 1 — where the maths actually saved someone**

```text
A student's model scored 99% and they were delighted.

Someone asked: "what is the base rate of the positive class?"
It was 99%. The model was predicting "no" every time.

THAT QUESTION IS STATISTICS - and it is the difference between
shipping a useless model and catching the problem in ten seconds.
```

**Scenario 2 — where you can safely not know the maths**

```text
You do not need to derive the Random Forest algorithm to use one well.

You DO need to know:
  - what it does (many trees vote)
  - when it fails (it cannot extrapolate beyond its training range)
  - how to evaluate it honestly (Session 8)

That is engineering knowledge, not mathematical derivation.
```

**Scenario 3 — a realistic learning plan**

```text
NOW           mean, median, std, quartiles, correlation
              -> you already have these

WITH SESSION 5 probability basics, and what predict_proba means

WITH SESSION 8 sampling variation, confidence intervals
              -> why a 0.003 difference is not a result

WITH SESSION 9 vectors, matrices, dot products, gradients
              -> conceptually, so the NumPy network makes sense

LATER         derivations, optimisation theory, the maths of
              regularisation - only if you go deeper

Nobody learns it all first. You learn it when you meet the need.
```

## ✏️ Tasks

1. For one numeric column, compute mean, median, std and IQR, and say what they tell you.
2. Compute a dot product by hand and with NumPy, and confirm they match.
3. Run the gradient descent example and change the learning rate from 0.3 to 1.5. What happens?
4. Explain in your own words what a gradient tells you, without using the word derivative.
5. Write your own maths learning plan, ordered by when you will actually need each piece.

<details><summary>Solutions</summary>

```python
import pandas as pd, numpy as np
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
col = pd.read_csv(BASE + "loan_data_10k.csv")["person_income"]

q1, q3 = col.quantile([.25, .75])                                      # 1
print(f"mean {col.mean():,.0f}  median {col.median():,.0f}  "
      f"std {col.std():,.0f}  IQR {q3-q1:,.0f}")
# Mean well above median -> skewed. The std is large relative to the
# median -> a wide spread. Both matter for Session 3's scaling choice.

marks, weights = np.array([78, 92, 45]), np.array([0.3, 0.5, 0.2])     # 2
print(marks @ weights, "==", 78*.3 + 92*.5 + 45*.2)

def loss(w): return (w - 3) ** 2                                       # 3
for lr in [0.3, 1.5]:
    w = 10.0
    for _ in range(6):
        g = (loss(w + .001) - loss(w)) / .001
        w -= lr * g
    print(f"lr={lr}: final w = {w:.3f}")
# lr=0.3 walks smoothly to 3. lr=1.5 OVERSHOOTS and diverges - the loss
# grows instead of shrinking. That is Session 9's learning-rate lesson.

# 4 - A gradient tells you which direction makes the loss go UP, and by
#     how steeply. So you step the OPPOSITE way, and repeat. It is
#     feeling which way the ground slopes under your feet in fog.

# 5 - Statistics now (you use it daily); probability with Session 5;
#     sampling and confidence intervals with Session 8; vectors,
#     matrices and gradients with Session 9; derivations only if you
#     go deeper. Do NOT wait until the maths is "finished" - nobody
#     finishes.
```
</details>

## ❓ MCQs

**Q1.** Which area of maths do you use most in day-to-day ML?
- (a) Calculus  (b) Statistics  (c) Topology  (d) Number theory

**Q2.** A gradient tells you…
- (a) The answer  (b) Which direction changes the loss, and how steeply  (c) The accuracy  (d) The number of layers

**Q3.** Every row of your DataFrame is…
- (a) A matrix  (b) A vector  (c) A scalar  (d) A gradient

**Q4.** A model scores 99% and the base rate is 99%. The model is probably…
- (a) Excellent  (b) Predicting the majority class every time  (c) Overfitted  (d) Undertrained

**Q5.** Should you finish learning the maths before starting ML?
- (a) Yes  (b) No — learn it as you meet the need for it  (c) Only calculus  (d) Only linear algebra

<details><summary>Answers</summary>

**A1 — (b) Statistics.** Mean, median, spread, correlation — every single day.

**A2 — (b).** Feeling which way the ground slopes, in fog.

**A3 — (b) A vector.** The whole table is a matrix — your `X`.

**A4 — (b).** **That question is statistics, and it takes ten seconds.**

**A5 — (b).** Nobody finishes. You learn it when you meet the need.
</details>

---

# Part F — Building AI

# 15. AI Programming Languages & Ecosystem

**Python won, and it is worth knowing why — because the reason tells you what to learn next.**

| Language | Used for | Why |
|---|---|---|
| **Python** | **Almost everything** | Every major library is Python-first |
| R | Statistics, academic research | Excellent statistical tooling |
| SQL | **Getting the data** | Your data lives in a database |
| Julia | Numerical research | Fast, but a small ecosystem |
| C++ | The engine room | PyTorch and TensorFlow are C++ underneath |
| JavaScript | Models in a browser | TensorFlow.js |

> **Python did not win on speed — it is slow.** It won because the fast parts are written in C, and Python is the comfortable handle on them. **You write Python; C does the work.**

## The Python ecosystem, in the order you meet it

| Library | Does | Session |
|---|---|---|
| **NumPy** | Arrays and fast numeric work | 2 |
| **Pandas** | Tables, loading, cleaning | 2 |
| **Matplotlib / Seaborn** | Charts | 2 |
| **scikit-learn** | Classical ML — the workhorse | 3–8 |
| **TensorFlow / Keras** | Deep learning | 9 |
| **PyTorch** | Deep learning, research-first | 9 |
| **Transformers** | Pretrained models from Hugging Face | 12 |
| **Streamlit** | Turning a model into a web app | 5, 11 |
| **joblib** | Saving and loading trained models | 5 |

> **SQL deserves more respect than it gets on course syllabi.** In a real job, a large share of the work is getting the right rows out of a database before Python ever sees them.

## The tools around the code

| Tool | For |
|---|---|
| **Jupyter / Colab** | Exploring, with output beside the code |
| **VS Code / PyCharm** | Writing real programs |
| **Git and GitHub** | Version control, and your portfolio |
| **conda / venv** | Keeping project dependencies separate |
| **Docker** | Making it run identically elsewhere |

## 📘 Examples

**Example 1 — the ecosystem in one workflow**

```python
# illustrative: a syntax reference, not runnable as written.
import pandas as pd                      # load and clean
import numpy as np                       # numeric work
import matplotlib.pyplot as plt          # look at it
from sklearn.ensemble import RandomForestClassifier   # model it
import joblib                            # save it
# then: streamlit run app.py             # ship it
```

**Six libraries, and that is a complete project.**

**Example 2 — why Python is fast enough**

```python
import numpy as np, time

a = np.arange(1_000_000)

t0 = time.time(); total = sum(a); py = time.time() - t0        # Python loop
t0 = time.time(); total = a.sum(); np_ = time.time() - t0      # C underneath

print(f"pure Python : {py:.4f}s")
print(f"NumPy       : {np_:.4f}s")
print(f"NumPy is roughly {py / max(np_, 1e-9):.0f}x faster")
print("\nYou wrote Python. C did the work. That is the whole trick.")
```

**Example 3 — where SQL fits**

```sql
-- illustrative: SQL, run against your database, not in Python
SELECT customer_id, SUM(amount) AS total_spend, COUNT(*) AS orders
FROM   transactions
WHERE  order_date >= '2026-01-01'
GROUP  BY customer_id;
```

```python
# illustrative: a syntax reference, not runnable as written.
df = pd.read_sql(query, connection)      # the result arrives as a DataFrame
```

**The database does the aggregating on millions of rows; Pandas receives thousands.** **Doing it the other way round is a common and expensive mistake.**

**Example 4 — checking what you have**

```python
import importlib
for lib in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn", "streamlit"]:
    try:
        m = importlib.import_module(lib)
        print(f"{lib:<14}{getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"{lib:<14}NOT INSTALLED")
```

## 🌍 Scenarios

**Scenario 1 — a realistic project's tool list**

```text
SQL          pull last year's orders from the warehouse
Pandas       clean and join
Seaborn      explore and plot
scikit-learn train and evaluate
joblib       save the model
Streamlit    a page the sales team can actually use
Git          version it all, and show it to an employer
```

**Scenario 2 — why environments matter**

```text
Project A needs scikit-learn 1.0. Project B needs 1.9.

Install both globally and one of them breaks, silently, in a way that
takes an afternoon to diagnose.

  conda create -n projectA python=3.12
  conda create -n projectB python=3.12

Two separate environments, no conflict. It is why this course insists
your prompt shows (genai) before you install anything.
```

**Scenario 3 — the language question, answered honestly**

```text
"Should I learn R as well?"

If you are doing academic statistics, R is excellent and you will meet
it. For everything else - industry ML, deep learning, deployment,
GenAI - Python is where the ecosystem is.

LEARN PYTHON PROPERLY FIRST. Add SQL second, because that is where
your data actually lives. Everything else is optional.
```

## ✏️ Tasks

1. Check which of the course libraries are installed in your environment and print their versions.
2. Time a pure-Python sum against a NumPy sum on a million values.
3. Write a SQL query that aggregates a table, and say why doing it in SQL beats doing it in Pandas.
4. List the tools you would use for a complete project, from data to deployed app.
5. Explain why two projects might need separate environments.

<details><summary>Solutions</summary>

```python
import importlib, numpy as np, time

for lib in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn",     # 1
            "streamlit", "joblib"]:
    try:
        m = importlib.import_module(lib)
        print(f"{lib:<14}{getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"{lib:<14}NOT INSTALLED")

a = np.arange(1_000_000)                                               # 2
t0 = time.time(); sum(a); py = time.time() - t0
t0 = time.time(); a.sum(); npt = time.time() - t0
print(f"\npython {py:.4f}s   numpy {npt:.4f}s   -> ~{py/max(npt,1e-9):.0f}x")

# 3 - SELECT customer_id, SUM(amount) FROM transactions
#     WHERE order_date >= '2026-01-01' GROUP BY customer_id;
#     The DATABASE aggregates millions of rows and returns thousands.
#     Pulling all 10 million rows into Pandas first wastes memory,
#     network time, and often does not fit at all.

# 4 - SQL -> Pandas -> Seaborn -> scikit-learn -> joblib -> Streamlit,
#     with Git across the whole thing.

# 5 - Different projects pin different library versions. Installed
#     globally they conflict, and the failure is silent and confusing.
#     A separate environment per project removes the problem entirely.
```
</details>

## ❓ MCQs

**Q1.** Why did Python win for ML?
- (a) It is the fastest language  (b) The fast parts are written in C, and Python is the comfortable handle on them  (c) It is the oldest  (d) It is the only option

**Q2.** Which language should you learn second, after Python?
- (a) C++  (b) SQL — your data lives in a database  (c) Julia  (d) JavaScript

**Q3.** Aggregating ten million rows is better done…
- (a) In Pandas after loading everything  (b) In SQL, so only the summary reaches Python  (c) In Excel  (d) By hand

**Q4.** Why use separate conda environments per project?
- (a) Speed  (b) Different projects need different library versions, and globally they conflict silently  (c) Disk space  (d) It is required

**Q5.** Which library is the workhorse of Sessions 3–8?
- (a) TensorFlow  (b) scikit-learn  (c) PyTorch  (d) Transformers

<details><summary>Answers</summary>

**A1 — (b).** **You write Python; C does the work.**

**A2 — (b) SQL.** A large share of real work is getting the right rows out.

**A3 — (b) In SQL.** Doing it the other way round is common and expensive.

**A4 — (b).** The failure is silent and takes an afternoon to diagnose.

**A5 — (b) scikit-learn.**
</details>

---

# 16. Local, Cloud, Edge & No-Code Platforms

**Where does the code actually run?** The answer changes cost, speed, privacy and who can build it.

## The three places

| | Local | Cloud | Edge |
|---|---|---|---|
| Runs on | Your laptop | A rented server | The device itself |
| Cost | Free, once you own it | Per hour or per call | Free, once deployed |
| Power | Limited | **Effectively unlimited** | **Very limited** |
| **Your data** | **Never leaves** | Leaves your machine | **Never leaves** |
| Works offline | ✅ | ❌ | ✅ |
| Best for | Learning, small data, privacy | Training big models | Phones, cameras, sensors |
| Examples | Jupyter on your laptop | Colab, AWS, GCP | Face unlock, a smart camera |

🧠 **Analogy: cooking at home, hiring a restaurant kitchen, or a camping stove.** Home is free and limited. The restaurant kitchen is powerful and charged by the hour. **The camping stove goes with you and works with no electricity — but you are not roasting a whole lamb on it.**

## Choosing

```text
Learning, or a small dataset?           -> LOCAL
Need a GPU you do not own?              -> CLOUD  (start with Colab, free)
Data must not leave the building?       -> LOCAL or your own private cloud
Must work with no internet?             -> EDGE
Millions of predictions a day?          -> CLOUD
Prediction must happen in milliseconds
  on the device?                        -> EDGE
```

> **This course runs on local and Colab.** Both are free, and both are enough for everything through Session 12.

## No-code AI platforms

**You do not always need to write the code.**

| Platform | Does |
|---|---|
| **Google Teachable Machine** | Image, sound and pose classifiers in a browser — Topic 9 |
| **Google AutoML / Vertex AI** | Trains a model on your data, tuning it for you |
| **Azure ML Designer** | Drag-and-drop pipelines |
| **Amazon SageMaker Canvas** | Point-and-click model building |
| **Orange, KNIME** | Visual workflow tools, free |
| **Hugging Face Spaces** | Host and share a model demo |

### When no-code is genuinely the right answer

| Use it | Do not use it |
|---|---|
| Proving an idea is worth pursuing | When you need to control preprocessing |
| A demo for a non-technical audience | When you must explain exactly what it did |
| Teaching the concepts (Topic 9) | When it must run inside your own systems |
| You have no engineer and a deadline | When you need to debug why it is wrong |

> **No-code tools are excellent at showing that something is possible, and poor at telling you why it went wrong.** **Use them to decide whether to build; use code to build.**

## 📘 Examples

**Example 1 — the same task, three places**

```text
CLASSIFYING A PHOTO

  LOCAL  load the model in Python on your laptop.
         Free, private, works offline, limited by your hardware.

  CLOUD  send the image to an API.
         Powerful, scalable, costs per call, and THE IMAGE LEAVES
         YOUR MACHINE.

  EDGE   the model runs on the phone itself.
         Instant, private, works in a tunnel - and it must be small
         enough to fit.
```

**Example 2 — why Colab exists**

```text
Training an image model on a laptop CPU : hours or days
The same model on a Colab GPU           : minutes

Colab gives you a GPU free, in a browser, with nothing to install.
For a student that is the single most useful tool in this list.
```

**Example 3 — an edge constraint, made concrete**

```text
A phone's face unlock must:
  - answer in under a second
  - work with no signal
  - never send your face to a server
  - fit in a few tens of megabytes

That is why edge models are SMALL and QUANTISED - the same idea
Session 12 covers for open language models.
```

**Example 4 — the no-code decision**

```text
"Can we tell good mangoes from bad from a photograph?"

  DAY 1  Teachable Machine, 60 photos, 20 minutes.
         Answer: yes, roughly 85% - the idea is worth pursuing.

  MONTH 1 Build it properly in code: more data, real evaluation,
         a fairness check, and a deployable model.

The no-code step SAVED a month of building something that might not
have worked. It did not replace the building.
```

## 🌍 Scenarios

**Scenario 1 — a hospital**

```text
CONSTRAINT  patient data must not leave the building

  Cloud API           ruled out immediately, whatever its accuracy
  Local / private     the only options
  Edge                for a device in the ward

The DEPLOYMENT CHOICE was decided by policy before anyone compared
model quality. Session 12 returns to this.
```

**Scenario 2 — a student's realistic setup**

```text
EXPLORING     Jupyter locally - fast, offline, no quota
TRAINING      Colab when you need a GPU
SHARING       Streamlit Community Cloud, or Hugging Face Spaces
STORING       GitHub

Total cost: nothing.
```

**Scenario 3 — a small business with no engineer**

```text
NEED     predict which customers are about to leave
HAVE     a spreadsheet, and no data scientist

  START   a no-code AutoML tool on the spreadsheet
  RESULT  a usable model in an afternoon, and a number good enough
          to justify hiring someone

The alternative was doing nothing. No-code beat nothing.
```

## ✏️ Tasks

1. For five scenarios of your choice, decide local, cloud or edge, and give the deciding reason.
2. Name three constraints that force an edge deployment.
3. Use Teachable Machine (Topic 9) and list three things it did *not* let you control.
4. Describe a situation where no-code is the right first step and code is the right second step.
5. Explain why a hospital might rule out a cloud API before comparing accuracy.

<details><summary>Solutions</summary>

```text
1  Learning on a small CSV          -> LOCAL, free and fast
   Training an image model          -> CLOUD, you need a GPU
   Face unlock on a phone           -> EDGE, instant and private
   Serving a million predictions/day-> CLOUD, it must scale
   Patient records analysis         -> LOCAL, data cannot leave

2  Must work offline; must answer in milliseconds; the data must never
   leave the device. Any one of the three forces edge.

3  Teachable Machine does not let you choose the architecture, see or
   control the preprocessing, set the train/test split, choose the
   metric, or inspect why a prediction was made. That is the trade:
   speed and simplicity in exchange for control and explanation.

4  "Can we detect a crop disease from photographs?"
   NO-CODE first: 60 photos, 20 minutes, and you learn whether the
     idea is viable at all.
   CODE second: proper data collection, evaluation, fairness check
     and deployment.
   The no-code step saves you a month of building the wrong thing.

5  Sending patient data to an external API may breach data protection
   rules and hospital policy regardless of how accurate the model is.
   The constraint is legal, not technical, so it decides FIRST.
```
</details>

## ❓ MCQs

**Q1.** Which option keeps data on the device and works offline?
- (a) Cloud  (b) Edge  (c) Both  (d) Neither

**Q2.** Why is Colab so useful for a student?
- (a) It is faster than any laptop  (b) It gives a free GPU in a browser with nothing to install  (c) It is offline  (d) It is private

**Q3.** No-code tools are best used for…
- (a) Production systems  (b) Deciding quickly whether an idea is worth building  (c) Debugging  (d) Fairness auditing

**Q4.** A hospital rules out a cloud API. The reason is most likely…
- (a) Cost  (b) Data must not leave the building  (c) Speed  (d) Accuracy

**Q5.** Edge models must be…
- (a) As large as possible  (b) Small enough to fit and fast enough to answer on-device  (c) Cloud-connected  (d) Written in C++

<details><summary>Answers</summary>

**A1 — (b) Edge.** Local also keeps data in-house, but edge means the device itself.

**A2 — (b).** Hours on a CPU become minutes on a GPU, for free.

**A3 — (b).** **Excellent at showing something is possible, poor at telling you why it went wrong.**

**A4 — (b).** A legal and policy constraint, decided before accuracy is discussed.

**A5 — (b).** Which is why they are small and quantised — the same idea as Session 12's open models.
</details>

---

# 17. AIML Dataset Repositories

**You cannot learn Machine Learning without data, and you should not invent it.**

## Where to get datasets

| Repository | What is there | Best for |
|---|---|---|
| **Kaggle Datasets** | Tens of thousands, with notebooks | **Start here** — real data, real examples |
| **UCI ML Repository** | The classic academic benchmarks | Small, clean, well-documented |
| **Hugging Face Datasets** | Text, image, audio, one-line loading | NLP and deep learning |
| **Google Dataset Search** | A search engine across repositories | Finding something specific |
| **data.gov.in** | Indian government open data | Local, relevant projects |
| **AWS / Google public datasets** | Very large, cloud-hosted | Big data work |
| **`sklearn.datasets`** | A handful built in | Instant testing, no download |
| **This course** | [`datasets/`](../../../datasets/) | Everything used in these sessions |

## What makes a dataset good for a project

| Check | Why |
|---|---|
| **Is there a clear target column?** | Without one you cannot do supervised learning |
| **Enough rows?** | Under a few hundred and your results will be noise |
| **Documented?** | You must know what each column means |
| **Licensed for your use?** | Especially if the project is public |
| **Not already solved to death?** | Titanic and Iris teach; they do not impress |

> ⚠️ **Download it and open it before you commit to a project.** This is the commonest capstone failure: a topic chosen from an interesting title, and the data turns out to be 40 rows, or paywalled, or missing the column the whole idea depended on.

## Loading without downloading

```python
# illustrative: a syntax reference, not runnable as written.
import pandas as pd

# From a URL - works identically on your laptop and in Colab
df = pd.read_csv("https://raw.githubusercontent.com/.../data.csv")

# Built into scikit-learn
from sklearn.datasets import load_iris, fetch_california_housing

# From Hugging Face
from datasets import load_dataset
data = load_dataset("imdb", split="train[:100]")
```

## 📘 Examples

**Example 1 — this course's own datasets**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

for name in ["loan_data_10k.csv",
             "classification/diabetes_prediction_dataset.csv",
             "regression/advertising.csv",
             "clustering/Mall_Customers.csv"]:
    df = pd.read_csv(BASE + name, nrows=5)
    print(f"{name:<52}{len(df.columns):>3} columns")
```

**Reading `nrows=5` first is a cheap way to see the columns without downloading the whole file.**

**Example 2 — a dataset built into scikit-learn**

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
print(df.shape)
print(df.head(3))
print("\nNo download, no internet. Useful for testing code quickly.")
```

**Example 3 — the checks to run before committing**

```python
def dataset_fit_check(df, target):
    """Would this dataset actually support a project?"""
    ok = True
    print(f"Rows       : {len(df):,}", end="")
    if len(df) < 300:
        print("   TOO FEW - results will be noise"); ok = False
    else:
        print("   fine")

    print(f"Columns    : {len(df.columns)}")

    if target not in df.columns:
        print(f"Target     : '{target}' NOT PRESENT"); ok = False
    else:
        n = df[target].nunique()
        kind = "classification" if n <= 10 else "regression"
        print(f"Target     : '{target}', {n} distinct -> {kind}")
        if kind == "classification":
            bal = df[target].value_counts(normalize=True).min()
            print(f"Balance    : minority class {bal:.1%}"
                  f"{'   IMBALANCED - use recall/F1' if bal < 0.2 else ''}")

    miss = df.isna().mean().max()
    print(f"Worst gap  : {miss:.1%} missing"
          f"{'   needs work' if miss > 0.3 else ''}")
    print(f"\nVERDICT: {'usable' if ok else 'NOT SUITABLE - find another'}")

BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
dataset_fit_check(pd.read_csv(BASE + "loan_data_10k.csv"), "loan_status")
```

**Example 4 — the datasets folder in this repository**

```text
datasets/
├── classification/   adult, diabetes, heart failure, iris, loan
├── regression/       advertising, cardekho, salary
├── clustering/       Mall_Customers
├── cv/               image-classification.zip  (Topic 9's demo)
├── nlp/              text datasets
└── prepreprocessing/ pre_data.csv  (deliberately messy)
```

## 🌍 Scenarios

**Scenario 1 — choosing a capstone dataset properly**

```text
WRONG ORDER
  1. pick an exciting topic
  2. look for data
  3. discover there is none, three weeks in

RIGHT ORDER
  1. browse Kaggle or data.gov.in for datasets that interest you
  2. DOWNLOAD two or three and open them
  3. run the fit check above on each
  4. choose the topic the data can actually support
```

**Scenario 2 — the licence question**

```text
A dataset may be free to DOWNLOAD and still not free to USE in a
public project or a commercial product.

Check before you publish:
  - can it be redistributed?
  - is attribution required?
  - does it contain personal data? (then extra rules apply)

Session 12 covers this for models. The same care applies to data.
```

**Scenario 3 — why not Titanic**

```text
Titanic and Iris are excellent for LEARNING - small, clean, and
every question has been answered somewhere.

That is also why they do not make a good capstone. An employer has
seen a hundred Titanic notebooks.

Use them to practise. Use something local and specific - your city's
data, your college's data - for the project you show people.
```

## ✏️ Tasks

1. Find three datasets on Kaggle or data.gov.in that interest you and record their size and target column.
2. Load a dataset with `nrows=5` and list its columns without downloading it all.
3. Write a `dataset_fit_check()` function and run it on two datasets.
4. Load a built-in scikit-learn dataset and print its shape.
5. Check the licence of one public dataset and say whether you could use it in a public project.

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.datasets import load_iris
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

peek = pd.read_csv(BASE + "loan_data_10k.csv", nrows=5)                # 2
print(peek.columns.tolist())
# nrows=5 shows you the columns without pulling the whole file.

def dataset_fit_check(df, target):                                     # 3
    print(f"rows {len(df):,}  columns {len(df.columns)}")
    if target in df.columns:
        n = df[target].nunique()
        print(f"target '{target}': {n} distinct -> "
              f"{'classification' if n <= 10 else 'regression'}")
    print("worst missing:", f"{df.isna().mean().max():.1%}")
    print("verdict:", "usable" if len(df) >= 300 and target in df.columns
          else "NOT SUITABLE")

dataset_fit_check(pd.read_csv(BASE + "loan_data_10k.csv"), "loan_status")

iris = load_iris(as_frame=True).frame                                  # 4
print(iris.shape)

# 5 - Look for the licence on the dataset's page. Free to DOWNLOAD is
#     not the same as free to REDISTRIBUTE or to use commercially, and
#     personal data carries extra obligations regardless of licence.
```
</details>

## ❓ MCQs

**Q1.** What should you do before committing to a capstone topic?
- (a) Write the proposal  (b) Download the data and open it  (c) Choose an algorithm  (d) Build the app

**Q2.** A dataset has 40 rows. It is…
- (a) Fine  (b) Too small — your results will be noise  (c) Ideal for beginners  (d) Better than 10,000 rows

**Q3.** Why avoid Titanic for a capstone?
- (a) It is too hard  (b) It is excellent for learning, but an employer has seen a hundred of them  (c) It is unlicensed  (d) It has no target

**Q4.** Free to download means…
- (a) Free to use anywhere  (b) Not necessarily free to redistribute or use commercially  (c) Public domain  (d) No licence exists

**Q5.** `pd.read_csv(url, nrows=5)` is useful because…
- (a) It is more accurate  (b) You see the columns without pulling the whole file  (c) It cleans the data  (d) It is required

<details><summary>Answers</summary>

**A1 — (b).** **The commonest capstone failure is choosing a topic before checking the data exists.**

**A2 — (b).** Under a few hundred rows and you are measuring noise.

**A3 — (b).** Practise on it; show something local and specific.

**A4 — (b).** Check redistribution, attribution, and whether it holds personal data.

**A5 — (b).** A cheap look before you commit.
</details>

---

# Part G — The Workflow

# 18. The Machine Learning Workflow

**Every machine learning project — a bank's loan system, a hospital's screening tool, a road-safety camera, a news classifier — follows the same seven stages in the same order.**

**The stages never change. What fills them changes completely.** This topic explains each stage properly, and then shows how it plays out on four genuinely different kinds of data.

```text
1. PROBLEM DEFINITION              what exactly are we predicting, and why?
2. DATA COLLECTION                 where does the data come from?
3. DATA CLEANING & PREPROCESSING   getting it into a usable state
4. FEATURE ENGINEERING & SELECTION deciding what the model gets to see
5. MODEL SELECTION & TRAINING      choosing an approach and fitting it
6. MODEL EVALUATION & TUNING       is it any good, honestly?
7. MODEL DEPLOYMENT                putting it where someone can use it
```

> **Stages 2 to 4 are roughly 70% of the work on a real project.** Nobody puts that in the demo video, and it is why Sessions 2 and 3 came before this one.

---

## The four datasets we will follow

**To see why the stages behave so differently, we will carry four projects through all seven.**

| | Dataset | Kind of data | The question |
|---|---|---|---|
| **A** | `loan_data_10k.csv` | **Tabular** — 10,000 rows, 14 columns | Will this loan application be approved? |
| **B** | `diabetes_prediction_dataset.csv` | **Tabular**, but only 8.5% positive | Is this patient likely to have diabetes? |
| **C** | `cv/helmet_dataset` | **Images** — 60 training photographs | Is this rider wearing a helmet? |
| **D** | `nlp/bbc-text.csv` | **Text** — 2,225 articles, ~337 words each | Which section does this article belong to? |

**They were chosen because they differ on the things that actually change your decisions:** how much data there is, how balanced it is, what a mistake costs, and whether the raw input is already numbers.

---

## Stage 1 — Problem definition

### What this stage is

**Problem definition is turning a wish into a question a model can answer.**

A wish sounds like *"use AI to reduce bad loans"*. A question sounds like *"given the information on an application form, predict whether this loan will be approved"*. **The difference is that the second one names an input, an output and a moment in time.**

### What you must settle before writing any code

| Decision | Why it cannot wait |
|---|---|
| **What is the target?** | It determines whether this is regression or classification, and everything downstream |
| **What information exists at the moment of prediction?** | Anything recorded *after* the decision cannot be used — that is leakage |
| **What does success mean, numerically?** | "Good accuracy" is not a target. 85%? Better than the current process? |
| **What does a mistake cost, and to whom?** | This decides your metric more than any other consideration |
| **What is this system *not*?** | A decision-support tool and an autonomous decider are different products |

> **The last row is the one people skip, and it changes what you build.** Writing *"this is not an automatic approver — a person decides"* at stage 1 means you will design for a human reader at stage 7. Deciding that at the end is too late.

### The four projects

**A — Loan.** *Given an application, predict approval.* The target is `loan_status`, which is 0 or 1, so this is **binary classification**. It is roughly 50/50, so accuracy will be a fair headline number. **The system is decision support: it gives a loan officer a consistent first opinion, and the officer decides.**

**B — Diabetes.** *Given routine health measurements, predict whether a patient likely has diabetes.* Also binary classification — but here **the two mistakes are wildly unequal.** Missing a diabetic patient may delay treatment for years; a false alarm means one confirmatory blood test. **Success is therefore defined as high recall**, and the system is a *screening* tool, never a diagnosis.

**C — Helmet.** *Given a photograph of a rider, decide whether they are wearing a helmet.* Binary classification again, but the input is an image rather than a row. **The purpose shapes everything:** a tool that flags photographs for a human reviewer needs different accuracy from one that issues automatic fines. **We define it as flagging for review**, which makes a false alarm cheap.

**D — BBC news.** *Given an article, predict which section it belongs to.* **Five categories, so this is multi-class classification** — and that alone changes the metrics, because "accuracy" now averages over five classes that may not be equally easy.

| | A — Loan | B — Diabetes | C — Helmet | D — News |
|---|---|---|---|---|
| Task type | Binary classification | Binary classification | Binary classification | **Multi-class** (5) |
| Target balance | 50 / 50 | **8.5% positive** | 30 / 30 | Roughly even |
| Costly mistake | False approval | **False negative** — a missed case | Depends on the use | Neither, particularly |
| Success measure | Accuracy, plus recall | **Recall** | Precision if it fines, recall if it flags | Accuracy and per-class F1 |
| What it is *not* | An auto-approver | A diagnosis | An automatic fine | An editor |

---

## Stage 2 — Data collection

### What this stage is

**Data collection is getting hold of the raw material, and establishing that you are allowed to use it.**

It sounds administrative. **It is where projects most often die**, and always for the same reason: someone chose the question before checking whether the data existed.

### The four questions

1. **Does the data exist, and can you actually get it?** Not "is it on the internet somewhere" — have you downloaded it and opened it?
2. **Is there enough?** A few hundred rows will give you noise, not results.
3. **Do the labels exist, and who made them?** Supervised learning needs answers, and someone has to have produced them.
4. **Are you permitted to use it for this purpose?** Consent, licence, and personal-data rules all live here.

> **The cost and quality of labels is often the real constraint on a project — more than the choice of algorithm.** Some labels arrive free; some cost a specialist's time per example.

### The four projects

**A — Loan.** 10,000 historical applications with their outcomes. **The labels arrived by themselves** — whether a loan was approved is simply recorded. That is the easiest possible labelling situation, and it is why tabular business problems are a good place to start.

**B — Diabetes.** 100,000 patient records. The labels come from confirmed diagnoses, which are reliable but **arrive slowly** — a patient's status may only be established months after the measurements were taken. **And this is personal medical data**, so consent and data-protection rules are not optional extras; they may decide where the model is allowed to run at all.

**C — Helmet.** **Sixty training photographs.** That is the striking fact about this project. Thirty riders with helmets, thirty without. Somebody had to photograph or gather them, and somebody had to sort them into two folders — **labelling images is manual work, and it is why image projects so often stall here.**

**D — BBC news.** 2,225 articles, each already tagged with its section. **The labels were a by-product of publishing** — the newspaper had to file each article somewhere anyway. **When labels are a by-product of an existing process, you have found a good project.**

| | A — Loan | B — Diabetes | C — Helmet | D — News |
|---|---|---|---|---|
| Volume | 10,000 rows | 100,000 rows | **60 images** | 2,225 articles |
| Where labels came from | Recorded automatically | Clinical diagnosis | **Hand-sorted into folders** | A by-product of publishing |
| Cost per label | Effectively zero | Expensive, and slow | Manual, but cheap | Effectively zero |
| Legal weight | Financial regulation | **Medical privacy** | Photographs of people | Low |
| Biggest risk | Data may not reflect today | Consent and residency | **Far too few examples** | Articles may be dated |

---

## Stage 3 — Data cleaning and preprocessing

### What this stage is

**Cleaning is fixing what is wrong. Preprocessing is putting what remains into the shape a model can consume.** Together they are usually the largest single part of the work.

**Session 3 covered this in full for tabular data:** missing values, duplicates, outliers, encoding, scaling, and the train-test split. The important thing to see here is that **the same idea applies to images and text, but the operations look nothing alike.**

### Why the operations differ so much

**A model consumes numbers.** How far your raw data is from being numbers determines how much work this stage takes.

- **A table** is *already* numbers, apart from its text columns. Cleaning is repair work.
- **An image** is already numbers too — a grid of pixel values — but they need to be made *consistent*: same size, same scale, same orientation.
- **Text** is not numbers at all. It has to be converted, and how you convert it is a significant decision in itself.

### The four projects

**A — Loan.** Remove the one impossible age of 144. Fill three missing cells. Encode five text columns — one of which, education level, is genuinely ordered and must not be encoded alphabetically. Scale the numeric columns after splitting. **This is exactly Session 3.**

**B — Diabetes.** The same operations, and one additional decision that dominates: **because only 8.5% of patients are positive, the split must be stratified**, or a test set could end up with too few positive cases to measure anything. **The imbalance is not a data-quality problem — it is a real property of the world — but it changes how you handle every later stage.**

**C — Helmet.** Cleaning means something different here. **Every image must be resized to identical dimensions**, because a model needs a fixed input size. Pixel values are scaled from 0–255 down to 0–1. And with only sixty images, **augmentation becomes near-essential**: flipped, rotated and brightness-adjusted copies multiply the effective dataset. **A flipped photograph of a helmet is still a helmet — but you must check that claim for your own data, because flipping a photograph of text would not be.**

**D — BBC news.** Text needs the most transformation. Articles are lowercased, punctuation handled, and very common words like *the* and *and* usually removed because they appear everywhere and distinguish nothing. **Then the text must become numbers** — either by counting words, or by converting to embeddings that place similar meanings near each other. **With a median length of 337 words, you also have to decide how much of each article to keep.**

| | A — Loan | B — Diabetes | C — Helmet | D — News |
|---|---|---|---|---|
| Missing values | Fill or drop | Fill or drop | **Not applicable** | Empty articles |
| The main operation | Encode and scale | Encode and scale | **Resize and normalise pixels** | **Tokenise into numbers** |
| Special concern | Impossible values | **Stratify the split** | **Augment — 60 is very few** | Stop words, document length |
| Result the model sees | A row of numbers | A row of numbers | A fixed-size pixel array | A vector per document |

---

## Stage 4 — Feature engineering and selection

### What this stage is

**Feature engineering is creating new inputs that make the pattern easier to see. Feature selection is deciding which inputs to keep.**

**The two halves pull in opposite directions on purpose**: engineering adds columns that carry more meaning, selection removes columns that carry little.

### Why this stage is where domain knowledge pays

**A model can only find patterns among the columns you give it.** If the useful quantity is a *relationship* between two columns rather than either one alone, you have to construct it. **No amount of tuning discovers a ratio you never built.**

### The four projects

**A — Loan.** The single most useful feature is not in the raw data: **the ratio of loan amount to income.** A loan of ₹800,000 is modest against a large income and impossible against a small one, and **neither column alone carries that.** A banker would tell you this immediately; the algorithm would not.

**B — Diabetes.** Much of the feature engineering has already been done by medicine. **BMI is itself an engineered feature** — weight divided by height squared — invented precisely because neither weight nor height alone is informative. **HbA1c is a deliberately constructed measure of long-term blood sugar.** You are inheriting decades of domain work.

**C — Helmet.** **This is the great difference with deep learning: you do not engineer features at all.** A convolutional network learns its own — early layers find edges, later ones find shapes, and eventually something that responds to the presence of a helmet. **You gave up control and gained the ability to work with data you could never have described in columns.**

**D — BBC news.** The choice of text representation *is* the feature engineering. Counting words treats each word independently. Weighting rarer words more heavily emphasises the terms that distinguish sections — *goal* and *striker* mark a sports article far better than *the* does. **Embeddings go further and capture meaning, so that *football* and *soccer* sit near each other.** Each choice produces genuinely different features.

| | A — Loan | B — Diabetes | C — Helmet | D — News |
|---|---|---|---|---|
| Who creates the features | **You** | Medicine already did | **The network itself** | Your choice of representation |
| A good example | loan ÷ income | BMI, HbA1c | Learned edge and shape detectors | Word weighting or embeddings |
| Domain knowledge needed | High | Very high | Low | Moderate |
| Can you explain a feature? | Yes, easily | Yes | **Often not** | Partly |

---

## Stage 5 — Model selection and training

### What this stage is

**Model selection is choosing the kind of model. Training is fitting it to the data.**

**Training is usually the shortest part of the whole project** — often a single line of code and a few seconds of computation. The thinking is in the choosing.

### The rule that saves the most time

**Start with a baseline, then the simplest real model, and only then something more powerful.**

**A baseline is a model that does something trivial** — always predicting the most common class, for instance. **It costs three lines, and it tells you the score that anything you build must clearly beat.** A model that cannot beat "always guess the majority" has learned nothing, whatever its accuracy looks like.

### How the choice differs

**The single biggest factor is whether your data is a table.**

**A — Loan.** A table, so a Random Forest is the natural choice: it handles mixed types, needs no scaling, trains in seconds, and reports which columns mattered. **Logistic regression is worth trying first because it is explainable — and in lending, explainability may be a legal requirement rather than a nicety.**

**B — Diabetes.** Also a table, so the same family applies. But the imbalance changes how you train: **you may weight the rare class more heavily**, so the model stops being rewarded for ignoring it.

**C — Helmet.** A table-based model cannot use raw pixels sensibly, so this needs a neural network. **But with only sixty images, training one from scratch is hopeless** — it would simply memorise them. **The standard answer is transfer learning:** take a network already trained on millions of general images, and retrain only its final layer on your sixty. **You inherit its knowledge of edges, textures and shapes, and teach it only the last step.**

**D — BBC news.** Once text is converted to numbers, classical models work surprisingly well — **Naive Bayes has been a strong baseline for text classification for decades and trains in moments.** A fine-tuned language model will usually do better, at considerably more cost.

| | A — Loan | B — Diabetes | C — Helmet | D — News |
|---|---|---|---|---|
| Baseline | Always approve | Always predict "no" | Always guess "helmet" | Always guess "sport" |
| Sensible first model | Logistic regression | Logistic regression | **A pretrained network** | Naive Bayes |
| Likely best | Random Forest | Random Forest, class-weighted | **Transfer learning** | Fine-tuned language model |
| Training time | Seconds | Seconds | Minutes | Seconds to hours |
| Main risk | — | Ignoring the rare class | **Memorising 60 images** | Overfitting to vocabulary |

---

## Stage 6 — Model evaluation and tuning

### What this stage is

**Evaluation is finding out how good the model really is. Tuning is trying to improve it.**

**They are in the same stage because they are dangerous together.** Every time you look at a score and change something, you learn a little about your test data — and your estimate becomes slightly less honest.

### Choosing the metric

**This is where stage 1 pays off.** You decided what a mistake costs; the metric is how you count them.

| Metric | Answers | Right when |
|---|---|---|
| **Accuracy** | What fraction did I get right? | The classes are balanced |
| **Precision** | When I say yes, how often am I right? | A false alarm is expensive |
| **Recall** | Of all the real cases, how many did I catch? | **A miss is expensive** |
| **F1** | One number balancing both | You need a single figure |

> **Accuracy is the default and it is frequently the wrong choice.** On the diabetes data, a model that predicts "no diabetes" for every patient scores **91.5%** — and finds nobody. That number looks like success on a dashboard and represents total failure.

### Evaluating honestly

**One train-test split gives one number, and that number moves depending on which rows happened to land where.** Cross-validation repeats the split several times and reports a mean and a spread. **Reporting the spread matters: a difference smaller than it is not a result.**

### The four projects

**A — Loan.** Balanced classes, so accuracy is meaningful — **but report precision and recall too**, because approving a bad loan and declining a good customer are different failures with different costs.

**B — Diabetes.** **Recall is the headline.** How many diabetic patients did we find? The 91.5% accuracy trap makes this vivid, and it is why the metric was decided at stage 1.

**C — Helmet.** With sixty training images, the test set is tiny — perhaps a dozen photographs. **A single misclassification moves the score by eight percentage points.** Any number computed here carries enormous uncertainty, and **saying so is part of doing the work honestly.**

**D — BBC news.** Five classes, so the useful output is a **per-class breakdown**. Overall accuracy can hide the fact that four sections are recognised easily and one is not — and the confusion matrix will often show which two categories the model mixes up, which is genuinely informative.

| | A — Loan | B — Diabetes | C — Helmet | D — News |
|---|---|---|---|---|
| Headline metric | Accuracy + F1 | **Recall** | Recall, with a large caveat | Per-class F1 |
| The trap | — | **91.5% by predicting "no"** | **The test set is far too small** | One weak class hidden by the average |
| Confidence in the number | Good | Good | **Poor — and say so** | Reasonable |

---

## Stage 7 — Model deployment

### What this stage is

**Deployment is putting the model where the people who need it can actually use it.**

**A model in a notebook helps nobody.** This stage is where a piece of analysis becomes a piece of software, and it raises questions that have nothing to do with accuracy.

### What deployment actually requires

| Question | Why it matters |
|---|---|
| **Who uses this, and on what screen?** | A loan officer's tool looks nothing like a camera system |
| **Does it answer instantly, or overnight?** | Real-time and batch are different engineering problems |
| **Where does it run?** | A server, or the device itself — privacy and connectivity decide this |
| **What does the user see?** | A label, a probability, or an explanation |
| **What happens when it is wrong?** | There must be a route to a human |
| **How will anyone notice it degrading?** | Models decay as the world moves away from their training data |

> **That last question is the one most projects never answer.** A model quietly gets worse as reality drifts, and without monitoring nobody finds out until something goes badly wrong.

### The four projects

**A — Loan.** A simple internal web page. The officer enters the details, sees a decision **with its confidence**, and reads a plain-English explanation of the main factors. **Showing the probability rather than a bare yes/no is what keeps a human genuinely in the loop.**

**B — Diabetes.** A screening tool used inside a clinic. **It must display a probability, never a diagnosis**, and it must state plainly on screen what it is not. **If patient data cannot leave the building, the model runs locally — and that constraint is decided before anyone compares accuracy.**

**C — Helmet.** Likely to run **on the camera itself**, because sending video to a server is expensive and slow. That forces the model to be small and fast, which in turn constrains stage 5. **Deployment decided the model choice, not the other way round.**

**D — BBC news.** Batch, not real-time. Articles can be classified overnight, and a moderate delay costs nothing. **This is the easiest deployment of the four** — and it means you can afford a larger, slower, more accurate model.

| | A — Loan | B — Diabetes | C — Helmet | D — News |
|---|---|---|---|---|
| Where it runs | Internal server | **Locally, for privacy** | **On the camera** | Batch job |
| Response time | Seconds | Seconds | **Milliseconds** | Overnight |
| What the user sees | Decision + confidence + reason | **A probability, never a diagnosis** | A flagged image | A suggested tag |
| Monitor for | New applicant types | Population drift | New helmet styles, weather | New topics and vocabulary |

---

## What the four projects show

**The seven stages are fixed. Almost every judgement inside them is not.**

| Stage | The thing that differed most |
|---|---|
| **1. Problem definition** | What a mistake costs — and therefore what success means |
| **2. Data collection** | Where labels came from, and what they cost |
| **3. Cleaning & preprocessing** | How far the raw data was from being numbers |
| **4. Features** | Whether *you* build them or the model learns them |
| **5. Model & training** | Whether the data is a table |
| **6. Evaluation** | Which metric, decided by stage 1 |
| **7. Deployment** | Where it runs — which sometimes decides stage 5 |

**Three lessons worth carrying forward:**

1. **Stage 1 decides stage 6.** You cannot choose a metric sensibly without having decided what a mistake costs. **The diabetes project shows what happens when you skip that thinking: 91.5% accuracy and zero patients found.**

2. **The amount of data changes what is possible, not just what is easy.** Sixty images rules out training a network from scratch and makes transfer learning and augmentation necessary rather than optional.

3. **Deployment constraints can reach backwards.** The helmet model must be small because it runs on a camera — **a stage 7 requirement that determines a stage 5 decision.** Thinking about deployment at the start is not premature; it is how you avoid building the wrong thing.

## 📘 Examples

**Example 1 — a problem definition that would be approved**

> **Question:** given the information on a loan application form, predict whether the application will be approved.
> **Target:** `loan_status`, binary — this is classification.
> **Success:** clearly beat "approve everyone"; report recall as well as accuracy; be explainable to an applicant.
> **This is not:** an automatic approver. A loan officer decides.

**Every line does work.** The target fixes the model family, the success criterion fixes the metric, and the final line fixes the interface you will build at stage 7.

**Example 2 — the same definition done badly**

> **Question:** use AI to improve lending.
> **Data:** we will find some.
> **Success:** high accuracy.

**Three failures.** The question names no target and no moment of prediction, the data does not exist yet, and "high accuracy" is meaningless without a baseline to beat.

**Example 3 — how the metric follows from the cost**

| Project | Cost of a false negative | Cost of a false positive | Metric that follows |
|---|---|---|---|
| Diabetes screening | A missed illness, possibly for years | One extra blood test | **Recall** |
| Spam filtering | Junk reaches the inbox | **A job offer is lost** | **Precision** |
| Helmet flagging | An unsafe rider is missed | A reviewer glances at one photo | **Recall** |
| News tagging | An article is mis-filed | An article is mis-filed | Balanced — F1 |

**Read the two cost columns and the metric chooses itself.** That is what stage 1 is for.

**Example 4 — where the work actually goes**

```text
  Problem definition        5%
  Data collection          15%
  Cleaning & preprocessing 40%   <-
  Feature engineering      15%   <-  together, about 70%
  Model selection           5%
  Evaluation & tuning      10%
  Deployment               10%
```

**The stage everyone pictures when they imagine machine learning — choosing and fitting the model — is the smallest one.**

## 🌍 Scenarios

**Scenario 1 — the same data, three different projects**

```text
ONE loan dataset:

  "Will this application be approved?"   -> classification; recall matters
  "How much will this customer borrow?"  -> regression; RMSE matters
  "What kinds of borrower do we have?"   -> clustering; no target at all

STAGE 1 DECIDES EVERYTHING THAT FOLLOWS. Get it wrong and stages 2 to 7
are all correct answers to the wrong question.
```

**Scenario 2 — a project that died at stage 2**

```text
A team chose "predict crop yield from satellite images".

Stage 1 was excellent - a clear target, a real user, a measurable outcome.

Stage 2 was never done. The satellite imagery was licensed and costly,
and the yield labels existed for only 40 fields.

They discovered this in week three, and the project ended there.

DOWNLOAD THE DATA BEFORE YOU COMMIT TO THE QUESTION.
```

**Scenario 3 — deployment reaching backwards into model choice**

```text
The helmet system must run on the camera, because streaming video to a
server is too slow and too expensive.

That constraint means the model must fit in a few tens of megabytes and
answer in milliseconds.

Which rules out the largest and most accurate networks - at STAGE 5,
because of a requirement from STAGE 7.

This is why you ask "where will this run?" at the beginning.
```

## ✏️ Tasks

1. Write a full stage 1 definition for a problem at your own college: target, task type, success criterion, cost of each kind of mistake, and what the system is *not*.
2. For a dataset of your choice, answer the four data-collection questions in writing — including where the labels would come from and what they would cost.
3. Take the four projects in this topic and rank them by how much work stage 3 would take. **Justify the top and the bottom.**
4. For a problem you care about, name two features you would engineer from domain knowledge, and say why a model could not discover them alone.
5. Choose a metric for a problem of your own and defend it using the two cost columns from Example 3. Then write the monitoring plan: what three numbers would tell you it had stopped working?

<details><summary>Solutions</summary>

```text
1  A good answer names the target column, says whether it is regression
   or classification, states success as a NUMBER to beat rather than
   "high accuracy", says which of the two mistakes is worse and why,
   and ends with a sentence about what the system is not.
   The last line is the one that changes what you build.

2  Does the data exist and have you opened it; is there enough; do the
   labels exist and who made them; are you allowed to use it for this
   purpose. If any answer is "we will find out later", the project is
   not ready to start.

3  Most work : the helmet images - resizing, normalising, and heavy
                augmentation to compensate for only 60 examples.
     then     : the BBC text - lowercasing, stop words, and a real
                decision about how to turn words into numbers.
     then     : diabetes - ordinary tabular cleaning, plus stratifying
                the split because of the 8.5% imbalance.
   Least work : loan - ordinary tabular cleaning, and that is all.
   The ranking follows one thing: how far the raw data is from numbers.

4  For loans: loan-to-income ratio, and income per year of employment.
   A model can only find patterns among the columns it is GIVEN. If the
   useful quantity is a RELATIONSHIP between two columns, somebody has
   to construct it. No amount of tuning discovers a ratio you never built.

5  Metric follows cost. If a miss is worse than a false alarm, report
   recall; if a false alarm is worse, report precision.
   A monitoring plan might watch: the predicted positive rate against
   its historical value; the score measured on newly confirmed cases;
   and whether incoming data still resembles the training population.
   Any of the three moving is a trigger to review.
```
</details>

## ❓ MCQs

**Q1.** Which stages take roughly 70% of the work on a real project?
- (a) Model selection and training  (b) Data collection, cleaning and feature engineering  (c) Deployment  (d) Problem definition

**Q2.** What should a stage 1 definition include that people usually leave out?
- (a) The algorithm  (b) What the system is *not*  (c) The programming language  (d) The training time

**Q3.** A model predicts "no diabetes" for every patient and scores 91.5% accuracy. This shows…
- (a) An excellent model  (b) That accuracy is the wrong metric when classes are imbalanced  (c) A coding error  (d) Too little data

**Q4.** With only 60 training images, training a network from scratch is…
- (a) Ideal  (b) Hopeless — it would memorise them; use transfer learning instead  (c) Faster  (d) More accurate

**Q5.** In the helmet project, deployment on the camera constrained…
- (a) Nothing  (b) The model choice at stage 5, because it must be small and fast  (c) The labels  (d) The metric

<details><summary>Answers</summary>

**A1 — (b).** Stages 2 to 4. **The stage everyone imagines — fitting the model — is the smallest one.**

**A2 — (b) What it is *not*.** Writing "this is not an automatic approver" at stage 1 means you design for a human reader at stage 7.

**A3 — (b).** It found **zero** patients. That number looks like success on a dashboard and represents total failure.

**A4 — (b).** Take a network trained on millions of general images and retrain only its last layer.

**A5 — (b).** **A stage 7 requirement determining a stage 5 decision** — which is why you ask "where will this run?" at the beginning.
</details>

---

# 19. ML & AI APIs

**You will not write machine learning algorithms yourself.** You will use libraries that other people have written, tested and optimised over many years — and **choosing the right one is a real decision that this topic prepares you to make.**

---

## What an API is, in this context

**An API is a ready-made toolbox that someone else built, with a documented set of handles you can pull.**

🧠 **Analogy: a kettle.** You do not build a heating element, a thermostat and a safety cut-off every time you want tea. **You press a switch.** The kettle's "API" is that switch, the lid and the spout — the parts you are meant to touch. Everything inside is somebody else's careful work.

**In practice, an ML library gives you:**

- **Algorithms** that are already written and heavily tested
- **A consistent way to use them**, so learning one teaches you the rest
- **Speed**, because the slow parts are written in a faster language underneath
- **Correctness**, because thousands of people have found the bugs before you

> **Writing your own algorithm is an excellent way to learn** — you will do exactly that in Session 9. **It is a poor way to build something people depend on.** A library that has been used by millions has had its edge cases found; yours has not.

---

## The four libraries

**These are the four you will hear named constantly.** Each exists for a different job.

### scikit-learn

**The library for classical machine learning on tables.**

If your data has rows and columns — a spreadsheet, a CSV, a database table — **this is almost always the right answer.** It covers the whole workflow from Topic 18: splitting data, filling gaps, scaling, encoding, training, and measuring how well it did.

**It is the library used throughout Sessions 3 to 8 of this course.** You have already used parts of it in Session 3.

### TensorFlow

**A library for deep learning, built for large-scale production use.**

Deep learning means models with many layers, used for data that is not a neat table — images, audio, video, text. **TensorFlow's particular strength is everything around the model:** tools for serving it to millions of users, and for shrinking it to run on a phone or a camera.

### Keras

**A friendly way of writing TensorFlow.**

> ⚠️ **Keras is not a competitor to TensorFlow.** It sits *on top of* it. **You write Keras; TensorFlow does the work underneath.** This confuses almost everyone at first.

**Keras exists because raw TensorFlow is verbose.** Keras lets you describe a network in a few readable lines, and it is the usual choice when learning deep learning or building something quickly.

### PyTorch

**A library for deep learning, built research-first.**

PyTorch gives you more direct control over what happens during training. **You write more code, and in exchange you can change any step of the process** — which is exactly what you need when you are trying something nobody has tried before.

**Most new deep learning research is published in PyTorch**, so if you go further into the field you will meet it.

---

## The one thing they all share

**Every one of these libraries follows the same shape:**

```text
1. CREATE    make a model, and set how it should behave
2. TRAIN     show it the training data, so it learns
3. PREDICT   give it new data, and get answers back
4. MEASURE   check those answers against the truth
```

**Learn that shape once and every library becomes familiar.** The names differ slightly, but the sequence never does.

> **You have already built this pattern yourself.** At the end of Session 1 you wrote a small class with a method that learned from data and a method that made predictions. **That is the whole idea — a library is the same shape, done properly and at scale.**

---

## How to choose

**One question does most of the work: is your data a table?**

```text
Is your data rows and columns?
   YES -> scikit-learn.  Stop here.
   NO  |
       Is it images, audio, video, or text?
          YES -> you need a deep learning library
                    learning or prototyping      -> Keras
                    research, or unusual ideas   -> PyTorch
                    large-scale deployment       -> TensorFlow
```

**This connects directly to Topic 13.** Structured data goes to scikit-learn; unstructured data goes to a deep learning library. **The type of your data chooses your library, not the other way round.**

| Library | Built for | Reach for it when | Not the tool for |
|---|---|---|---|
| **scikit-learn** | Classical ML on **tables** | Rows and columns | Images, audio, raw text |
| **TensorFlow** | Deep learning, production scale | Serving many users; running on phones | A simple spreadsheet problem |
| **Keras** | Writing TensorFlow readably | Learning deep learning; building fast | Very unusual architectures |
| **PyTorch** | Deep learning, research-first | Full control; trying new ideas | A simple spreadsheet problem |

---

## ⚠️ The mistake beginners make

**Deep learning sounds more advanced, so people reach for it first.**

**On a table of a few thousand rows, a classical model from scikit-learn will usually:**

- **train in seconds** rather than many minutes
- **score as well or better** than a neural network
- **tell you which columns mattered**, which a neural network generally cannot
- **need no special hardware**

> **Choose the tool that fits the data, not the one that sounds impressive.** **Session 5 measures this directly on the loan dataset** — you will see the numbers rather than take it on trust.

---

## What does not change

**Whichever library you pick, the seven stages from Topic 18 are unaffected.**

You still define the problem, collect the data, clean it, decide what the model sees, train it, evaluate it honestly, and deploy it. **The library is a detail inside stage 5.**

> **The workflow is the skill. The library is a tool you can swap.** Someone who understands Topic 18 can pick up a new library in a week. Someone who only knows one library's commands cannot do the reverse.

## 📘 Examples

**Example 1 — the same question, four different answers**

| The situation | The library | Why |
|---|---|---|
| A hospital predicting readmission from 30 columns of patient records | **scikit-learn** | It is a table, and the result must be explainable |
| A startup sorting 200,000 product photographs | **Keras** | Images, and they want something working quickly |
| A research group testing a new idea about how networks should learn | **PyTorch** | They must be able to change the training process itself |
| A model that has to run on a phone with no internet | **TensorFlow** | Its tools for shrinking and deploying models are the strongest |

**Example 2 — why "Keras or TensorFlow?" is the wrong question**

**It is like asking "steering wheel or car?".**

**Keras is how you give instructions. TensorFlow is what carries them out.** When you write a network in Keras, TensorFlow is doing the actual computation underneath. **Choosing Keras *is* choosing TensorFlow, with a more comfortable way of writing it.**

**Example 3 — the four libraries against the four projects from Topic 18**

| Project | Data | Library | Reason |
|---|---|---|---|
| **A — Loan approval** | A table of 10,000 rows | **scikit-learn** | Structured data, and lending decisions must be explainable |
| **B — Diabetes screening** | A table of 100,000 rows | **scikit-learn** | Structured data, and a clinician needs to understand it |
| **C — Helmet detection** | 60 photographs | **Keras or PyTorch** | Images cannot be handled by a table-based library |
| **D — News classification** | 2,225 articles | **Either** | Once text becomes numbers, scikit-learn works well; a deep model may do better |

**Notice that two of the four are plain scikit-learn.** **Most working machine learning in the world is classical ML on tables** — which is exactly what this course spends Sessions 5 to 8 teaching.

**Example 4 — what you are actually buying**

**When you use a library instead of writing your own algorithm, you gain:**

| | |
|---|---|
| **Years of testing** | Edge cases found by millions of users |
| **Speed** | The heavy computation runs in a faster language underneath |
| **A shared vocabulary** | Your code reads like everyone else's, so others can help |
| **Documentation** | Someone has already written down what every setting does |

**And you give up:** knowing exactly what is happening inside. **Session 9 gives that back for one algorithm, by having you build a neural network from scratch.**

## 🌍 Scenarios

**Scenario 1 — the choice made for a reason that is not accuracy**

```text
A bank is deciding how to build its loan model.

  A deep learning model MIGHT score a fraction of a point higher.
  It cannot easily explain WHY it declined an application.

  Regulation requires that an applicant can be told the reason.

The bank chooses a classical model from scikit-learn - not because it
scored better, but because it can be EXPLAINED.

Accuracy is one consideration among several, and often not the deciding one.
```

**Scenario 2 — the library chosen by the deployment**

```text
The helmet-detection system must run on the camera itself, because
sending video to a server would be too slow and too expensive.

That means the model must be small and fast.

TensorFlow has the strongest tools for shrinking a model to fit on a
small device - so the DEPLOYMENT requirement chose the library, before
anyone compared accuracy.

This is Topic 18's stage 7 reaching back into stage 5 again.
```

**Scenario 3 — starting simple, and being glad of it**

```text
A student has a spreadsheet of 3,000 rows and wants to predict a
yes/no outcome.

  They could spend a week learning a deep learning library.
  Or they could use scikit-learn this afternoon.

The scikit-learn version trains in seconds, works, and tells them which
columns mattered - which teaches them something about their problem.

If it turns out not to be good enough, they have lost an afternoon and
learned a great deal. Starting with the complicated option risks losing
a week and learning less.
```

## ✏️ Tasks

1. In your own words, explain the difference between Keras and TensorFlow to someone who has never heard of either. **Use an analogy of your own, not the one in this topic.**
2. For each of the four Topic 18 projects, name the library you would choose and give one sentence of justification. **Then name the single fact that would change your mind.**
3. List three things you gain by using a library instead of writing an algorithm yourself, and one thing you give up.
4. Someone tells you they want to use deep learning for a 2,000-row spreadsheet. **Write three sentences explaining why a classical library is likely the better choice** — without using the word "advanced".
5. Take a problem from your own life or college, decide which of the four libraries fits, and write a short paragraph defending the choice. **Your reason must not be "it is the most powerful".**

<details><summary>Solutions</summary>

```text
1  A good answer makes clear that Keras and TensorFlow are not
   alternatives - one runs on top of the other. Any analogy works if it
   captures that relationship: a steering wheel and an engine, a light
   switch and the wiring, a menu and a kitchen.

2  Loan     -> scikit-learn; it is a table, and the decision must be
                explainable to an applicant.
   Diabetes -> scikit-learn; a table again, and a clinician must be able
                to understand what drove the result.
   Helmet   -> Keras or PyTorch; images cannot be handled by a
                table-based library.
   News     -> either; once text becomes numbers, scikit-learn works
                well, though a deep model may do better.
   What would change my mind: the data turning out not to be a table;
   a hard limit on response time; a rule that data cannot leave a device.

3  GAIN: years of testing and bug-fixing; speed, because the heavy work
     runs in a faster language underneath; a shared vocabulary, so other
     people can read and help with your code; and documentation.
   GIVE UP: knowing exactly what is happening inside. Session 9 gives
     that back for one algorithm by having you build it yourself.

4  On a small table, a classical model trains in seconds rather than
   minutes, usually scores at least as well, and can tell you which
   columns mattered - which teaches you something about the problem.
   It also needs no special hardware. If it proves not good enough, you
   have lost an afternoon rather than a week.

5  A good answer gives a reason connected to the DATA (is it a table?),
   the CONSTRAINTS (must it be explainable, fast, or run offline?), or
   the SITUATION (how much time and hardware do you have?).
   "It is the most powerful" is not a reason - it is a description.
```
</details>

## ❓ MCQs

**Q1.** Your data is a table of rows and columns. Which library should you reach for?
- (a) TensorFlow  (b) PyTorch  (c) scikit-learn  (d) Keras

**Q2.** What is the relationship between Keras and TensorFlow?
- (a) They are competitors  (b) Keras runs on top of TensorFlow — you write Keras, TensorFlow does the work  (c) TensorFlow runs on top of Keras  (d) They are unrelated

**Q3.** Which four steps do all these libraries share?
- (a) Open, read, write, close  (b) Create, train, predict, measure  (c) Load, clean, plot, save  (d) Import, install, update, remove

**Q4.** On a small table, choosing deep learning over a classical model usually means…
- (a) A much better score  (b) Longer training, no better result, and less explanation of which columns mattered  (c) Less code  (d) No difference at all

**Q5.** A bank picks a classical model over a deep one. The most likely reason is…
- (a) It scored higher  (b) An applicant must be able to be told why they were declined  (c) It is newer  (d) It uses less memory

<details><summary>Answers</summary>

**A1 — (c) scikit-learn.** Rows and columns: stop there. It is what Sessions 5 to 8 use.

**A2 — (b).** **Keras is not a competitor to TensorFlow — it sits on top of it.** Choosing Keras *is* choosing TensorFlow, with a more comfortable way of writing it.

**A3 — (b) Create, train, predict, measure.** Learn the shape once and every library becomes familiar — and you already built it yourself in Session 1.

**A4 — (b).** **Choose the tool that fits the data, not the one that sounds impressive.** Session 5 shows you the actual numbers.

**A5 — (b) Explainability.** Accuracy is one consideration among several, and often not the deciding one.
</details>

---

# ✅ Before you move on

**Artificial Intelligence**

- [ ] I can define AI in one sentence
- [ ] I can place AI, ML, Deep Learning and GenAI correctly inside one another
- [ ] I know a rule-based chess engine is AI but **not** ML
- [ ] I know everything deployed today is Narrow AI
- [ ] I can name real applications and say which are predictive and which generative
- [ ] I know most working AI is classical ML on tables

**Machine Learning**

- [ ] I can state the swap: data + rules → answers, versus data + answers → rules
- [ ] I know what a machine, learning, and Machine Learning each mean
- [ ] I have trained an image classifier with Teachable Machine
- [ ] I know a classifier can only answer with the classes it was given
- [ ] I can tell supervised, unsupervised and reinforcement learning apart
- [ ] I can decide regression versus classification from the target column
- [ ] **I know the arithmetic test for whether a number is really a category**

**Building it**

- [ ] I can recognise structured, unstructured and semi-structured data
- [ ] I know which maths I need now and which can wait
- [ ] I know why Python won, and that SQL is worth learning second
- [ ] I can choose between local, cloud and edge, and say why
- [ ] I check a dataset by opening it, **before** committing to a topic
- [ ] I can name the seven workflow stages, and know which takes longest
- [ ] I compute a baseline before celebrating any score
- [ ] I can choose between scikit-learn, TensorFlow, PyTorch and Keras

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-04-intro-ml-ai.ipynb) | Every example above, runnable |
| [Teachable Machine](https://teachablemachine.withgoogle.com/) | Train a real classifier with no code |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
| [Session 5 — Supervised Learning](session-05-regression.md) | Where you build the models properly |
