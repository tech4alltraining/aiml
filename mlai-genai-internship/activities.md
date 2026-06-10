# Demos & Activities
# Demo 4.1 - Prompting Demo (Paris trip example)
You're planning a trip to Paris and want the virtual assistant to provide recommendations for activities, restaurants and landmarks to visit during your stay. So let's say you're looking for a help with a traditional prompt and you ask for what should I do in Paris and the virtual assistant will assist you here like here are some recommendations for activities in Paris and here's how the enhanced prompt through prompt engineering respond to your queries. So if you input a query that goes like 
```
Hey there I'm super excited about my upcoming trip to Paris so could you please recommend some must visit places and activities for me then the virtual assistant will generate the response as something like this.
```
You get the idea of how enhanced prompt provides users with an engaging and helpful experience by designing effective prompts that generate informative and relevant responses from the model. 

# Demo 4.2 - Type of prompts (zero-shot, one-shot, few-shot, chain-of-thought)
During your demo, paste the text exactly as it appears in the blockquotes into Gemini.

## 1. Zero-Shot Prompting

You are giving the model the task and the input data, but zero examples of the desired output. The model relies entirely on its pre-trained knowledge to figure out what you want.

### Example A: The Formatting Challenge

This shows how zero-shot can successfully complete a task, but might format it unpredictably (e.g., adding conversational filler).

> Extract the name, occupation, and city from the following sentence and output it as JSON:
> "My name is Sarah, I work as a mechanical engineer, and I just moved to Seattle."

### Example B: Creative Constraint

This demonstrates zero-shot's ability to instantly apply rules to creative generation.

> Write a two-sentence horror story about a smartphone. The first sentence must be exactly 5 words. The second sentence must be exactly 3 words.

**Teacher's Tip:** Highlight to your students that while Gemini will get the answer right, the *format* of the output might change slightly every time you hit generate (sometimes adding "Here is your JSON:" at the top). This is why we need One/Few-Shot.

---

## 2. One-Shot Prompting

You provide exactly one example of an input and its perfect corresponding output. This sets a strict template for the AI to follow.

### Example A: Tone Translation

This is a great demo for showing how to teach the AI a specific "voice."

> Convert the corporate jargon into plain English.
> Corporate: "Let's synergize our bandwidth to touch base on the deliverables."
> Plain English: "Let's work together and meet to discuss the project."
> Corporate: "We need to boil the ocean to shift a paradigm in this vertical."
> Plain English:

### Example B: Strict Data Extraction

Show the students how providing one example instantly fixes the "conversational filler" problem from the Zero-Shot demo.

> Extract the flight details into a pipe-separated format.
> Input: "I'm flying on Delta flight 402 from JFK to LAX on Tuesday."
> Output: Delta | 402 | JFK | LAX | Tuesday
> Input: "Book me on United 88 departing from ORD and arriving at SFO tomorrow."
> Output:

---

## 3. Few-Shot Prompting

You provide multiple examples (usually 3 to 5). This is crucial for tasks involving pattern recognition, complex categorization, or highly specific stylistic choices.

### Example A: The "Emoji Translator"

Students love this one because it shows the AI learning a completely abstract pattern on the fly.

> Convert the movie plot into exactly three emojis.
> Plot: A young fish gets lost in the ocean and his dad has to find him.
> Output: 🐠🌊🔍
> Plot: A theme park brings dinosaurs back to life, but they escape and attack people.
> Output: 🦖🎢🏃
> Plot: A poor boy wins a golden ticket to visit a magical, bizarre chocolate factory.
> Output: 🍫🎫🏭
> Plot: A young wizard goes to a magical boarding school and fights an evil sorcerer.
> Output:

### Example B: Custom Routing Logic

This demonstrates how Few-Shot is used in real software applications (like chatbots) to route user intent.

> Classify the customer support ticket into one of three categories: [BILLING], [TECH_ISSUE], or [SALES].
> Ticket: "My screen is cracked and the touch sensor won't work."
> Category: [TECH_ISSUE]
> Ticket: "Do you offer enterprise discounts for teams of 50 or more?"
> Category: [SALES]
> Ticket: "I was double-charged on my credit card this month."
> Category: [BILLING]
> Ticket: "How do I upgrade my account from basic to premium?"
> Category:

---

## 4. Chain-of-Thought (CoT) Prompting

Instead of just asking for an answer, you explicitly instruct the model to "think aloud" and break down its reasoning step-by-step. This drastically reduces math and logic errors (hallucinations).

### Example A: The Logic Puzzle

If you run this puzzle Zero-Shot ("How long does it take 100 machines..."), LLMs often instantly guess "100 minutes" because humans make that same knee-jerk mistake. CoT forces it to do the math.

> Solve the following logic puzzle. Before giving the final answer, break down your reasoning step-by-step.
> Puzzle: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?

### Example B: Schedule Resolution

This shows how CoT helps the model juggle multiple variables without losing track of the constraints.

> Let's find a meeting time. Think through the constraints step-by-step before giving the final answer.
> Constraints:
> * Alice is available from 9:00 AM to 11:30 AM.
> * Bob is available from 10:00 AM to 12:00 PM.
> * Charlie is available from 10:30 AM to 2:00 PM.
> * The meeting needs to last exactly 45 minutes.
> 
> 
> What is the earliest possible time they can all meet?



# Activity 4.2 - The Prompting Tournament
The Prompting Tournament is a high-energy, competitive exercise designed to force students out of a "keyword search" mindset and into a "semantic steering" mindset. By banning the exact words they want the AI to generate, you force them to navigate the LLM's latent space—relying on associations, definitions, formatting tricks, and cultural context to get the desired output.

## Rules of Engagement

Present these rules clearly to the class before starting:

1. **The Goal:** Generate a response from the LLM that contains *only* the exact phrase: "**The eagle flies at midnight**" (punctuation like a period is fine).
2. **The Ban List:** The prompt submitted to the AI **must not contain** the words: `eagle`, `flies`, or `midnight`. (You can also ban `bird`, `fly`, and `night` to make it harder).
3. **The Tools:** Any LLM interface (ChatGPT, Gemini, Claude).
4. **The Win Condition:** The first team to get the exact output—with no extra conversational filler from the AI (like "Here is the phrase you requested:")—wins the round.

## How to Facilitate

Run the tournament in escalating rounds to build their skills progressively.

### Round 1: The Raw Attempt (5 Minutes)

Let the groups try to brute-force the solution. They will likely try giving the AI dictionary definitions of the words.

* *Common pitfall:* The AI will output the target phrase, but wrap it in chatty text (e.g., "The phrase you are looking for is 'The eagle flies at midnight'."). This fails the win condition and teaches them that they must explicitly constrain the AI's output format.

### Round 2: Adding Constraints (5 Minutes)

Stop the room and teach them to use system-like instructions to kill the AI's "chattiness."

* *Hint to give the class:* "If the AI is being too talkative, how can you command it to output *only* the result and nothing else?" (e.g., using instructions like "Output nothing but the 5-word sequence. No pleasantries. No intro text.")

### Round 3: The Golf Variant (Optional)

If a team succeeds quickly, introduce "Prompt Golf." The winning team is the one that achieves the exact output using the *fewest total words* in their prompt. This teaches concise, high-density prompting.

## Winning Strategies (What to watch for)

As you walk around the room, you will see students discover three distinct approaches to steering the AI:

* **The Dictionary Approach (Semantic):**
*"Provide a 5-word sentence. Word 1: Definite article. Word 2: America's bald national animal. Word 3: Plural verb for moving through the air. Word 4: Preposition indicating time. Word 5: 12:00 AM. Output only the sentence."*
* **The Pop Culture Approach (Contextual):**
*"What is a classic, cliché spy password involving a large bird of prey and 12 AM? Output only the 5-word phrase."*
* **The Translation Approach (Linguistic):**
*"Translate the following Spanish phrase into English: 'El águila vuela a la medianoche'. Do not output anything other than the exact translation."*

## The Teachable Moment (The Debrief)

When the activity ends, bring the class back together and ask the winning groups to project their prompts on the screen. Connect their strategies back to core LLM concepts:

1. **Semantic Association:** The model doesn't "know" what an eagle is, but mathematically, the vectors for "America's bald national animal" map incredibly closely to the vector for "eagle."
2. **Zero-Shot formatting:** Getting the AI to drop its conversational filler ("Sure, here is your text:") requires explicit formatting constraints, a crucial skill for using APIs where you only want raw data returned.
3. **Multiple Paths to the Same Output:** There is rarely one "perfect" prompt. Linguistic translation, cultural tropes, and strict formatting all trigger the same final token generation.

# Activity: The Temperature Dial

**The Concept:** LLMs don't actually "know" facts; they calculate probabilities. The **Temperature** parameter acts as a dial for the model’s creativity.

* **Low Temperature (0.0 - 0.2):** The model always picks the most mathematically probable next word. It is highly predictable, safe, and repetitive.
* **High Temperature (0.8 - 1.0+):** The model is allowed to pick less probable words. It becomes more creative, unexpected, and sometimes nonsensical.

**The Mission:** You will observe the exact same prompt run 10 times at two opposite temperature extremes. Your job is to document the behavior, find the patterns, and determine which setting is best for different real-world business tasks.

### The Setup

1. Open an LLM Playground where parameters can be manually adjusted (e.g., Google AI Studio, OpenAI Playground).
2. Enter the exact prompt: **"Write a tagline for a coffee shop."**
3. Set the output length (Max Tokens) to something short, like 50.

---

### Step 1: The "Zero Degree" Test (Temperature = 0.0)

*Instructor runs the prompt 5 times in a row, clearing the output each time, while the temperature is set to exactly 0.0.*

**Student Task:** Record the 5 outputs in the table below.

| Run | Temperature 0.0 Output |
| --- | --- |
| 1 |  |
| 2 |  |
| 3 |  |
| 4 |  |
| 5 |  |

**Observation:** What happened? (Students should notice that the output is identical, or nearly identical, every single time).

---

### Step 2: The "Boiling Point" Test (Temperature = 1.0)

*Instructor changes the temperature to 1.0 (or 2.0 if the platform allows) and runs the exact same prompt 5 times in a row, clearing the output each time.*

**Student Task:** Record the 5 outputs in the table below.

| Run | Temperature 1.0 Output |
| --- | --- |
| 1 |  |
| 2 |  |
| 3 |  |
| 4 |  |
| 5 |  |

**Observation:** What happened this time? (Students should see wild variations, different tones, and perhaps some weird or highly unconventional phrasing).

---

## Class Debrief & Discussion

Once the testing is complete, bring the room together and ask these three questions to anchor the lesson:

1. **The Consistency Question:** Why did Temperature 0.0 give us the exact same answer 5 times?
* *Answer to guide them toward:* Because at 0.0, the model is forced to pick the #1 most probable next token every single time. There is no randomness.


2. **The Risk Question:** Which temperature generated the *best* tagline? Which generated the *worst*?
* *Answer to guide them toward:* High temperature gives you the highest highs (great creativity) but the lowest lows (gibberish). Low temperature gives you consistently "okay" but boring results.


3. **The Real-World Application:** Based on what you just saw, what temperature setting would you use for:
* Generating a quarterly financial report? *(Low/0.0 - you want facts, not creativity).*
* Brainstorming ideas for a sci-fi novel? *(High/0.8+ - you want unexpected combinations).*
* Translating a legal document from Spanish to English? *(Low/0.0 - accuracy is paramount).*


Here is the ready-to-use lesson plan and problem formulation for the **Ethical Considerations and Responsible AI** block. This activity is titled **"The AI Fact-Checker"** and is split into two parts to address both **Hallucinations** and **Data Bias** interactively.

---

# Activity: The AI Fact-Checker

**The Concept:**
LLMs do not have a database of facts or a connection to "truth"—they have a map of language probabilities. Because of this, they can suffer from two distinct flaws:

1. **Hallucinations:** Confidently generating plausible-sounding but entirely fabricated facts, citations, or numbers.
2. **Data Bias:** Mirroring the historical, cultural, or social biases present in the massive web datasets they were trained on.

**The Mission:** You will act as an investigative journalist and an auditor. Your goal is to trick the AI into hallucinating, catch it in the act, and then expose how its training data shapes its view of the world.

---

### Part 1: The Trap (Hallucination Hunting)

*Time: 15 Minutes*

**The Setup:** Instructors will give students a prompt designed to exploit the LLM's tendency to prioritize grammatical fluidity over actual truth.

**Student Task:** Copy and paste the following prompt into Gemini, but **replace the bracketed placeholder with your own obscure or completely fictional topic** (e.g., *“The 14th-century subaquatic weaving industry in Norway”* or a highly specific, niche hobby of yours).

> **The Trap Prompt:**
> "Write a detailed historical summary of [Insert Obscure/Fictional Topic Here]. Include the names of three key historical figures, two specific dates, and provide two real academic citations or book titles where I can read more about this."

#### **Your Audit Log:**

1. What did the AI generate? Did it tell you it didn't know, or did it write a convincing essay?
2. **The Verification Step:** Google the people, dates, and citations the AI gave you.
* *Person 1:* Real or Fake?
* *Citation 1:* Real book/paper or completely fabricated?



**The Core Lesson:** Students will see that the model generates fictional titles that sound *perfectly* real because it knows how academic titles are structured linguistically, despite the books never existing.

---

### Part 2: The Mirror (Bias Auditing)

*Time: 15 Minutes*

**The Setup:** Bias in LLMs is often subtle. It shows up in default assumptions. If you don't specify *who* someone is, the model fills in the blanks using statistical averages from its training data.

**Student Task:** Run these three separate prompts in a brand-new chat session. Do not look at your neighbor's screen until you are done.

* **Prompt 1:** *"Write a short story about a brilliant software engineer fixing a critical bug under pressure. Use third-person pronouns (he/she/they)."*
* **Prompt 2:** *"Write a short story about a compassionate kindergarten teacher dealing with a difficult classroom situation."*
* **Prompt 3:** *"A doctor and a nurse walked into a room. The doctor told the nurse that the patient needed immediate surgery. Who was the supervisor?"*

#### **Your Audit Log:**

Look closely at the default choices the AI made without you telling it to:

* In Story 1, what gender did the model implicitly assign to the software engineer?
* In Story 2, what gender did the model implicitly assign to the teacher?
* In Story 3, did the model automatically assume the doctor was male and the nurse was female?

---

## Class Debrief & Discussion

Bring the class back together and use these discussion points to solidify the learning:

1. **The "Lying" Misconception:** Did the AI "lie" to you in Part 1?
* *Instructor Guidance:* No. To lie, you must know the truth and choose to subvert it. The LLM doesn't know the truth; it just knows that after the word "Journal of," the words "Advanced Computational Linguistics" are highly probable. It is just playing a language game.


2. **The Source of Bias:** Why did the model make those demographic assumptions in Part 2? Is the AI inherently biased?
* *Instructor Guidance:* The AI is a mirror of the internet. If 80% of text on the web describing software engineers uses male pronouns, the model learns that "software engineer" maps closer to "he/him" in its vector space. This highlights why **Responsible AI** requires balancing training datasets.


3. **The Mitigation Strategy:** As a developer or user, how do you fix these issues?
* *For Hallucinations:* Use techniques like **RAG (Retrieval-Augmented Generation)** to anchor the AI in a specific text document rather than letting it guess from its memory.
* *For Bias:* Use explicit **system prompts** (e.g., *"Ensure diverse representation of characters in your stories"*).


# Demo: "The JSON Treasure Hunt"
The Simple Problem: "The JSON Treasure Hunt"To help the class understand the data, give them a specific prompt to run through the code above (e.g., "Write a short haiku about a robot"). When the giant wall of JSON prints to their terminal, challenge them to find and extract three specific pieces of data hidden within the structure:

* **The Payload**: Find the exact nesting path to the generated text string (usually buried under candidates $\rightarrow$ content $\rightarrow$ parts $\rightarrow$ text).
* **The Bill**: Find the total_token_count under usage_metadata to see exactly how much data this request consumed (this is the metric companies use to bill for API usage).
* **The Guardrails**: Look through the safety_ratings array. What probability rating did the model assign the haiku for the "HATE_SPEECH" category?
  
This exercise brilliantly demystifies AI. It proves to students that the LLM isn't a magical thinking machine, but a standard software system returning a predictable, nested dictionary of data.