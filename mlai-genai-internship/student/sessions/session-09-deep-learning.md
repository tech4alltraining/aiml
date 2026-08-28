# Session 9 — Deep Learning

**Artificial Neural Networks · Activation Functions · Loss Functions · Backpropagation · Optimizers · From Deep Learning to Generative AI**

| | |
|---|---|
| **Notebook** | [session-09-deep-learning.ipynb](../notebooks/session-09-deep-learning.ipynb) |
| **Previous** | [Session 8 — Model Evaluation & Improvement](session-08-evaluation-tuning.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **You will build a working neural network in about twenty lines of NumPy** — forward pass, loss, backpropagation and all. Once you have seen those twenty lines, every deep learning framework is just a faster version of them.
>
> This is also the session where the course turns towards Generative AI. **A language model is a neural network. Everything here is the foundation for everything after.**

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Explain what a single neuron computes
2. Show why a network **needs** hidden layers, with a problem linear models cannot solve
3. Choose between ReLU, sigmoid, tanh and softmax, and say why ReLU is the default
4. Pick the right loss function for regression, binary and multi-class problems
5. Explain backpropagation and gradient descent in plain English
6. **Write a neural network from scratch in NumPy and train it**
7. Compare SGD, Momentum and Adam
8. Explain how deep learning becomes Generative AI

---

## The six topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Neurons and networks](#1-from-a-neuron-to-a-network) | Without hidden layers you cannot solve XOR |
| 2 | [Activation functions](#2-activation-functions) | No activation = no matter how deep, still linear |
| 3 | [Loss functions](#3-loss-functions) | The loss is what the network is actually trying to reduce |
| 4 | [Backpropagation](#4-backpropagation-and-gradient-descent) | Blame flows backwards; weights step downhill |
| 5 | [Optimizers](#5-optimizers) | Adam is the safe default, not always the winner |
| 6 | [DL → GenAI](#6-from-deep-learning-to-generative-ai) | Predict the next token, repeatedly |

---

# 1. From a neuron to a network

**One neuron does three things:** multiply each input by a weight, add them up with a bias, and pass the result through an activation function.

```text
inputs        weights
  x1 ----w1----\
  x2 ----w2-----> [ sum + bias ] --> [ activation ] --> output
  x3 ----w3----/
```

```python
output = activation(x1*w1 + x2*w2 + x3*w3 + bias)
```

🧠 **Analogy: deciding whether to go out tonight.** You weigh several factors — *is it raining?* (weight −5), *are my friends going?* (weight +3), *do I have work tomorrow?* (weight −2). You add them up, and if the total clears your personal threshold, you go. **That is one neuron.** Your bias is how much you like going out in general.

## Why one neuron is not enough

A single neuron draws **one straight line**. Some problems cannot be separated by any straight line.

**XOR** is the classic: output 1 when exactly one input is 1.

| x1 | x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | 0 |

Try to draw one straight line separating the two 1s from the two 0s. **You cannot** — they sit on opposite diagonals.

Measured:

| Model | XOR accuracy |
|---|---|
| Logistic Regression | **0.50** (pure chance) |
| MLP with 4 hidden units | **1.00** |

**This single result is why hidden layers exist.** A hidden layer lets the network bend the space until a straight line *does* work.

> **Historical note:** this exact limitation, published in 1969, stalled neural network research for over a decade. The fix — hidden layers trained by backpropagation — is what you will implement in Topic 4.

## 📘 Examples

**Example 1 — one neuron by hand**

```python
import numpy as np

def neuron(inputs, weights, bias):
    return max(0, np.dot(inputs, weights) + bias)      # ReLU activation

print(neuron([1, 0, 1], [-5, 3, -2], 4))    # raining, no friends, work tomorrow
print(neuron([0, 1, 0], [-5, 3, -2], 4))    # dry, friends going, free tomorrow
```

**Example 2 — XOR, proven**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

X = np.array([[0,0], [0,1], [1,0], [1,1]], float)
y = np.array([0, 1, 1, 0])

print(LogisticRegression().fit(X, y).score(X, y))                             # 0.50
print(MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000,
                    random_state=1).fit(X, y).score(X, y))                    # 1.00
```

**Example 3 — the shape of a network**

```python
MLPClassifier(hidden_layer_sizes=(16, 8))
#                                  ^^  ^
#     input layer: however many features you have
#     hidden layer 1: 16 neurons
#     hidden layer 2: 8 neurons
#     output layer: added automatically (1 for binary, n for n classes)
```

**"Deep" learning simply means more than one hidden layer.** There is no other threshold.

## ✏️ Practice

1. Compute one neuron's output by hand for inputs `[1, 2]`, weights `[0.5, -1]`, bias `0.3`.
2. Run the XOR comparison. What does logistic regression score?
3. Try `hidden_layer_sizes=(1,)` on XOR. Does one hidden neuron suffice?
4. Draw the XOR points on paper and try to separate them with one line.
5. How many layers make a network "deep"?

<details><summary>Solutions</summary>

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

print(max(0, np.dot([1, 2], [0.5, -1]) + 0.3))     # 1  -> 0.5 - 2 + 0.3 = -1.2 -> ReLU -> 0

X = np.array([[0,0], [0,1], [1,0], [1,1]], float)
y = np.array([0, 1, 1, 0])
print("logistic:", LogisticRegression().fit(X, y).score(X, y))            # 2 -> 0.50
print("MLP (4) :", MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000,
                                 random_state=1).fit(X, y).score(X, y))   # 1.00

for h in [1, 2, 4]:                                                        # 3
    s = MLPClassifier(hidden_layer_sizes=(h,), max_iter=5000,
                      random_state=1).fit(X, y).score(X, y)
    print(f"hidden={h}: {s}")
# One neuron is still ONE straight line -- it cannot solve XOR.
# You need at least two hidden neurons to bend the space.

# 4 - Impossible. The 1s sit at (0,1) and (1,0); the 0s at (0,0) and (1,1).
#     They are on opposite diagonals, so no single line separates them.

# 5 - More than one hidden layer. There is no other threshold.
```
</details>

## ❓ MCQs

**Q1.** A single neuron computes…
- (a) A random number  (b) activation(weighted sum of inputs + bias)  (c) The average of its inputs  (d) A decision tree split

**Q2.** Logistic regression scores 0.50 on XOR because…
- (a) Too little data  (b) XOR is not linearly separable and one neuron draws one line  (c) A bug  (d) Wrong metric

**Q3.** What does a hidden layer let a network do?
- (a) Run faster  (b) Bend the space so a straight line can separate the classes  (c) Use less memory  (d) Avoid scaling

**Q4.** `hidden_layer_sizes=(16, 8)` means…
- (a) 16 features and 8 classes  (b) Two hidden layers, of 16 and 8 neurons  (c) 16 epochs, 8 batches  (d) 24 layers

**Q5.** What makes a network "deep"?
- (a) Many neurons  (b) More than one hidden layer  (c) Using a GPU  (d) Over 1M parameters

**Q6.** Can one hidden neuron solve XOR?
- (a) Yes  (b) No — one neuron is still one straight line  (c) Only with more epochs  (d) Only with ReLU

<details><summary>Answers</summary>

**A1 — (b).** Multiply, add, activate. That is the whole operation.

**A2 — (b).** The two 1s sit on opposite diagonals. **0.50 on a balanced two-class problem is pure chance.**

**A3 — (b).** It transforms the space until the problem *becomes* linearly separable.

**A4 — (b).** The input and output layers are added automatically.

**A5 — (b).** No other threshold — "deep" is a much less impressive word than it sounds.

**A6 — (b).** You need at least two.
</details>

## 🎯 Tasks

**Task 1 — The XOR investigation.** Try 1, 2, 4 and 16 hidden neurons on XOR, each with five random seeds. **Report how often each succeeds.** You will find that even a capable network sometimes fails from a bad start — connect that to `n_init` in Session 7.

**Task 2 — Neuron by hand.** On paper, compute the output of a 2-input, 2-hidden-neuron, 1-output network for one input, with weights you choose. **Show every multiplication.** Doing this once by hand makes Topic 4 far easier.

---

# 2. Activation Functions

**The activation function is what makes a neural network more than an expensive linear model.**

🧠 **Analogy: a decision threshold.** A weighted sum gives you a *score*. The activation is what you do with it — ignore anything negative, squash it into a probability, or centre it around zero. **Without one, stacking layers is pointless: a chain of linear steps is just one bigger linear step.**

| Function | Output range | Use it | Why |
|---|---|---|---|
| **ReLU** `max(0, z)` | 0 to ∞ | **Hidden layers — the default** | Fast, and gradients survive deep stacks |
| **Sigmoid** | 0 to 1 | Binary output layer | Reads directly as a probability |
| **Tanh** | −1 to 1 | Sometimes hidden layers | Zero-centred |
| **Softmax** | Sums to 1 | Multi-class output layer | Probabilities across all classes |

> **The rule you can memorise:** **ReLU everywhere hidden. Sigmoid for one output. Softmax for many.**

## Why ReLU won

Sigmoid squashes everything into 0–1, so its gradient is tiny almost everywhere. Stack ten sigmoid layers and multiply ten tiny numbers together — **the gradient reaching the first layer is effectively zero and it never learns.** This is the **vanishing gradient problem**, and it is why deep networks did not work for years.

ReLU's gradient is exactly 1 for any positive input. **Multiply 1 by itself ten times and you still have 1.**

## 📘 Examples

**Example 1 — the four functions**

```python
import numpy as np

def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def tanh(z):    return np.tanh(z)
def softmax(z):
    e = np.exp(z - z.max())        # subtract max for numerical stability
    return e / e.sum()

print(relu(np.array([-2, -0.5, 0, 0.5, 2])))     # [0.  0.  0.  0.5 2. ]
print(softmax(np.array([2.0, 1.0, 0.1])).round(3))  # [0.659 0.242 0.099]
```

**Example 2 — no activation means no depth**

```python
# Two linear layers with NO activation between them
h = X @ W1 + b1
out = h @ W2 + b2
#   = (X @ W1 + b1) @ W2 + b2
#   = X @ (W1 @ W2) + (b1 @ W2 + b2)
#   = X @ W_combined + b_combined     <- a SINGLE linear layer
```

**Ten layers without activations collapse into one.** The activation is not a detail — it is the thing that makes depth mean anything.

**Example 3 — measured on a real problem**

| Activation | Test accuracy | Iterations to converge |
|---|---|---|
| ReLU | 0.9300 | 789 |
| Tanh | 0.9400 | 1110 |
| Sigmoid (`logistic`) | 0.9100 | 399 |

**On a small two-layer network the difference is minor** — tanh even edges ahead here. **ReLU's advantage appears in deep networks**, where sigmoid's vanishing gradients become fatal. **Do not over-read a small result like this one.**

## ✏️ Practice

1. Implement all four functions and plot them from −5 to 5.
2. Confirm softmax sums to 1.
3. Compare the three activations on the moons data. Which converges fastest?
4. Explain in your own words why stacking linear layers without activations is pointless.
5. Why does sigmoid cause vanishing gradients?

<details><summary>Solutions</summary>

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

z = np.linspace(-5, 5, 11)                                             # 1
print("relu   ", relu(z).round(2))
print("sigmoid", sigmoid(z).round(3))
print("tanh   ", np.tanh(z).round(3))
print("softmax sums to:", softmax(np.array([2.0, 1.0, 0.1])).sum())    # 2 -> 1.0

X, y = make_moons(400, noise=.2, random_state=42)                      # 3
X = StandardScaler().fit_transform(X)
a, b, c, d = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
for act in ["relu", "tanh", "logistic"]:
    m = MLPClassifier(hidden_layer_sizes=(16, 8), activation=act,
                      max_iter=2000, random_state=42).fit(a, c)
    print(f"{act:<10} test {m.score(b, d):.4f}   iterations {m.n_iter_}")

# 4 - A linear function of a linear function is still linear. Ten stacked
#     linear layers algebraically collapse into one. The activation is the
#     ONLY thing that makes depth mean anything.

# 5 - Sigmoid squashes everything into 0..1, so its gradient is tiny almost
#     everywhere (max 0.25). Stack ten layers and you multiply ten tiny
#     numbers: the gradient reaching layer 1 is effectively zero, so it
#     never learns. ReLU's gradient is exactly 1 for positive inputs --
#     multiply 1 by itself ten times and you still have 1.
```
</details>

## ❓ MCQs

**Q1.** Which is the default for hidden layers?
- (a) Sigmoid  (b) ReLU  (c) Softmax  (d) Linear

**Q2.** Which belongs on a multi-class output layer?
- (a) ReLU  (b) Sigmoid  (c) Softmax  (d) Tanh

**Q3.** Stacking linear layers with no activation gives you…
- (a) A deep network  (b) The equivalent of one linear layer  (c) An error  (d) Better accuracy

**Q4.** Why does sigmoid cause vanishing gradients in deep networks?
- (a) It is slow  (b) Its gradient is tiny, and many tiny numbers multiplied together approach zero  (c) It outputs negatives  (d) It needs scaling

**Q5.** `relu(-3)` returns…
- (a) −3  (b) 0  (c) 3  (d) 0.047

**Q6.** Softmax outputs…
- (a) One number  (b) Values summing to 1 across all classes  (c) Values from −1 to 1  (d) Only 0 or 1

**Q7.** Tanh slightly beat ReLU on a small 2-layer network. You should conclude…
- (a) Tanh is better  (b) On small networks the choice barely matters; ReLU's edge is in deep ones  (c) ReLU is broken  (d) Always use tanh

<details><summary>Answers</summary>

**A1 — (b) ReLU.**

**A2 — (c) Softmax.** Sigmoid handles a single yes/no output.

**A3 — (b).** The algebra collapses. **Depth without activations is an illusion.**

**A4 — (b).** This is why deep networks did not work for years.

**A5 — (b) 0.** `max(0, -3)`.

**A6 — (b).** That is what makes it readable as a probability distribution.

**A7 — (b).** **Do not over-read a small result.** One 400-row problem does not overturn the general guidance.
</details>

## 🎯 Tasks

**Task 1 — Plot all four.** Plot ReLU, sigmoid, tanh and softmax, **and their gradients**, from −5 to 5. **Annotate where sigmoid's gradient goes near zero** and connect it to the vanishing gradient problem.

**Task 2 — Prove the collapse.** With NumPy, build two linear layers with no activation and show numerically that the result equals a single layer with `W1 @ W2`. **Then insert a ReLU and show the equality breaks.**

---

# 3. Loss Functions

**The loss is the single number the network is trying to make smaller.** Everything else — the gradients, the weight updates, the whole training process — exists to reduce it.

🧠 **Analogy: your score in a round of golf.** Lower is better, and every decision you make aims to reduce it. **Choose the wrong scoring system and you will optimise for the wrong thing.**

| Problem | Loss | In scikit-learn / Keras |
|---|---|---|
| Regression | **Mean Squared Error** | `mse` |
| Regression, outliers present | **Mean Absolute Error** | `mae` |
| Binary classification | **Binary cross-entropy** | `binary_crossentropy` |
| Multi-class | **Categorical cross-entropy** | `categorical_crossentropy` |

> **Loss is not the same as your metric.** The network optimises the *loss*; you report the *metric*. Accuracy cannot be a loss because it has no useful gradient — nudging a weight slightly changes no predicted label at all, so there is nothing to follow downhill.

## 📘 Examples

**Example 1 — the losses in code**

```python
import numpy as np

def mse(y, p):  return np.mean((y - p) ** 2)
def mae(y, p):  return np.mean(np.abs(y - p))
def bce(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)        # never log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```

**Example 2 — cross-entropy punishes confident mistakes**

```python
y = np.array([1.0])

for p in [0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:
    print(f"true=1, predicted={p:<5}  loss = {bce(y, np.array([p])):.4f}")
```

| Predicted | Loss |
|---|---|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| **0.01** | **4.605** |

**Being confidently wrong is punished far harder than being unsure.** A model that says "99% certain" and is wrong takes a much bigger penalty than one that says "50/50". **That is exactly the incentive you want.**

**Example 3 — why not accuracy?**

```python
# Nudge a weight by 0.0001.
#   Cross-entropy: 0.6931 -> 0.6929    a direction to follow
#   Accuracy:      0.8500 -> 0.8500    no signal at all
```

**Accuracy is a step function.** Gradient descent needs a smooth slope, and a staircase has none.

## ✏️ Practice

1. Implement MSE, MAE and binary cross-entropy.
2. Compute cross-entropy for predictions 0.9, 0.5 and 0.01 when the truth is 1.
3. On predictions `[2, 4, 6]` versus truth `[2, 4, 20]`, compare MSE and MAE. Which reacts more?
4. Why can accuracy not be used as a loss function?
5. Which loss for predicting one of five categories?

<details><summary>Solutions</summary>

```python
import numpy as np

def mse(y, p): return np.mean((y - p) ** 2)
def mae(y, p): return np.mean(np.abs(y - p))
def bce(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

for p in [0.9, 0.5, 0.01]:                                             # 2
    print(f"predicted {p:<5} -> loss {bce(np.array([1.0]), np.array([p])):.4f}")
# Being CONFIDENTLY WRONG is punished far harder than being unsure.

y = np.array([2, 4, 20.]); p = np.array([2, 4, 6.])                    # 3
print(f"MSE {mse(y, p):.2f}   MAE {mae(y, p):.2f}")
# MSE reacts far more: it SQUARES the error, so the single big miss
# dominates. Use MAE when outliers should not dominate training.

# 4 - Accuracy is a STEP function. Nudge a weight slightly and no predicted
#     label changes, so accuracy does not move and there is no gradient to
#     follow. Gradient descent needs a smooth slope; a staircase has none.

# 5 - Categorical cross-entropy, with softmax on the output layer.
```
</details>

## ❓ MCQs

**Q1.** Which loss for binary classification?
- (a) MSE  (b) Binary cross-entropy  (c) MAE  (d) Accuracy

**Q2.** Cross-entropy loss for a confident wrong prediction (0.01 when truth is 1) is…
- (a) Near zero  (b) Very large (~4.6)  (c) Exactly 1  (d) Negative

**Q3.** Why can accuracy not serve as a loss?
- (a) It is slow  (b) It is a step function with no useful gradient  (c) It needs labels  (d) It is not in sklearn

**Q4.** For regression with a few extreme outliers, prefer…
- (a) MSE  (b) MAE  (c) Cross-entropy  (d) Softmax

**Q5.** Loss and metric differ in that…
- (a) They are the same  (b) The network optimises the loss; you report the metric  (c) The metric is always lower  (d) Loss is only for regression

**Q6.** Predicting one of five categories calls for…
- (a) Binary cross-entropy  (b) Categorical cross-entropy with softmax  (c) MSE  (d) MAE

<details><summary>Answers</summary>

**A1 — (b) Binary cross-entropy.**

**A2 — (b) ~4.605.** Confident mistakes are punished hardest — exactly the incentive you want.

**A3 — (b).** Gradient descent needs a slope to walk down.

**A4 — (b) MAE.** MSE squares errors, so one extreme value dominates training.

**A5 — (b).** They are often different on purpose.

**A6 — (b).**
</details>

## 🎯 Tasks

**Task 1 — The loss curve.** Plot binary cross-entropy against predicted probability from 0.01 to 0.99, for a true label of 1. **Mark where the loss explodes** and write one sentence on what that incentive teaches the network.

**Task 2 — MSE versus MAE under outliers.** Fit the same regression twice, once optimising MSE and once MAE, on data containing three extreme outliers. **Show how differently the two fitted lines behave** and say which you would use for house prices, and why.

---

# 4. Backpropagation and gradient descent

**This is how a network actually learns.** Two ideas, both simple:

🧠 **Analogy: walking downhill in fog.** You cannot see the valley. But you can feel which way the ground slopes under your feet, and take a step that way. Repeat. **That is gradient descent, and the step size is the learning rate.**

🧠 **Analogy: assigning blame after a failed group project.** The final output was wrong. How much was each person responsible? You work backwards from the outcome through the chain of contributions. **That is backpropagation** — the chain rule from calculus, applied layer by layer.

```text
FORWARD   input -> layer 1 -> layer 2 -> prediction -> LOSS
BACKWARD  how much did each weight contribute to that loss?  <---
UPDATE    every weight steps a little in the direction that reduces it
```

Repeat for many **epochs** (one epoch = one pass over the training data).

## 📘 Examples

**Example 1 — the entire training loop, in NumPy**

```python
import numpy as np

def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

rng = np.random.default_rng(0)
H = 8
W1 = rng.normal(0, np.sqrt(2 / 2), (2, H)); b1 = np.zeros(H)     # He initialisation
W2 = rng.normal(0, np.sqrt(2 / H), (H, 1)); b2 = np.zeros(1)
lr = 0.5
Y = y_train.reshape(-1, 1)

for epoch in range(1501):
    # ---- FORWARD: compute the prediction and the loss
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    p  = sigmoid(z2)
    loss = -np.mean(Y * np.log(p + 1e-9) + (1 - Y) * np.log(1 - p + 1e-9))

    # ---- BACKWARD: how much did each weight contribute to that loss?
    dz2 = (p - Y) / len(X_train)          # gradient at the output
    dW2 = a1.T @ dz2
    db2 = dz2.sum(0)
    da1 = dz2 @ W2.T                      # push the blame back one layer
    dz1 = da1 * (z1 > 0)                  # ReLU's gradient: 1 if positive, else 0
    dW1 = X_train.T @ dz1
    db1 = dz1.sum(0)

    # ---- UPDATE: step every weight downhill
    W1 -= lr * dW1;  b1 -= lr * db1
    W2 -= lr * dW2;  b2 -= lr * db2
```

**That is a complete neural network.** Every framework you will ever use — PyTorch, TensorFlow, Keras — is a faster, more general version of these twenty lines.

**Example 2 — watching it learn**

| Epoch | Loss | Test accuracy |
|---|---|---|
| 0 | 0.7807 | 0.5000 |
| 300 | 0.3075 | 0.9000 |
| **600** | 0.1495 | **0.9800** ← best |
| 900 | 0.0892 | 0.9500 |
| 1200 | 0.0741 | 0.9500 |
| 1500 | **0.0672** | 0.9500 |

**Look carefully at the last three rows.** The training loss keeps falling — 0.089, 0.074, 0.067 — while test accuracy has already peaked at epoch 600 and *fallen back*.

**That is overfitting, visible live in the training log.** Session 8's lesson, now inside a neural network: **a falling training loss is not proof of a improving model.** The fix is *early stopping* — keep the weights from the epoch with the best validation score, not the last epoch.

**Example 3 — the learning rate**

| Learning rate | What happens |
|---|---|
| Too small (0.0001) | Learns correctly but agonisingly slowly |
| **About right (0.1–0.5)** | **Converges steadily** |
| Too large (10) | Overshoots the valley; loss bounces or explodes to NaN |

**If your loss becomes `nan`, your learning rate is almost always too high.**

## ✏️ Practice

1. Run the training loop. What loss do you reach after 1,500 epochs?
2. Print test accuracy every 300 epochs. When does it peak?
3. Set `lr = 0.001`. How many epochs to reach the same loss?
4. Set `lr = 50`. What happens?
5. Change `H` from 8 to 2, then to 64. What changes?

<details><summary>Solutions</summary>

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

X, y = make_moons(400, noise=.2, random_state=42)
X = StandardScaler().fit_transform(X)
a, b, c, d = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)

def train(lr=0.5, H=8, epochs=1501, verbose=True):
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2/2), (2, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, np.sqrt(2/H), (H, 1)); b2 = np.zeros(1)
    Y = c.reshape(-1, 1)
    for ep in range(epochs):
        z1 = a @ W1 + b1; a1 = relu(z1); z2 = a1 @ W2 + b2; p = sigmoid(z2)
        loss = -np.mean(Y*np.log(p+1e-9) + (1-Y)*np.log(1-p+1e-9))
        dz2 = (p - Y) / len(a)
        dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        dW1 = a.T @ dz1;  db1 = dz1.sum(0)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
        if verbose and ep % 300 == 0:
            acc = ((sigmoid(relu(b @ W1 + b1) @ W2 + b2) > .5).ravel() == d).mean()
            print(f"  epoch {ep:>5}  loss {loss:.4f}  test acc {acc:.4f}")
    return loss

print("lr=0.5:");   train()                                            # 1, 2
# Loss reaches ~0.067. Test accuracy PEAKS at epoch 600 (0.98) and then
# FALLS BACK to 0.95 while the loss keeps dropping -- overfitting, visible
# live in the training log. The fix is early stopping.

print("\\nlr=0.001:"); train(lr=0.001, verbose=False)                   # 3
print("  final loss:", round(train(lr=0.001, verbose=False), 4))
# Far higher after the same 1,500 epochs -- it needs many times longer.

print("\\nlr=50:"); print("  final loss:", train(lr=50, verbose=False)) # 4
# The loss bounces or blows up. If your loss becomes nan, the learning
# rate is almost always too high.

for H in [2, 8, 64]:                                                   # 5
    print(f"H={H:<3} final loss {train(H=H, verbose=False):.4f}")
# H=2 underfits (too little capacity). H=64 fits the training data harder
# but does not help the test score -- Session 8's complexity curve again.
```
</details>

## ❓ MCQs

**Q1.** Backpropagation computes…
- (a) The prediction  (b) How much each weight contributed to the loss  (c) The accuracy  (d) The learning rate

**Q2.** One **epoch** means…
- (a) One weight update  (b) One full pass over the training data  (c) One layer  (d) One neuron

**Q3.** Your loss becomes `nan` during training. The most likely cause is…
- (a) Too little data  (b) The learning rate is too high  (c) Wrong activation  (d) Too few epochs

**Q4.** Training loss keeps falling while test accuracy peaks and declines. This is…
- (a) Normal and fine  (b) Overfitting — stop at the peak  (c) Underfitting  (d) A bug

**Q5.** The learning rate controls…
- (a) The number of layers  (b) How big a step the weights take each update  (c) The batch size  (d) The activation

**Q6.** In the backward pass, `dz1 = da1 * (z1 > 0)` implements…
- (a) The loss  (b) ReLU's gradient: 1 where the input was positive, 0 elsewhere  (c) Softmax  (d) Dropout

**Q7.** The right response to the epoch-600 peak is…
- (a) Train longer  (b) Early stopping — keep the best-validation weights, not the last ones  (c) Raise the learning rate  (d) Add layers

<details><summary>Answers</summary>

**A1 — (b).** It assigns blame backwards through the layers using the chain rule.

**A2 — (b) One full pass over the training data.**

**A3 — (b).** Almost always. Lower it by a factor of ten and try again.

**A4 — (b) Overfitting.** **A falling training loss is not proof of an improving model.**

**A5 — (b).** The step size in the walking-downhill analogy.

**A6 — (b).** ReLU passes the gradient through where it was active and blocks it where it was not.

**A7 — (b) Early stopping.**
</details>

## 🎯 Tasks

**Task 1 — The learning rate sweep.** Train with lr = 0.001, 0.01, 0.1, 0.5, 5 and 50. **Plot the loss curve for each on one chart** and label which is too small, about right, and unstable.

**Task 2 — Implement early stopping.** Modify the training loop to track validation accuracy each epoch, keep a copy of the best weights, and restore them at the end. **Report the accuracy you save compared with using the final weights.**

**Task 3 — Explain it to a friend.** Write half a page explaining backpropagation to someone who knows no calculus. **Use your own analogy, not the ones here.** If you can do this, you understand it.

---

# 5. Optimizers

**The optimizer decides how to turn a gradient into an actual weight update.** Plain gradient descent uses the gradient directly. The others are smarter about it.

| Optimizer | Idea | 🧠 Analogy |
|---|---|---|
| **SGD** | Step directly down the gradient | Walking downhill one careful step at a time |
| **Momentum** | Keep some velocity from previous steps | A **ball rolling** downhill — it powers through small bumps |
| **RMSProp** | Give each weight its own step size | Small steps on steep ground, big steps on flat |
| **Adam** | Momentum **and** per-weight step sizes | The ball, with adaptive shoe sizes. **The default** |

> **Start with Adam. Change only if you have a reason.**

## 📘 Examples

**Example 1 — measured on the moons data**

| Optimizer | Learning rate | Test accuracy | Iterations |
|---|---|---|---|
| Adam | 0.001 | 0.9300 | 789 |
| **SGD** | 0.001 | **0.9500** | 538 |
| SGD | 0.1 | 0.9300 | **173** |
| L-BFGS | — | 0.9100 | 104 |

**Adam did not win.** Plain SGD scored higher, and SGD with a larger learning rate converged in a fifth of the iterations.

**So why is Adam the default?** Because it is **robust**: it works acceptably across a huge range of problems without tuning. SGD beat it here because someone (you) picked a good learning rate for this specific problem. **Adam is the safe choice, not the optimal one** — and on a 400-row toy problem, all four land within 4 points of each other anyway.

> This is the same lesson as Session 8's tuning result: **the differences here are inside the noise, and treating them as a ranking would be a mistake.**

**Example 2 — batch size**

```python
MLPClassifier(batch_size=32)     # update the weights every 32 rows
```

| Batch size | Behaviour |
|---|---|
| 1 (pure SGD) | Very noisy updates, can escape poor minima, slow |
| **32 – 256** | **The practical range** |
| All rows | Smooth but slow, and needs a lot of memory |

**Example 3 — the practical starting point**

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),   # start small; grow if underfitting
    activation="relu",             # the default for hidden layers
    solver="adam",                 # the safe optimizer
    learning_rate_init=0.001,      # Adam's usual starting point
    batch_size=32,
    early_stopping=True,           # <- stop at the validation peak
    validation_fraction=0.1,
    random_state=42,
)
```

> **`early_stopping=True` is the single most valuable setting here.** It implements exactly the fix for the epoch-600 problem you saw in Topic 4.

## ✏️ Practice

1. Compare `adam`, `sgd` and `lbfgs` on the moons data.
2. Try SGD with learning rates 0.001, 0.01 and 0.1. What changes?
3. Turn on `early_stopping=True`. How do the iterations and score change?
4. Compare batch sizes 8, 32 and 256.
5. Given the numbers above, is Adam "the best optimizer"? Defend your answer.

<details><summary>Solutions</summary>

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

X, y = make_moons(400, noise=.2, random_state=42)
X = StandardScaler().fit_transform(X)
a, b, c, d = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)

for solver in ["adam", "sgd", "lbfgs"]:                                # 1
    m = MLPClassifier(hidden_layer_sizes=(16, 8), solver=solver,
                      max_iter=2000, random_state=42).fit(a, c)
    print(f"{solver:<7} test {m.score(b, d):.4f}  iters {m.n_iter_}")

for lr in [0.001, 0.01, 0.1]:                                          # 2
    m = MLPClassifier(hidden_layer_sizes=(16, 8), solver="sgd", learning_rate_init=lr,
                      max_iter=2000, random_state=42).fit(a, c)
    print(f"sgd lr={lr:<6} test {m.score(b, d):.4f}  iters {m.n_iter_}")
# A larger learning rate converges in far fewer iterations here.

for es in [False, True]:                                               # 3
    m = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, early_stopping=es,
                      random_state=42).fit(a, c)
    print(f"early_stopping={str(es):<6} test {m.score(b,d):.4f}  iters {m.n_iter_}")
# Early stopping halts far sooner and protects against the epoch-600 problem.

for bs in [8, 32, 256]:                                                # 4
    m = MLPClassifier(hidden_layer_sizes=(16, 8), batch_size=bs,
                      max_iter=2000, random_state=42).fit(a, c)
    print(f"batch={bs:<4} test {m.score(b,d):.4f}  iters {m.n_iter_}")

# 5 - NO. Adam is the ROBUST default, not the best on every problem. Here
#     SGD beat it. And all four results sit within a few points on 100 test
#     rows -- well inside the noise Session 8 taught you to measure.
#     Treating this table as a ranking would be a mistake.
```
</details>

## ❓ MCQs

**Q1.** Which optimizer should you start with?
- (a) SGD  (b) Adam  (c) L-BFGS  (d) RMSProp

**Q2.** Momentum helps by…
- (a) Reducing the learning rate  (b) Carrying velocity from previous steps, powering through small bumps  (c) Adding layers  (d) Scaling features

**Q3.** SGD beat Adam on this dataset. This means…
- (a) Adam is bad  (b) Adam is the robust default, not always the optimum; and here the gaps are inside the noise  (c) Always use SGD  (d) The test was wrong

**Q4.** The practical batch size range is…
- (a) 1  (b) 32 – 256  (c) The whole dataset  (d) 10,000+

**Q5.** What does `early_stopping=True` do?
- (a) Stops after one epoch  (b) Halts when validation score stops improving, keeping the best weights  (c) Reduces the learning rate  (d) Removes layers

**Q6.** Adam combines…
- (a) Two loss functions  (b) Momentum and per-weight adaptive step sizes  (c) Two networks  (d) Grid and random search

<details><summary>Answers</summary>

**A1 — (b) Adam.** Change only when you have a reason.

**A2 — (b).** The rolling-ball analogy.

**A3 — (b).** **The differences are inside the noise.** Session 8's discipline applies here too.

**A4 — (b) 32 – 256.**

**A5 — (b).** It is the built-in fix for the epoch-600 overfitting you saw in Topic 4.

**A6 — (b).** Hence the name: **ada**ptive **m**oment estimation.
</details>

## 🎯 Tasks

**Task 1 — The optimizer bake-off.** Compare Adam, SGD, SGD+momentum and L-BFGS on a dataset of your choice, **with five seeds each**, and report mean ± std. **Then say honestly which differences are real** — you now have the tools to know.

**Task 2 — The full recipe.** Build a network using the practical starting point above, on a real dataset. Tune one thing at a time and keep a log of what each change did. **Note every change that was smaller than your noise.**

---

# 6. From Deep Learning to Generative AI

**Everything you have just built is the foundation of ChatGPT.** The pieces are the same; the scale and one architectural idea are different.

## The step that changes everything

A classifier predicts a **label**. A language model predicts **the next token** — and then feeds its own output back in and predicts again.

```text
"The capital of France is"        -> "Paris"
"The capital of France is Paris"  -> "."
"The capital of France is Paris." -> "It"
```

**Generation is prediction, run in a loop.** That is genuinely the whole trick.

🧠 **Analogy: predictive text on your phone, scaled up enormously.** Your keyboard suggests the next word from the last two or three. An LLM does the same thing, having read a large fraction of the public internet, and considering thousands of previous words at once.

## What had to be invented

| Component | Purpose |
|---|---|
| **Tokens** | Text split into pieces (roughly ¾ of a word each) |
| **Embeddings** | Each token becomes a vector, so similar words sit near each other |
| **Attention** | Lets the model weigh *which earlier words matter* for the next one |
| **Transformer** | The architecture stacking attention and feed-forward layers |
| **Scale** | Billions of parameters, trained on enormous text corpora |

**Attention is the key invention** (2017, *"Attention Is All You Need"*). Earlier models read text strictly left to right and forgot the beginning of long passages. Attention lets every word look at every other word directly.

🧠 **Analogy for attention: reading a long sentence.** *"The trophy did not fit in the suitcase because **it** was too big."* What does "it" refer to? You resolve that by looking back and weighting the earlier words. **That weighting is attention.**

## What is the same, and what is different

| | Your NumPy network | GPT-class model |
|---|---|---|
| Neurons, weights, biases | ✅ | ✅ Identical idea |
| Activation functions | ✅ | ✅ (GELU rather than ReLU) |
| Loss function | Binary cross-entropy | Cross-entropy over the vocabulary |
| Backpropagation | ✅ | ✅ Identical idea |
| Optimizer | SGD | Adam variants |
| Parameters | ~40 | Hundreds of billions |
| Architecture | Two dense layers | Stacked transformer blocks |
| Training cost | Under a second | Millions of dollars |

> **You are not missing a magical extra ingredient. You have seen the whole machine — the difference is scale and the attention mechanism.**

## 📘 Examples

**Example 1 — generation is a loop**

```python
# The entire generation algorithm, in pseudocode
text = "The capital of France is"
for _ in range(20):
    probabilities = model(text)          # a probability for every possible token
    next_token = sample(probabilities)   # pick one
    text = text + next_token             # feed it back in
```

**Example 2 — temperature, which you will use in Session 10**

```python
# probabilities -> which token gets chosen
#   temperature 0.0   always the single most likely token   (repetitive, safe)
#   temperature 0.7   mostly likely tokens, some variety    (the usual default)
#   temperature 1.5   adventurous, sometimes incoherent
```

**Temperature is just how sharply you sample from that probability distribution.** Nothing more mystical than that.

**Example 3 — what "parameters" means**

```python
# Your moons network:
#   W1: 2 x 8 = 16,  b1: 8,  W2: 8 x 1 = 8,  b2: 1   -> 33 parameters
#
# A small open LLM:      ~7,000,000,000 parameters
# A frontier model:      hundreds of billions
#
# Same kind of number. Same backpropagation. Roughly 200 billion times more of them.
```

## ✏️ Practice

1. Count the parameters in your moons network by hand. Check against the formula.
2. Explain next-token prediction to someone in two sentences.
3. What problem does attention solve that older sequence models had?
4. Name three things your NumPy network shares with an LLM.
5. What does temperature control?

<details><summary>Solutions</summary>

```python
# 1 - W1 is 2x8 = 16, b1 is 8, W2 is 8x1 = 8, b2 is 1  ->  33 parameters.
#     General formula for a dense layer: (inputs x outputs) + outputs.

# 2 - The model is given some text and predicts which token comes next.
#     It then adds that token to the text and predicts again -- so
#     generation is just prediction run in a loop.

# 3 - Older models read strictly left to right and forgot the beginning of
#     long passages. Attention lets every word look at every other word
#     directly, so "it" can be linked to "trophy" thirteen words earlier.

# 4 - Neurons with weights and biases; activation functions; a loss;
#     backpropagation; gradient-based optimizers. All identical in kind.

# 5 - How sharply the next token is sampled from the probability
#     distribution. Low = repetitive and safe, high = varied and riskier.
```
</details>

## ❓ MCQs

**Q1.** How does a language model generate text?
- (a) It looks up answers in a database  (b) It predicts the next token and feeds its output back in, repeatedly  (c) It searches the web  (d) It copies its training data

**Q2.** What problem does attention solve?
- (a) Slow training  (b) Earlier models forgot the beginning of long passages  (c) Memory use  (d) Overfitting

**Q3.** Which is **not** shared between your NumPy network and an LLM?
- (a) Backpropagation  (b) Activation functions  (c) The transformer architecture  (d) A loss function

**Q4.** Temperature controls…
- (a) Training speed  (b) How sharply the next token is sampled  (c) The number of layers  (d) The learning rate

**Q5.** A dense layer with 2 inputs and 8 outputs has how many parameters?
- (a) 16  (b) 24  (c) 10  (d) 8

**Q6.** The main difference between your network and GPT-class models is…
- (a) A secret algorithm  (b) Scale and the attention mechanism  (c) A different kind of mathematics  (d) Nothing

<details><summary>Answers</summary>

**A1 — (b).** **Generation is prediction, run in a loop.**

**A2 — (b).** Every word can look at every other word directly.

**A3 — (c) The transformer architecture.** Everything else is shared.

**A4 — (b).** Nothing more mystical than that.

**A5 — (b) 24.** (2 × 8) weights + 8 biases.

**A6 — (b).** **You have seen the whole machine.**
</details>

## 🎯 Tasks

**Task 1 — Count a real architecture.** Take a network with layers 100 → 64 → 32 → 10 and compute its total parameters by hand. **Then verify with `sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)`.**

**Task 2 — The bridge document.** Write one page mapping each component of your NumPy network onto its equivalent in an LLM, **and state clearly which component has no equivalent in what you built.** You will refer back to this in Sessions 10 and 11.

**Task 3 — Predictive text by hand.** Take any paragraph, hide the last word, and ask three people to predict it. **Record how often they agree.** That agreement rate is roughly what a language model is learning to reproduce, at scale.

---

# ✅ Before you move on

- [ ] I can explain what a single neuron computes
- [ ] I know why XOR needs a hidden layer, and I have proved it
- [ ] I use ReLU for hidden layers, sigmoid for one output, softmax for many
- [ ] I know why a network with no activations collapses to a linear model
- [ ] I can pick the loss for regression, binary and multi-class problems
- [ ] I know why accuracy cannot be a loss function
- [ ] **I have written and trained a neural network in NumPy**
- [ ] I can spot overfitting in a training log and fix it with early stopping
- [ ] I know Adam is the robust default, not always the winner
- [ ] I can explain how next-token prediction turns a network into a text generator

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-09-deep-learning.ipynb) | Every example above, runnable |
| [TensorFlow Playground](https://playground.tensorflow.org/) | Watch a network learn, in your browser, no code |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
