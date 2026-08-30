# Session 7 — Unsupervised Learning

**Clustering · Association Rule Mining · Dimensionality Reduction · k-Means, Hierarchical Clustering, DBSCAN**

| | |
|---|---|
| **Notebook** | [session-07-unsupervised.ipynb](../notebooks/session-07-unsupervised.ipynb) |
| **Previous** | [Session 6 — Augmentation & Feature Engineering](session-06-augmentation-feature-engg-red.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Everything so far had an answer key.** From here there is none. Nobody can tell you your clustering is "correct" — which makes judgement, not code, the hard part of this session.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Say what unsupervised learning is, and name its three families
2. Run k-Means and **choose k with evidence** rather than by guessing
3. Read an elbow plot and a silhouette score
4. Use hierarchical clustering and read a dendrogram
5. Use DBSCAN, and explain what it does that k-Means cannot
6. Explain a case where **the metric picks the wrong answer** — and why you must look at the plot
7. Compute support, confidence and lift, and read a market-basket rule
8. Use PCA to see high-dimensional data in two dimensions

---

## The five topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Unsupervised learning](#1-what-unsupervised-learning-is) | No answer key, so no accuracy score |
| 2 | [k-Means](#2-k-means-and-choosing-k) | You must **choose** k, and you can choose it with evidence |
| 3 | [Hierarchical & DBSCAN](#3-hierarchical-clustering-and-dbscan) | DBSCAN finds shapes and admits "noise" |
| 4 | [Association rules](#4-association-rule-mining) | **Lift**, not confidence, is the interesting number |
| 5 | [Dimensionality reduction](#5-dimensionality-reduction) | Two dimensions you can see beat thirteen you cannot |

---

# 1. What unsupervised learning is

**No labels. No answer key. No accuracy score.** You give the algorithm data and it finds structure.

🧠 **Analogy: a box of loose photographs.** Supervised learning is a box already sorted into labelled envelopes — "family", "holidays", "work" — and you learn to file new photos correctly. Unsupervised learning is an unsorted box: **you decide the piles yourself.** Two people would produce different piles, and neither would be wrong.

| Family | Question it answers | Tools |
|---|---|---|
| **Clustering** | *What natural groups exist?* | k-Means, Hierarchical, DBSCAN |
| **Association rules** | *What goes with what?* | Apriori, FP-Growth |
| **Dimensionality reduction** | *Can I see this in 2D?* | PCA, t-SNE, UMAP |

> **The hard part is not the code — it is knowing whether the answer is useful.** A clustering with a great silhouette score and no business meaning is worthless. A clustering with a mediocre score that marketing can act on is valuable.

## 📘 Examples

**Example 1 — the missing `y`**

```python
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X, y)     # supervised   - needs y
km  = KMeans(n_clusters=5).fit(X)        # unsupervised - NO y anywhere
```

**That missing `y` is the whole difference.** There is nothing to score against.

**Example 2 — so how do you judge it?**

```python
from sklearn.metrics import silhouette_score
print(silhouette_score(X_scaled, km.labels_))
```

Silhouette runs from −1 to 1 and asks: *is each point closer to its own group than to the nearest other group?*

| Score | Reading |
|---|---|
| Above 0.5 | Strong, well-separated groups |
| 0.25 – 0.5 | Reasonable structure |
| Below 0.25 | Weak — the groups may not be real |

**Example 3 — the mall customers**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
mall = pd.read_csv(BASE + "clustering/Mall_Customers.csv")
print(mall.columns.tolist())
# ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
```

Nobody labelled these customers. **The segments do not exist until you find them.**

## ✏️ Practice

1. Load the mall data. How many rows and columns?
2. Which two columns would you cluster on, and why those?
3. Why can you not report accuracy for a clustering?
4. Scale the two columns and run `KMeans(n_clusters=5)`. How many customers per cluster?
5. Give a real business question for each of the three families.

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

mall = pd.read_csv(BASE + "clustering/Mall_Customers.csv")
print(mall.shape)                                                      # 1
# 2 - Income and Spending Score. They are the two the business can ACT on:
#     they map directly onto "who should get which offer".

# 3 - There is no correct answer to compare against. Accuracy needs labels.

X = mall[["Annual Income (k$)", "Spending Score (1-100)"]]             # 4
Xs = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(Xs)
print(pd.Series(labels).value_counts().sort_index())

# 5 - Clustering: "which customer segments should get which offer?"
#     Association: "which products should sit next to each other?"
#     Reduction:   "can I plot my 30-column dataset to spot the outliers?"
```
</details>

## ❓ MCQs

**Q1.** What distinguishes unsupervised from supervised learning?
- (a) It is faster  (b) There is no target column  (c) It uses neural networks  (d) It needs more data

**Q2.** Why can you not report accuracy for a clustering?
- (a) It is too slow  (b) There is no correct answer to compare against  (c) sklearn does not support it  (d) You can

**Q3.** A silhouette score of 0.55 indicates…
- (a) 55% accuracy  (b) Strong, well-separated groups  (c) Weak structure  (d) An error

**Q4.** Two analysts cluster the same data and get different groupings. This means…
- (a) One is wrong  (b) It can be legitimate — clustering has no single right answer  (c) The data is bad  (d) They used different libraries

**Q5.** Which family answers *"what products are bought together?"*
- (a) Clustering  (b) Association rule mining  (c) Dimensionality reduction  (d) Regression

**Q6.** A clustering scores 0.62 silhouette but the groups mean nothing to the business. It is…
- (a) A success — the score is high  (b) Not useful; the score is not the goal  (c) Overfitted  (d) Underfitted

<details><summary>Answers</summary>

**A1 — (b).** No `y` anywhere in the code.

**A2 — (b).** Accuracy needs an answer key, and there isn't one.

**A3 — (b).** Above 0.5 means points sit comfortably closer to their own group than to any other.

**A4 — (b).** Different scalings, different features and different k all give defensible answers.

**A5 — (b) Association rule mining.**

**A6 — (b).** **The score is a sanity check, not the objective.** A clustering nobody can act on is worthless.
</details>

## 🎯 Tasks

**Task 1 — The unlabelled question.** Write five questions your college could answer *without* any labelled data, and say which family each needs. **For each, state what a useful answer would let someone actually do differently.**

**Task 2 — Two defensible answers.** Cluster the mall data twice — once on income and spending, once including age. Produce both segment profiles. **Argue that both are legitimate**, then say which you would present and why.

---

# 2. k-Means and choosing k

k-Means splits the data into **k** groups by repeatedly moving k centre points.

🧠 **Analogy: k food stalls at a festival.** Drop k stalls at random. Everyone walks to their nearest stall. Each stall then moves to the middle of its own crowd. People re-choose. Repeat until nothing moves. **You must decide how many stalls to put out before you start — that is k, and the algorithm cannot choose it for you.**

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
```

> ⚠️ **Always scale before k-Means.** It measures distance, so an income column in the tens of thousands drowns out a score column under 100 — exactly the Session 5 kNN problem.

## Choosing k with evidence

| Method | What it shows | How to read it |
|---|---|---|
| **Elbow** | Inertia (tightness) against k | The bend where extra groups stop helping |
| **Silhouette** | Separation against k | **The peak** |
| **Business sense** | Can you name and act on the groups? | Five segments is a campaign; forty is not |

## 📘 Examples

**Example 1 — the two curves, measured on the mall data**

| k | Inertia | Silhouette |
|---|---|---|
| 2 | 269.69 | 0.3213 |
| 3 | 157.70 | 0.4666 |
| 4 | 108.92 | 0.4939 |
| **5** | **65.57** | **0.5547** ← peak |
| 6 | 55.06 | 0.5399 |
| 7 | 44.86 | 0.5281 |
| 8 | 37.23 | 0.4552 |

**Inertia always falls** as k rises — with k = 200 every customer is their own cluster and inertia is zero. That is why inertia alone can never choose k; you look for the **bend**. Silhouette does have a peak, and here **both methods agree on k = 5**.

**When your two methods agree, you have a real answer. When they disagree, look at the plot and think about the business.**

**Example 2 — naming the segments**

```python
mall["cluster"] = labels
print(mall.groupby("cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1))
```

The five groups have clear identities:

| Segment | Income | Spending | What to do |
|---|---|---|---|
| Careful | High | Low | Persuade — they can afford more |
| Target | High | High | Protect — these are your best customers |
| Standard | Mid | Mid | The bulk; steady offers |
| Careless | Low | High | Watch credit risk |
| Sensible | Low | Low | Low priority |

> **This table is the deliverable — not the silhouette score.** A cluster you cannot name is a cluster you cannot use.

**Example 3 — why `n_init` matters**

```python
KMeans(n_clusters=5, n_init=10, random_state=42)
```

k-Means starts from random positions and can settle into a poor arrangement. `n_init=10` runs it ten times and keeps the best. **Always set `random_state` too, or your report will not reproduce.**

## ✏️ Practice

1. Compute inertia for k = 2…8 and plot it. Where is the elbow?
2. Compute silhouette for the same range. Where is the peak?
3. Run k-Means **without** scaling. How does the silhouette change?
4. Profile the five clusters and give each a one-word name.
5. Run with `n_init=1` and several different `random_state` values. Do you always get the same answer?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

mall = pd.read_csv(BASE + "clustering/Mall_Customers.csv")
X = mall[["Annual Income (k$)", "Spending Score (1-100)"]]
Xs = StandardScaler().fit_transform(X)

for k in range(2, 9):                                                  # 1, 2
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    print(f"k={k}  inertia {km.inertia_:8.2f}  silhouette {silhouette_score(Xs, km.labels_):.4f}")
# Elbow at 5; silhouette peaks at 5. Both agree -> a real answer.

km_raw = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X)       # 3
print("unscaled silhouette:", round(silhouette_score(X, km_raw.labels_), 4))
print("scaled   silhouette:", round(silhouette_score(Xs,
      KMeans(n_clusters=5, n_init=10, random_state=42).fit(Xs).labels_), 4))
# Here the two columns happen to have similar ranges, so the damage is mild.
# On income-vs-age it would be severe. SCALE ANYWAY - it costs one line.

mall["cluster"] = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(Xs)
print(mall.groupby("cluster")[["Annual Income (k$)",                   # 4
                               "Spending Score (1-100)"]].mean().round(1))
# Careful / Target / Standard / Careless / Sensible

for seed in [0, 1, 2, 3, 4]:                                           # 5
    km = KMeans(n_clusters=5, n_init=1, random_state=seed).fit(Xs)
    print(f"seed {seed}: inertia {km.inertia_:.2f}")
# With n_init=1 the inertia varies between seeds: it sometimes settles in a
# worse arrangement. n_init=10 runs it ten times and keeps the best.
```
</details>

## ❓ MCQs

**Q1.** Why can inertia alone never choose k?
- (a) It is slow to compute  (b) It always falls as k rises — at k = n it is zero  (c) It needs labels  (d) It only works for k < 5

**Q2.** Where do you read k off a silhouette curve?
- (a) The lowest point  (b) The peak  (c) The first value  (d) Where it crosses zero

**Q3.** Why scale before k-Means?
- (a) It runs faster  (b) It measures distance, so a large-range column dominates  (c) sklearn requires it  (d) To remove outliers

**Q4.** What does `n_init=10` do?
- (a) Makes 10 clusters  (b) Runs the algorithm 10 times from different starts and keeps the best  (c) Uses 10 features  (d) Runs 10 iterations

**Q5.** Elbow and silhouette both point to k = 5. This means…
- (a) Nothing  (b) Two independent methods agree, so you have a real answer  (c) k must be 5 for all datasets  (d) The data is bad

**Q6.** Your clustering scores well but you cannot describe the groups in words. You should…
- (a) Ship it — the score is good  (b) Treat it as not yet useful and try different features or k  (c) Increase k  (d) Ignore the profile

<details><summary>Answers</summary>

**A1 — (b).** With one cluster per point, inertia is zero and the model is useless. You look for the **bend**, not the minimum.

**A2 — (b) The peak.** Higher silhouette means better-separated groups.

**A3 — (b).** Same reason kNN needs it in Session 5.

**A4 — (b).** k-Means can settle into a poor arrangement from a bad random start.

**A5 — (b).** Agreement between independent methods is the strongest evidence you get without labels.

**A6 — (b).** **A cluster you cannot name is a cluster you cannot use.**
</details>

## 🎯 Tasks

**Task 1 — The segment report.** Cluster the mall customers, choose k with both methods, profile each segment, **name each one**, and write a one-line marketing action per segment. **The deliverable is the table of named segments, not the score.**

**Task 2 — Does age change the story?** Re-run including `Age` as a third feature. Do the segments survive? **Report which customers moved between segments and what that tells you.**

**Task 3 — Break it deliberately.** Cluster on income and spending **without scaling**, using a version of income multiplied by 1,000. Show how badly the segments degrade and explain exactly why.

---

# 3. Hierarchical clustering and DBSCAN

k-Means has two real limitations: **you must choose k in advance**, and **it only finds round blobs**. These two algorithms answer those limitations.

## Hierarchical clustering

🧠 **Analogy: a family tree, built upwards.** Start with everyone alone. Repeatedly join the two closest groups. Keep going until everyone is in one group. **The full history is the dendrogram — and you cut it at whatever height you like.**

```python
from sklearn.cluster import AgglomerativeClustering
labels = AgglomerativeClustering(n_clusters=5).fit_predict(X_scaled)
```

**The advantage:** the dendrogram shows you every possible k at once. You choose *after* seeing the structure, not before.

## DBSCAN

🧠 **Analogy: finding towns from a map of houses at night.** Anywhere houses are densely packed is a town. Isolated houses in the countryside belong to no town at all — and DBSCAN is the only algorithm here that will **say so**.

```python
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.4, min_samples=5).fit_predict(X_scaled)
# label -1 means NOISE - not in any cluster
```

| Parameter | Meaning |
|---|---|
| `eps` | How close counts as "neighbouring" |
| `min_samples` | How many neighbours make a dense region |

| | k-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Choose k in advance? | **Yes** | No — cut the tree after | **No** |
| Finds odd shapes? | No, round blobs only | Somewhat | **Yes** |
| Handles outliers? | No — forces every point into a group | No | **Yes — labels them noise** |
| Scales to large data? | **Yes** | Poorly | Moderately |

## 📘 Examples

**Example 1 — all three on the mall data**

| Method | Silhouette |
|---|---|
| k-Means, k = 5 | **0.5547** |
| Hierarchical, k = 5 | 0.5538 |
| DBSCAN, eps = 0.4 | 0.4133 (4 clusters, 15 noise points) |

**k-Means and hierarchical agree almost exactly** — two different algorithms finding the same structure is strong evidence the structure is real.

**DBSCAN does worse here, and that is expected.** The mall segments are round blobs, which is exactly what k-Means is built for. **Using the fancier algorithm because it is fancier is a mistake.**

**Example 2 — where DBSCAN wins, and where the metric lies**

```python
from sklearn.datasets import make_moons
X_moons, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
```

Two interleaving crescents. Measured:

| Method | Silhouette | What it actually found |
|---|---|---|
| k-Means, k = 2 | **0.4863** | Sliced both crescents in half — **wrong** |
| DBSCAN, eps = 0.3 | 0.3298 | The two crescents exactly — **right** |

**Read that again. The metric prefers the wrong answer.**

Silhouette is built on distance to a cluster centre, so it quietly assumes clusters are round. A crescent's own centre is outside the crescent. **The metric is biased toward exactly the kind of cluster k-Means produces.**

> **This is why you always plot your clusters.** A number cannot tell you that a crescent is a crescent. When the plot and the metric disagree, **the plot wins** — and you should be able to explain why.

**Example 3 — DBSCAN's sensitivity to `eps`**

| `eps` | Clusters | Noise points | Silhouette |
|---|---|---|---|
| 0.2 | 7 | 77 | 0.1406 |
| 0.3 | 7 | 35 | 0.3161 |
| **0.4** | **4** | **15** | **0.4133** |
| 0.5 | 2 | 8 | 0.3504 |
| 0.6 | 1 | 5 | — |

**Small `eps` fragments the data and calls most of it noise. Large `eps` melts everything into one blob.** DBSCAN removes the burden of choosing k and hands you the burden of choosing `eps` instead. **There is no free lunch here — only a different choice.**

## ✏️ Practice

1. Run all three on the mall data at k = 5 and compare silhouettes.
2. Sweep DBSCAN `eps` from 0.2 to 0.6. How do clusters and noise change?
3. On `make_moons`, plot k-Means and DBSCAN side by side. Which is visually right?
4. Compute silhouette for both on the moons. **Which does the metric prefer, and is it correct?**
5. How many mall customers does DBSCAN call noise? Look at a few — are they unusual?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

mall = pd.read_csv(BASE + "clustering/Mall_Customers.csv")
Xs = StandardScaler().fit_transform(mall[["Annual Income (k$)", "Spending Score (1-100)"]])

km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(Xs)          # 1
ag = AgglomerativeClustering(n_clusters=5).fit(Xs)
db = DBSCAN(eps=0.4, min_samples=5).fit(Xs)
print("kMeans      ", round(silhouette_score(Xs, km.labels_), 4))
print("hierarchical", round(silhouette_score(Xs, ag.labels_), 4))
print("DBSCAN      ", round(silhouette_score(Xs, db.labels_), 4))
# kMeans and hierarchical agree almost exactly -> the structure is real.

for eps in [0.2, 0.3, 0.4, 0.5, 0.6]:                                  # 2
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(Xs)
    n = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"eps={eps}  clusters {n}  noise {(lab == -1).sum()}")

Xm, _ = make_moons(n_samples=300, noise=0.06, random_state=42)         # 3, 4
kl = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Xm)
dl = DBSCAN(eps=0.3, min_samples=5).fit_predict(Xm)
print("moons kMeans silhouette:", round(silhouette_score(Xm, kl), 4))  # 0.4863
print("moons DBSCAN silhouette:", round(silhouette_score(Xm, dl), 4))  # 0.3298
# The METRIC prefers kMeans -- and the metric is WRONG. Silhouette assumes
# round clusters, so it is biased against crescents. PLOT IT and you can
# see DBSCAN recovered the two crescents exactly.

noise = mall[DBSCAN(eps=0.4, min_samples=5).fit_predict(Xs) == -1]     # 5
print(noise[["Annual Income (k$)", "Spending Score (1-100)"]])
# They sit in the sparse gaps between the dense blobs -- genuinely unusual
# customers. k-Means would have forced each into a segment regardless.
```
</details>

## ❓ MCQs

**Q1.** Which algorithm does **not** need k chosen in advance?
- (a) k-Means  (b) DBSCAN  (c) Both need it  (d) Neither needs it

**Q2.** In DBSCAN output, what does the label `-1` mean?
- (a) The first cluster  (b) Noise — the point is in no cluster  (c) An error  (d) A missing value

**Q3.** On `make_moons`, silhouette rates k-Means (0.486) above DBSCAN (0.330), yet DBSCAN is right. Why?
- (a) DBSCAN is broken  (b) Silhouette assumes round clusters, so it is biased against crescents  (c) The data is bad  (d) Silhouette needs labels

**Q4.** DBSCAN scored worse than k-Means on the mall data because…
- (a) DBSCAN is a worse algorithm  (b) Those segments really are round blobs, which suits k-Means  (c) The data was unscaled  (d) `eps` cannot be tuned

**Q5.** What is a dendrogram?
- (a) A scatter plot  (b) A tree showing every merge, so you can cut at any k  (c) A confusion matrix  (d) A histogram

**Q6.** Increasing DBSCAN's `eps` tends to…
- (a) Create more clusters  (b) Merge clusters and reduce noise  (c) Have no effect  (d) Always improve silhouette

**Q7.** When your metric and your plot disagree, you should…
- (a) Trust the metric — it is objective  (b) Look at the plot and understand why the metric is misled  (c) Ignore both  (d) Change datasets

<details><summary>Answers</summary>

**A1 — (b) DBSCAN.** It finds however many dense regions exist. You choose `eps` instead — a different burden, not no burden.

**A2 — (b) Noise.** DBSCAN is the only one of the three that will say "this point belongs to nothing".

**A3 — (b).** Silhouette measures distance to a centre; a crescent's centre lies outside the crescent.

**A4 — (b).** **Do not use the fancier algorithm because it is fancier.** Match the algorithm to the shape of your data.

**A5 — (b).** Its great advantage: you choose k *after* seeing the structure.

**A6 — (b).** Larger neighbourhoods absorb more points, eventually into a single blob.

**A7 — (b).** A number cannot tell you a crescent is a crescent. **Always plot your clusters.**
</details>

## 🎯 Tasks

**Task 1 — Three algorithms, one dataset.** Run all three on a dataset of your choice. Produce a comparison table and **two scatter plots side by side**. Recommend one, and say what would change your mind.

**Task 2 — Reproduce the metric failure.** Build a dataset where silhouette prefers the visually wrong clustering. **Explain in a paragraph exactly what assumption the metric is making** and when that assumption is safe.

**Task 3 — The eps sweep.** Sweep DBSCAN's `eps`, and plot clusters-found and noise-count against it on twin axes. **Mark the value you would choose and justify it with more than the silhouette score.**

---

# 4. Association Rule Mining

**Which things appear together?** This is the "customers who bought X also bought Y" engine.

🧠 **Analogy: watching a supermarket's trolleys all day.** You notice bread and butter keep turning up together. But so do bread and *shopping bags* — because almost every trolley has a bag. **The first is a real pattern; the second is just popularity.** Telling those apart is the entire point of this topic.

## The three numbers

For a rule **A → B**:

| Number | Formula | Question |
|---|---|---|
| **Support** | baskets with A **and** B ÷ all baskets | *How often does this happen at all?* |
| **Confidence** | baskets with A and B ÷ baskets with A | *When A happens, how often does B?* |
| **Lift** | confidence ÷ support(B) | ***Is this more than B's popularity alone?*** |

**Lift is the one that matters:**

- **Lift > 1** — A genuinely makes B more likely. **Interesting.**
- **Lift ≈ 1** — no relationship; B is just common.
- **Lift < 1** — A makes B *less* likely.

> **High confidence with lift ≈ 1 is the classic beginner's trap.** "90% of people who buy bread also buy shopping bags" sounds like a finding. If 90% of *all* baskets contain a bag, you have discovered nothing.

## 📘 Examples

**Example 1 — the three numbers, by hand**

```python
def support(itemset):
    return sum(1 for b in baskets if itemset <= b) / len(baskets)

def rule(A, B):
    conf = support(A | B) / support(A)
    return support(A | B), conf, conf / support(B)      # support, confidence, lift

s, c, l = rule({"bread"}, {"butter"})
print(f"bread -> butter   support {s:.3f}  confidence {c:.3f}  lift {l:.2f}")
```

**Example 2 — Apriori, and the one idea it rests on**

```python
MIN_SUP = 0.05

freq = {frozenset([i]) for i in items if support({i}) >= MIN_SUP}   # single items
all_freq, k = set(freq), 2
while freq and k <= 3:
    cand = {a | b for a in freq for b in freq if len(a | b) == k}    # grow by one
    freq = {c for c in cand if support(set(c)) >= MIN_SUP}          # keep frequent
    all_freq |= freq
    k += 1
```

**The Apriori insight, in one sentence:** *if {bread, butter} is rare, then {bread, butter, jam} cannot possibly be common.* So you never bother testing it. **That single observation is what makes the problem tractable** — otherwise you would test every subset of every basket.

**Example 3 — rules found in 500 simulated baskets**

| Antecedent | Consequent | Support | Confidence | **Lift** |
|---|---|---|---|---|
| butter, jam | bread | 0.068 | 0.567 | **1.97** |
| bread, jam | butter | 0.068 | 0.850 | 1.77 |
| rice, tea | sugar | 0.056 | 0.824 | 1.75 |
| **bread** | **butter** | **0.232** | **0.806** | **1.68** |
| butter, tea | sugar | 0.086 | 0.782 | 1.66 |

The bread → butter rule was **deliberately planted** in the generated data, and Apriori recovered it: confidence 0.806, lift 1.68. **Sort by lift, not confidence** — the highest-confidence rules are often just about popular items.

> **In production:** `pip install mlxtend`, then `apriori()` and `association_rules()`. But you now know what those functions do, which is the part that matters.

## ✏️ Practice

1. Compute support, confidence and lift for bread → butter.
2. Find a rule with high confidence but lift near 1. Why is it uninteresting?
3. Sort all rules by lift and then by confidence. Do the top fives differ?
4. Raise `MIN_SUP` from 0.05 to 0.15. How many rules survive?
5. State the Apriori pruning rule in your own words.

<details><summary>Solutions</summary>

```python
import numpy as np
from itertools import combinations

rng = np.random.default_rng(7)
items = ["bread", "milk", "eggs", "butter", "jam", "tea", "coffee", "sugar", "rice", "oil"]
baskets = []
for _ in range(500):
    b = {str(x) for x in rng.choice(items, size=rng.integers(2, 5), replace=False)}
    if "bread" in b and rng.random() < .75: b.add("butter")     # planted rule
    if "tea" in b and rng.random() < .70: b.add("sugar")        # planted rule
    baskets.append(b)

def support(s): return sum(1 for b in baskets if s <= b) / len(baskets)

def rule(A, B):
    conf = support(A | B) / support(A)
    return support(A | B), conf, conf / support(B)

s, c, l = rule({"bread"}, {"butter"})                                  # 1
print(f"bread -> butter  sup {s:.3f}  conf {c:.3f}  lift {l:.2f}")

for other in ["milk", "eggs", "oil"]:                                  # 2
    s2, c2, l2 = rule({"bread"}, {other})
    print(f"bread -> {other:<6} conf {c2:.3f}  lift {l2:.2f}")
# A rule with lift near 1 tells you nothing: the consequent is simply
# common, and the antecedent did not change its odds.

MIN_SUP, MIN_CONF = 0.05, 0.5                                          # 3, 4
def mine(min_sup):
    freq = {frozenset([i]) for i in items if support({i}) >= min_sup}
    allf, k = set(freq), 2
    while freq and k <= 3:
        cand = {a | b for a in freq for b in freq if len(a | b) == k}
        freq = {c for c in cand if support(set(c)) >= min_sup}
        allf |= freq; k += 1
    out = []
    for s_ in allf:
        if len(s_) < 2: continue
        for r in range(1, len(s_)):
            for ante in combinations(sorted(s_), r):
                A = set(ante); B = set(s_) - A
                conf = support(set(s_)) / support(A)
                if conf >= MIN_CONF:
                    out.append((sorted(A), sorted(B), support(set(s_)), conf, conf / support(B)))
    return out

rules = mine(MIN_SUP)
print("\\ntop by LIFT      :", [f"{','.join(a)}->{','.join(b)}" for a, b, _, _, _
                                in sorted(rules, key=lambda r: -r[4])[:5]])
print("top by CONFIDENCE:", [f"{','.join(a)}->{','.join(b)}" for a, b, _, _, _
                                in sorted(rules, key=lambda r: -r[3])[:5]])
# They differ. High-confidence rules often just point at popular items.

print("\\nrules at min_sup 0.05:", len(rules))
print("rules at min_sup 0.15:", len(mine(0.15)))

# 5 - If an itemset is rare, every LARGER set containing it must be at
#     least as rare. So once {bread, butter} falls below the threshold you
#     never need to test {bread, butter, jam}. That prune is what makes
#     the search feasible.
```
</details>

## ❓ MCQs

**Q1.** Which of the three numbers tells you a rule is genuinely interesting?
- (a) Support  (b) Confidence  (c) Lift  (d) All equally

**Q2.** A rule has confidence 0.90 and lift 1.02. It means…
- (a) A strong finding  (b) The consequent is simply very common — nothing learned  (c) An error  (d) A negative relationship

**Q3.** Lift below 1 means…
- (a) No relationship  (b) A makes B *less* likely  (c) A causes B  (d) Low support

**Q4.** State the Apriori pruning rule.
- (a) Drop rare items at the end  (b) If an itemset is rare, every larger set containing it is also rare  (c) Only test pairs  (d) Sort by confidence

**Q5.** Raising the minimum support threshold…
- (a) Finds more rules  (b) Finds fewer but more common rules  (c) Has no effect  (d) Increases lift

**Q6.** Support answers which question?
- (a) *When A happens, how often does B?*  (b) *How often does this combination happen at all?*  (c) *Is this beyond B's popularity?*  (d) *How many items are there?*

<details><summary>Answers</summary>

**A1 — (c) Lift.** It is the only one that compares against the consequent's baseline popularity.

**A2 — (b).** **The classic beginner's trap.** If 88% of all baskets contain the item anyway, 90% is not news.

**A3 — (b).** Buying A actively makes B less likely — sometimes a useful finding in itself.

**A4 — (b).** That prune is what makes the search tractable.

**A5 — (b).** Fewer, more reliable, less surprising rules. It is a trade like everything else in this course.

**A6 — (b).** Support is raw frequency; confidence is conditional; lift is the comparison.
</details>

## 🎯 Tasks

**Task 1 — Your own basket data.** Build 200 baskets from a shop you know (a canteen, a stationery shop). **Plant one rule deliberately** and check your miner recovers it. Report support, confidence and lift.

**Task 2 — The popularity trap.** Find a rule in your data with high confidence and lift ≈ 1. **Write the misleading version of the finding and then the honest one.** This is exactly how bad dashboards get built.

**Task 3 — Threshold sensitivity.** Plot rules-found against minimum support. **Choose a threshold and justify it** — too low and you drown in noise, too high and you only rediscover the obvious.

---

# 5. Dimensionality Reduction

**You cannot plot thirteen dimensions. You can plot two.**

🧠 **Analogy: the shadow of a teapot.** A teapot is three-dimensional; its shadow is flat. **A well-chosen angle casts a shadow you can still recognise as a teapot; a bad angle casts a blob.** PCA finds the angle that keeps the most information.

| Tool | Best for | Caution |
|---|---|---|
| **PCA** | Fast, linear, reversible; good default | Only captures straight-line structure |
| **t-SNE** | Beautiful cluster visualisations | Distances **between** clusters are meaningless |
| **UMAP** | Faster than t-SNE, keeps more global shape | Extra install |

> ⚠️ **Session 6 warning still applies:** PCA is for *seeing* and for *linear/distance models*. It often makes tree models slightly worse, because it rotates the axes they split along.

## 📘 Examples

**Example 1 — four dimensions down to two**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(iris.drop(columns=["species"]))
coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
```

Plotting `coords` shows the three iris species separating clearly — **from four columns you could never have plotted at once.**

**Example 2 — how much did you keep?**

```python
p = PCA().fit(Xs)
print(p.explained_variance_ratio_.round(3))
print(p.explained_variance_ratio_.cumsum().round(3))
```

If the first two components hold 95% of the variance, your 2-D picture is a fair summary. **If they hold 40%, the picture is misleading and you should say so.**

**Example 3 — clustering in reduced space**

```python
coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(coords)
```

Reducing first can help distance-based algorithms by removing noise dimensions. **Always report the variance kept alongside the result.**

## ✏️ Practice

1. Reduce iris to 2 components and plot, coloured by species.
2. How much variance do the first two components hold?
3. Cluster the reduced data with k=3. Do the clusters match the real species?
4. Reduce the mall data to 2 components including `Age` and `Gender`. What appears?
5. Why must you scale before PCA?

<details><summary>Solutions</summary>

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

iris = pd.read_csv(BASE + "classification/iris.csv")
Xs = StandardScaler().fit_transform(iris.drop(columns=["species"]))

p = PCA(n_components=2, random_state=42)                               # 1, 2
coords = p.fit_transform(Xs)
print("variance per component:", p.explained_variance_ratio_.round(3))
print("total kept:", round(p.explained_variance_ratio_.sum(), 3))
# The first two components hold the large majority -> the 2-D picture is fair.

labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(coords)  # 3
print("agreement with true species (ARI):",
      round(adjusted_rand_score(iris["species"], labels), 4))
# High, but not 1.0: two of the three species genuinely overlap.
# NOTE: we can only check this because iris HAPPENS to have labels.
# In a real unsupervised problem you would not have this luxury.

# 5 - PCA maximises VARIANCE. An unscaled column with a large range has a
#     large variance purely because of its units, so PCA would point its
#     first component straight at that column and ignore the rest.
```
</details>

## ❓ MCQs

**Q1.** Why scale before PCA?
- (a) It runs faster  (b) PCA maximises variance, so a large-unit column would dominate  (c) It is required by sklearn  (d) To handle missing values

**Q2.** `explained_variance_ratio_.sum()` for two components is 0.40. Your 2-D plot is…
- (a) An excellent summary  (b) Missing most of the structure — say so when you present it  (c) Wrong  (d) Perfect

**Q3.** In a t-SNE plot, the distance *between* two clusters…
- (a) Is meaningful  (b) Is not meaningful and should not be interpreted  (c) Equals Euclidean distance  (d) Is always 1

**Q4.** PCA components are…
- (a) Original columns, reordered  (b) New axes built from combinations of the originals  (c) Cluster labels  (d) Row indices

**Q5.** PCA before a Random Forest usually…
- (a) Helps a lot  (b) Slightly hurts, because it rotates the axes trees split along  (c) Has no effect  (d) Is required

**Q6.** The teapot-shadow analogy makes which point?
- (a) PCA is slow  (b) A good projection preserves recognisable structure; a bad one destroys it  (c) PCA needs labels  (d) Shadows are 3-D

<details><summary>Answers</summary>

**A1 — (b).** Otherwise the first component just points at whichever column has the biggest units.

**A2 — (b).** **Always report the variance kept.** A 2-D plot holding 40% of the variance can mislead badly.

**A3 — (b).** t-SNE preserves local neighbourhoods, not global geometry. A common misreading.

**A4 — (b).** That is why you lose interpretability — no component is "credit score".

**A5 — (b).** As measured in Session 6.

**A6 — (b).** PCA looks for the angle that casts the most informative shadow.
</details>

## 🎯 Tasks

**Task 1 — The 2-D map.** Take any dataset with 8+ numeric columns, reduce to 2-D, and plot it. **State the variance kept in the caption** and write two sentences on what the picture does and does not show.

**Task 2 — Reduce then cluster.** Cluster a dataset both raw and after PCA. Compare silhouettes and runtimes. **Say which you would use, and note that a higher silhouette in reduced space is partly an artefact of removing dimensions.**

**Task 3 — The full unsupervised pipeline.** On one dataset: reduce to 2-D, cluster, profile and name the clusters, and produce one annotated plot. **This is a complete unsupervised analysis and a strong capstone component.**

---

# ✅ Before you move on

- [ ] I can explain unsupervised learning and name its three families
- [ ] I know why there is no accuracy score here
- [ ] I can choose k with an elbow plot **and** a silhouette curve
- [ ] I can profile and **name** my clusters — the real deliverable
- [ ] I know what DBSCAN does that k-Means cannot, and when not to use it
- [ ] I have seen a case where the metric prefers the wrong answer
- [ ] **I always plot my clusters**
- [ ] I can compute support, confidence and lift, and I sort by lift
- [ ] I can spot the high-confidence, lift ≈ 1 trap
- [ ] I can reduce to 2-D with PCA and report the variance kept

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-07-unsupervised.ipynb) | Every example above, runnable |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
