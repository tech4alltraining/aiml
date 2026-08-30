"""Generate the figures embedded in Session 7 (clustering).

Run from this directory:
    python generate_plots_s7.py
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

OUT = pathlib.Path(__file__).parent
DATA = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/clustering/Mall_Customers.csv"
LOCAL = pathlib.Path(__file__).parents[4] / "datasets" / "clustering" / "Mall_Customers.csv"

def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)

df = pd.read_csv(LOCAL if LOCAL.exists() else DATA)
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ---------------------------------------------------------------- 1. segments
km = KMeans(n_clusters=5, random_state=0, n_init=10).fit(X)
df["Cluster"] = km.labels_
centers = km.cluster_centers_

# name each cluster from where its centre sits
def name_of(cx, cy):
    if 40 < cx < 75 and 40 < cy < 65:
        return "Standard"
    if cx >= 60 and cy >= 60:
        return "Target"
    if cx >= 60 and cy < 40:
        return "Careful"
    if cx < 45 and cy < 40:
        return "Sensible"
    return "Careless"

fig, ax = plt.subplots(figsize=(8.5, 6))
colours = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
for c in range(5):
    sub = df[df["Cluster"] == c]
    ax.scatter(sub["Annual Income (k$)"], sub["Spending Score (1-100)"],
               s=42, alpha=.85, color=colours[c],
               label=f"{name_of(*centers[c])}  (n={len(sub)})")
ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=260, c="black",
           edgecolor="white", linewidth=1.6, zorder=5, label="centroids")
for cx, cy in centers:
    ax.annotate(name_of(cx, cy), (cx, cy), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=9.5, weight="bold")
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score (1-100)")
ax.set_title("k-Means (k=5) on mall customers — five segments a marketer can act on")
ax.legend(loc="center left", fontsize=9); ax.grid(alpha=.3)
save(fig, "s7-mall-segments.png")

# ------------------------------------------------------- 2. elbow + silhouette
ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    m = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
    inertias.append(m.inertia_)
    sils.append(silhouette_score(X, m.labels_))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
a1.plot(list(ks), inertias, "o-", color="#4c72b0")
a1.axvline(5, color="crimson", ls="--", lw=1.4)
a1.annotate("elbow at k=5", (5, inertias[3]), textcoords="offset points",
            xytext=(28, 34), color="crimson", weight="bold",
            arrowprops=dict(arrowstyle="->", color="crimson"))
a1.set_xlabel("k"); a1.set_ylabel("inertia (within-cluster sum of squares)")
a1.set_title("Elbow method — look for the bend"); a1.grid(alpha=.3)

a2.plot(list(ks), sils, "s-", color="#55a868")
a2.axvline(5, color="crimson", ls="--", lw=1.4)
a2.annotate(f"peak {max(sils):.3f} at k=5", (5, max(sils)),
            textcoords="offset points", xytext=(28, -30), color="crimson",
            weight="bold", arrowprops=dict(arrowstyle="->", color="crimson"))
a2.set_xlabel("k"); a2.set_ylabel("silhouette score")
a2.set_title("Silhouette — look for the highest point"); a2.grid(alpha=.3)
fig.suptitle("Two methods, one answer: k = 5", fontsize=13, y=1.03)
fig.tight_layout(); save(fig, "s7-elbow-silhouette.png")

# ------------------------------------------------------------- 3. scaling
X2 = df[["Age", "Annual Income (k$)"]]
raw_labels = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(X2)
X2s = StandardScaler().fit_transform(X2)
sc_labels = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(X2s)
ari = adjusted_rand_score(raw_labels, sc_labels)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, lab, title in [(a1, raw_labels, "Unscaled — income (var 690) drowns out age (var 195)"),
                       (a2, sc_labels, "Scaled — both columns get an equal vote")]:
    for c in range(4):
        m = lab == c
        ax.scatter(X2["Age"][m], X2["Annual Income (k$)"][m], s=40, alpha=.85,
                   color=colours[c])
    ax.set_xlabel("Age"); ax.set_title(title, fontsize=10.5); ax.grid(alpha=.3)
a1.set_ylabel("Annual Income (k$)")
fig.suptitle(f"Scaling changes the answer — the two agree on only {ari:.0%} of pairs (ARI)",
             fontsize=12.5, y=1.03)
fig.tight_layout(); save(fig, "s7-scaling-matters.png")

# ------------------------------------------------------------- 4. dendrogram
Z = linkage(X, method="complete")
fig, ax = plt.subplots(figsize=(13, 5.5))
dendrogram(Z, ax=ax, color_threshold=90, no_labels=True)
ax.axhline(90, color="crimson", ls="--", lw=1.5)
ax.annotate("cut here -> 5 clusters", (10, 92), color="crimson", weight="bold")
ax.set_title("Hierarchical clustering dendrogram (complete linkage)")
ax.set_xlabel("customers"); ax.set_ylabel("Euclidean distance at merge")
fig.tight_layout(); save(fig, "s7-dendrogram.png")

# -------------------------------------------------------- 5. linkage methods
fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
for ax, method in zip(axes.ravel(), ["ward", "complete", "average", "single"]):
    dendrogram(linkage(X, method=method), ax=ax, no_labels=True)
    lab = AgglomerativeClustering(n_clusters=5, linkage=method).fit_predict(X)
    sizes = sorted(pd.Series(lab).value_counts().tolist(), reverse=True)
    ax.set_title(f"{method}  ->  cluster sizes {sizes}", fontsize=10.5)
    ax.set_ylabel("distance")
fig.suptitle("The linkage rule decides the answer — single linkage chains everything into one blob",
             fontsize=12.5, y=1.0)
fig.tight_layout(); save(fig, "s7-linkage-methods.png")

# ------------------------------------------------------- 6. algorithms compared
Xb, yb = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)
Xm, ym = make_moons(n_samples=300, noise=0.06, random_state=42)
rows = [("round blobs", Xb, yb, 4, 0.8), ("two crescents", Xm, ym, 2, 0.25)]
fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
for r, (name, Xd, yd, k, eps) in enumerate(rows):
    preds = [
        ("k-Means", KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xd)),
        ("Hierarchical (single)", AgglomerativeClustering(n_clusters=k, linkage="single").fit_predict(Xd)),
        ("DBSCAN", DBSCAN(eps=eps, min_samples=5).fit_predict(Xd)),
    ]
    for c, (title, lab) in enumerate(preds):
        ax = axes[r, c]
        for cl in sorted(set(lab)):
            m = lab == cl
            ax.scatter(Xd[m, 0], Xd[m, 1], s=22, alpha=.85,
                       color="lightgrey" if cl == -1 else colours[cl % 5],
                       label="noise" if cl == -1 else None)
        ax.set_title(f"{title} on {name}\nARI vs truth = {adjusted_rand_score(yd, lab):.2f}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if -1 in lab:
            ax.legend(fontsize=8, loc="lower right")
fig.suptitle("No algorithm wins everywhere — the shape of your data decides",
             fontsize=13, y=1.0)
fig.tight_layout(); save(fig, "s7-algorithms-compared.png")

# ---------------------------------------------------- 7. the metric is wrong
km_m = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(Xm)
db_m = DBSCAN(eps=0.25, min_samples=5).fit_predict(Xm)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, lab, title in [
        (a1, km_m, f"k-Means\nsilhouette {silhouette_score(Xm, km_m):.3f}  |  ARI {adjusted_rand_score(ym, km_m):.3f}"),
        (a2, db_m, f"DBSCAN\nsilhouette {silhouette_score(Xm, db_m):.3f}  |  ARI {adjusted_rand_score(ym, db_m):.3f}")]:
    for cl in sorted(set(lab)):
        m = lab == cl
        ax.scatter(Xm[m, 0], Xm[m, 1], s=26, alpha=.85,
                   color="lightgrey" if cl == -1 else colours[cl % 5])
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("The higher silhouette is the WRONG answer — always plot it",
             fontsize=13, y=1.02)
fig.tight_layout(); save(fig, "s7-metric-wrong.png")

print("\nAll Session 7 plots generated.")
