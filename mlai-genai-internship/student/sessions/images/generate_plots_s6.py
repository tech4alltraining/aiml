"""Generate the Session 6 figures (augmentation, feature engineering, reduction)."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, pathlib
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

OUT = pathlib.Path(__file__).parent
def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  wrote", name)

iris = load_iris(); X, y = iris.data, iris.target
NAMES = [n.replace(" (cm)", "") for n in iris.feature_names]

# 1. class distribution before / oversampled / SMOTE
rng = np.random.default_rng(42)
mask = np.r_[np.where(y == 0)[0][:50], np.where(y == 1)[0][:25]]
Xi, yi = X[mask], y[mask]

def random_oversample(Xa, ya, seed=42):
    r = np.random.default_rng(seed)
    counts = pd.Series(ya).value_counts()
    target = counts.max()
    Xs, ys = [Xa], [ya]
    for cls, n in counts.items():
        if n < target:
            idx = np.where(ya == cls)[0]
            extra = r.choice(idx, target - n, replace=True)
            Xs.append(Xa[extra]); ys.append(ya[extra])
    return np.vstack(Xs), np.concatenate(ys)

def smote(Xa, ya, k=5, seed=42):
    from sklearn.neighbors import NearestNeighbors
    r = np.random.default_rng(seed)
    counts = pd.Series(ya).value_counts(); target = counts.max()
    Xs, ys = [Xa], [ya]
    for cls, n in counts.items():
        if n >= target: continue
        pts = Xa[ya == cls]
        nn = NearestNeighbors(n_neighbors=min(k, len(pts)) ).fit(pts)
        idx = nn.kneighbors(pts, return_distance=False)[:, 1:]
        need = target - n
        base = r.integers(0, len(pts), need)
        nbr = idx[base, r.integers(0, idx.shape[1], need)]
        lam = r.random((need, 1))
        Xs.append(pts[base] + lam * (pts[nbr] - pts[base]))
        ys.append(np.full(need, cls))
    return np.vstack(Xs), np.concatenate(ys)

Xo, yo = random_oversample(Xi, yi)
Xs_, ys_ = smote(Xi, yi)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (yy, title) in zip(axes, [(yi, "Before — imbalanced"),
                                  (yo, "Random over-sampling"),
                                  (ys_, "SMOTE")]):
    c = pd.Series(yy).value_counts().sort_index()
    ax.bar([f"class {i}" for i in c.index], c.values, color=["#4c72b0", "#dd8452"])
    for i, v in enumerate(c.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=11)
    ax.set_title(title); ax.set_ylim(0, 60); ax.grid(alpha=.3, axis="y")
axes[0].set_ylabel("samples")
fig.tight_layout(); save(fig, "s6-class-balance.png")

# 2. what the two methods actually do, in feature space
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, (Xa, ya, title) in zip([a1, a2],
        [(Xo, yo, "Random over-sampling — copies sit ON existing points"),
         (Xs_, ys_, "SMOTE — new points sit BETWEEN existing ones")]):
    orig_n = len(Xi)
    ax.scatter(Xa[:orig_n, 2], Xa[:orig_n, 3], c=yi, cmap="coolwarm",
               s=45, edgecolor="k", linewidth=.4, label="original")
    ax.scatter(Xa[orig_n:, 2], Xa[orig_n:, 3], marker="x", s=70,
               color="green", linewidth=1.6, label="synthetic")
    ax.set_xlabel("petal length"); ax.set_ylabel("petal width")
    ax.set_title(title, fontsize=10); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); save(fig, "s6-oversample-vs-smote.png")

# 3. image augmentation
from PIL import Image, ImageEnhance
img = np.zeros((72, 72), dtype=np.uint8)
img[14:58, 20:29] = 255; img[49:58, 20:52] = 255
img[31:40, 20:52] = 255; img[31:58, 43:52] = 255
base = Image.fromarray(img)
r2 = np.random.default_rng(0)
views = {"original": base,
         "flip": base.transpose(Image.FLIP_LEFT_RIGHT),
         "rotate 20°": base.rotate(20),
         "crop + resize": base.crop((8, 8, 64, 64)).resize((72, 72)),
         "brighter": ImageEnhance.Brightness(base).enhance(1.7),
         "noise": Image.fromarray(np.clip(img + r2.normal(0, 45, img.shape), 0, 255).astype(np.uint8))}
fig, axes = plt.subplots(1, 6, figsize=(14, 2.7))
for ax, (n, im) in zip(axes, views.items()):
    ax.imshow(im, cmap="gray"); ax.set_title(n, fontsize=10); ax.axis("off")
fig.suptitle("One image, six training examples", fontsize=12, y=1.06)
fig.tight_layout(); save(fig, "s6-image-augmentation.png")

# 4. projection methods side by side
Xsc = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xsc); Xp = pca.transform(Xsc)
lda = LDA(n_components=2).fit(X, y); Xl = lda.transform(X)
Xt = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (Z, title) in zip(axes, [
        (Xp, f"PCA — unsupervised\n{pca.explained_variance_ratio_.sum():.1%} of variance kept"),
        (Xl, f"LDA — uses the labels\n{lda.explained_variance_ratio_.sum():.1%} of separation kept"),
        (Xt, "t-SNE — for visualising only\ndistances between clusters mean nothing")]):
    for cls, name in enumerate(iris.target_names):
        ax.scatter(Z[y == cls, 0], Z[y == cls, 1], s=32, alpha=.85, label=name)
    ax.set_title(title, fontsize=10.5); ax.grid(alpha=.3)
axes[0].legend()
fig.tight_layout(); save(fig, "s6-projection-methods.png")

# 5. filter methods agree
scores = {}
scores["ANOVA F"] = SelectKBest(f_classif, k="all").fit(X, y).scores_
scores["chi²"] = SelectKBest(chi2, k="all").fit(X, y).scores_
scores["mutual info"] = SelectKBest(mutual_info_classif, k="all").fit(X, y).scores_
scores["forest importance"] = RandomForestClassifier(
    n_estimators=200, random_state=42).fit(X, y).feature_importances_

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (name, s) in zip(axes, scores.items()):
    s = np.asarray(s, dtype=float)
    ax.barh(NAMES, s / s.max(), color="#55a868")
    ax.set_title(name, fontsize=11); ax.set_xlim(0, 1.08)
    ax.grid(alpha=.3, axis="x")
fig.suptitle("Four different methods, one answer: the petals carry the signal",
             fontsize=12.5, y=1.04)
fig.tight_layout(); save(fig, "s6-filter-methods-agree.png")

# 6. what reduction costs
ks = [1, 2, 3, 4]
sel_scores, pca_scores = [], []
for k in ks:
    sel = SelectKBest(f_classif, k=k).fit_transform(X, y)
    sel_scores.append(cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), sel, y, cv=5).mean())
    pc = PCA(n_components=k).fit_transform(Xsc)
    pca_scores.append(cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), pc, y, cv=5).mean())
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(ks, sel_scores, "o-", label="SelectKBest — keeps real columns")
ax.plot(ks, pca_scores, "s-", label="PCA — replaces columns")
ax.axhline(sel_scores[-1], color="grey", ls="--", lw=1, label=f"all 4 features ({sel_scores[-1]:.3f})")
ax.set_xticks(ks); ax.set_xlabel("features / components kept")
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Reduction is a trade — and PCA is not always the better half")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); save(fig, "s6-reduction-curve.png")

print("\nAll Session 6 plots generated.")
