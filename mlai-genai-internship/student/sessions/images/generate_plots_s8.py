"""Generate the figures embedded in Session 8 (evaluation & tuning).

Run from this directory:
    python generate_plots_s8.py
"""
import pathlib, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate, KFold,
    StratifiedKFold, LeaveOneOut, validation_curve, learning_curve,
    GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

OUT = pathlib.Path(__file__).parent
ROOT = pathlib.Path(__file__).parents[4] / "datasets"
BLUE, ORANGE, GREEN, RED = "#4c72b0", "#dd8452", "#55a868", "#c44e52"

def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)

# ---------------------------------------------------------------- data
# Session 3 preprocessing: impossible values -> missing -> LabelEncoder
heart = pd.read_csv(ROOT / "classification" / "heart_failure_raw.csv")
heart.loc[heart["age"] > 120, "age"] = np.nan
for c in ["age", "ejection_fraction", "serum_creatinine"]:
    heart[c] = heart[c].fillna(heart[c].median())
le = LabelEncoder()
for c in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
          "treatment_type", "DEATH_EVENT"]:
    heart[c] = le.fit_transform(heart[c])
Xh = heart.drop(columns=["DEATH_EVENT"])
yh = heart["DEATH_EVENT"]

cars = pd.read_csv(ROOT / "regression" / "cardekho_preprocessed.csv")
cars_clean = cars[(cars["seats"] > 0) & (cars["km_driven"] <= 1_000_000)].reset_index(drop=True)
FEATS = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
Xc, yc = cars_clean[FEATS], cars_clean["selling_price"]
yc_log = np.log(yc)

# ------------------------------------------------- 1. holdout instability
seeds, accs = list(range(10)), []
for s in seeds:
    a, b, c, d = train_test_split(Xh, yh, test_size=.2, random_state=s, stratify=yh)
    accs.append(make_pipeline(MinMaxScaler(), SVC()).fit(a, c).score(b, d))

fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar([str(s) for s in seeds], accs, color=BLUE, alpha=.85)
bars[int(np.argmax(accs))].set_color(GREEN)
bars[int(np.argmin(accs))].set_color(RED)
ax.axhline(np.mean(accs), color="black", ls="--", lw=1.3,
           label=f"mean {np.mean(accs):.3f}")
ax.annotate(f"{max(accs):.3f}", (int(np.argmax(accs)), max(accs)),
            ha="center", va="bottom", weight="bold", color=GREEN)
ax.annotate(f"{min(accs):.3f}", (int(np.argmin(accs)), min(accs)),
            ha="center", va="bottom", weight="bold", color=RED)
ax.set_ylim(0.55, 0.9)
ax.set_xlabel("random_state — the ONLY thing that changed")
ax.set_ylabel("test accuracy")
ax.set_title(f"The same model, the same data: a {max(accs)-min(accs):.1%} swing from the seed alone")
ax.legend(); ax.grid(alpha=.3, axis="y")
save(fig, "s8-holdout-instability.png")

# --------------------------------------------------------- 2. CV variants
kf = KFold(5, shuffle=True, random_state=42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
pipe = lambda: make_pipeline(MinMaxScaler(), SVC())
s_kf = cross_val_score(pipe(), Xh, yh, cv=kf)
s_skf = cross_val_score(pipe(), Xh, yh, cv=skf)
s_loo = cross_val_score(pipe(), Xh, yh, cv=LeaveOneOut(), n_jobs=-1)
a, b, c, d = train_test_split(Xh, yh, test_size=.2, random_state=42, stratify=yh)
holdout = pipe().fit(a, c).score(b, d)

fig, ax = plt.subplots(figsize=(9.5, 4.8))
names = ["Holdout\n(1 split)", "5-fold\nKFold", "5-fold\nStratified", "LOOCV\n(299 models)"]
means = [holdout, s_kf.mean(), s_skf.mean(), s_loo.mean()]
errs = [0, s_kf.std(), s_skf.std(), 0]
ax.bar(names, means, yerr=errs, capsize=6,
       color=[RED, BLUE, GREEN, ORANGE], alpha=.85)
for x, folds in [(1, s_kf), (2, s_skf)]:
    ax.scatter([x] * len(folds), folds, color="black", zorder=5, s=26,
               label="individual folds" if x == 1 else None)
for i, (m, e) in enumerate(zip(means, errs)):
    ax.text(i, m + e + .012, f"{m:.4f}", ha="center", weight="bold", fontsize=10)
ax.set_ylim(0.6, 0.88); ax.set_ylabel("accuracy")
ax.set_title("Four ways to estimate the same model's accuracy — and they disagree")
ax.legend(); ax.grid(alpha=.3, axis="y")
save(fig, "s8-cv-variants.png")

# ------------------------------------------------------ 3. the fit spectrum
kf_c = KFold(5, shuffle=True, random_state=42)

def fit_scores(feats, **kw):
    r = cross_validate(DecisionTreeRegressor(random_state=42, **kw),
                       cars_clean[feats], cars_clean["selling_price"],
                       cv=kf_c, scoring="r2", return_train_score=True, n_jobs=-1)
    return r["train_score"].mean(), r["test_score"].mean()

specs = [("UNDERFIT\n1 feature, depth 1", fit_scores(["vehicle_age"], max_depth=1)),
         ("GOOD FIT\n4 features, depth 5", fit_scores(
             ["vehicle_age", "km_driven", "engine", "max_power"],
             max_depth=5, min_samples_leaf=10)),
         ("OVERFIT\n6 features, no limit", fit_scores(FEATS, max_depth=None,
                                                      min_samples_leaf=1))]
fig, ax = plt.subplots(figsize=(9.5, 5))
x = np.arange(3); w = .36
ax.bar(x - w/2, [s[1][0] for s in specs], w, label="Train R²", color=BLUE, alpha=.9)
ax.bar(x + w/2, [s[1][1] for s in specs], w, label="5-fold CV R²", color=ORANGE, alpha=.9)
for i, (_, (tr, te)) in enumerate(specs):
    ax.text(i - w/2, tr + .015, f"{tr:.3f}", ha="center", fontsize=9.5)
    ax.text(i + w/2, te + .015, f"{te:.3f}", ha="center", fontsize=9.5)
    ax.annotate(f"gap {tr-te:+.3f}", (i, max(tr, te) + .09), ha="center",
                weight="bold", fontsize=10.5,
                color=RED if tr - te > .3 else "black")
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in specs])
ax.set_ylim(0, 1.18); ax.set_ylabel("R²")
ax.set_title("Underfitting, good fit, overfitting — read the GAP, not the train score")
ax.set_xlabel("car prices, 15,240 rows, 5-fold cross-validated")
ax.legend(loc="upper left"); ax.grid(alpha=.3, axis="y")
save(fig, "s8-fit-spectrum.png")

# --------------------------------------------------- 4. validation curve
depths = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30]
tr_s, te_s = validation_curve(DecisionTreeRegressor(random_state=42), Xc, yc_log,
    param_name="max_depth", param_range=depths, cv=kf, scoring="r2", n_jobs=-1)
raw_tr, raw_te = validation_curve(DecisionTreeRegressor(random_state=42), Xc, yc,
    param_name="max_depth", param_range=depths, cv=kf, scoring="r2", n_jobs=-1)
single = raw_te.mean(axis=1)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
a1.plot(depths, single, "s-", color=RED)
a1.set_title("Raw price — R² is dominated by a few luxury cars", fontsize=11)
a1.set_xlabel("max_depth"); a1.set_ylabel("5-fold CV R²"); a1.grid(alpha=.3)
a2.plot(depths, tr_s.mean(1), "o-", color=BLUE, label="train R²")
a2.plot(depths, te_s.mean(1), "s-", color=GREEN, label="5-fold CV R²")
a2.fill_between(depths, te_s.mean(1), tr_s.mean(1), color=RED, alpha=.12,
                label="the gap = overfitting")
best = depths[int(np.argmax(te_s.mean(1)))]
a2.axvline(best, color="crimson", ls="--", lw=1.3)
a2.annotate(f"best depth {best}\nCV R² {te_s.mean(1).max():.3f}", (best, .55),
            color="crimson", weight="bold", fontsize=10)
a2.set_title("log(price) — the same sweep, now readable", fontsize=11)
a2.set_xlabel("max_depth"); a2.legend(loc="lower right"); a2.grid(alpha=.3)
fig.suptitle("A validation curve you cannot read is telling you about your TARGET", fontsize=13, y=1.02)
fig.tight_layout(); save(fig, "s8-validation-curve.png")

# ------------------------------------------------------ 5. learning curve
sizes, l_tr, l_te = learning_curve(
    RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1), Xc, yc,
    train_sizes=np.linspace(.1, 1.0, 6), cv=kf, scoring="r2", n_jobs=-1,
    shuffle=True, random_state=42)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(sizes, l_tr.mean(1), "o-", color=BLUE, label="train R²")
ax.plot(sizes, l_te.mean(1), "s-", color=GREEN, label="5-fold CV R²")
ax.fill_between(sizes, l_te.mean(1), l_tr.mean(1), color=RED, alpha=.12,
                label="the gap")
ax.annotate("the curve has flattened —\nmore rows will not help",
            (sizes[-1], l_te.mean(1)[-1]), textcoords="offset points",
            xytext=(-160, -46), weight="bold", fontsize=10,
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("training rows used"); ax.set_ylabel("R²")
ax.set_title("Learning curve — would more data help?")
ax.legend(loc="center right"); ax.grid(alpha=.3)
save(fig, "s8-learning-curve.png")

# --------------------------------------------------- 6. grid vs random
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
gx, gy = np.meshgrid(np.linspace(.1, .9, 5), np.linspace(.1, .9, 5))
rng = np.random.default_rng(3)
rx, ry = rng.uniform(.05, .95, 25), rng.uniform(.05, .95, 25)
for ax, (px, py), title in [(a1, (gx.ravel(), gy.ravel()), "Grid search — 25 trials, only 5 distinct values of each"),
                            (a2, (rx, ry), "Random search — 25 trials, 25 distinct values of each")]:
    ax.scatter(px, py, s=55, color=BLUE, zorder=3)
    ax.plot(np.linspace(0, 1, 200),
            .55 + .32 * np.exp(-((np.linspace(0, 1, 200) - .62) ** 2) / .012) * 0 + 0 * np.linspace(0, 1, 200),
            alpha=0)
    xs = np.linspace(0, 1, 200)
    ax.plot(xs, .08 + .9 * np.exp(-((xs - .62) ** 2) / .02), color=GREEN, lw=2,
            alpha=.55, label="what actually matters (parameter 1)")
    ax.axvline(.62, color=GREEN, ls=":", lw=1.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("parameter that matters"); ax.set_ylabel("parameter that does not")
    ax.set_title(title, fontsize=10.5); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
fig.suptitle("Same budget, very different coverage of the parameter that matters",
             fontsize=12.5, y=1.02)
fig.tight_layout(); save(fig, "s8-grid-vs-random.png")

# --------------------------------------------------------- 7. KNN tuning
from imblearn.over_sampling import SMOTE
Xtr, Xte, ytr, yte = train_test_split(Xh, yh, test_size=.2, random_state=42, stratify=yh)
Xbal, ybal = SMOTE(random_state=42).fit_resample(Xtr, ytr)     # TRAIN only
sc = MinMaxScaler().fit(Xbal)
Xtr_s, Xte_s = sc.transform(Xbal), sc.transform(Xte)
ks = list(range(1, 20))
tr_a, te_a, cv_a = [], [], []
for k in ks:
    kn = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, ybal)
    tr_a.append(kn.score(Xtr_s, ybal)); te_a.append(kn.score(Xte_s, yte))
    cv_a.append(cross_val_score(make_pipeline(MinMaxScaler(),
        KNeighborsClassifier(n_neighbors=k)), Xbal, ybal, cv=5).mean())

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(ks, tr_a, "o-", color=BLUE, label="train accuracy")
ax.plot(ks, te_a, "s-", color=RED, label="TEST accuracy (must not be used to choose k)")
ax.plot(ks, cv_a, "^-", color=GREEN, label="5-fold CV on train (the honest chooser)")
k_test, k_cv = ks[int(np.argmax(te_a))], ks[int(np.argmax(cv_a))]
ax.axvline(k_test, color=RED, ls="--", lw=1.2)
ax.axvline(k_cv, color=GREEN, ls="--", lw=1.2)
ax.annotate(f"peeking picks k={k_test}", (k_test, max(te_a)), textcoords="offset points",
            xytext=(8, 18), color=RED, weight="bold", fontsize=9.5)
ax.annotate(f"CV picks k={k_cv}", (k_cv, max(cv_a)), textcoords="offset points",
            xytext=(-30, -34), color=GREEN, weight="bold", fontsize=9.5)
ax.set_xticks(ks); ax.set_xlabel("k (n_neighbors)"); ax.set_ylabel("accuracy")
ax.set_title("Choosing k: the test set is not allowed to vote")
ax.legend(fontsize=9); ax.grid(alpha=.3)
save(fig, "s8-knn-tuning.png")

# ------------------------------------------- 8. heart failure: tree depth
ha, hb, hc, hd = train_test_split(Xh, yh, test_size=.2, random_state=42, stratify=yh)
depths = [1, 2, 3, 4, 5, 6, 8, 10, 15]
h_tr, h_te = [], []
for dep in depths:
    m = DecisionTreeClassifier(max_depth=dep, random_state=42).fit(ha, hc)
    h_tr.append(accuracy_score(hc, m.predict(ha)))
    h_te.append(accuracy_score(hd, m.predict(hb)))
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(depths, h_tr, "o-", color=BLUE, label="Training Accuracy")
ax.plot(depths, h_te, "s-", color=ORANGE, label="Testing Accuracy")
ax.fill_between(depths, h_te, h_tr, color=RED, alpha=.12, label="the gap")
ax.annotate("best test score is at depth 1 —\nthe 'underfitting' model wins",
            (1, h_te[0]), textcoords="offset points", xytext=(60, -46),
            weight="bold", fontsize=9.5, arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy")
ax.set_title("Underfitting vs Good Fit vs Overfitting — heart failure")
ax.legend(); ax.grid(alpha=.3)
save(fig, "s8-heart-tree-depth.png")

# ----------------------------------- 9. what four impossible rows cost
kf5 = KFold(5, shuffle=True, random_state=42)
configs = [("underfit\ndepth 1", ["vehicle_age"], dict(max_depth=1)),
           ("good fit\ndepth 5", ["vehicle_age", "km_driven", "engine", "max_power"],
            dict(max_depth=5, min_samples_leaf=10)),
           ("overfit\nno limit", FEATS, dict(max_depth=None, min_samples_leaf=1)),
           ("regularised\ndepth 10", FEATS, dict(max_depth=10, min_samples_leaf=10))]
raw_cv, clean_cv, raw_err, clean_err = [], [], [], []
for _, feats, kw in configs:
    for df, means, errs in [(cars, raw_cv, raw_err), (cars_clean, clean_cv, clean_err)]:
        r = cross_validate(DecisionTreeRegressor(random_state=42, **kw), df[feats],
                           df["selling_price"], cv=kf5, scoring="r2", n_jobs=-1)
        means.append(r["test_score"].mean()); errs.append(r["test_score"].std())
fig, ax = plt.subplots(figsize=(9.5, 4.8))
xx = np.arange(len(configs)); w = .36
ax.bar(xx - w/2, raw_cv, w, yerr=raw_err, capsize=5, color=RED, alpha=.85,
       label="raw — 2 cars with seats=0, 2 driven over 1,000,000 km")
ax.bar(xx + w/2, clean_cv, w, yerr=clean_err, capsize=5, color=GREEN, alpha=.85,
       label="after removing those 4 rows")
ax.set_xticks(xx); ax.set_xticklabels([c[0] for c in configs])
ax.set_ylabel("5-fold CV R²"); ax.set_ylim(0, 1.05)
ax.set_title("Four impossible rows out of 15,244 — and what they were costing")
ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
save(fig, "s8-impossible-rows-cost.png")

print("\nAll Session 8 plots generated.")
