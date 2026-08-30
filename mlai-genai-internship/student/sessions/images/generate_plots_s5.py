"""Generate the Session 5 (regression) and 5B (classification) figures.

Every figure is the output of code that appears in the guides, run against
the real datasets - so what a student sees on the page is what they get.
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, pathlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

OUT = pathlib.Path(__file__).parent
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  wrote", name)

# ======================================================== REGRESSION
sal = pd.read_csv(BASE + "regression/salary_data.csv").dropna()
X, y = sal.drop("Salary", axis=1), sal["Salary"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
m = LinearRegression().fit(Xtr, ytr); pred = m.predict(Xte)

# 1. the trainer's plot - the fitted line over the test data
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(Xte, yte, color="black", s=28, label="Actual data", alpha=.7)
order = np.argsort(Xte.values.ravel())
ax.plot(Xte.values.ravel()[order], pred[order], color="blue", lw=1.5,
        marker="*", markersize=7, label="Regression line")
ax.set_xlabel("Experience (years)"); ax.set_ylabel("Salary")
ax.set_title(f"Linear Regression on Salary Data   (R² = {r2_score(yte, pred):.3f})")
ax.legend(); ax.grid(alpha=.3)
save(fig, "s5-salary-regression-line.png")

# 2. predicted vs actual, and residuals - the two diagnostic plots
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
a1.scatter(yte, pred, alpha=.6, edgecolor="k", linewidth=.4)
lims = [min(yte.min(), pred.min()) - 5000, max(yte.max(), pred.max()) + 5000]
a1.plot(lims, lims, "r--", lw=1.4, label="perfect prediction")
a1.set_xlabel("Actual salary"); a1.set_ylabel("Predicted salary")
a1.set_title("Predicted vs actual"); a1.legend(); a1.grid(alpha=.3)
a1.set_xlim(lims); a1.set_ylim(lims)

resid = yte - pred
a2.scatter(pred, resid, alpha=.6, edgecolor="k", linewidth=.4)
a2.axhline(0, color="red", ls="--", lw=1.4)
a2.set_xlabel("Predicted salary"); a2.set_ylabel("Error (actual − predicted)")
a2.set_title("Residuals — look for a shapeless cloud"); a2.grid(alpha=.3)
fig.tight_layout(); save(fig, "s5-salary-diagnostics.png")

# 3. advertising - each channel against sales
ad = pd.read_csv(BASE + "regression/advertising.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, ch in zip(axes, ["TV", "Radio", "Newspaper"]):
    ax.scatter(ad[ch], ad["Sales"], alpha=.6, edgecolor="k", linewidth=.3)
    ax.set_title(f"{ch}   (r = {ad[ch].corr(ad['Sales']):.3f})")
    ax.set_xlabel(f"{ch} spend"); ax.grid(alpha=.3)
axes[0].set_ylabel("Sales")
fig.tight_layout(); save(fig, "s5-advertising-channels.png")

# 4. the car model - why R2 0.66 is not good enough
car = pd.read_csv(BASE + "regression/cardekho_dataset.csv")
car = car.drop(columns=["Unnamed: 0", "car_name", "model"])
for c in ["brand", "seller_type", "fuel_type", "transmission_type"]:
    car[c] = LabelEncoder().fit_transform(car[c])
Xc, yc = car.drop("selling_price", axis=1), car["selling_price"]
a, b, c_, d_ = train_test_split(Xc, yc, test_size=.2, random_state=42)
pc = LinearRegression().fit(a, c_).predict(b)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
a1.scatter(d_, pc, alpha=.25, s=12, edgecolor="none")
lim = [0, 4_000_000]
a1.plot(lim, lim, "r--", lw=1.4, label="perfect prediction")
a1.set_xlim(lim); a1.set_ylim(lim)
a1.set_xlabel("Actual price"); a1.set_ylabel("Predicted price")
a1.set_title(f"Original target   (R² = {r2_score(d_, pc):.3f})")
a1.legend(); a1.grid(alpha=.3)

yl = np.log1p(yc)
a2_, b2, c2, d2 = train_test_split(Xc, yl, test_size=.2, random_state=42)
pl = LinearRegression().fit(a2_, c2).predict(b2)
a2.scatter(d2, pl, alpha=.25, s=12, edgecolor="none", color="seagreen")
lim2 = [min(d2.min(), pl.min()), max(d2.max(), pl.max())]
a2.plot(lim2, lim2, "r--", lw=1.4)
a2.set_xlabel("Actual log(price)"); a2.set_ylabel("Predicted log(price)")
a2.set_title(f"Log-transformed target   (R² = {r2_score(d2, pl):.3f})")
a2.grid(alpha=.3)
fig.tight_layout(); save(fig, "s5-car-log-transform.png")

# ==================================================== CLASSIFICATION
df = pd.read_csv(BASE + "loan_data_10k.csv").dropna()
NUM = ["person_age", "person_income", "person_emp_exp", "loan_amnt",
       "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]

# 5. box plots of every numeric feature, before outlier removal
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for ax, f in zip(axes.flat, NUM):
    sns.boxplot(x=df[f], ax=ax, color="#7fb3d5")
    ax.set_title(f, fontsize=10); ax.set_xlabel("")
fig.suptitle("Outlier check — every numeric feature", fontsize=13, y=1.01)
fig.tight_layout(); save(fig, "s5b-boxplots-before.png")

for f in NUM:
    q1, q3 = np.percentile(df[f], [25, 75]); iqr = q3 - q1
    df = df[(df[f] >= q1 - 1.5*iqr) & (df[f] <= q3 + 1.5*iqr)]

for c in ["person_gender", "person_education", "person_home_ownership",
          "loan_intent", "previous_loan_defaults_on_file"]:
    df[c] = LabelEncoder().fit_transform(df[c])
df[NUM] = MinMaxScaler().fit_transform(df[NUM])

Xl, yl2 = df.drop("loan_status", axis=1), df["loan_status"]
Xtr, Xte, ytr, yte = train_test_split(Xl, yl2, test_size=.20, random_state=42)

# 6. confusion matrix for logistic regression, annotated
lr = LogisticRegression().fit(Xtr, ytr); p = lr.predict(Xte)
cm = confusion_matrix(yte, p)
fig, ax = plt.subplots(figsize=(6.4, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"], annot_kws={"size": 15})
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j + .5, i + .74, labels[i][j], ha="center", fontsize=11,
                color="crimson", weight="bold")
ax.set_title("Confusion matrix — Logistic Regression", fontsize=12)
save(fig, "s5b-confusion-matrix.png")

# 7. all six models compared
models = [("Logistic\nRegression", LogisticRegression()),
          ("kNN\n(k=5)", KNeighborsClassifier(n_neighbors=5)),
          ("Decision\nTree", DecisionTreeClassifier(random_state=42)),
          ("SVM\n(linear)", SVC(kernel="linear")),
          ("Gaussian\nNB", GaussianNB()),
          ("Random\nForest", RandomForestClassifier(n_estimators=200, random_state=42))]
rows = []
for name, mm in models:
    mm.fit(Xtr, ytr); pp = mm.predict(Xte)
    rows.append((name, accuracy_score(yte, pp), precision_score(yte, pp),
                 recall_score(yte, pp), f1_score(yte, pp)))

names = [r[0] for r in rows]
x = np.arange(len(names)); w = .2
fig, ax = plt.subplots(figsize=(12, 4.8))
for i, (lab, col) in enumerate(zip(["accuracy", "precision", "recall", "f1"],
                                   ["#4c72b0", "#dd8452", "#55a868", "#c44e52"])):
    ax.bar(x + (i - 1.5) * w, [r[i + 1] for r in rows], w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(.6, 1.0); ax.set_ylabel("score")
ax.set_title("Six classifiers on the same data — no single winner on every metric")
ax.legend(ncol=4); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); save(fig, "s5b-model-comparison.png")

# 8. the precision/recall trade-off, made visual
gnb = GaussianNB().fit(Xtr, ytr); pg = gnb.predict(Xte)
dt = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr); pd_ = dt.predict(Xte)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, pp, title in [(a1, pg, "Gaussian Naive Bayes"), (a2, pd_, "Decision Tree")]:
    c = confusion_matrix(yte, pp)
    sns.heatmap(c, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
    ax.set_title(f"{title}\nrecall {recall_score(yte, pp):.3f}   "
                 f"precision {precision_score(yte, pp):.3f}", fontsize=11)
fig.suptitle("Same data, opposite personalities", fontsize=13, y=1.03)
fig.tight_layout(); save(fig, "s5b-precision-recall-tradeoff.png")

print("\nAll Session 5 / 5B plots generated.")
