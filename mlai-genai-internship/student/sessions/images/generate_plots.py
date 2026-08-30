"""Generate Session 3's plots from the exact code in the guide."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import pathlib

OUT = pathlib.Path(__file__).parent
dataset_url = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/prepreprocessing/pre_data.csv"

# --- reproduce the walkthrough exactly up to the box-plot step
df = pd.read_csv(dataset_url)
df2 = df.copy()
df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Salary'] = df2['Salary'].fillna(df2['Salary'].mean())
df2['Country'] = df2['Country'].fillna(df2['Country'].mode()[0])
df2 = df2.drop_duplicates()

q1, q3 = np.percentile(df2['Salary'], [25, 75])
iqr = q3 - q1
lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
clean = df2[(df2['Salary'] >= lower_bound) & (df2['Salary'] <= upper_bound)]


def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)

# 1. Age box plot - no outliers
fig, ax = plt.subplots(figsize=(6, 3.2))
sns.boxplot(x=df2['Age'], ax=ax, color="#7fb3d5")
ax.set_title("Box plot of Age — no outliers", fontsize=11)
ax.set_xlabel("Age")
save(fig, "s3-boxplot-age.png")

# 2. Salary box plot - the box squashed by the outlier
fig, ax = plt.subplots(figsize=(6, 3.2))
sns.boxplot(x=df2['Salary'], ax=ax, color="#f5b7b1")
ax.set_title("Box plot of Salary — two values flagged", fontsize=11)
ax.set_xlabel("Salary")
save(fig, "s3-boxplot-salary.png")

# 3. Salary after removing the outliers, for comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
sns.boxplot(x=df2['Salary'], ax=axes[0], color="#f5b7b1")
axes[0].set_title("Before — the box is squashed flat", fontsize=11)
axes[0].set_xlabel("Salary")
sns.boxplot(x=clean['Salary'], ax=axes[1], color="#a9dfbf")
axes[1].set_title("After removal — the real spread is visible", fontsize=11)
axes[1].set_xlabel("Salary")
fig.tight_layout()
save(fig, "s3-boxplot-salary-before-after.png")

# 4. How to read a box plot - an annotated diagram on the real Age data
age = clean['Age']
q1a, med, q3a = np.percentile(age, [25, 50, 75])
iqra = q3a - q1a
lo_w = age[age >= q1a - 1.5 * iqra].min()
hi_w = age[age <= q3a + 1.5 * iqra].max()

fig, ax = plt.subplots(figsize=(9, 3.8))
sns.boxplot(x=age, ax=ax, color="#7fb3d5", width=.38)

# Labels above the box, clear of it
for x, label in [(q1a, "Q1  (25%)"), (med, "median"), (q3a, "Q3  (75%)")]:
    ax.annotate(label, xy=(x, -.19), xytext=(x, -.42), ha="center", fontsize=9.5,
                arrowprops=dict(arrowstyle="->", lw=1.1))

for x, label in [(lo_w, "lower whisker"), (hi_w, "upper whisker")]:
    ax.annotate(label, xy=(x, 0), xytext=(x, -.42), ha="center", fontsize=8.5,
                color="dimgrey", arrowprops=dict(arrowstyle="->", lw=.9, color="dimgrey"))

# IQR span below the box, on its own line
ax.annotate("", xy=(q1a, .30), xytext=(q3a, .30),
            arrowprops=dict(arrowstyle="<->", lw=1.4, color="crimson"))
ax.text((q1a + q3a) / 2, .38, "IQR — the middle 50% of the values",
        ha="center", va="top", fontsize=9.5, color="crimson")

ax.set_title("How to read a box plot", fontsize=12, pad=14)
ax.set_xlabel("Age")
ax.set_xlim(age.min() - 4.5, age.max() + 4.5)   # room for the whisker labels
ax.set_ylim(.52, -.58)                          # inverted so "above" is above
ax.set_yticks([])
save(fig, "s3-boxplot-explained.png")

# 5. Scaling: the same data before and after
from sklearn.preprocessing import MinMaxScaler, StandardScaler
vals = clean[['Age', 'Salary']]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, data, title in [
    (axes[0], vals, "Original — wildly different ranges"),
    (axes[1], pd.DataFrame(MinMaxScaler().fit_transform(vals), columns=vals.columns),
     "MinMaxScaler — everything in 0 to 1"),
    (axes[2], pd.DataFrame(StandardScaler().fit_transform(vals), columns=vals.columns),
     "StandardScaler — centred on 0"),
]:
    ax.boxplot([data['Age'], data['Salary']], labels=['Age', 'Salary'])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.3, axis="y")
fig.tight_layout()
save(fig, "s3-scaling-comparison.png")
print("\nAll plots generated.")
