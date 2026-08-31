"""Generate the figures embedded in Session 9 (neural networks).

Run from this directory:
    python generate_plots_s9.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse

OUT = pathlib.Path(__file__).parent
BLUE, ORANGE, GREEN, RED, PURPLE = "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"

def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)

def sig(z): return 1 / (1 + np.exp(-z))

# ------------------------------------------------ 1. biological vs artificial
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

# biological
a1.set_xlim(0, 10); a1.set_ylim(0, 6); a1.axis("off")
a1.add_patch(Circle((3.2, 3), .85, fc=BLUE, alpha=.35, ec=BLUE, lw=2))
a1.text(3.2, 3, "cell\nbody", ha="center", va="center", fontsize=9.5, weight="bold")
for dy, lab in [(1.5, ""), (0.75, ""), (0, ""), (-0.75, ""), (-1.5, "")]:
    a1.plot([1.1, 2.4], [3 + dy * 1.05, 3 + dy * .55], color=BLUE, lw=2)
a1.text(1.0, 5.0, "dendrites\n(signals in)", ha="center", fontsize=9.5, color=BLUE)
a1.plot([4.05, 7.4], [3, 3], color=BLUE, lw=3)
a1.text(5.7, 3.35, "axon", ha="center", fontsize=9.5, color=BLUE)
for dy in (-.7, 0, .7):
    a1.plot([7.4, 8.6], [3, 3 + dy], color=BLUE, lw=2)
a1.text(9.0, 3, "synapses\n(signal out)", ha="center", va="center", fontsize=9.5, color=BLUE)
a1.set_title("Biological neuron", fontsize=13, weight="bold")

# artificial
a2.set_xlim(0, 10); a2.set_ylim(0, 6); a2.axis("off")
for i, (yy, lab, w) in enumerate([(4.6, "x₁", "w₁"), (3.0, "x₂", "w₂"), (1.4, "x₃", "w₃")]):
    a2.add_patch(Circle((1.3, yy), .42, fc="white", ec=ORANGE, lw=2))
    a2.text(1.3, yy, lab, ha="center", va="center", fontsize=11)
    a2.annotate("", (3.05, 3), (1.72, yy), arrowprops=dict(arrowstyle="->", lw=1.8, color=ORANGE))
    a2.text((1.72 + 3.05) / 2, (yy + 3) / 2 + .18, w, fontsize=10, color=ORANGE, weight="bold")
a2.add_patch(Ellipse((4.15, 3), 2.1, 1.7, fc=ORANGE, alpha=.25, ec=ORANGE, lw=2))
a2.text(4.15, 3.28, "Σ  wᵢxᵢ + b", ha="center", fontsize=11, weight="bold")
a2.text(4.15, 2.62, "then f( · )", ha="center", fontsize=10)
a2.annotate("", (7.6, 3), (5.25, 3), arrowprops=dict(arrowstyle="->", lw=2.4, color=ORANGE))
a2.add_patch(Circle((8.1, 3), .45, fc="white", ec=ORANGE, lw=2))
a2.text(8.1, 3, "a", ha="center", va="center", fontsize=11)
a2.text(9.3, 3, "output", ha="center", va="center", fontsize=9.5, color=ORANGE)
a2.text(4.15, 1.15, "bias b: how easily it fires", ha="center", fontsize=9, style="italic")
a2.set_title("Artificial neuron (perceptron)", fontsize=13, weight="bold")

fig.suptitle("One is an analogy for the other — not a copy of it", fontsize=12.5, y=1.0)
fig.tight_layout(); save(fig, "s9-neuron-mapping.png")

# --------------------------------------------------------- 2. activations
z = np.linspace(-6, 6, 400)
fns = [("Sigmoid  1/(1+e⁻ᶻ)", sig(z), "squashes to (0, 1) — output layer, binary"),
       ("Tanh", np.tanh(z), "squashes to (−1, 1) — centred on zero"),
       ("ReLU  max(0, z)", np.maximum(0, z), "the hidden-layer default"),
       ("Leaky ReLU", np.where(z > 0, z, .05 * z), "never fully dies")]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, (name, vals, note), c in zip(axes, fns, [BLUE, PURPLE, GREEN, ORANGE]):
    ax.plot(z, vals, color=c, lw=2.6)
    ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel(note, fontsize=8.5)
    ax.grid(alpha=.3); ax.set_ylim(-1.4, 3)
fig.suptitle("Without one of these, a hundred layers collapse into one straight line",
             fontsize=12.5, y=1.04)
fig.tight_layout(); save(fig, "s9-activations.png")

# ------------------------------------- 3. the play network, with real numbers
def draw_net(ax, values=True):
    nodes = {"x1": (1, 4.2), "x2": (1, 1.8), "h1": (4, 4.2), "h2": (4, 1.8), "o": (7, 3)}
    labels = {"x1": "sunny\nx₁ = 1", "x2": "feel good\nx₂ = 0",
              "h1": "h₁", "h2": "h₂", "o": "play?"}
    for k, (xx, yy) in nodes.items():
        c = BLUE if k.startswith("x") else (GREEN if k.startswith("h") else ORANGE)
        ax.add_patch(Circle((xx, yy), .48, fc=c, alpha=.3, ec=c, lw=2.2))
        ax.text(xx, yy, labels[k], ha="center", va="center", fontsize=9, weight="bold")
    # (from, to, label, position along the edge) - crossing edges get
    # different positions so their labels do not land on the same point
    edges = [("x1", "h1", "0.8", .50), ("x1", "h2", "0.6", .30),
             ("x2", "h1", "0.4", .72), ("x2", "h2", "0.9", .50),
             ("h1", "o", "0.7", .50), ("h2", "o", "-0.5", .50)]
    for a, b, w, pos in edges:
        (x0, y0), (x1_, y1_) = nodes[a], nodes[b]
        ax.annotate("", (x1_ - .48, y1_), (x0 + .48, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="grey"))
        ax.text(x0 + pos * (x1_ - x0), y0 + pos * (y1_ - y0) + .18, w,
                fontsize=9, color="dimgrey", ha="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.85))
    if values:
        ax.text(4, 5.15, "z=0.90  a=0.7109", ha="center", fontsize=9, color=GREEN, weight="bold")
        ax.text(4, .85, "z=0.40  a=0.5987", ha="center", fontsize=9, color=GREEN, weight="bold")
        ax.text(7, 3.95, "z=0.2483\na=0.5618", ha="center", fontsize=9, color=ORANGE, weight="bold")
    ax.set_xlim(0, 8.6); ax.set_ylim(0, 6); ax.axis("off")

fig, ax = plt.subplots(figsize=(9.5, 5))
draw_net(ax)
ax.set_title("Should I play? — 2 inputs, 2 hidden, 1 output\n"
             "biases b₁=0.1  b₂=−0.2  b₃=0.05", fontsize=12, weight="bold")
fig.tight_layout(); save(fig, "s9-play-network.png")

# ----------------------------------------------------- 4. forward / backward
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8))
draw_net(a1, values=True)
a1.annotate("", (7.6, .6), (.6, .6), arrowprops=dict(arrowstyle="->", lw=3, color=BLUE))
a1.text(4, .15, "FORWARD — compute the prediction", ha="center", fontsize=11,
        weight="bold", color=BLUE)
draw_net(a2, values=False)
a2.text(4, 5.15, "δ₁ = +0.0199", ha="center", fontsize=9.5, color=RED, weight="bold")
a2.text(4, .85, "δ₂ = −0.0166", ha="center", fontsize=9.5, color=RED, weight="bold")
a2.text(7, 3.95, "δₒ = 0.1383", ha="center", fontsize=9.5, color=RED, weight="bold")
a2.annotate("", (.6, .6), (7.6, .6), arrowprops=dict(arrowstyle="->", lw=3, color=RED))
a2.text(4, .15, "BACKWARD — spread the blame", ha="center", fontsize=11,
        weight="bold", color=RED)
fig.suptitle("Backpropagation is one round trip: predict, then assign blame",
             fontsize=13, y=1.0)
fig.tight_layout(); save(fig, "s9-backprop-flow.png")

# ------------------------------------------------------ 5. gradient descent
w = np.linspace(-3, 3, 400)
loss = w ** 2 + .3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, lr, title, col in [(axes[0], .05, "Learning rate too SMALL\ncrawls — many steps wasted", BLUE),
                           (axes[1], .35, "Learning rate JUST RIGHT\nsettles quickly", GREEN),
                           (axes[2], 1.02, "Learning rate too LARGE\novershoots and diverges", RED)]:
    ax.plot(w, loss, color="grey", lw=2)
    p = -2.6
    pts = [p]
    for _ in range(9):
        p = p - lr * 2 * p
        pts.append(p)
    ax.plot(pts, [x ** 2 + .3 for x in pts], "o-", color=col, ms=6, lw=1.4)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("weight"); ax.set_ylabel("loss")
    ax.set_ylim(0, 10); ax.grid(alpha=.3)
fig.suptitle("The learning rate is the size of the step down the hill", fontsize=13, y=1.02)
fig.tight_layout(); save(fig, "s9-learning-rate.png")

# ------------------------------------------------------ 6. training curve
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
targets = np.array([0, 0, 0, 1], dtype=float)
rng = np.random.default_rng(0)
Wa = rng.normal(0, 1, (2, 2)); ba = np.zeros(2)
Wb = rng.normal(0, 1, 2); bb = 0.0
history = []
for epoch in range(4001):
    total = 0
    for xi, ti in zip(data, targets):
        ah = sig(xi @ Wa + ba); ao = sig(ah @ Wb + bb)
        total += .5 * (ao - ti) ** 2
        do = (ao - ti) * ao * (1 - ao)
        dh = do * Wb * ah * (1 - ah)
        Wb -= .5 * do * ah; bb -= .5 * do
        Wa -= .5 * np.outer(xi, dh); ba -= .5 * dh
    history.append(total)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(history, color=BLUE, lw=2)
ax.set_yscale("log")
ax.set_xlabel("epoch (one pass over all four examples)")
ax.set_ylabel("total loss (log scale)")
ax.set_title("4,000 rounds of predict-and-correct on the play problem")
preds = [sig(sig(xi @ Wa + ba) @ Wb + bb) for xi in data]
ax.text(1600, history[10],
        "final predictions\n"
        f"(0,0) → {preds[0]:.4f}   (0,1) → {preds[1]:.4f}\n"
        f"(1,0) → {preds[2]:.4f}   (1,1) → {preds[3]:.4f}",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec=BLUE, alpha=.9))
ax.grid(alpha=.3)
fig.tight_layout(); save(fig, "s9-training-curve.png")

# -------------------------------------------------------- 7. output heads
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
specs = [
    ("BINARY\nwill this patient survive?", ["survive?"], [0.87],
     "1 neuron · sigmoid\nread as a probability\nthreshold at 0.5", BLUE),
    ("MULTICLASS\nwhich iris species?", ["setosa", "versicolor", "virginica"],
     [0.0067, 0.1829, 0.8104],
     "3 neurons · softmax\nprobabilities sum to 1\ntake the largest", GREEN),
    ("REGRESSION\nwhat is this car worth?", ["price"], [559000],
     "1 neuron · NO activation\nthe number itself", ORANGE),
]
for ax, (title, names, vals, note, col) in zip(axes, specs):
    ax.barh(names, [v if max(vals) <= 1 else 1 for v in vals], color=col, alpha=.85)
    for i, v in enumerate(vals):
        txt = f"{v:,.0f}" if v > 1 else f"{v:.4f}"
        ax.text(.02, i, txt, va="center", fontsize=11, weight="bold", color="white")
    ax.set_xlim(0, 1.05); ax.set_xticks([])
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel(note, fontsize=9.5)
fig.suptitle("The same network body — only the output layer changes", fontsize=13, y=1.02)
fig.tight_layout(); save(fig, "s9-output-heads.png")

print("\nAll Session 9 plots generated.")
