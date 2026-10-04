"""Paper figures from the runs' metrics.jsonl (re-run after new runs: python paper/make_figures.py).

Writes PDF (vector, for LaTeX) + PNG (preview) into paper/figures/. Every number plotted comes straight from
runs/<run>/metrics.jsonl or, for the TrayPack3 bar chart, from the tables in QC_EXPERIMENT_LOG.md (cited inline).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# validated categorical slots 1-3 (blue, orange, aqua; all-pairs CVD-safe); aqua is < 3:1 on white,
# so every series also gets its own marker + line style and a direct label
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(run):
    with open(os.path.join(ROOT, "runs", run, "metrics.jsonl")) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def series(rows, key, offset=0):
    pts = [(r["step"] - offset, r[key]) for r in rows if key in r and r["step"] >= offset]
    return [p[0] / 1e6 for p in pts], [p[1] for p in pts]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---- Fig 1: Unstack3 success vs online env steps --------------------------------------------------
# QC runs log step = 200k offline gradient steps + online env steps; x = online env steps for all three.
qc_mean, qc_min, sq = load("unstack3_500"), load("unstack3_500_qmin"), load("unstack3_squint_g09")
fig, ax = plt.subplots(figsize=(3.4, 2.2))
curves = [
    (qc_min, 200_000, "QC-FQL, min target (64 eps)", BLUE, "-", "o"),
    (qc_mean, 200_000, "QC-FQL, mean target (16 eps)", ORANGE, "--", "s"),
    (sq, 0, "Squint SAC, no demos (64 eps, stopped at 1M)", AQUA, ":", "^"),
]
for rows, off, label, c, ls, mk in curves:
    x, y = series(rows, "eval/success_at_end", off)
    ax.plot(x, y, color=c, ls=ls, marker=mk, ms=3.2, label=label)
ax.set_xlabel("online environment steps (M)")
ax.set_ylabel("success rate (end of episode)")
ax.set_ylim(-0.03, 1.0)
ax.set_xlim(0, 2.05)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
save(fig, "unstack3_success")

# ---- Fig 2: Unstack3 critic overestimation: online q_max vs the true ceiling --------------------
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for rows, label, c, ls, mk in ((qc_mean, "mean target", ORANGE, "--", "s"), (qc_min, "min target", BLUE, "-", "o")):
    x, y = series(rows, "train_rl/q_max", 200_000)
    ax.plot(x, y, color=c, ls=ls, marker=mk, ms=3.2, label=label)
ax.axhline(100, color=INK2, lw=0.9)
ax.text(1.02, 100.5, r"max. achievable $Q = 1/(1-\gamma) = 100$", color=INK2, fontsize=6.5, va="bottom")
ax.set_xlabel("online environment steps (M)")
ax.set_ylabel(r"critic $Q_{\max}$ (training batch)")
ax.set_xlim(0, 2.05)
ax.legend(loc="lower right", frameon=False)
save(fig, "unstack3_qmax")

# ---- Fig 3: TrayPack3, clean vs recovery demos (end of training, 16 eval episodes) -------------
# QC_EXPERIMENT_LOG.md, `tp3_recovery` final, observation 17: tp3_r2 (100 clean demos, 256x3) vs
# tp3_recovery (300 recovery demos, 512x4). Note: 3 factors changed between them (see log obs. 6).
labels = ["≥1 cube", "≥2 cubes", "3 cubes"]
clean, recovery = [0.19, 0.00, 0.00], [0.88, 0.38, 0.06]
fig, ax = plt.subplots(figsize=(3.4, 2.0))
w = 0.36
xs = range(len(labels))
b1 = ax.bar([i - w / 2 - 0.01 for i in xs], clean, w, color=ORANGE, label="clean (tp3_r2)")
b2 = ax.bar([i + w / 2 + 0.01 for i in xs], recovery, w, color=BLUE, label="recovery (tp3_recovery)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(list(xs), labels)
ax.set_ylabel("fraction of eval episodes")
ax.set_ylim(0, 1.0)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", frameon=False)
save(fig, "traypack3_recovery_demos")

# ---- Fig 4: TrayPack3 tp3_800_h400 partial progress vs full success ----------------------------
tp = load("tp3_800_h400")
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for key, label, c, ls, mk in (("eval/num_correct_ge1_once", "≥1 cube", BLUE, "-", "o"),
                              ("eval/num_correct_ge2_once", "≥2 cubes", ORANGE, "--", "s"),
                              ("eval/success_at_end", "full success", AQUA, ":", "^")):
    x, y = series(tp, key, 300_000)
    ax.plot(x, y, color=c, ls=ls, marker=mk, ms=2.8, label=label)
ax.set_xlabel("online environment steps (M)")
ax.set_ylabel("fraction of eval episodes (64)")
ax.set_ylim(-0.02, 1.0)
ax.set_xlim(0, 2.05)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
save(fig, "traypack3_h400_progress")
