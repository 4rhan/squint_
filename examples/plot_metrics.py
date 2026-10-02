"""Plot every metric a train_squint_qc run logged to runs/<exp_name>/metrics.jsonl.

Two dashboards plus one PNG per metric group (one panel per metric), written to runs/<exp_name>/plots/:
  summary.png     success, return, stage progress, critic loss, actor losses, Q (offline + online on one axis)
  rewards.png     every reward term summed over an episode, eval and training, and the total return
  eval.png        success, return, num_correct / stage flags
  eval_rew.png    eval return split by reward term (raw units summed over an episode)
  train_rew.png   the same split for training episodes
  train_rl.png    critic/actor losses, Q values during online RL
  offline.png     losses/Q during offline pretraining
  train.png       training-episode stats from ManiSkill (return, success, ...)
A dashed line marks the end of offline pretraining. Pass several runs to overlay them.
train_squint_qc.py re-runs this in the background while training (--plot_every_sec), so the PNGs stay current.

    python -m examples.plot_metrics runs/tp3_r2
    python -m examples.plot_metrics runs/tp3_full_envfix runs/tp3_r2 --out runs/compare_plots
"""
import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir):
    series = defaultdict(lambda: ([], []))
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            step = row.pop("step")
            for k, v in row.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    series[k][0].append(step); series[k][1].append(v)
    return series


def _save(fig, path):
    """Write via a temp file + rename, so a viewer never sees a half-written PNG while training updates it."""
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=100); plt.close(fig)
    os.replace(tmp, path)


def _offline_end(series):
    off = series.get("offline/critic_loss")
    return max(off[0]) if off and off[0] else None


def _panel(ax, data, keys, title, logy=False, legend_keys=True):
    """Plot `keys` (each may exist in any run) on one axis. Lines are labelled by key, and by run too when
    several runs are overlaid."""
    multi = len(data) > 1
    styles = ["-", "--", ":", "-."]
    colors = plt.cm.tab20.colors if len(keys) > 10 else plt.cm.tab10.colors
    drew = False
    for ri, (run, series) in enumerate(data.items()):
        for ki, key in enumerate(keys):
            if key not in series:
                continue
            x, y = series[key]
            if logy:
                x, y = zip(*[(a, b) for a, b in zip(x, y) if b > 0]) if any(b > 0 for b in y) else ([], [])
            if not x:
                continue
            # drop the group prefix only when every key in the panel shares it (eval/x vs train/x stay apart)
            label = (key.split("/", 1)[-1] if len({k.split("/")[0] for k in keys}) == 1 else key) if legend_keys else None
            if multi:
                label = f"{run}: {label}" if label else run
            ax.plot(x, y, styles[ri % len(styles)], color=colors[ki % len(colors)], lw=1.2,
                    marker="o" if len(x) < 40 else None, ms=2.5, label=label)
            drew = True
        end = _offline_end(series)
        if end is not None:
            ax.axvline(end, color="grey", ls="--", lw=0.8)
    if logy and drew:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    if drew:
        ax.legend(fontsize=6, ncol=2 if len(keys) > 6 else 1, loc="best")
    else:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes, color="grey")
    return drew


def _keys(data, prefix, exclude=()):
    return sorted({k for s in data.values() for k in s if k.startswith(prefix) and k not in exclude})


def plot_dashboards(data, out):
    both = lambda name: [f"offline/{name}", f"train_rl/{name}"]  # offline then online, one curve each
    stage = [k for k in _keys(data, "eval/") if k.endswith("_once") and k != "eval/success_once"]
    panels = [
        ("success (eval: 16 eps; train: rollouts)",
         ["eval/success_at_end", "eval/success_once", "train/success_at_end", "train/success_once"], False),
        ("return per episode", ["eval/return", "train/return"], False),
        ("eval: stage reached at least once", stage, False),
        ("critic loss", both("critic_loss"), True),
        ("actor losses", both("bc_flow_loss") + both("distill_loss") + both("q_loss"), True),
        ("Q values", both("q_mean") + both("q_min") + both("q_max"), False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (title, keys, logy) in zip(axes.flat, panels):
        _panel(ax, data, keys, title, logy=logy)
    fig.suptitle(f"summary  ({', '.join(data)})  - dashed line: end of offline pretraining", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out, "summary.png"))

    ev, tr = _keys(data, "eval_rew/", ("eval_rew/total",)), _keys(data, "train_rew/", ("train_rew/total",))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    _panel(axes[0], data, ev, "eval: reward terms (raw units, summed over an episode)")
    _panel(axes[1], data, tr, "train: reward terms (raw units, summed over an episode, mean over envs)")
    _panel(axes[2], data, ["eval_rew/total", "train_rew/total", "eval/return", "train/return"],
           "total (raw) and return (normalized env reward)")
    fig.suptitle(f"rewards  ({', '.join(data)})", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out, "rewards.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories (runs/<exp_name>)")
    p.add_argument("--out", default=None, help="output dir (default: <first run>/plots)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    out = args.out or os.path.join(args.runs[0], "plots")
    os.makedirs(out, exist_ok=True)
    data = {os.path.basename(os.path.normpath(r)): load(r) for r in args.runs}
    plot_dashboards(data, out)

    groups = defaultdict(set)
    for series in data.values():
        for k in series:
            groups[k.split("/")[0] if "/" in k else "other"].add(k)
    for group, keys in sorted(groups.items()):
        keys = sorted(keys)
        cols = min(4, len(keys)); rows = math.ceil(len(keys) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), squeeze=False)
        for ax, key in zip(axes.flat, keys):
            for run, series in data.items():
                if key in series:
                    x, y = series[key]
                    ax.plot(x, y, marker="o" if len(x) < 40 else None, ms=3, lw=1.3, label=run)
                off = series.get("offline/critic_loss")
                if off and off[0]:
                    ax.axvline(max(off[0]), color="grey", ls="--", lw=0.8)
            ax.set_title(key, fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        if len(data) > 1:
            axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{group}  ({', '.join(data)})", fontsize=11)
        fig.tight_layout()
        path = os.path.join(out, f"{group}.png")
        _save(fig, path)
        if not args.quiet:
            print(f"wrote {path}  ({len(keys)} metrics)")


if __name__ == "__main__":
    main()
