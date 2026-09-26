"""Plot every metric a train_squint_qc run logged to runs/<exp_name>/metrics.jsonl.

One PNG per metric group, one panel per metric, written to runs/<exp_name>/plots/:
  eval.png        success, return, num_correct / stage flags
  eval_rew.png    eval return split by reward term (raw units summed over an episode)
  train_rew.png   the same split for training episodes
  train_rl.png    critic/actor losses, Q values during online RL
  offline.png     losses/Q during offline pretraining
  train.png       training-episode stats from ManiSkill (return, success, ...)
A dashed line marks the end of offline pretraining. Pass several runs to overlay them.

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories (runs/<exp_name>)")
    p.add_argument("--out", default=None, help="output dir (default: <first run>/plots)")
    args = p.parse_args()
    out = args.out or os.path.join(args.runs[0], "plots")
    os.makedirs(out, exist_ok=True)
    data = {os.path.basename(os.path.normpath(r)): load(r) for r in args.runs}

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
        fig.savefig(path, dpi=100); plt.close(fig)
        print(f"wrote {path}  ({len(keys)} metrics)")


if __name__ == "__main__":
    main()
