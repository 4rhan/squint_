"""Plot SQUINT pilot learning curves from local metrics.csv.

Usage: python -m examples.plot_pilot_curves runs/pilot16_SO101StackCube-v1__s1 --out pilot_stackcube.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        try:
            step = int(float(r["step"]))
        except Exception:
            continue
        for k, v in r.items():
            if k == "step" or v == "":
                continue
            try:
                out.setdefault(k, []).append((step, float(v)))
            except Exception:
                pass
    return {k: (list(zip(*v))[0], list(zip(*v))[1]) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = load(os.path.join(args.run_dir, "metrics.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(os.path.basename(args.run_dir.rstrip("/")))
    specs = [("eval/success_once", "eval/success_at_end"),
             ("eval/return", "train/return"),
             ("eval/max_num_correct", "eval/num_correct"),
             ("time/sps", "time/gpu_mem_gb")]
    for ax, (a, b) in zip(axes.flat, specs):
        for key in (a, b):
            if key in d:
                x, y = d[key]
                ax.plot(x, y, label=key)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    out = args.out or os.path.join(args.run_dir, "curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"wrote {out}; keys: {sorted(d)[:12]}...")


if __name__ == "__main__":
    main()
