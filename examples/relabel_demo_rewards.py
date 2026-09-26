"""Recompute the rewards stored in a demo file under the env's current reward function.

The demo HDF5 stores the rewards seen at collection time and the critic trains on them, so after a
reward change the demos must be relabelled. Each demo is replayed from its seed on the CPU sim (where
replay is exact: same states, same success), the new normalized_dense reward is recorded, and a copy of
the file is written with only `rewards` replaced. The replay is checked against the recorded joint
state, and a demo that no longer ends in success is reported.

With --plot N, the per-step reward terms (info['rew_<term>'], raw units) of the first N demos are
plotted to <out>.terms.png, which shows what a correct placement looks like to the critic.

    python -m examples.relabel_demo_rewards demos/qc/SO101TrayPack3-v2.h5 demos/qc/SO101TrayPack3-v2_r2.h5 --reward_version 2 --plot 3
"""
import argparse
import json
from collections import defaultdict

import gymnasium as gym
import h5py
import numpy as np
import torch

import envs  # noqa: F401  registers tasks


def replay(env, g, collect_terms):
    base = env.unwrapped
    env.reset(seed=int(g.attrs["seed"]))
    rec = g["obs/state"][()]
    rewards, terms, worst, success = [], defaultdict(list), 0.0, False
    for t, a in enumerate(g["actions"][()]):
        _, r, _, _, info = env.step(torch.as_tensor(a)[None])
        rewards.append(float(torch.as_tensor(r).reshape(-1)[0]))
        q = base.agent.robot.get_qpos()[0].cpu().numpy()
        tgt = base.agent.controller._target_qpos[0].cpu().numpy()
        worst = max(worst, float(np.abs(q - rec[t + 1, :6]).max()), float(np.abs(tgt - rec[t + 1, 6:12]).max()))
        success = bool(torch.as_tensor(info["success"]).reshape(-1)[0])
        if collect_terms:
            for k, v in info.items():
                if k.startswith("rew_"):
                    terms[k[4:]].append(float(torch.as_tensor(v).reshape(-1)[0]))
    return np.asarray(rewards, dtype=np.float32), terms, worst, success


def plot_terms(all_terms, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(all_terms)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), squeeze=False)
    for ax, (name, terms) in zip(axes[:, 0], all_terms):
        for k, v in terms.items():
            if k == "total":
                ax.plot(v, color="black", lw=2, label="total")
            elif np.abs(v).max() > 0:
                ax.plot(v, lw=1.2, label=k)
        ax.set_title(f"{name}: raw reward terms per step (max 14)")
        ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(ncol=6, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst")
    p.add_argument("--reward_version", type=int, default=None, help="TrayPack reward version (default: env default)")
    p.add_argument("--plot", type=int, default=0, help="plot per-step reward terms of the first N demos")
    p.add_argument("--state_tol", type=float, default=1e-3)
    args = p.parse_args()

    with h5py.File(args.src, "r") as src:
        meta = json.loads(src.attrs["meta"])
        kw = dict(obs_mode="state", sim_backend="cpu", reward_mode="normalized_dense", reconfiguration_freq=1)
        if meta.get("domain_randomization"):
            kw["domain_randomization"] = True
        if args.reward_version is not None:
            kw["reward_version"] = args.reward_version
        env = gym.make(meta["env_id"], num_envs=1, **kw)
        names = sorted(src.keys(), key=lambda k: int(k.split("_")[-1]))
        bad, plotted, old_ret, new_ret, worst_drop = [], [], [], [], []
        with h5py.File(args.dst, "w") as dst:
            dst.attrs["meta"] = json.dumps(dict(meta, reward_relabelled_from=args.src,
                                                reward_version=getattr(env.unwrapped, "reward_version", None)))
            for i, name in enumerate(names):
                rew, terms, worst, success = replay(env, src[name], True)
                # worst reward drop: total vs its running max over the previous 10 steps, ignoring
                # steps where the table penalty fires (that is the solver's own contact, not a placement)
                tot, tab = np.asarray(terms["total"]), np.asarray(terms.get("table_pen", np.zeros(len(rew))))
                drops = [tot[t] - tot[max(0, t - 10):t].max() for t in range(1, len(tot)) if tab[t] == 0 and tab[t - 1] == 0]
                worst_drop.append(min(drops) if drops else 0.0)
                if worst > args.state_tol or not success:
                    bad.append(f"{name} (seed {src[name].attrs['seed']}): state err {worst:.4f}, success {success}")
                src.copy(src[name], dst, name=name)
                del dst[name]["rewards"]
                dst[name].create_dataset("rewards", data=rew)
                old_ret.append(float(src[name]["rewards"][()].sum())); new_ret.append(float(rew.sum()))
                if i < args.plot:
                    plotted.append((f"{name} seed {src[name].attrs['seed']}", {k: np.asarray(v) for k, v in terms.items()}))
                print(f"\r{i + 1}/{len(names)}", end="", flush=True)
        env.close()
    print(f"\nrelabelled {len(names)} demos -> {args.dst}")
    print(f"demo return (normalized): old mean {np.mean(old_ret):.1f} -> new mean {np.mean(new_ret):.1f}")
    wd = np.asarray(worst_drop)
    print(f"worst reward drop per demo (raw, excl. table contact): mean {wd.mean():.2f}, "
          f"median {np.median(wd):.2f}, worst {wd.min():.2f}; demos with a drop > 1.0: {(wd < -1.0).sum()}/{len(wd)}")
    print("all replays reproduce the recording and succeed" if not bad else "PROBLEMS:\n  " + "\n  ".join(bad))
    if plotted:
        plot_terms(plotted, args.dst + ".terms.png")


if __name__ == "__main__":
    main()
