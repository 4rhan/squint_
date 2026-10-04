"""Replay demos and measure how good each one is (not only whether it passes).

verify_demos answers "does this demo solve the task?". This answers "is it a good demo to learn
from?": for every trajectory it replays the recorded actions with the recorded seed and records

  len, slack            steps used and steps left before the horizon
  success, t_success    replay ends in success / first success step
  state_err             max proprio divergence from the recording (replay determinism)
  t_obj<k>              first step object k counts as placed (TrayPack/Rearrange: per_object_correct)
  undone                an object was placed and later stopped being placed (knocked out)
  grasps, regrasps      grasp onsets overall / onsets beyond one per object (retries, recoveries)
  drops                 grasps that ended with the object not placed (dropped or released early)
  table, tray           steps the robot touches the table / tray
  ret                   replayed return (env reward mode/version as given)

and writes <out>.csv plus a summary with flags (fail, tight on time, undone, many retries).

Usage:
  python -m examples.demo_quality demos/qc/SO101TrayPack3-recovery800.h5 --workers 8 --out logs/tp3_800_quality
"""
import argparse
import csv
import json
from multiprocessing import Pool

import h5py
import numpy as np


def _measure(job):
    path, names, reward_version = job
    import torch
    torch.set_num_threads(1)
    from examples.collect_new_tasks_demos import make_env

    rows = []
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        env, horizon, *_ = make_env(meta["env_id"], meta.get("render_size", 128),
                                    meta.get("domain_randomization", False),
                                    meta.get("requested_control_mode"), reward_version=reward_version)
        u = env.unwrapped
        cols = slice(6, 12) if meta.get("domain_randomization") else slice(0, 12)
        for name in names:
            g = f[name]
            rec_state = g["obs/state"][()]
            acts = g["actions"][()]
            env.reset(seed=int(g.attrs["seed"]))
            # reconfiguration_freq=1 rebuilds the actors on every reset: fetch handles after it
            objs, tray = list(getattr(u, "cubes", [])), getattr(u, "tray", None)
            n = len(objs)
            t_place = [-1] * n
            prev_placed = np.zeros(n, bool)
            prev_grasp = np.zeros(n, bool)
            grasps, drops, undone, table, tray_t = np.zeros(n, int), 0, 0, 0, 0
            worst, ret, success, t_success = 0.0, 0.0, False, -1
            for t, a in enumerate(acts):
                obs, rew, term, trunc, info = env.step(a)
                s = obs["state"][0].cpu().numpy()
                worst = max(worst, float(np.abs(s[cols] - rec_state[t + 1, cols]).max()))
                ret += float(torch.as_tensor(rew).reshape(-1)[0])
                success = bool(torch.as_tensor(info["success"]).reshape(-1)[0])
                if success and t_success < 0:
                    t_success = t + 1
                per = info.get("per_object_correct")
                placed = (per[0, :n].cpu().numpy() if per is not None else np.zeros(n, bool)).astype(bool)
                grasp = np.array([bool(u.agent.is_grasping(o)[0]) for o in objs], bool)
                for k in range(n):
                    if placed[k] and t_place[k] < 0:
                        t_place[k] = t + 1
                    if grasp[k] and not prev_grasp[k]:
                        grasps[k] += 1
                    if prev_grasp[k] and not grasp[k] and not placed[k]:
                        # released or slipped; counted only if it is not seated a few steps later
                        drops += int(not bool(getattr(u, "_in_comp", np.zeros((1, n), bool))[0, k]))
                    if prev_placed[k] and not placed[k] and not grasp[k]:
                        undone += 1
                prev_placed, prev_grasp = placed, grasp
                table += int(bool(u.agent.is_touching(u.table_scene.table)[0]))
                if tray is not None:
                    tray_t += int(bool(u.agent.is_touching(tray)[0]))
            T = len(acts)
            rows.append(dict(traj=name, seed=int(g.attrs["seed"]), len=T, slack=horizon - T,
                             success=int(success), t_success=t_success, state_err=round(worst, 5),
                             **{f"t_obj{k}": t_place[k] for k in range(n)},
                             undone=undone, grasps=int(grasps.sum()), regrasps=int(np.maximum(grasps - 1, 0).sum()),
                             drops=drops, table=table, tray=tray_t, ret=round(ret, 3),
                             rec_ret=round(float(g["rewards"][()].sum()), 3)))
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reward_version", type=int, default=None, help="TrayPack reward variant for `ret`")
    p.add_argument("--out", default=None, help="output prefix (default: <file> without .h5)")
    a = p.parse_args()
    out = a.out or a.file[:-3]
    with h5py.File(a.file, "r") as f:
        names = sorted([k for k in f.keys() if k.startswith("traj")], key=lambda s: int(s.split("_")[-1]))
    names = names[:a.limit] if a.limit else names
    chunks = [names[i::a.workers] for i in range(a.workers)]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(_measure, [(a.file, c, a.reward_version) for c in chunks if c]) for r in rs]
    rows.sort(key=lambda r: int(r["traj"].split("_")[-1]))
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    col = lambda k: np.array([r[k] for r in rows])
    L = col("len")
    pct = lambda x: f"mean {x.mean():.1f}  p50 {np.median(x):.0f}  p90 {np.percentile(x, 90):.0f}  max {x.max()}"
    lines = [f"{a.file}: {len(rows)} demos"]
    lines.append(f"  replay success     {col('success').sum()}/{len(rows)}  (state_err max {col('state_err').max():.4f})")
    lines.append(f"  length             {pct(L)}")
    lines.append(f"  slack (horizon-T)  <=5: {(col('slack') <= 5).sum()}  <=20: {(col('slack') <= 20).sum()}")
    for k in [c for c in rows[0] if c.startswith("t_obj")]:
        t = col(k)
        lines.append(f"  {k} placed at     {pct(t[t > 0]) if (t > 0).any() else 'never'}   (never: {(t < 0).sum()})")
    for k in ("grasps", "regrasps", "drops", "undone", "table", "tray", "ret"):
        lines.append(f"  {k:<18} {pct(col(k).astype(float))}")
    bad = dict(fail=col("success") == 0, diverge=col("state_err") > 1e-3, tight=col("slack") <= 5,
               undone=col("undone") > 0, many_retries=col("regrasps") >= 4)
    lines.append("  flags: " + ", ".join(f"{k} {v.sum()}" for k, v in bad.items()))
    any_bad = np.zeros(len(rows), bool)
    for v in bad.values():
        any_bad |= v
    lines.append(f"  clean demos (no flag): {(~any_bad).sum()}/{len(rows)}; "
                 f"clean among first 500: {(~any_bad[:500]).sum()}/{min(500, len(rows))}")
    text = "\n".join(lines)
    print(text)
    with open(out + ".summary.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
