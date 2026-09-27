"""Verify demo files written by collect_new_tasks_demos.

For every trajectory:
  1. static checks: T actions vs T+1 observations, dtypes/shapes, actions finite and inside
     the action space, rewards finite, length <= horizon, seeds unique;
  2. replay (unless --no-replay): reset a fresh env built exactly like the collector's with
     the recorded seed, step the recorded actions, and require that the replayed
     proprioceptive state tracks the recorded one and that the episode ends in success.
     This proves the saved actions themselves solve the task.

Exit code is non-zero if any trajectory fails, so it can gate a pipeline.

Usage:
  python -m examples.verify_demos demos/qc/SO101TrayPack3-v1.h5
  python -m examples.verify_demos demos/qc/*.h5 --workers 16
  python -m examples.verify_demos demos/qc/SO101Rearrange3-v1.h5 --no-replay     # static checks only
  python -m examples.verify_demos demos/qc/SO101TrayPack3-v1.h5 --video demo_videos  # + one mp4 per demo

--video writes <dir>/<env_id>_traj<i>_seed<s>_{PASS,FAIL}.mp4: the scene camera during the
replay next to the recorded wrist image, with step / reward / success overlaid.
"""
import argparse
import json
import sys

import h5py
import numpy as np


def static_checks(g, meta, act_low, act_high):
    errs = []
    T = g["actions"].shape[0]
    for key, n in (("obs/rgb", T + 1), ("obs/state", T + 1), ("rewards", T), ("terminated", T), ("truncated", T)):
        if g[key].shape[0] != n:
            errs.append(f"{key} has {g[key].shape[0]} rows, expected {n}")
    if g["obs/rgb"].dtype != np.uint8:
        errs.append(f"obs/rgb dtype {g['obs/rgb'].dtype}")
    a = g["actions"][()]
    if not np.isfinite(a).all():
        errs.append("non-finite actions")
    elif (a < act_low - 1e-6).any() or (a > act_high + 1e-6).any():
        errs.append(f"actions outside action space [{a.min():.3f}, {a.max():.3f}]")
    if not np.isfinite(g["rewards"][()]).all() or not np.isfinite(g["obs/state"][()]).all():
        errs.append("non-finite rewards/state")
    horizon = int(g.attrs.get("horizon", meta.get("horizon", 10 ** 9)))
    if T > horizon:
        errs.append(f"length {T} > horizon {horizon}")
    if g["truncated"][()][:-1].any():
        errs.append("truncated before the last step")
    term = g["terminated"][()]
    if not term[-1]:
        errs.append("last step is not terminated (success)")
    elif not term[np.argmax(term):].all():
        errs.append("terminated flag switches off again after success")
    # The solver's final hold keeps stepping after success, so the episode ends with a
    # short run of terminated=True steps; valid, but loaders cutting at the first flag drop them.
    return errs


def post_success_steps(g):
    term = g["terminated"][()]
    return int(len(term) - np.argmax(term) - 1) if term.any() else 0


def _frame(scene, wrist, text):
    from PIL import Image, ImageDraw
    h = scene.shape[0]
    wrist = np.asarray(Image.fromarray(wrist).resize((h, h), Image.NEAREST))
    img = Image.fromarray(np.concatenate([scene, wrist], axis=1))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def _save_video(frames, video_dir, meta, name, seed, ok):
    import os
    import imageio.v2 as imageio
    os.makedirs(video_dir, exist_ok=True)
    out = os.path.join(video_dir, f"{meta['env_id']}_{name}_seed{seed}_{'PASS' if ok else 'FAIL'}.mp4")
    imageio.mimsave(out, frames, fps=10, macro_block_size=1)


def replay(path, names, state_tol, video_dir=None):
    """Replay trajectories `names` of `path`; returns {name: [errors]}."""
    import torch
    torch.set_num_threads(1)
    from examples.collect_new_tasks_demos import make_env

    out = {}
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        env, *_ = make_env(meta["env_id"], meta.get("render_size", 128), meta.get("domain_randomization", False),
                           meta.get("requested_control_mode"))
        for name in names:
            g = f[name]
            errs = []
            rec_state = g["obs/state"][()]
            obs, _ = env.reset(seed=int(g.attrs["seed"]))
            # state = noisy_qpos(6) + target_qpos(6); with domain randomization the qpos part is
            # re-noised on every call, so only the controller target is compared then.
            cols = slice(6, 12) if meta.get("domain_randomization") else slice(0, 12)
            worst, success = 0.0, False
            rec_rgb = g["obs/rgb"] if video_dir else None
            frames = []

            def snap(t, r=0.0):
                scene = env.render()
                scene = (scene[0] if scene.ndim == 4 else scene)
                scene = scene.cpu().numpy() if hasattr(scene, "cpu") else np.asarray(scene)
                frames.append(_frame(scene.astype(np.uint8), rec_rgb[t][..., :3],
                                     f"{name} seed {g.attrs['seed']}  step {t}/{len(g['actions'])}  "
                                     f"reward {r:.2f}  success {success}"))

            if video_dir:
                snap(0)
            for t, a in enumerate(g["actions"][()]):
                obs, rew, term, trunc, info = env.step(a)
                s = obs["state"][0].cpu().numpy()
                worst = max(worst, float(np.abs(s[cols] - rec_state[t + 1, cols]).max()))
                success = bool(torch.as_tensor(info["success"]).reshape(-1)[0])
                if video_dir:
                    snap(t + 1, float(torch.as_tensor(rew).reshape(-1)[0]))
            if worst > state_tol:
                errs.append(f"replay diverges from recording (max state err {worst:.4f} > {state_tol})")
            if not success:
                errs.append("replay does not end in success")
            out[name] = errs
            if video_dir:
                _save_video(frames, video_dir, meta, name, int(g.attrs["seed"]), not errs)
        env.close()
    return out


def _replay_job(args):
    return replay(*args)


def verify(path, do_replay, workers, state_tol, video_dir=None):
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        names = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
        ci = meta.get("control_info", {})
        low = np.asarray(ci.get("action_low", -1.0)); high = np.asarray(ci.get("action_high", 1.0))
        results = {n: static_checks(f[n], meta, low, high) for n in names}
        seeds = [int(f[n].attrs["seed"]) for n in names]
        lengths = [f[n]["actions"].shape[0] for n in names]
        extra = [post_success_steps(f[n]) for n in names]
    dup = {s for s in seeds if seeds.count(s) > 1}
    for n, s in zip(names, seeds):
        if s in dup:
            results[n].append(f"duplicate seed {s}")

    if do_replay and names:
        chunks = [names[w::workers] for w in range(workers) if names[w::workers]]
        if len(chunks) == 1:
            replays = [replay(path, chunks[0], state_tol, video_dir)]
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(len(chunks)) as pool:
                replays = pool.map(_replay_job, [(path, c, state_tol, video_dir) for c in chunks])
        for r in replays:
            for n, errs in r.items():
                results[n] += errs

    bad = {n: e for n, e in results.items() if e}
    print(f"\n{path}  [{meta['env_id']}]")
    print(f"  {len(names)} demos, length mean {np.mean(lengths):.0f} / max {max(lengths)} "
          f"(horizon {meta.get('horizon')}), replay {'on' if do_replay else 'OFF'}")
    if names:
        print(f"  note: steps recorded after first success: max {max(extra)} (terminated stays True)")
    for n, errs in bad.items():
        print(f"  FAIL {n} (seed {seeds[names.index(n)]}): " + "; ".join(errs))
    print(f"  => {len(names) - len(bad)}/{len(names)} passed")
    return not bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--no-replay", action="store_true", help="static checks only (fast)")
    p.add_argument("--workers", type=int, default=1, help="parallel replay processes")
    p.add_argument("--state-tol", type=float, default=1e-3,
                   help="max allowed |replayed - recorded| joint state (rad)")
    p.add_argument("--video", default=None, metavar="DIR", help="save one mp4 per demo into DIR (needs replay)")
    args = p.parse_args()
    if args.video and args.no_replay:
        p.error("--video needs the replay; drop --no-replay")
    ok = [verify(f, not args.no_replay, args.workers, args.state_tol, args.video) for f in args.files]
    print(f"\n{sum(ok)}/{len(ok)} files fully passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
