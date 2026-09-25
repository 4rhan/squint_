"""Replay demos from a collect_qc_demos .h5 file in the simulator and save videos, to check them by eye.

The demo files only hold 16x16 images, so this re-runs each demo's recorded actions from its recorded seed in
an env built the way the collector built it, renders the normal 512x512 camera, and overlays the step and the
task progress. It also checks the recording: the replay should end in success and reproduce the recorded
ground-truth object positions (a large mismatch means the actions do not reproduce the demo).

    python -m examples.replay_qc_demos demos/qc/place3.h5 --traj 0 1 2 --outdir demo_videos
    python -m examples.replay_qc_demos demos/qc/place3.h5 --traj 0 --vis      # watch live in the simulator window
Writes <outdir>/<env_id>_traj<i>.mp4 for each demo and <outdir>/<env_id>_sheet.png (8 frames per demo).
"""
import argparse
import json
import os

import envs  # noqa: F401  registers the SO101 tasks
import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


def make_env(meta, backend, vis=False):
    kwargs = dict(obs_mode=meta.get("obs_mode", "rgb+segmentation"), render_mode="human" if vis else "rgb_array",
                  sim_backend=backend,
                  sensor_configs=dict(width=meta.get("render_size", 128), height=meta.get("render_size", 128)),
                  reconfiguration_freq=1)
    if meta.get("control_mode"):
        kwargs["control_mode"] = meta["control_mode"]
    if meta.get("domain_randomization", True):
        kwargs["domain_randomization"] = True
    if meta.get("cube_half_used"):
        h = meta["cube_half_used"]
        kwargs["domain_randomization_config"] = dict(cube_half_size_range=(h, h))
    return gym.make(meta["env_id"], num_envs=1, **kwargs)


def frame(env):
    img = env.render()
    img = img[0] if img.ndim == 4 else img
    return np.asarray(img.cpu() if torch.is_tensor(img) else img).astype(np.uint8)


def annotate(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def object_positions(e):
    if hasattr(e, "cubes"):
        return np.stack([c.pose.p[0].cpu().numpy() for c in e.cubes])
    return np.stack([getattr(e, n).pose.p[0].cpu().numpy() for n in ("item", "itemA", "itemB", "itemC") if hasattr(e, n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--traj", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--outdir", default="demo_videos")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--sim-backend", default="cpu")
    ap.add_argument("--vis", action="store_true", help="open the live simulator window instead of writing videos")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        meta = json.loads(f.attrs["meta"])
        env = make_env(meta, args.sim_backend, args.vis)
        e = env.unwrapped
        sheet = []
        for i in args.traj:
            g = f[f"traj_{i}"]
            actions, seed = g["actions"][:], int(g.attrs["seed"])
            env.reset(seed=seed)
            frames = [] if args.vis else [annotate(frame(env), f"{meta['env_id']}  demo {i} (seed {seed})  step 0/{len(actions)}")]
            info = {}
            for t, a in enumerate(actions):
                _, _, _, _, info = env.step(torch.as_tensor(a[None]))
                if args.vis:
                    e.render_human()
                    continue
                extra = f"  in bin {int(info['num_in_bin'].item())}/3" if "num_in_bin" in info else ""
                frames.append(annotate(frame(env), f"{meta['env_id']}  demo {i}  step {t + 1}/{len(actions)}{extra}"))
            ok = bool(info["success"].reshape(-1)[0]) if "success" in info else None
            msg = f"demo {i}: replay success={ok}"
            if "gt" in g and any(k.endswith("_pose") for k in g["gt"]):
                rec = g["gt/cube_pose"][-1, :, :3] if "cube_pose" in g["gt"] else np.stack(
                    [g["gt/" + k][-1, :3] for k in ("item_pose", "itemA_pose", "itemB_pose", "itemC_pose") if k in g["gt"]])
                diff = 1000 * np.abs(object_positions(e) - rec).max()
                msg += f", max object position difference vs recording {diff:.1f} mm" + ("" if diff < 5 else "  <-- does NOT reproduce")
            print(msg)
            if args.vis:
                continue
            path = os.path.join(args.outdir, f"{meta['env_id']}_traj{i}.mp4")
            imageio.mimsave(path, frames, fps=args.fps)
            sheet.append(np.concatenate([frames[j] for j in np.linspace(0, len(frames) - 1, 8).astype(int)], axis=1))
        if args.vis:
            return
        Image.fromarray(np.concatenate(sheet, axis=0)).resize((2048, 256 * len(sheet))).save(
            os.path.join(args.outdir, f"{meta['env_id']}_sheet.png"))
    print(f"saved videos and contact sheet to {args.outdir}/")


if __name__ == "__main__":
    main()
