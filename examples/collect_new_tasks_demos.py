"""Collect scripted demonstrations for new SO-101 tasks in QC-compatible HDF5.

Runs the motion-planning solver directly in the training controller
(pd_joint_target_delta_pos, normalized actions, 10 Hz, declared limits) so
recordings match training: obs_mode rgb+segmentation, 128px, FlattenRGBD,
no colour jitter. Clean 128px stored; training area-downsamples to 16/32/64.
No teleporting; no absolute-position commands labelled as delta.

Output per episode group traj_<i>:
  obs/rgb    (T+1,H,W,C) uint8   (128px clean)
  obs/state  (T+1,S) float32     (flattened training layout: noisy_qpos+target_qpos)
  actions    (T,A) float32       (normalized delta in env action space)
  rewards    (T,) float32        (normalized_dense)
  terminated (T,) bool, truncated (T,) bool
Root attrs: JSON metadata with ACTUAL settings (task/version, seeds,
domain_randomization bool, control_mode/limits, sim/control freq, camera,
obs layout, reward mode). Episode boundaries via groups + term/trunc (T+1 obs).

Usage:
  python -m examples.collect_new_tasks_demos -e SO101Tower3Cube-v1 -n 10 -o demos/qc/SO101Tower3Cube-v1.h5
  python -m examples.collect_new_tasks_demos --all -n 5 --outdir demos/qc
  python -m examples.collect_new_tasks_demos -e SO101Rearrange3-v1 -n 20 --start-seed 100 --domain-randomization -o demos/qc/rearr3.h5
"""
import argparse
import importlib
import json
import os
from collections import Counter

import envs  # noqa: F401 registers tasks
import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from tqdm import tqdm

SOLVERS = {
    "SO101Tower3Cube-v1": "examples.motionplanning.so101.solutions.tower3_cube:solve",
    "SO101Tower2Cube-v1": "examples.motionplanning.so101.solutions.tower2_cube:solve",
    "SO101TrayPack1-v1": "examples.motionplanning.so101.solutions.tray_pack:solve_pack1",
    "SO101TrayPack2-v1": "examples.motionplanning.so101.solutions.tray_pack:solve_pack2",
    "SO101TrayPack3-v1": "examples.motionplanning.so101.solutions.tray_pack:solve",
    "SO101Rearrange3-v1": "examples.motionplanning.so101.solutions.rearrange:solve3",
    "SO101Rearrange2-v1": "examples.motionplanning.so101.solutions.rearrange:solve2",
}
SOLVER_MODULES = {
    "SO101Tower3Cube-v1": "examples.motionplanning.so101.solutions.tower3_cube",
    "SO101Tower2Cube-v1": "examples.motionplanning.so101.solutions.tower2_cube",
    "SO101TrayPack1-v1": "examples.motionplanning.so101.solutions.tray_pack",
    "SO101TrayPack2-v1": "examples.motionplanning.so101.solutions.tray_pack",
    "SO101TrayPack3-v1": "examples.motionplanning.so101.solutions.tray_pack",
    "SO101Rearrange3-v1": "examples.motionplanning.so101.solutions.rearrange",
    "SO101Rearrange2-v1": "examples.motionplanning.so101.solutions.rearrange",
}


class Recorder(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rgb = [obs["rgb"][0].cpu().numpy().astype(np.uint8)]
        self.state = [obs["state"][0].cpu().numpy().astype(np.float32)]
        self.actions, self.rewards, self.terms, self.truncs = [], [], [], []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rgb.append(obs["rgb"][0].cpu().numpy().astype(np.uint8))
        self.state.append(obs["state"][0].cpu().numpy().astype(np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        r = torch.as_tensor(reward).reshape(-1)[0]
        self.rewards.append(float(r))
        self.terms.append(bool(torch.as_tensor(terminated).reshape(-1)[0]))
        self.truncs.append(bool(torch.as_tensor(truncated).reshape(-1)[0]))
        return obs, reward, terminated, truncated, info


def make_env(env_id, render_size=128, domain_randomization=False, control_mode=None,
             reward_mode="normalized_dense"):
    kwargs = dict(obs_mode="rgb+segmentation", render_mode="rgb_array", sim_backend="cpu",
                  sensor_configs=dict(width=render_size, height=render_size),
                  reward_mode=reward_mode, reconfiguration_freq=1)
    if control_mode is not None:
        kwargs["control_mode"] = control_mode
    if domain_randomization:
        kwargs["domain_randomization"] = True
    env = gym.make(env_id, num_envs=1, **kwargs)
    horizon = gym_utils.find_max_episode_steps_value(env)
    actual_dr = getattr(env.unwrapped, "domain_randomization", False)
    actual_ctrl = env.unwrapped.agent.control_mode if hasattr(env.unwrapped, "agent") else kwargs.get("control_mode")
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return Recorder(env), horizon, actual_dr, actual_ctrl


def load_solver(env_id):
    mod, fn = SOLVERS[env_id].split(":")
    return importlib.import_module(mod).__dict__[fn]


def collect_one(env_id, num_traj, out, start_seed=0, max_attempts=None, render_size=128,
                domain_randomization=False, control_mode=None):
    solve = load_solver(env_id)
    sol_mod = importlib.import_module(SOLVER_MODULES[env_id])
    env, horizon, actual_dr, actual_ctrl = make_env(
        env_id, render_size, domain_randomization, control_mode)
    unw = env.unwrapped
    try:
        ctrl = unw.agent.controller.config
        ctrl_info = dict(control_mode=unw.agent.control_mode,
                         action_low=unw.single_action_space.low.tolist(),
                         action_high=unw.single_action_space.high.tolist(),
                         delta_upper=list(getattr(ctrl, "upper", [])),
                         delta_lower=list(getattr(ctrl, "lower", [])))
    except Exception:
        ctrl_info = {}
    try:
        sim_freq = unw._sim_config.sim_freq; control_freq = unw._sim_config.control_freq
    except Exception:
        sim_freq, control_freq = 100, 10
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    saved = attempts = too_long = 0
    lengths = []
    fail_cats = Counter()
    seed = start_seed
    pbar = tqdm(total=num_traj, desc=env_id)
    with h5py.File(out, "w") as f:
        meta = dict(env_id=env_id, horizon=horizon, render_size=render_size,
                    obs_mode="rgb+segmentation", reward_mode="normalized_dense",
                    domain_randomization=bool(actual_dr),
                    requested_domain_randomization=bool(domain_randomization),
                    control_info=ctrl_info, requested_control_mode=control_mode,
                    sim_freq=sim_freq, control_freq=control_freq,
                    source_domain="sim", obs_layout="flattened rgb(128,H,W,C uint8)+state(12 float32: noisy_qpos6+target_qpos6)",
                    camera="wrist 128px FOV71deg + per-episode pose/FOV noise iff domain_randomization=true",
                    note="training controller only (normalized delta @10Hz); clean images, no jitter")
        f.attrs["meta"] = json.dumps(meta)
        while saved < num_traj and (max_attempts is None or attempts < max_attempts):
            attempts += 1
            fail_stage, fail_reason = None, None
            try:
                res = solve(env, seed=seed)
                if res == -1 and hasattr(sol_mod, "LAST_FAIL"):
                    fail_stage = sol_mod.LAST_FAIL.get("stage")
                    fail_reason = sol_mod.LAST_FAIL.get("reason")
            except Exception as err:
                print(f"[{env_id} seed {seed}] solver exception: {err}")
                res = -1
                fail_stage, fail_reason = "exception", str(err)[:80]
            cur_seed = seed
            seed += 1
            if res == -1:
                fail_cats[f"plan_fail:{fail_stage}:{fail_reason}"] += 1
                continue
            try:
                ok = bool(res[-1]["success"].reshape(-1)[0])
            except Exception:
                ok = bool(res[-1]["success"])
            if not ok:
                # Categorize unsuccessful execution (placed but checks fail).
                try:
                    info = env.unwrapped.evaluate()
                    n = int(info.get("num_correct", -1).reshape(-1)[0]) if hasattr(info.get("num_correct", 0), "reshape") else int(info.get("num_correct", -1))
                except Exception:
                    n = -1
                fail_cats[f"success_false:num_correct={n}"] += 1
                continue
            T = len(env.actions)
            if T > horizon:
                too_long += 1
                fail_cats["too_long"] += 1
                continue
            g = f.create_group(f"traj_{saved}")
            g.create_dataset("obs/rgb", data=np.stack(env.rgb), compression="gzip")
            g.create_dataset("obs/state", data=np.stack(env.state))
            g.create_dataset("actions", data=np.stack(env.actions))
            g.create_dataset("rewards", data=np.asarray(env.rewards, dtype=np.float32))
            g.create_dataset("terminated", data=np.asarray(env.terms, dtype=np.bool_))
            g.create_dataset("truncated", data=np.asarray(env.truncs, dtype=np.bool_))
            g.attrs["seed"] = cur_seed
            g.attrs["horizon"] = horizon
            lengths.append(T)
            saved += 1
            pbar.update(1)
            pbar.set_postfix(success_rate=f"{saved/attempts:.2f}", too_long=too_long, mean_len=f"{np.mean(lengths):.0f}")
    env.close()
    print(f"[{env_id}] Saved {saved}/{attempts} (too_long {too_long}) mean_len {np.mean(lengths) if lengths else 0:.0f} -> {out}")
    print(f"[{env_id}] failures: {dict(fail_cats)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--env-id", default=None, choices=sorted(SOLVERS))
    p.add_argument("--all", action="store_true")
    p.add_argument("-n", "--num-traj", type=int, default=5)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--outdir", default="demos/qc")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=200)
    p.add_argument("--render-size", type=int, default=128)
    p.add_argument("--domain-randomization", action="store_true",
                   help="explicitly enable domain randomization (default off)")
    p.add_argument("--control-mode", default=None,
                   help="explicit controller (default None = env default pd_joint_target_delta_pos)")
    args = p.parse_args()
    ids = sorted(SOLVERS) if args.all else [args.env_id or "SO101Tower3Cube-v1"]
    for eid in ids:
        out = args.out if (args.out and not args.all) else f"{args.outdir}/{eid}.h5"
        collect_one(eid, args.num_traj, out, start_seed=args.start_seed,
                    max_attempts=args.max_attempts, render_size=args.render_size,
                    domain_randomization=args.domain_randomization,
                    control_mode=args.control_mode)


if __name__ == "__main__":
    main()
