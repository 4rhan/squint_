"""Check a QC-FQL checkpoint through the deployment path, in sim, before running it on the real robot.

The env is built exactly like deploy.py builds its sim twin (128 px sensors, no domain randomization,
FlattenRGBDObservationWrapper), and the policy is qc_agent.QCDeployAgent, the same agent deploy_qc.py uses.
Actions go through deploy.py's `clip(action * action_scale, -1, 1)`. Reports success per episode, so you can
see how --action_scale and --qc_exec_steps change the behaviour before trying them on the arm.

    python -m examples.check_deploy_qc --checkpoint runs/lift_qc_dr200/ckpt_best.pt
    python -m examples.check_deploy_qc --checkpoint ... --action_scale 0.45 --max_episode_steps 150
    python -m examples.check_deploy_qc --checkpoint ... --qc_exec_steps 2 --video_dir videos/deploy_check

Notes: in sim one step is one 10 Hz control step. deploy.py runs the real arm at --control_freq 30 with
--action_scale 0.15, i.e. ~45% of the sim joint speed per second, so a scaled run here needs more steps
(--max_episode_steps). --domain_randomization turns on the training randomization (camera, joint noise, cube).
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401  (registers tasks)
from qc_agent import QCDeployAgent


@dataclass
class Args:
    checkpoint: str
    env_id: str = "SO101LiftCube-v1"
    episodes: int = 20
    seed: int = 0
    """episode i uses layout seed + i"""
    max_episode_steps: int = 50
    action_scale: float = 1.0
    """deploy.py multiplies every action by this (its default is 0.15 at 30 Hz); 1.0 = as in training"""
    qc_exec_steps: Optional[int] = None
    image_size: int = 128
    """sensor size, as deploy.py's --image_size"""
    domain_randomization: bool = False
    """deploy.py builds its sim twin without randomization"""
    sim_backend: str = "auto"
    video_dir: Optional[str] = None
    """save a wrist-camera + scene mp4 per episode here"""


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # same kwargs as deploy.main(), except reward/eval info is needed here to score the episode
    env = gym.make(args.env_id, obs_mode="rgb+segmentation", render_mode="rgb_array",
                   max_episode_steps=args.max_episode_steps, domain_randomization=args.domain_randomization,
                   control_mode="pd_joint_target_delta_pos", sim_backend=args.sim_backend,
                   sensor_configs=dict(width=args.image_size, height=args.image_size))
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    obs, _ = env.reset(seed=args.seed)
    agent = QCDeployAgent(env, {k: v for k, v in obs.items()}, args.checkpoint, exec_steps=args.qc_exec_steps,
                          device=device).to(device)
    if args.video_dir:
        import os
        import imageio
        os.makedirs(args.video_dir, exist_ok=True)

    # per-task stage flags reported as "reached at any step of the episode" (only those the env provides)
    stage_keys = ["is_item_grasped", "item_lifted",                                   # Lift
                  "itemA_picked_clean", "is_itemA_on_table", "itemB_picked_clean",   # Unstack3
                  "is_itemB_on_table", "tower_knocked"]
    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        agent.reset()
        frames, success_once, first_success = [], False, None
        stages = {}
        for t in range(args.max_episode_steps):
            action = agent.get_action({k: v.to(device) for k, v in obs.items()}).cpu().numpy()
            action = np.clip(action * args.action_scale, -1, 1)  # deploy.py
            obs, _, _, _, info = env.step(action)
            ok = bool(info["success"].reshape(-1)[0])
            if ok and not success_once:
                success_once, first_success = True, t + 1
            for k in stage_keys:
                if k in info:
                    stages[k] = stages.get(k, False) or bool(info[k].reshape(-1)[0])
            if args.video_dir:
                scene = env.render()[0].cpu().numpy()
                # the wrist image the policy sees (before its 16 px downsample), scaled to the scene height
                wrist = cv2.resize(obs["rgb"][0].cpu().numpy(), (scene.shape[0], scene.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                frames.append(np.concatenate([scene, wrist], axis=1))
        results.append((ok, success_once, first_success, stages))
        print(f"episode {ep:2d} (seed {args.seed + ep}): success at end {ok!s:5}  once {success_once!s:5}  "
              f"first success step {first_success}  " + "  ".join(f"{k} {int(v)}" for k, v in stages.items()))
        if args.video_dir:
            imageio.mimsave(f"{args.video_dir}/episode_{ep}.mp4", frames, fps=10)

    r = np.array([[x[0], x[1]] for x in results], dtype=float)
    steps = [x[2] for x in results if x[2] is not None]
    print(f"\n{args.episodes} episodes, action_scale {args.action_scale}, exec_steps {agent.exec_steps}, "
          f"{args.max_episode_steps} steps, DR {args.domain_randomization}:")
    print(f"  success at end {r[:, 0].mean():.2f}  success once {r[:, 1].mean():.2f}  "
          f"first success step mean {np.mean(steps) if steps else float('nan'):.1f}")
    print("  reached during the episode: " + "  ".join(
        f"{k} {np.mean([x[3].get(k, False) for x in results]):.2f}" for k in results[0][3]))
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
