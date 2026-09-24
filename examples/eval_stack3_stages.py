"""Diagnose *why* SO101Stack3Cube-v1 training saturates: per-stage success breakdown.

Loads a trained checkpoint (from train_squint.py) and rolls it out in sim, tracking
per-episode whether each sub-goal was ever hit:
    - itemB ever placed on itemC (stage 1)
    - itemA ever grasped (did it even attempt stage 2?)
    - itemA ever placed on itemB (stage 2)
    - full task success

This tells you whether the plateau is because the policy:
    (a) never attempts to grasp itemA after finishing stage 1 (reward local optimum), or
    (b) attempts it but fails to place it (grasping/placing is hard, needs more training
        or reward/geometry tuning), or
    (c) places it but fails the final static/robot-static success gate (needs more
        "settle" time or looser success thresholds)

Usage:
    python examples/eval_stack3_stages.py --checkpoint runs/baseline/ckpt.pt \
        --num_envs 32 --num_episodes 200
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro
import gymnasium as gym

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper

import utils
# Add tasks
import envs
import mani_skill.envs

from train_squint import DeployAgent


@dataclass
class Args:
    checkpoint: str = "runs/baseline/ckpt.pt"
    """Path to a checkpoint saved by train_squint.py."""
    env_id: str = "SO101Stack3Cube-v1"
    control_mode: str = "pd_joint_target_delta_pos"
    obs_mode: str = "rgb+segmentation"
    num_envs: int = 32
    """Parallel envs to roll out with (requires GPU sim for >1)."""
    num_episodes: int = 200
    """Total episodes to evaluate across all envs (rounded up to a multiple of num_envs)."""
    render_size: int = 128
    image_size: int = 16
    sim_backend: str = "gpu"
    domain_randomization: bool = True
    seed: int = 0
    flag_keys: tuple[str, ...] = ("is_itemB_on_itemC", "is_itemA_grasped", "is_itemA_on_itemB")
    """Boolean info flags to report the 'ever hit during an episode' rate of. For SO101Place3Cube-v1 use
    --flag-keys in_bin_ge1 in_bin_ge2 in_bin_ge3."""


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode=None,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        domain_randomization=args.domain_randomization,
        sensor_configs=dict(width=args.render_size, height=args.render_size),
        reconfiguration_freq=1,  # re-randomize objects every reset, like eval in train_squint.py
    )

    env = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(env)

    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if args.render_size != args.image_size:
        env = utils.DownsampleObsWrapper(env, target_size=args.image_size)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    obs, info = env.reset(seed=args.seed)

    agent = DeployAgent(env, sample_obs=obs, target_image_size=args.image_size, device=device)
    agent.load_checkpoint(args.checkpoint)
    agent.eval()

    num_envs = args.num_envs
    n_rounds = int(np.ceil(args.num_episodes / num_envs))

    keys = ["success", *args.flag_keys]
    totals = {k: 0 for k in keys}
    n_done = 0

    print(f"[eval] device={device}, num_envs={num_envs}, episodes~{n_rounds * num_envs}, "
          f"max_episode_steps={max_episode_steps}, tracking={keys}")

    for round_idx in range(n_rounds):
        obs, info = env.reset()
        ever = {k: torch.zeros(num_envs, dtype=torch.bool, device=device) for k in keys}

        for step in range(max_episode_steps):
            with torch.no_grad():
                action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            for k in keys:
                ever[k] |= info[k]

        for k in keys:
            totals[k] += int(ever[k].sum().item())
        n_done += num_envs
        print(f"[eval] round {round_idx + 1}/{n_rounds}: " + " ".join(f"{k}={ever[k].float().mean().item():.2f}" for k in keys))

    env.close()

    print("\n===== Per-stage breakdown over", n_done, "episodes (fraction of episodes where it EVER happened) =====")
    for k in keys:
        print(f"{k:28s} {totals[k] / n_done:.3f}")

    # Automatic reading, only for the Stack3 flags
    if tuple(args.flag_keys) == ("is_itemB_on_itemC", "is_itemA_grasped", "is_itemA_on_itemB"):
        s1, gr, s2, ok = (totals[k] / n_done for k in ("is_itemB_on_itemC", "is_itemA_grasped", "is_itemA_on_itemB", "success"))
        if s1 > 0.8 and gr < 0.2:
            print("\n-> Diagnosis: finishes stage 1 but almost never attempts the third cube (reward local optimum or the cube is not visible).")
        elif gr > 0.5 and s2 < 0.2:
            print("\n-> Diagnosis: grasps the third cube often but rarely lands it on itemB (placement precision / knocks the tower over).")
        elif s2 > 0.5 and ok < 0.2:
            print("\n-> Diagnosis: lands the third cube but rarely satisfies the final static/released success gate.")


if __name__ == "__main__":
    main(tyro.cli(Args))
