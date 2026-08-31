"""Open a task in the interactive SAPIEN viewer (a real, resizable 3D window you can
orbit/pan/zoom with the mouse), instead of the small cv2 preview used by visualize_sim.py.

Usage:
    python examples/view_in_sapien.py SO101ToolSweep-v1
    python examples/view_in_sapien.py SO101ToolSweep-v1 --robot-uids so101 --domain-randomization
    python examples/view_in_sapien.py SO101ToolSweep-v1 --pause   # start paused so you can look around first

Viewer controls (once the window opens):
    - Left mouse drag: orbit camera
    - Right mouse drag / scroll: pan / zoom
    - "p": pause/unpause the simulation
    - "g": toggle move/rotate gizmo on the selected object (click an object first)
    - Close the window (or Ctrl+C in the terminal) to exit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
logging.disable(level=logging.WARN)

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import sapien
import tyro

# Register this repo's custom SO-100/SO-101 tasks and robots, plus stock ManiSkill tasks.
import envs
import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv


@dataclass
class Args:
    env_id: tyro.conf.Positional[str] = "SO101ToolSweep-v1"
    """Task to load, e.g. SO101ToolSweep-v1, SO101PegInsertionSide-v1, SO101PlaceCube-v1, SO101StackCube-v1, SO101LiftCube-v1, SO101ReachCube-v1"""

    robot_uids: Optional[str] = None
    """Robot to use (so100 or so101). Defaults to the task's default robot."""

    control_mode: Optional[str] = None
    """Control mode. Defaults to the task's default control mode."""

    domain_randomization: bool = False
    """Enable the task's domain randomization (lighting, friction, robot color, etc)."""

    num_envs: int = 1
    """Number of parallel envs to load into the same scene. Only useful for eyeballing domain
    randomization variety side by side; >1 disables per-env wrist camera following."""

    seed: Optional[int] = None
    """Seed for reproducible resets/actions."""

    policy: str = "random"
    """How to drive the robot: 'random' actions, or 'still' (zero action, useful with --pause
    to just look at the scene)."""

    shader: str = "default"
    """Render shader: 'default' (fast) or 'rt'/'rt-fast' (ray-traced, much slower but photoreal)."""

    pause: bool = False
    """Start the viewer paused so you can look around before the sim starts moving."""

    max_steps: Optional[int] = None
    """Stop after this many steps total (default: run until you close the window)."""


def main(args: Args):
    parallel_in_single_scene = args.num_envs > 1

    env_kwargs = dict(
        obs_mode="state",
        render_mode="human",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend="physx_cpu" if args.num_envs == 1 else "auto",
        parallel_in_single_scene=parallel_in_single_scene,
        domain_randomization=args.domain_randomization,
        reconfiguration_freq=None,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    env: BaseEnv = gym.make(args.env_id, **env_kwargs)

    print("env_id:", args.env_id)
    print("robot_uids:", env.unwrapped.robot_uids)
    print("control_mode:", env.unwrapped.control_mode)
    print("action space:", env.action_space)
    print()
    print("A SAPIEN viewer window should now open. Drag with the mouse to orbit,")
    print("scroll/right-drag to zoom/pan, press 'p' to pause/unpause.")

    seed = args.seed
    obs, info = env.reset(seed=seed, options=dict(reconfigure=True))

    viewer = env.render()
    if isinstance(viewer, sapien.utils.Viewer):
        viewer.paused = args.pause
    env.render()

    step = 0
    try:
        while args.max_steps is None or step < args.max_steps:
            if args.policy == "still":
                action = np.zeros(env.action_space.shape)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            done = bool((terminated | truncated).any())
            if done:
                print(f"step {step}: episode finished (success={info.get('success')}), resetting")
                env.reset()

            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
