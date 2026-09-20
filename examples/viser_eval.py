"""Headless live viewer for ManiSkill eval rollouts, served over the browser via viser.

Useful when the machine actually running the sim/training has no attached display and you
only reach it over SSH / a VPN like ZeroTier: run this script ON that machine, then open the
printed URL from a browser on your own PC.

    - viser binds 0.0.0.0 by default, so if your training box and your PC are both on the
      same ZeroTier network, you can just open http://<training box's ZeroTier IP>:8080
      directly - no SSH tunnel needed.
    - If you'd rather not open the port on the VPN interface, SSH-tunnel it instead:
          ssh -L 8080:localhost:8080 user@<training box>
      then open http://localhost:8080 on your PC.

Usage (run on the machine with the GPU / the sim):
    python examples/viser_eval.py --checkpoint runs/baseline/ckpt.pt --env_id SO101Stack3Cube-v1
    python examples/viser_eval.py --env_id SO101Stack3Cube-v1   # no checkpoint -> random agent,
                                                                  # useful to sanity check the viewer itself
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro
import gymnasium as gym
import viser

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.utils.visualization.misc import tile_images

import utils
# Add tasks
import envs
import mani_skill.envs

from train_squint import DeployAgent


@dataclass
class Args:
    env_id: str = "SO101Stack3Cube-v1"
    checkpoint: Optional[str] = None
    """Path to a train_squint.py checkpoint (e.g. runs/baseline/ckpt.pt). If None, runs a
    random agent - useful to sanity check the viewer/streaming itself works."""
    control_mode: str = "pd_joint_target_delta_pos"
    obs_mode: str = "rgb+segmentation"
    num_envs: int = 4
    """Number of parallel envs to run and tile into the live view."""
    render_size: int = 128
    """Resolution both the wrist-cam obs and third-person render are captured at."""
    image_size: int = 16
    """Resolution the policy actually sees (must match what it was trained with)."""
    sim_backend: str = "gpu"
    domain_randomization: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    fps: float = 15.0
    """Playback speed cap - sim steps much faster than this, throttled so the stream is watchable."""
    seed: int = 0


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        domain_randomization=args.domain_randomization,
        sensor_configs=dict(width=args.render_size, height=args.render_size),
        human_render_camera_configs=dict(width=args.render_size, height=args.render_size),
        reconfiguration_freq=1,
    )
    max_episode_steps = gym_utils.find_max_episode_steps_value(env)
    base_env = env.unwrapped

    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if args.render_size != args.image_size:
        env = utils.DownsampleObsWrapper(env, target_size=args.image_size)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    obs, info = env.reset(seed=args.seed)

    agent = None
    if args.checkpoint is not None:
        agent = DeployAgent(env, sample_obs=obs, target_image_size=args.image_size, device=device)
        agent.load_checkpoint(args.checkpoint)
        agent.eval()
    else:
        print("[viser_eval] No --checkpoint given: running a RANDOM agent (viewer sanity check only).")

    action_dim = env.action_space.shape[-1]

    server = viser.ViserServer(host=args.host, port=args.port, label=f"squint eval: {args.env_id}")
    image_handle = server.gui.add_image(
        image=np.zeros((args.render_size, args.render_size * 2, 3), dtype=np.uint8),
        label="obs (wrist cam, policy-res) | render (third-person)",
    )
    step_text = server.gui.add_text("step", initial_value="0 / 0")
    success_text = server.gui.add_text("success", initial_value="-")
    stage_text = server.gui.add_text("stage flags", initial_value="-")

    print(f"[viser_eval] serving at http://{args.host}:{args.port}")
    print(f"[viser_eval] from your PC, open http://<this machine's ZeroTier IP>:{args.port}")

    step_in_episode = 0
    dt = 1.0 / args.fps
    while True:
        tstart = time.time()
        if agent is not None:
            with torch.no_grad():
                action = agent.get_action(obs)
        else:
            action = torch.rand((args.num_envs, action_dim)) * 2 - 1

        obs, reward, terminated, truncated, info = env.step(action)
        step_in_episode += 1

        # Build the tiled live frame: wrist-cam obs (what the policy actually sees) next to
        # the third-person render, same layout as examples/visualize_sim.py.
        render_rgb = base_env.render()  # (N, H, W, 3)
        obs_rgb = obs['rgb']
        if obs_rgb.shape[-1] != 3 and obs_rgb.shape[-1] % 3 == 0:
            obs_rgb = obs_rgb[..., :3]
        if obs_rgb.shape[1:3] != render_rgb.shape[1:3]:
            obs_rgb = torch.nn.functional.interpolate(
                obs_rgb.permute(0, 3, 1, 2).float(), size=render_rgb.shape[1:3], mode="nearest"
            ).permute(0, 2, 3, 1).to(torch.uint8)
        paired = torch.cat([obs_rgb, render_rgb], dim=2)
        nrows = int(np.sqrt(args.num_envs))
        frame = tile_images(paired, nrows=nrows).cpu().numpy().astype(np.uint8)
        image_handle.image = frame

        step_text.value = f"{step_in_episode} / {max_episode_steps}"
        if "success" in info:
            success_text.value = str(info["success"].tolist())
        stage_bits = []
        for key in ["is_itemB_on_itemC", "is_itemA_grasped", "is_itemA_on_itemB"]:
            if key in info:
                stage_bits.append(f"{key}={info[key].tolist()}")
        if stage_bits:
            stage_text.value = " | ".join(stage_bits)

        if step_in_episode >= max_episode_steps:
            obs, info = env.reset()
            step_in_episode = 0

        elapsed = time.time() - tstart
        if elapsed < dt:
            time.sleep(dt - elapsed)


if __name__ == "__main__":
    main(tyro.cli(Args))
