"""Squint + Q-chunking (flow policy): offline pretraining on HDF5 demos, then online RL with 50/50 sampling.

    python train_squint_qc.py --env_id SO101StackCube-v1 --demo_path demos/.../demos.h5 --horizon 5

Env / logging / eval plumbing is shared with train_squint.py; the agent lives in qc_agent.py and the
chunk replay + HDF5 loader in qc_data.py.
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torchvision
import tqdm
import tyro
import wandb
from collections import deque

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import envs  # noqa: F401  (registers tasks)
import mani_skill.envs  # noqa: F401
import utils
from qc_agent import QCAgent, QCConfig
from qc_data import ChunkBuffer, color_jitter, load_h5_demos
from train_squint import Args, Logger, evaluate


@dataclass
class QCArgs(Args):
    agent_name: Optional[str] = "squint_qc"
    exp_name: Optional[str] = "qc_baseline"
    gamma: float = 0.9
    """discount per env step (the h-step backup uses gamma**horizon). Squint's tuned value for these
    50-step dense-reward tasks; the official QC's 0.99 targets long sparse-reward OGBench tasks."""
    tau: float = 0.01
    """target smoothing coefficient (Squint's value)"""
    policy_frequency: int = 1
    """unused by QC (actor + critic update together); kept so the base Args stay valid"""

    demo_path: Optional[str] = None
    """HDF5 file with demonstrations (schema in qc_data.py). Required unless --offline_steps 0."""
    max_demo_trajs: Optional[int] = None
    offline_steps: int = 50_000
    """gradient steps of offline pretraining on the demos before any environment interaction"""
    offline_ratio: float = 0.5
    """fraction of each online batch drawn from the demos (rest from the online replay); 0 disables demos online"""
    num_updates: int = 256
    """gradient steps per parallel env step (Squint's update-to-data ratio)"""
    learning_starts: int = 5_000
    """online steps to collect before online updates begin (Squint's value)"""
    batch_size: int = 512

    horizon: int = 5
    """action chunk length h"""
    actor_type: str = "distill-ddpg"
    """'distill-ddpg' or 'best-of-n'"""
    actor_num_samples: int = 32
    flow_steps: int = 10
    alpha: float = 100.0
    """BC/distillation coefficient (tune per task)"""
    q_agg: str = "mean"
    hidden_dim: int = 512
    num_layers: int = 4
    lr: float = 3e-4


class ChunkExecutor:
    """Turns a chunk policy into a per-step get_action(rgb, state) closure: queries the agent every
    `horizon` steps and replays the chunk open-loop in between (as in qc's main_online.py)."""

    def __init__(self, agent, action_scale, action_bias, deterministic=False):
        self.agent, self.scale, self.bias = agent, action_scale, action_bias
        self.deterministic = deterministic
        self.chunk, self.i = None, 0

    def reset(self):
        self.chunk = None

    def __call__(self, rgb, state):
        if self.chunk is None or self.i >= self.agent.cfg.horizon:
            self.chunk, self.i = self.agent.act(rgb, state, deterministic=self.deterministic), 0
        a = self.chunk[:, self.i]
        self.i += 1
        return a * self.scale + self.bias


if __name__ == "__main__":
    args = tyro.cli(QCArgs)
    args.num_total_iterations = int(args.total_timesteps // args.num_envs)
    assert args.offline_steps == 0 or args.demo_path, "--demo_path is required when --offline_steps > 0"
    assert 0.0 <= args.offline_ratio <= 1.0

    run_name = args.exp_name or f"{args.env_id}__qc__{args.seed}__{int(time.time())}"
    model_path = os.path.abspath(f"runs/{run_name}/ckpt.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environments (same construction as train_squint.py) ────────────────
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu",
                      sensor_configs=dict(width=args.render_size, height=args.render_size))
    eval_env_kwargs = dict(env_kwargs, human_render_camera_configs=dict(
        shader_pack="default", width=args.render_size, height=args.render_size))
    if args.control_mode is not None:
        env_kwargs["control_mode"] = eval_env_kwargs["control_mode"] = args.control_mode
    if args.env_domain_randomization:
        env_kwargs["domain_randomization"] = eval_env_kwargs["domain_randomization"] = True
    if args.stage2_start_prob > 0:
        env_kwargs["stage2_start_prob"] = args.stage2_start_prob

    train_envs = gym.make(args.env_id, num_envs=args.num_envs, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **eval_env_kwargs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(train_envs)
    train_envs = FlattenRGBDObservationWrapper(train_envs, rgb=True, depth=False, state=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=True)
    if args.render_size != args.image_size:
        train_envs = utils.DownsampleObsWrapper(train_envs, target_size=args.image_size)
        eval_envs = utils.DownsampleObsWrapper(eval_envs, target_size=args.image_size)
    if args.apply_jitter:
        train_envs = utils.ColorJitterWrapper(train_envs)
        eval_envs = utils.ColorJitterWrapper(eval_envs)
    if isinstance(train_envs.action_space, gym.spaces.Dict):
        train_envs = FlattenActionSpaceWrapper(train_envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    eval_output_dir = None
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=False, save_video=True,
                                  trajectory_name="trajectory", max_steps_per_video=max_episode_steps, video_fps=20)
    train_envs = ManiSkillVectorEnv(train_envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)

    space = train_envs.unwrapped.single_action_space
    assert isinstance(space, gym.spaces.Box)
    n_act = int(np.prod(space.shape))
    n_obs = (args.image_size, args.image_size, train_envs.unwrapped.single_observation_space['rgb'].shape[2])
    n_state = int(np.prod(train_envs.unwrapped.single_observation_space['state'].shape))
    act_scale = torch.as_tensor((space.high - space.low) / 2.0, dtype=torch.float32, device=device)
    act_bias = torch.as_tensor((space.high + space.low) / 2.0, dtype=torch.float32, device=device)

    logger = Logger(log_wandb=args.track)
    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), name=run_name,
                   group=args.wandb_group, tags=[args.wandb_group, args.agent_name, args.env_id, f"seed={args.seed}"])

    # ── Agent ──────────────────────────────────────────────────────────────
    cfg = QCConfig(horizon=args.horizon, actor_type=args.actor_type, actor_num_samples=args.actor_num_samples,
                   flow_steps=args.flow_steps, alpha=args.alpha, hidden_dim=args.hidden_dim,
                   actor_layers=args.num_layers, critic_hidden_dim=args.hidden_dim, critic_layers=args.num_layers,
                   num_q=args.num_q, q_agg=args.q_agg, gamma=args.gamma, tau=args.tau, lr=args.lr)
    agent = QCAgent(cfg, n_obs, n_state, n_act, device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        agent.load_state_dicts(ckpt)
        print(f"Loaded checkpoint {args.checkpoint}")
    executor = ChunkExecutor(agent, act_scale, act_bias)
    # evaluation acts deterministically, like Squint's get_eval_action (training rollouts stay stochastic)
    eval_executor = ChunkExecutor(agent, act_scale, act_bias, deterministic=True)

    # Demo frames are stored clean; online frames come out of ColorJitterWrapper already jittered.
    # Apply the same per-observation jitter to every demo batch so both sources match.
    demo_jitter = torchvision.transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)  # = utils.ColorJitterWrapper

    def sample_offline(n):
        b = offline.sample(n, cfg.horizon, cfg.gamma)
        if args.apply_jitter:
            for key in ("rgb", "next_rgb"):
                b[key] = color_jitter(b[key], demo_jitter)
        return b

    def save(step):
        if args.save_model:
            torch.save(dict(agent.state_dicts(), global_step=step, horizon=cfg.horizon), model_path)

    # ── Offline data + pretraining ─────────────────────────────────────────
    offline = None
    if args.demo_path:
        offline = load_h5_demos(args.demo_path, device, args.image_size, act_scale, act_bias,
                                bootstrap_at_done=args.bootstrap_at_done, max_trajs=args.max_demo_trajs)
        assert offline.rgb.shape[2:] == n_obs and offline.state.shape[-1] == n_state, (
            f"demo obs {tuple(offline.rgb.shape[2:])}/{offline.state.shape[-1]} do not match env {n_obs}/{n_state}")
    if args.offline_steps > 0 and not args.checkpoint:
        for step in tqdm.trange(args.offline_steps, desc="offline pretrain"):
            info = agent.update(sample_offline(args.batch_size))
            if step % 1000 == 0:
                logger.log({f"offline/{k}": v.item() for k, v in info.items()}, step=step)
            if args.eval_freq > 0 and step > 0 and step % (args.eval_freq // 10) == 0:
                eval_executor.reset()
                evaluate(args, eval_envs, eval_executor, logger, eval_output_dir, max_episode_steps, step, tqdm.tqdm(total=0))
        save(0)

    # ── Online ─────────────────────────────────────────────────────────────
    rb = ChunkBuffer(args.buffer_size // args.num_envs, args.num_envs, n_obs, n_state, n_act, device)
    n_off = int(round(args.batch_size * args.offline_ratio)) if offline is not None else 0
    n_on = args.batch_size - n_off

    def sample_batch():
        parts = []
        if n_on > 0:
            parts.append(rb.sample(n_on, cfg.horizon, cfg.gamma))
        if n_off > 0:
            parts.append(sample_offline(n_off))
        return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}

    log_off = args.offline_steps if not args.checkpoint else 0  # keep wandb steps monotonic across phases
    obs, _ = train_envs.reset(seed=args.seed)
    eval_envs.reset(seed=args.seed)
    executor.reset()
    global_step, d = 0, {}
    avg_returns = deque(maxlen=20)
    pbar = tqdm.tqdm(total=args.total_timesteps, desc="steps")

    for iteration in range(args.num_total_iterations + 2):
        if args.eval_freq > 0 and ((global_step - args.num_envs) // args.eval_freq) < (global_step // args.eval_freq):
            eval_executor.reset()
            evaluate(args, eval_envs, eval_executor, logger, eval_output_dir, max_episode_steps, log_off + global_step, pbar)
            save(log_off + global_step)

        with torch.no_grad():
            env_action = executor(obs['rgb'], obs['state'])  # normalised chunk -> env units
        norm_action = (env_action - act_bias) / act_scale

        next_obs, rewards, terminations, truncations, infos = train_envs.step(env_action)
        real_next = {'rgb': next_obs['rgb'].clone(), 'state': next_obs['state'].clone()}
        ep_end = terminations | truncations
        if args.bootstrap_at_done == 'never':
            dones = ep_end
        elif args.bootstrap_at_done == 'always':
            dones = torch.zeros_like(terminations)
        else:
            dones = terminations
        if "final_info" in infos:
            real_next['rgb'][ep_end] = infos["final_observation"]['rgb'][ep_end]
            real_next['state'][ep_end] = infos["final_observation"]['state'][ep_end]
        rb.add(obs['rgb'], obs['state'], real_next['rgb'], real_next['state'], norm_action, rewards, dones, ep_end)
        obs = next_obs
        if ep_end.any():
            executor.reset()  # a new episode started -> discard the remaining open-loop chunk

        if global_step > args.learning_starts and rb.size >= cfg.horizon:
            for _ in range(args.num_updates):
                info = agent.update(sample_batch())
            d.update({f"train_rl/{k}": v for k, v in info.items()})

        if "final_info" in infos:
            done_mask = infos["_final_info"]
            for k, v in infos["final_info"]["episode"].items():
                d[f"train/{k}"] = v[done_mask].float().mean()
            avg_returns.extend(infos["final_info"]["episode"]["return"][done_mask].tolist())
            sps = global_step / max(logger.wall_time, 1e-6)
            d["time/sps"] = sps
            pbar.set_description(f"{sps:.1f} sps, step={global_step}, return={np.mean(avg_returns):.2f}")
            logger.log(d={k: (v.item() if torch.is_tensor(v) else v) for k, v in d.items()}, step=log_off + global_step)

        pbar.update(args.num_envs)
        global_step += args.num_envs

    if args.save_model and os.path.exists(model_path):
        logger.upload_checkpoint(model_path, f"model_{args.agent_name}_{args.env_id}_{args.seed}")
    logger.close()
    try:
        train_envs.close(); eval_envs.close()
    except Exception:
        pass
