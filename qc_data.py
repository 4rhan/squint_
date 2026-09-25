"""Chunk-aware replay storage + HDF5 demo loading for Q-chunking.

`ChunkBuffer` stores transitions as [T, E, ...] (time-major, E parallel streams) so an action chunk is h
consecutive time-steps of one stream. The same class holds the offline HDF5 dataset (E=1, one long stream
with episode-end flags) and the online replay (E=num_envs, ring buffer over T).

HDF5 schema expected by `load_h5_demos` (one group per episode; examples/collect_qc_demos.py writes it
directly from an env built exactly like the training env):

    traj_<i>/obs/rgb      (T+1, H, W, C) uint8    same camera + channel layout the env produces
    traj_<i>/obs/state    (T+1, S)       float    same layout as the env's flattened 'state' obs
    traj_<i>/actions      (T, A)         float    in the env's action space (the training control mode,
                                                    by default normalised pd_joint_target_delta_pos in [-1, 1])
    traj_<i>/rewards      (T,)           float    same reward mode as training (normalized_dense)
    traj_<i>/terminated   (T,)           bool     (optional; ignored unless use_terminations, see load_h5_demos)
    traj_<i>/truncated    (T,)           bool     (optional, default all False)
"""
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F


class ChunkBuffer:
    def __init__(self, capacity_t, num_streams, rgb_shape, state_dim, act_dim, device):
        self.cap, self.E, self.device = capacity_t, num_streams, device
        kw = dict(device=device)
        shape = (capacity_t, num_streams)
        self.rgb = torch.zeros(*shape, *rgb_shape, dtype=torch.uint8, **kw)
        self.next_rgb = torch.zeros_like(self.rgb)
        self.state = torch.zeros(*shape, state_dim, **kw)
        self.next_state = torch.zeros_like(self.state)
        self.actions = torch.zeros(*shape, act_dim, **kw)  # normalised to [-1, 1]
        self.rewards = torch.zeros(*shape, **kw)
        self.dones = torch.zeros(*shape, **kw)  # 1 -> do not bootstrap past this step
        self.ep_end = torch.zeros(*shape, **kw)  # 1 -> last step of an episode (terminated or truncated)
        self.ptr = 0
        self.size = 0

    def extend(self, rgb, state, next_rgb, next_state, actions, rewards, dones, ep_end):
        """Append a block of k time-steps; every tensor is [k, E, ...]."""
        k = rgb.shape[0]
        idx = (self.ptr + torch.arange(k, device=self.device)) % self.cap
        self.rgb[idx], self.next_rgb[idx] = rgb, next_rgb
        self.state[idx], self.next_state[idx] = state.float(), next_state.float()
        self.actions[idx], self.rewards[idx] = actions.float(), rewards.float()
        self.dones[idx], self.ep_end[idx] = dones.float(), ep_end.float()
        self.ptr = (self.ptr + k) % self.cap
        self.size = min(self.size + k, self.cap)

    def add(self, rgb, state, next_rgb, next_state, actions, rewards, dones, ep_end):
        """Append one env step (every tensor is [E, ...])."""
        self.extend(*(x.unsqueeze(0) for x in (rgb, state, next_rgb, next_state, actions, rewards, dones, ep_end)))

    def sample(self, batch_size, horizon, gamma):
        """Sample `batch_size` chunks of `horizon` steps.

        Returned dict (B = batch_size, h = horizon):
          rgb/state         obs at the first step of the chunk
          actions [B,h,A]   the chunk (zero-padded past an episode end)
          valid   [B,h]     1 for chunk steps that are in the same episode as step 0 (used for the BC loss)
          rewards [B]       sum_k gamma^k r_k over valid steps (h-step return)
          masks   [B]       0 if a terminal was hit inside the chunk (no bootstrapping), else 1
          next_*            obs after the last valid step (bootstrap state)
          critic_valid [B]  0 if the chunk ends its episode *before* its last step; such chunks have no
                            well-defined (obs, h-action-chunk) -> h-step-return pair, so the TD loss skips them
        """
        assert self.size >= horizon, "buffer holds fewer steps than one chunk"
        dev = self.device
        # start offsets over the currently valid window (oldest .. newest-h+1); avoids crossing the write pointer
        oldest = (self.ptr - self.size) % self.cap
        t0 = (oldest + torch.randint(0, self.size - horizon + 1, (batch_size,), device=dev)) % self.cap
        e = torch.randint(0, self.E, (batch_size,), device=dev)
        idx = (t0[:, None] + torch.arange(horizon, device=dev)[None]) % self.cap  # [B,h]
        ee = e[:, None]

        ep_end = self.ep_end[idx, ee]
        ended_before = torch.cumsum(ep_end, 1) - ep_end  # ended strictly before step k
        valid = (ended_before == 0).float()
        actions = self.actions[idx, ee] * valid.unsqueeze(-1)
        dones = self.dones[idx, ee] * valid
        disc = gamma ** torch.arange(horizon, device=dev, dtype=torch.float32)
        rewards = (self.rewards[idx, ee] * valid * disc).sum(1)
        masks = 1.0 - dones.max(1).values
        last = (valid.sum(1) - 1).long()
        last_idx = idx[torch.arange(batch_size, device=dev), last]
        # ended before the final slot <=> some valid step other than the last one carries ep_end
        critic_valid = (ep_end[:, :-1].sum(1) == 0).float() if horizon > 1 else torch.ones(batch_size, device=dev)
        return dict(
            rgb=self.rgb[t0, e], state=self.state[t0, e], actions=actions, valid=valid, rewards=rewards,
            masks=masks, next_rgb=self.next_rgb[last_idx, e], next_state=self.next_state[last_idx, e],
            critic_valid=critic_valid,
        )


def color_jitter(rgb, jitter):
    """Apply a torchvision ColorJitter to a [B, H, W, C] uint8 batch (same as utils.ColorJitterWrapper)."""
    x = jitter(rgb.permute(0, 3, 1, 2).float() / 255.0)
    return (x.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8)


def _area_resize(rgb, size):
    if rgb.shape[1] == size and rgb.shape[2] == size:
        return rgb
    x = rgb.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="area")
    return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def load_h5_demos(path, device, image_size, action_scale, action_bias, bootstrap_at_done="always",
                  max_trajs: Optional[int] = None, use_terminations: bool = False) -> ChunkBuffer:
    """Load a ManiSkill-style .h5 demo file into a ChunkBuffer (E=1). See module docstring for the schema.

    Actions are mapped to [-1,1] via (a - bias) / scale, where scale/bias come from the env's action space.
    `bootstrap_at_done` mirrors train_squint's flag so demo `dones` follow the same convention as online data.
    `use_terminations` must match how the online env treats terminations: with the default
    ManiSkillVectorEnv(ignore_terminations=True) (partial_reset=False) success never ends an episode, so a
    demo's `terminated` flags (often set on the last several steps, while success holds) are ignored and
    the demo ends only at its last step. Honouring them would split each demo's tail into 1-step episodes.
    """
    scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
    bias = torch.as_tensor(action_bias, dtype=torch.float32, device=device)
    cols = {k: [] for k in ("rgb", "state", "next_rgb", "next_state", "actions", "rewards", "dones", "ep_end")}
    n_clipped = n_total = 0
    with h5py.File(path, "r") as f:
        keys = sorted((k for k in f.keys() if k.startswith("traj_")), key=lambda k: int(k.split("_")[1]))
        if max_trajs is not None:
            keys = keys[:max_trajs]
        assert keys, f"no traj_* groups in {path}"
        for k in keys:
            g = f[k]
            for need in ("obs/rgb", "obs/state", "actions", "rewards"):
                if need not in g:
                    raise KeyError(f"{path}:{k} is missing '{need}' (see qc_data.py docstring for the schema)")
            rgb = _area_resize(torch.as_tensor(g["obs/rgb"][:], device=device), image_size)
            state = torch.as_tensor(g["obs/state"][:], dtype=torch.float32, device=device)
            act = torch.as_tensor(g["actions"][:], dtype=torch.float32, device=device)
            T = act.shape[0]
            assert rgb.shape[0] == T + 1 and state.shape[0] == T + 1, f"{k}: obs must have T+1 entries"
            rew = torch.as_tensor(g["rewards"][:], dtype=torch.float32, device=device)
            term = torch.as_tensor(g["terminated"][:], device=device).bool() if "terminated" in g else torch.zeros(T, dtype=torch.bool, device=device)
            trunc = torch.as_tensor(g["truncated"][:], device=device).bool() if "truncated" in g else torch.zeros(T, dtype=torch.bool, device=device)
            if not use_terminations:
                term = torch.zeros_like(term)
            ep_end = torch.zeros(T, dtype=torch.bool, device=device)
            ep_end[-1] = True  # a demo always ends its episode at its last action
            ep_end |= term | trunc
            if bootstrap_at_done == "never":
                dones = term | trunc
            elif bootstrap_at_done == "always":
                dones = torch.zeros_like(term)
            else:
                dones = term
            a_norm = (act - bias) / scale
            n_clipped += (a_norm.abs() > 1).sum().item()
            n_total += a_norm.numel()
            cols["rgb"].append(rgb[:-1]); cols["next_rgb"].append(rgb[1:])
            cols["state"].append(state[:-1]); cols["next_state"].append(state[1:])
            cols["actions"].append(a_norm.clamp(-1, 1)); cols["rewards"].append(rew)
            cols["dones"].append(dones); cols["ep_end"].append(ep_end)
    cat = {k: torch.cat(v, 0).unsqueeze(1) for k, v in cols.items()}  # [N, 1, ...]
    N = cat["rgb"].shape[0]
    buf = ChunkBuffer(N, 1, cat["rgb"].shape[2:], cat["state"].shape[-1], cat["actions"].shape[-1], device)
    buf.extend(**cat)
    print(f"Loaded {len(keys)} demo trajectories ({N} steps) from {path}; "
          f"{100 * n_clipped / max(n_total, 1):.2f}% of action entries fell outside the env action range and were clipped")
    return buf
