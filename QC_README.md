# Squint + Q-Chunking (flow policy)

Files: `qc_agent.py` (agent) · `qc_data.py` (HDF5 loader + chunk replay) · `train_squint_qc.py` (training script)

## Big picture

```
 demos.h5 ──► load_h5_demos ──► OFFLINE ChunkBuffer (E=1)
                                        │
        ┌───────────────────────────────┤
        ▼                               ▼
 PHASE 1: offline pretrain        PHASE 2: online RL
 --offline_steps gradient steps   (loop over env steps)
 batch = 100% demos                     │
        │                               ▼
        │                    ChunkExecutor: every `horizon` steps
        │                    ask the agent for a chunk, replay it
        │                    open-loop one action per env step
        │                               │
        │                               ▼
        │                    train_envs.step ──► online ChunkBuffer
        │                               │        (E = num_envs)
        │                               ▼
        │                    --num_updates gradient steps per env step
        │                    batch = offline_ratio demos
        │                          + (1-offline_ratio) online replay
        ▼                               │
   save ckpt  ◄────── eval every --eval_freq (Squint's `evaluate`) ──┘
```

## One gradient step (`QCAgent.update`)

```
batch of chunks: rgb, state, actions[h,A], rewards(h-step), masks, next_rgb/state
        │
        ├─► encoder(rgb) ──► f ─────────────┬──────────────────────────────┐
        │                                   │ (detached)                   │
        ▼                                   ▼                              │
 CRITIC (also trains encoder)         ACTOR (bc_flow + onestep)            │
 target = R_h + γ^h · mask ·          1. BC flow loss:                     │
   Q_target(s_{t+h}, π(s_{t+h}))         x_t=(1-t)·noise + t·chunk         │
 loss = (Q(s, chunk) - target)²          predict velocity (chunk - noise)  │
   skipped if critic_valid = 0        2. Distill: onestep(noise) ≈         │
 then soft-update target (tau)           Euler-integrated flow (flow_steps)│
                                      3. Q loss: -Q(s, onestep(noise))     │
                                      loss = BC + alpha·distill + Q loss ◄─┘
```

Acting: `onestep(noise)` → chunk of `horizon` actions in [-1,1] → `a*scale+bias` → env.
(`best-of-n`: draw n flow chunks, keep the one with highest Q, no one-step net.)

## Parameters

**Data / phases** (`train_squint_qc.py`)
| flag | default | meaning |
|---|---|---|
| `--demo_path` | None | HDF5 demos (required if offline_steps>0) |
| `--max_demo_trajs` | None | use only first N demos |
| `--offline_steps` | 50000 | pretraining gradient steps on demos |
| `--offline_ratio` | 0.5 | share of demos in each online batch (0 = online only) |
| `--num_updates` | 64 | gradient steps per env step (UTD) |
| `--learning_starts` | 1000 | env steps before online updates begin |
| `--batch_size` | 256 | chunks per update |
| `--bootstrap_at_done` | always | how `dones` are set (same as Squint) |

**Agent** (`QCConfig`)
| flag | default | meaning |
|---|---|---|
| `--horizon` | 5 | chunk length h (also the h in γ^h) |
| `--actor_type` | distill-ddpg | or `best-of-n` |
| `--actor_num_samples` | 32 | n for best-of-n |
| `--flow_steps` | 10 | Euler steps for the flow |
| `--alpha` | 100 | distillation/BC weight — tune per task |
| `--q_agg` | mean | ensemble aggregation for target (`mean`/`min`) |
| `--num_q` | 2 | critic ensemble size |
| `--hidden_dim` / `--num_layers` | 512 / 4 | MLP size (actor + critic) |
| `--gamma` / `--tau` / `--lr` | 0.99 / 0.005 / 3e-4 | discount, target rate, Adam lr |

Env/eval flags (`--env_id`, `--num_envs`, `--eval_freq`, `--image_size`, …) are inherited from `train_squint.py`.

## Chunk sampling rules (`ChunkBuffer.sample`)
- `valid[k]` = step k is in the same episode as step 0 → used to mask the BC loss.
- `rewards` = Σ γᵏ·rₖ over valid steps; `masks` = 0 if a terminal occurs in the chunk.
- `critic_valid` = 0 if the episode ends before the last chunk step → critic skips it.

## HDF5 schema (per episode group `traj_<i>`)
```
obs/rgb (T+1,H,W,C) uint8 · obs/state (T+1,S) · actions (T,A) · rewards (T,)
terminated (T,) optional · truncated (T,) optional
```
Actions in env units (normalised to [-1,1] on load). rgb is area-resized to `--image_size`.

## Run
```
python train_squint_qc.py --env_id SO101StackCube-v1 --demo_path demos.h5 --num_envs 256 --horizon 5
```
Not yet supported: `deploy.py` (it only loads Squint's SAC actor).
