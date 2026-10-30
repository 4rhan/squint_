# Training Hyperparameters

This document summarizes the hyperparameters for the two training scripts in this repository: QC-FQL (`train_squint_qc.py`) and the Squint baseline (`train_squint.py`). Values are split into documented experiment settings and code defaults because some recommended run settings differ from the CLI defaults. A command-line argument or task-specific run can override either.

## QC-FQL

QC-FQL pretrains on demonstration chunks, then trains online with a mixture of demonstration and replay data.

### Documented multi-stage run settings

| Hyperparameter | Setting | Notes |
|---|---:|---|
| Action chunk length (`horizon`) | 5 | Actions per chunk |
| Discount (`gamma`) | 0.99 | Used for tasks with at least 150-step episodes; one TrayPack3 run used 0.995 |
| Actor and critic MLPs | 512 units × 4 layers | Earlier runs used 256 × 3 |
| Critic ensemble | 2 Q heads | Aggregation was `mean` in early runs; `min` is the recommended setting |
| Optimizer | Adam | |
| Learning rate (`lr`) | 3e-4 | Shared by QC actor/critic optimizers |
| Batch size | 512 chunks | |
| Target smoothing (`tau`) | 0.01 | Polyak update coefficient |
| Distillation weight (`alpha`) | 100 | Weight on flow distillation loss |
| Flow integration steps | 10 | Euler integration |
| Online updates per environment step | 256 | Update-to-data ratio |
| Actor update frequency | Every 4 critic updates | |
| Online demo fraction (`offline_ratio`) | 0.5 | Half demos, half online replay |
| Learning starts | 5,000 environment steps | |
| Offline pretraining | 150k–300k gradient steps | 200k for Unstack3 |
| Online training | 1.5M–5M environment steps | 2M for Unstack3 |

### QC-FQL code defaults

These are the defaults in `QCArgs` and `QCConfig`, before command-line or task-specific overrides:

| Hyperparameter | Default |
|---|---:|
| Action chunk length (`horizon`) | 5 |
| Actor type (`actor_type`) | `distill-ddpg` |
| Best-of-N samples (`actor_num_samples`) | 32 |
| Flow integration steps (`flow_steps`) | 10 |
| Distillation weight (`alpha`) | 100 |
| Actor network (`hidden_dim`, `num_layers`) | 256 units × 3 layers |
| Critic network (`critic_hidden_dim`, `critic_layers`) | 256 units × 3 layers |
| Critic ensemble (`num_q`) | 2 |
| Q aggregation (`q_agg`) | `mean` |
| Discount (`gamma`) | 0.9 |
| Target smoothing (`tau`) | 0.01 |
| Learning rate (`lr`) | 3e-4 |
| Batch size (`batch_size`) | 512 |
| Online updates per environment step (`num_updates`) | 256 |
| Actor update frequency (`policy_frequency`) | 4 |
| Learning starts (`learning_starts`) | 5,000 environment steps |
| Offline steps (`offline_steps`) | 50,000 gradient steps |
| Online demo fraction (`offline_ratio`) | 0.5 |
| Total online steps (`total_timesteps`) | 1,500,000 |
| Replay capacity (`buffer_size`) | 1,000,000 |
| BC loss trains shared image encoder (`bc_encoder_grad`) | Yes |

For longer tasks, the README recommends setting `gamma` to 0.99. The longer-task experiment configuration uses 512 × 4 actor/critic networks and `q_agg=min`; these are not the script defaults.

## Squint baseline

Defaults from `Args` in `train_squint.py`:

| Hyperparameter | Default |
|---|---:|
| Total environment steps (`total_timesteps`) | 1,500,000 |
| Parallel training environments (`num_envs`) | 1,024 |
| Replay capacity (`buffer_size`) | 1,000,000 |
| Batch size (`batch_size`) | 512 |
| Updates per environment step (`num_updates`) | 256 |
| Learning starts (`learning_starts`) | 5,000 environment steps |
| Policy learning rate (`policy_lr`) | 3e-4 |
| Q learning rate (`q_lr`) | 3e-4 |
| Entropy coefficient learning rate (`alpha_lr`) | 3e-4 |
| Policy update frequency (`policy_frequency`) | 4 updates |
| Target update frequency (`target_network_frequency`) | 1 update |
| Discount (`gamma`) | 0.9 |
| Target smoothing (`tau`) | 0.01 |
| Entropy coefficient (`alpha`) | 0.2 |
| Automatic entropy tuning (`autotune`) | Enabled |
| Number of Q networks (`num_q`) | 2 |
| Distributional critic atoms (`num_atoms`) | 101 |
| Distributional value support (`v_min`, `v_max`) | -20, 20 |
| Image input size (`image_size`) | 16 × 16 |

## Where these values come from

- QC-FQL run settings: `paper/PAPER_CONTEXT.md`, README recommendations, and `QC_EXPERIMENT_LOG.md`.
- QC-FQL CLI defaults: `train_squint_qc.py` (`QCArgs`) and `qc_agent.py` (`QCConfig`).
- Squint baseline defaults: `train_squint.py` (`Args`).

Run-specific overrides in commands, scripts, and experiment logs take precedence over the defaults above.
