# Squint + Q-Chunking: long-horizon SO-101 manipulation

This repo extends [**Squint**](https://aalmuzairee.github.io/squint) (fast visual RL for sim-to-real on the SO-101 arm)
to **multi-stage tasks** (stacking, unstacking, packing, rearranging several cubes), which plain RL from scratch does
not solve. The recipe:

1. **Scripted demonstrations.** A motion-planning solver performs each task in simulation and records demos in exactly
   the observation/action format the policy trains with.
2. **QC-FQL training.** A flow-matching policy that outputs **chunks of 5 actions** (Q-chunking) is pretrained on the
   demos (offline), then improved with RL (online), mixing demo and online data 50/50.
3. **Sim-to-real.** The policy sees only a 16×16 wrist-camera image and the robot's joint state, like Squint, so it
   can be deployed on the real arm.

```
envs/ (tasks)  ──►  scripted solver  ──►  demos (.h5)  ──►  train_squint_qc.py  ──►  checkpoint  ──►  real robot
                    examples/motionplanning   verify_demos      offline → online RL
```

## Results so far

Success = all objects placed and the robot at rest at the end of the episode, measured in eval episodes on a fixed
seed. Full details of every run are in [`QC_EXPERIMENT_LOG.md`](QC_EXPERIMENT_LOG.md).

| Task | Best success | Run | Notes |
|---|---|---|---|
| `SO101Unstack3Cube-v1` | **0.89** (64 eps) | `unstack3_500_qmin` | 500 recovery demos, `--q_agg min`; holds 0.66–0.89 to the end |
| `SO101LiftCube-v1` | 0.94 (16 eps) | `lift_qc_dr200` | with domain randomization, for real-robot deployment |
| `SO101TrayPack1-v1` | 1.00 (16 eps) | `tp1_r3` | 1-cube tray packing |
| `SO101TrayPack3-v1` | 0.12 (16 eps) | `tp3_rec800` | ≥1 cube placed in up to 0.94 of episodes; 3rd cube is the bottleneck |
| `SO101StackCube-v1` | ≤ 0.06 | `stack_fast500` | learns to grasp (0.94) but not to stack |
| `SO101Tower3Cube-v1` | 0 | `tower3_500` | never learned to grasp; cause not found |

What has made the difference so far:
- **Recovery demos** (noisy actions + deliberate missed grasps, so the demos show corrections): TrayPack3 ≥1 cube
  went from 0.19 to 0.88.
- **`--q_agg min`** (pessimistic critic target): stopped the critic overestimating and the late collapse on Unstack3
  (mean eval 0.62 → 0.78).
- Long offline pretraining does **not** help: online RL does the work. 150–200k offline steps is plenty.

## Installation

Requirements: an NVIDIA GPU (≥ 10 GB; an RTX 3060 12 GB works), and for deployment an SO-101 arm with a wrist
camera.

```bash
conda env create -f environment.yaml
conda activate squint
```

## Quick start (Unstack3)

Run everything from the repo root. On a remote box, run long jobs inside `tmux`.

**1. Watch the scripted solver do the task**
```bash
ENV_ID=SO101Unstack3Cube-v1 SEEDS="0 1 2" FAST=1 bash examples/view_task.sh           # live SAPIEN viewer
VIDEO=1 ENV_ID=SO101Unstack3Cube-v1 FAST=1 bash examples/view_task.sh                 # mp4s in task_videos/
```

**2. Collect demos, verify them and train, in one go**
```bash
EXP_NAME=unstack3_500_qmin OFFLINE_STEPS=200000 \
EXTRA_TRAIN_ARGS="--q_agg min --num_eval_envs 64" bash examples/run_unstack3_qc.sh
```
This collects 500 recovery demos (`demos/qc/SO101Unstack3Cube-recovery500.h5`, ~5 min on 8 workers, skipped if the
file exists), checks them with `examples/verify_demos.py`, then trains (offline 200k steps, online 2M steps,
~2 h on an RTX 3060).

**3. Watch training**
```bash
tail -f logs/run_unstack3_500_qmin.out        # progress
grep EVAL_REW logs/unstack3_500_qmin.log      # every reward term, per eval
```
Plots in `runs/<run>/plots/` update every 2 minutes and after each eval: `summary.png` (success, return, stage
progress, losses, Q values) and `rewards.png` (every reward term). To plot or compare runs afterwards:
```bash
python -m examples.plot_metrics runs/unstack3_500 runs/unstack3_500_qmin --out runs/compare
```

**Outputs** (`runs/` and `logs/` are git-ignored): `runs/<run>/ckpt_best.pt` (best eval), `ckpt.pt` (latest),
`metrics.jsonl` (every logged value), `plots/`, `videos/` (eval rollouts).

## Doing the steps by hand

**Collect demos** (any task with a solver, see the task table):
```bash
python -m examples.collect_new_tasks_demos -e SO101Unstack3Cube-v1 -n 500 --workers 8 --fast \
    --action-noise 0.2 --miss-prob 0.3 --start-seed 1000 -o demos/qc/my_demos.h5
python -m examples.verify_demos demos/qc/my_demos.h5            # static checks + replay
OUT=demos/qc/my_demos.h5 bash examples/demo_videos.sh          # videos of the first 10 demos
```
- `--fast`: policy-like solver speed, needed when demos would otherwise exceed the episode limit.
- `--action-noise`, `--miss-prob`: recovery demos (recommended).
- `--domain-randomization`: record with randomized visuals and physics (needed for real-robot policies).

**Train**
```bash
python train_squint_qc.py --env_id SO101Unstack3Cube-v1 --demo_path demos/qc/my_demos.h5 \
    --offline_steps 200000 --total_timesteps 2000000 --gamma 0.99 --horizon 5 \
    --hidden_dim 512 --num_layers 4 --q_agg min --num_eval_envs 64 --eval_seed 100 \
    --no-env_domain_randomization --exp_name my_run
```
Flags that matter most:

| Flag | Recommended | Why |
|---|---|---|
| `--gamma` | 0.99 for tasks ≥ 150 steps | with 0.9 the later stages are invisible to the critic |
| `--horizon` | 5 | action-chunk length |
| `--q_agg` | `min` | pessimistic TD target; prevents Q overestimation (see the log) |
| `--offline_steps` | 150k–200k | more pretraining didn't help on any task |
| `--num_eval_envs` | 64 | 16 episodes are too noisy to pick the best checkpoint |
| `--eval_seed` | 100 | same eval layouts at every eval, so evals are comparable |
| `--hidden_dim`, `--num_layers` | 512, 4 | network size used by every long-task run |
| `--env_domain_randomization` | off in sim experiments, on for real-robot policies | |
| `--plot_every_sec` | 120 | live plot refresh; 0 = only at the end |

How the agent works (losses, chunk sampling, demo file format): [`QC_README.md`](QC_README.md).

## Tasks

All tasks use the SO-101 arm, a wrist camera, and the joint-delta controller at 10 Hz. "Solver" = a scripted solver
exists, so demos can be collected.

| Task ID | What the robot does | Steps | Solver | Status |
|---|---|---|---|---|
| `SO101ReachCube/Can-v1` | reach a target object | 50 | – | original Squint |
| `SO101LiftCube/Can-v1` | pick up and lift | 50 | Lift (`collect_qc_demos`) | original Squint; QC 0.94 |
| `SO101PlaceCube/Can-v1` | pick and place into a bin | 50 | – | original Squint |
| `SO101StackCube/Can-v1` | stack the cube on the larger cube / can | 50 | StackCube | original Squint; QC doesn't stack yet |
| `SO101Stack3Cube-v1` | build a 3-cube tower | 150 | yes (`collect_qc_demos`) | early task, superseded by Tower3 |
| `SO101Place3Cube-v1` | put 3 cubes into a bin, in a row | 250 | yes (`collect_qc_demos`) | early task |
| `SO101Tower2Cube/3Cube-v1` | size-ordered tower at a marked spot | 200 / 300 | yes | Tower3 didn't learn |
| `SO101TrayPack1/2/3-v1` | put 1/2/3 cubes into assigned tray compartments | 150 / 200 / 300 | yes | TrayPack1 solved; TrayPack3 0.12 |
| `SO101Rearrange2/3-v1` | swap cubes between pockets using a buffer pocket | 300 / 400 | yes | Rearrange3 dropped (solver 12%) |
| `SO101Unstack3Cube-v1` | take a 3-cube tower apart, top cube first | 150 | yes | **0.89** |

Task definitions: `envs/`. Specs for Tower / TrayPack / Rearrange: [`TASK_SPEC.md`](TASK_SPEC.md). Domain
randomization parameters are shared in [`envs/base_random_env.py`](envs/base_random_env.py) and per task in each
env file. To see the original 8 tasks: `python examples/visualize_sim.py`.

## Original Squint training (no demos)

The upstream SAC trainer is unchanged and still works for the 50-step tasks:
```bash
python train_squint.py --env_id=SO101LiftCube-v1                               # ~15 min for 1.5M steps
python train_squint.py --env_id=SO101LiftCube-v1 --track --wandb_entity=YOUR_WANDB_USERNAME
```
`results/` holds the upstream Squint training curves for the 8 original tasks.

## Deployment on the real SO-101

> `deploy.py` on this branch runs **Squint SAC checkpoints** (`train_squint.py`). Deploying **QC-FQL checkpoints**
> (`deploy_qc.py`, `examples/check_deploy_qc.py`) is on the `feat/deployment` branch. Use `--qc_exec_steps 1`
> (replan every step) when scaling actions down on the real arm.

**Prerequisites:** a working SO-101 arm, a mounted wrist camera, and motors calibrated with
[LeRobot calibration](https://huggingface.co/docs/lerobot/en/so101).

1. **(Optional) 3D-print the objects** in `deploy_utils/blender_stls/` (bin: white, can: blue, cube: red,
   large cube: blue), or change the object colours in the sim tasks to match yours.
2. **Configure the robot:** edit [`deploy_utils/robot_config.py`](deploy_utils/robot_config.py) (ports, camera).
3. **Match the background:** in simulation, everything except the robot and objects is replaced by a background
   image (`envs/black_overlay.png`). If your table looks different, photograph it and point `rgb_overlay_path` in
   [`envs/base_random_env.py`](envs/base_random_env.py) at the photo.
4. **Align the camera:** `python deploy_utils/tune_camera.py`. Move the trackbars until the gripper and base line
   up in sim and real, press `p`, and copy the printed parameters into the wrist camera settings in
   `envs/base_random_env.py`.
5. **Deploy:**
   ```bash
   python deploy.py --checkpoint=path/to/ckpt.pt --env_id=SO101LiftCube-v1
   python deploy.py --checkpoint=wandb --env_id=SO101LiftCube-v1 --wandb_entity=YOUR_WANDB_USERNAME
   ```
   Keys: `s` skips the episode, `q` quits.

Tips: start with `--no-continuous_eval` (asks before each step) and with a Reach task; use a well-lit room without
sunlight; good motor calibration and camera alignment matter most for transfer.

## Project structure

```
├── train_squint_qc.py        # main trainer: QC-FQL (demos → offline → online RL), live plots
├── qc_agent.py               # QC-FQL agent: CNN encoder, flow policy, one-step actor, Q-ensemble critic
├── qc_data.py                # action-chunk replay buffer + demo .h5 loader
├── train_squint.py           # original Squint SAC trainer (also provides Args, Logger, evaluate)
├── utils.py                  # env wrappers: image downsampling, colour jitter
├── deploy.py                 # real-robot deployment (Squint SAC checkpoints)
├── environment.yaml          # conda environment
├── envs/                     # ManiSkill tasks (see the task table)
│   ├── base_random_env.py    # shared cameras, domain randomization, background overlay
│   ├── reach.py lift.py place.py stack.py                   # original Squint tasks
│   ├── stack3.py place3.py tower.py tray_pack.py rearrange.py unstack3.py   # multi-stage tasks
│   └── robot/                # SO-101 / SO-100 model, controllers, grasp checks
├── examples/
│   ├── motionplanning/so101/ # scripted solver (motionplanner.py, collision.py) + one solution per task
│   ├── collect_new_tasks_demos.py   # main demo collector (parallel, recovery noise)
│   ├── collect_qc_demos.py   # collector for Lift/Stack/Stack3/Place3/Unstack3 that also records privileged state
│   ├── verify_demos.py       # demo checks (shapes, lengths, replay ends in success)
│   ├── replay_qc_demos.py, demo_videos.sh   # demo videos
│   ├── relabel_demo_rewards.py              # recompute demo rewards after a reward change
│   ├── collect_tp3_recovery.sh              # generic collect → verify → train pipeline (used by the run scripts)
│   ├── run_unstack3_qc.sh, run_two_task_qc.sh   # experiment scripts
│   ├── plot_metrics.py       # matplotlib plots from runs/<run>/metrics.jsonl
│   ├── view_task.py/.sh      # watch the solver (live or mp4)
│   ├── check_wrist_visibility.py, visualize_sim.py, viser_eval.py   # visual checks / viewers
├── deploy_utils/             # real-robot interface, robot config, camera tuning, 3D-print files
├── results/                  # upstream Squint training curves
├── validation/               # wrist-camera snapshots from the Sep 26 task validation
└── docs/                     # upstream Squint project website
```

## Documentation

| File | Contents |
|---|---|
| [`QC_README.md`](QC_README.md) | how the QC-FQL agent works: losses, chunk sampling, demo format |
| [`QC_EXPERIMENT_LOG.md`](QC_EXPERIMENT_LOG.md) | every experiment: settings, results, analysis, decisions |
| [`DEMOS.md`](DEMOS.md) | demo collection and the HDF5 format |
| [`TASK_SPEC.md`](TASK_SPEC.md) | specification of Tower, TrayPack and Rearrange |
| [`VALIDATION.md`](VALIDATION.md) | what was verified for those tasks (Sep 26 snapshot) |

## Acknowledgments and citation

Built on **Squint** by [Abdulaziz Almuzairee](https://aalmuzairee.github.io) and
[Henrik I. Christensen](https://hichristensen.com) (UC San Diego) ([website](https://aalmuzairee.github.io/squint),
[paper](https://arxiv.org/abs/2602.21203)). The QC-FQL agent is a PyTorch port of the official Q-chunking (`qc`)
JAX implementation. Squint itself builds on [LeanRL](https://github.com/meta-pytorch/LeanRL),
[CleanRL](https://github.com/vwxyzjn/cleanrl), [ManiSkill3](https://github.com/haosulab/ManiSkill),
[LeRobot Sim2Real ManiSkill3](https://github.com/StoneT2000/lerobot-sim2real),
[FastTD3](https://github.com/younggyoseo/FastTD3), [FastSAC](https://github.com/amazon-far/holosoma) and
[LeRobot](https://github.com/huggingface/lerobot); SO-101 support in ManiSkill3 was started by
[@jackvial](https://github.com/jackvial).

If you use the Squint code, please cite:
```bibtex
@article{almuzairee2026squint,
      title={Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics},
      author={Almuzairee, Abdulaziz and Christensen, Henrik I.},
      journal={arXiv preprint arXiv:2602.21203},
      year={2026}
}
```

## License

[MIT](LICENSE). Dependencies are subject to their own licenses.
