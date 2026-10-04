#!/usr/bin/env bash
# QC-FQL on SO101Unstack3Cube-v1 (take a 3-cube tower apart: top cube to the table, then the middle one).
# Same protocol as run_two_task_qc.sh: recovery demos (noise 0.2, miss 0.3, FAST solver, no DR), 250k offline
# steps, 2M online steps, 512x4 networks, gamma 0.99, h=5, 16x16 wrist images, fixed eval seed.
# 1) collect the demos (skipped if the file exists), 2) verify them, 3) train. Stops if a step fails.
#
#   bash examples/run_unstack3_qc.sh                        # from the repo root, inside tmux
#   N_DEMOS=300 ONLINE_STEPS=3000000 bash examples/run_unstack3_qc.sh
#   EXP_NAME=unstack3_500_qmin EXTRA_TRAIN_ARGS="--q_agg min --num_eval_envs 64" bash examples/run_unstack3_qc.sh
#
# Logs: logs/collect_unstack3_recovery<N>.log, logs/<exp_name>.log, logs/run_<exp_name>.out. Run: runs/<exp_name>/.
set -uo pipefail
export PYTHONNOUSERSITE=1  # see collect_tp3_recovery.sh: a torch in ~/.local breaks the env's torchvision

N_DEMOS=${N_DEMOS:-500}
OFFLINE_STEPS=${OFFLINE_STEPS:-250000}
ONLINE_STEPS=${ONLINE_STEPS:-2000000}
EVAL_SEED=${EVAL_SEED:-100}
WORKERS=${WORKERS:-8}                  # nothing else should run during collection
DEMOS=${DEMOS:-demos/qc/SO101Unstack3Cube-recovery${N_DEMOS}.h5}
EXP_NAME=${EXP_NAME:-unstack3_${N_DEMOS}}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}  # more train_squint_qc.py flags, e.g. "--q_agg min --num_eval_envs 64"

cd "$(dirname "$0")/.."
mkdir -p logs
say() { echo "[$(date +%F\ %T)] $*"; }
say "start: $N_DEMOS demos -> $DEMOS, $OFFLINE_STEPS offline, $ONLINE_STEPS online, eval seed $EVAL_SEED, run $EXP_NAME"

if [ -e "$DEMOS" ]; then SKIP=1; say "demos exist, skipping collection"; else SKIP=0; fi
# ~50% of recovery attempts succeed (reach limits, noisy places, too long), so allow 4x attempts
TRAIN=1 SKIP_COLLECT="$SKIP" ENV_ID=SO101Unstack3Cube-v1 REWARD_VERSION=none FAST=1 \
    N="$N_DEMOS" WORKERS="$WORKERS" START_SEED=1000 MAX_ATTEMPTS=$((N_DEMOS * 4)) OUT="$DEMOS" \
    LOG="logs/collect_unstack3_recovery$N_DEMOS.log" EXP_NAME="$EXP_NAME" OFFLINE_STEPS="$OFFLINE_STEPS" \
    TRAIN_ARGS="--total_timesteps $ONLINE_STEPS --max_demo_trajs $N_DEMOS --eval_seed $EVAL_SEED $EXTRA_TRAIN_ARGS" \
    bash examples/collect_tp3_recovery.sh 2>&1 | tee "logs/run_$EXP_NAME.out"
status=${PIPESTATUS[0]}
say "finished with exit code $status"
exit "$status"
