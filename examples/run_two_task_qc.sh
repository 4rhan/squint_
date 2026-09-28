#!/usr/bin/env bash
# Two-task QC-FQL protocol (TrayPack3 + Tower3), same settings for both:
#   500 recovery demos (noise 0.2, miss 0.3), 250k offline steps, 2M online steps, 16 eval episodes at a fixed eval
#   seed (same layouts at every eval), 512x4 networks, gamma 0.99, h=5, 16x16 wrist images.
#   TrayPack3: reward v3, first 500 of demos/qc/SO101TrayPack3-recovery800.h5 (no new collection).
#   Tower3:    reward v2, 500 new demos.
# Strictly one job at a time (a second GPU sim or a RAM-hungry collector next to a training run has crashed it):
#   1) collect the Tower3 demos (skipped if the file exists), 2) train TrayPack3, 3) train Tower3.
# Stops if a step fails.
#
#   bash examples/run_two_task_qc.sh                        # from the repo root, inside tmux
#   ONLINE_STEPS=3000000 EVAL_SEED=7 bash examples/run_two_task_qc.sh
#
# Logs go to logs/ (git-ignored): logs/collect_*.log for collection, logs/<exp_name>.log for training,
# logs/run_two_task_qc.out for this script's own output. Runs go to runs/<exp_name>/.
set -uo pipefail
export PYTHONNOUSERSITE=1  # see collect_tp3_recovery.sh: a torch in ~/.local breaks the env's torchvision

N_DEMOS=${N_DEMOS:-500}
OFFLINE_STEPS=${OFFLINE_STEPS:-250000}
ONLINE_STEPS=${ONLINE_STEPS:-2000000}
EVAL_SEED=${EVAL_SEED:-100}            # seeds 1 and 2 give identical GPU layouts; 100 also differs from the train seed
TOWER_WORKERS=${TOWER_WORKERS:-8}      # nothing else runs during collection
TP3_DEMOS=${TP3_DEMOS:-demos/qc/SO101TrayPack3-recovery800.h5}
TOWER_DEMOS=${TOWER_DEMOS:-demos/qc/SO101Tower3Cube-recovery${N_DEMOS}.h5}
TP3_EXP=${TP3_EXP:-tp3_${N_DEMOS}}
TOWER_EXP=${TOWER_EXP:-tower3_${N_DEMOS}}

cd "$(dirname "$0")/.."
mkdir -p logs
COMMON="--total_timesteps $ONLINE_STEPS --max_demo_trajs $N_DEMOS --eval_seed $EVAL_SEED"
say() { echo "[$(date +%F\ %T)] $*"; }
say "start: $N_DEMOS demos, $OFFLINE_STEPS offline, $ONLINE_STEPS online, eval seed $EVAL_SEED"

# 1) Tower3 demos
if [ -e "$TOWER_DEMOS" ]; then
    say "step 1: $TOWER_DEMOS exists, skipping collection"
else
    say "step 1: collecting $N_DEMOS Tower3 demos on $TOWER_WORKERS workers -> $TOWER_DEMOS"
    N="$N_DEMOS" WORKERS="$TOWER_WORKERS" ENV_ID=SO101Tower3Cube-v1 REWARD_VERSION=2 START_SEED=1000 \
        OUT="$TOWER_DEMOS" LOG="logs/collect_tower3_recovery$N_DEMOS.log" \
        bash examples/collect_tp3_recovery.sh || { say "step 1 failed; stopping"; exit 1; }
    say "step 1 done"
fi

# 2) TrayPack3 training
say "step 2: training $TP3_EXP"
TRAIN=1 SKIP_COLLECT=1 ENV_ID=SO101TrayPack3-v1 REWARD_VERSION=3 OUT="$TP3_DEMOS" \
    LOG="logs/verify_$TP3_EXP.log" EXP_NAME="$TP3_EXP" OFFLINE_STEPS="$OFFLINE_STEPS" TRAIN_ARGS="$COMMON" \
    bash examples/collect_tp3_recovery.sh > "logs/run_$TP3_EXP.out" 2>&1 || { say "step 2 failed; stopping"; exit 1; }
say "step 2 done"

# 3) Tower3 training
say "step 3: training $TOWER_EXP"
TRAIN=1 SKIP_COLLECT=1 ENV_ID=SO101Tower3Cube-v1 REWARD_VERSION=2 OUT="$TOWER_DEMOS" \
    LOG="logs/verify_$TOWER_EXP.log" EXP_NAME="$TOWER_EXP" OFFLINE_STEPS="$OFFLINE_STEPS" TRAIN_ARGS="$COMMON" \
    bash examples/collect_tp3_recovery.sh > "logs/run_$TOWER_EXP.out" 2>&1 || { say "step 3 failed"; exit 1; }
say "step 3 done; all finished"
