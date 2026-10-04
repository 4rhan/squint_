#!/usr/bin/env bash
# Rearrange2 QC-FQL protocol (2 cubes, 3 pockets: swap A,B using the buffer pocket), same settings as the
# two-task run: recovery demos (noise 0.2, miss 0.3), 512x4 networks, gamma 0.99, h=5, 16x16 wrist images,
# 16 eval episodes at a fixed eval seed. Rearrange has no reward versions (REWARD_VERSION=none).
# One job at a time: 1) collect demos (skipped if the file exists), 2) train. Stops if a step fails.
#
#   bash examples/run_rearrange2_qc.sh                      # from the repo root, inside tmux
#   N_DEMOS=300 ONLINE_STEPS=3000000 bash examples/run_rearrange2_qc.sh
#
# Logs go to logs/ (git-ignored): logs/collect_rearr2_recovery<N>.log, logs/<exp_name>.log,
# logs/run_rearrange2_qc.out for this script's own output. Runs go to runs/<exp_name>/.
set -uo pipefail
export PYTHONNOUSERSITE=1  # see collect_tp3_recovery.sh: a torch in ~/.local breaks the env's torchvision

N_DEMOS=${N_DEMOS:-500}
OFFLINE_STEPS=${OFFLINE_STEPS:-250000}
ONLINE_STEPS=${ONLINE_STEPS:-2000000}
EVAL_SEED=${EVAL_SEED:-100}
WORKERS=${WORKERS:-8}                  # nothing else runs during collection
NOISE=${NOISE:-0.2}
MISS=${MISS:-0.3}
DEMOS=${DEMOS:-demos/qc/SO101Rearrange2-recovery${N_DEMOS}.h5}
EXP=${EXP:-rearr2_${N_DEMOS}}

cd "$(dirname "$0")/.."
mkdir -p logs
COMMON="--total_timesteps $ONLINE_STEPS --max_demo_trajs $N_DEMOS --eval_seed $EVAL_SEED"
say() { echo "[$(date +%F\ %T)] $*"; }
say "start: $N_DEMOS demos, $OFFLINE_STEPS offline, $ONLINE_STEPS online, eval seed $EVAL_SEED" \
    | tee -a logs/run_rearrange2_qc.out

# 1) demos
if [ -e "$DEMOS" ]; then
    say "step 1: $DEMOS exists, skipping collection" | tee -a logs/run_rearrange2_qc.out
else
    say "step 1: collecting $N_DEMOS Rearrange2 demos on $WORKERS workers -> $DEMOS" | tee -a logs/run_rearrange2_qc.out
    N="$N_DEMOS" WORKERS="$WORKERS" ENV_ID=SO101Rearrange2-v1 REWARD_VERSION=none START_SEED=1000 \
        NOISE="$NOISE" MISS="$MISS" MAX_ATTEMPTS=$((N_DEMOS * 6)) \
        OUT="$DEMOS" LOG="logs/collect_rearr2_recovery$N_DEMOS.log" \
        bash examples/collect_tp3_recovery.sh || { say "step 1 failed; stopping" | tee -a logs/run_rearrange2_qc.out; exit 1; }
    say "step 1 done" | tee -a logs/run_rearrange2_qc.out
fi

# 2) training
say "step 2: training $EXP" | tee -a logs/run_rearrange2_qc.out
TRAIN=1 SKIP_COLLECT=1 ENV_ID=SO101Rearrange2-v1 REWARD_VERSION=none OUT="$DEMOS" \
    LOG="logs/verify_$EXP.log" EXP_NAME="$EXP" OFFLINE_STEPS="$OFFLINE_STEPS" TRAIN_ARGS="$COMMON" \
    bash examples/collect_tp3_recovery.sh > "logs/run_$EXP.out" 2>&1 \
    || { say "step 2 failed" | tee -a logs/run_rearrange2_qc.out; exit 1; }
say "step 2 done; all finished" | tee -a logs/run_rearrange2_qc.out
