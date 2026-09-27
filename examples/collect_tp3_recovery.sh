#!/usr/bin/env bash
# Collect TrayPack3 recovery demos (noisy executed actions + deliberate first-grasp misses), recorded
# with the v3 dense reward so train_squint_qc.py can use them directly (no relabelling). With TRAIN=1,
# a QC-FQL run on the new demos starts automatically once they pass the checks.
#
#   bash examples/collect_tp3_recovery.sh                     # 300 demos, 8 workers, collect only
#   TRAIN=1 bash examples/collect_tp3_recovery.sh             # collect, verify, then train
#   N=100 WORKERS=6 bash examples/collect_tp3_recovery.sh
#   TRAIN=1 SKIP_COLLECT=1 bash examples/collect_tp3_recovery.sh   # train on an existing $OUT
#
# Run from the repo root inside tmux. Collectors get oom_score_adj=1000, so if RAM runs out the kernel
# kills them rather than a training run. Use <=4 workers while another training run is going (15 GB RAM
# box); 16 workers has hung before.
set -euo pipefail

ENV_ID=${ENV_ID:-SO101TrayPack3-v1}
N=${N:-300}
WORKERS=${WORKERS:-8}
START_SEED=${START_SEED:-1000}
NOISE=${NOISE:-0.2}
MISS=${MISS:-0.3}
REWARD_VERSION=${REWARD_VERSION:-3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-$((N * 5))}  # ~40% of TrayPack3 recovery attempts succeed
OUT=${OUT:-demos/qc/SO101TrayPack3-recovery.h5}
LOG=${LOG:-$HOME/collect_tp3_recovery.log}
SKIP_COLLECT=${SKIP_COLLECT:-0}

# training (TRAIN=1); extra train_squint_qc.py flags can be appended with TRAIN_ARGS="..."
TRAIN=${TRAIN:-0}
EXP_NAME=${EXP_NAME:-tp3_recovery}
OFFLINE_STEPS=${OFFLINE_STEPS:-150000}
HIDDEN_DIM=${HIDDEN_DIM:-512}
NUM_LAYERS=${NUM_LAYERS:-4}
TRAIN_ARGS=${TRAIN_ARGS:-}

cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

if [ "$SKIP_COLLECT" != 1 ]; then
    if [ -e "$OUT" ]; then
        echo "$OUT already exists; move it away, set OUT=..., or use SKIP_COLLECT=1" >&2
        exit 1
    fi
    echo "collecting $N $ENV_ID demos (noise $NOISE, miss $MISS, reward v$REWARD_VERSION) on $WORKERS workers -> $OUT"
    # subshell: only the collectors get the high OOM score, not the training run below
    (
        echo 1000 > /proc/self/oom_score_adj
        exec python -m examples.collect_new_tasks_demos -e "$ENV_ID" -n "$N" --workers "$WORKERS" \
            --start-seed "$START_SEED" --max-attempts "$MAX_ATTEMPTS" --reward-version "$REWARD_VERSION" \
            --action-noise "$NOISE" --miss-prob "$MISS" -o "$OUT"
    ) 2>&1 | tee "$LOG"
fi

python -m examples.verify_demos "$OUT" --no-replay 2>&1 | tail -5 | tee -a "$LOG"

if [ "$TRAIN" = 1 ]; then
    echo "training $EXP_NAME on $OUT (log: $HOME/$EXP_NAME.log)"
    # shellcheck disable=SC2086
    python train_squint_qc.py --env_id "$ENV_ID" --demo_path "$OUT" --reward_version "$REWARD_VERSION" \
        --gamma 0.99 --no-env_domain_randomization --horizon 5 \
        --offline_steps "$OFFLINE_STEPS" --hidden_dim "$HIDDEN_DIM" --num_layers "$NUM_LAYERS" \
        --exp_name "$EXP_NAME" $TRAIN_ARGS 2>&1 | tee "$HOME/$EXP_NAME.log"
fi
