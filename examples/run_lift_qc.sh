#!/usr/bin/env bash
# QC-FQL on SO101LiftCube-v1 for real-robot deployment (deploy_qc.py).
# Strictly one job at a time: 1) collect scripted demos (skipped if $DEMOS exists), 2) check them, 3) train.
#
# Unlike the TrayPack/Tower/StackCube runs, domain randomization is ON everywhere, as in Squint's sim2real
# setup: demos are recorded in randomized envs (camera pose/FOV noise, 5 deg joint-reading noise, cube size
# 22-28 mm, friction 0.1-0.5) and training/eval envs use the same randomization plus color jitter. The policy
# sees only the 16x16 wrist image + 12-dim proprio, which is exactly what deploy_qc.py feeds it on the robot.
#
#   bash examples/run_lift_qc.sh                         # from the repo root, inside tmux
#   N_DEMOS=100 ONLINE_STEPS=1000000 bash examples/run_lift_qc.sh
#   TRAIN=0 bash examples/run_lift_qc.sh                 # collect + check only
#
# Logs go to logs/ (git-ignored): logs/collect_$EXP_NAME.log, logs/$EXP_NAME.log. Run dir: runs/$EXP_NAME/
# (ckpt.pt = latest, ckpt_best.pt = best eval; either one works with deploy_qc.py).
set -euo pipefail
export PYTHONNOUSERSITE=1  # see collect_tp3_recovery.sh: a torch in ~/.local breaks the env's torchvision

ENV_ID=SO101LiftCube-v1
N_DEMOS=${N_DEMOS:-200}
WORKERS=${WORKERS:-8}
START_SEED=${START_SEED:-3000}          # StackCube used 1000+/2000+; keeps demo layouts distinct
MAX_ATTEMPTS=${MAX_ATTEMPTS:-$((N_DEMOS * 3))}  # local test: ~80% of randomized attempts saved
NOISE=${NOISE:-0}                        # recovery demos (e.g. 0.2 / 0.3) may not fit the 50-step limit
MISS=${MISS:-0}
# "table-black": envs/lift_overlay.png background + black cube. Earlier runs: lift_qc_dr200 (black background,
# red cube), lift_qc_table_dr200 (table background, red cube).
DEMOS=${DEMOS:-demos/qc/SO101LiftCube-table-black-dr-fast${N_DEMOS}.h5}

TRAIN=${TRAIN:-1}
EXP_NAME=${EXP_NAME:-lift_qc_table_black_dr${N_DEMOS}}
OFFLINE_STEPS=${OFFLINE_STEPS:-50000}    # stack_fast500: pretraining past ~100k made offline evals worse
ONLINE_STEPS=${ONLINE_STEPS:-1500000}    # Squint's default budget for Lift
GAMMA=${GAMMA:-0.9}                      # Squint's value for the 50-step tasks
HIDDEN_DIM=${HIDDEN_DIM:-512}
NUM_LAYERS=${NUM_LAYERS:-4}
EVAL_SEED=${EVAL_SEED:-100}
TRAIN_ARGS=${TRAIN_ARGS:-}

cd "$(dirname "$0")/.."
mkdir -p logs "$(dirname "$DEMOS")"
say() { echo "[$(date +%F\ %T)] $*"; }

# 1) demos: --fast (policy-like speed) so they fit the 50-step limit; too-long attempts are dropped
if [ -e "$DEMOS" ]; then
    say "step 1: $DEMOS exists, skipping collection"
else
    say "step 1: collecting $N_DEMOS $ENV_ID demos (DR on, fast, noise $NOISE, miss $MISS) on $WORKERS workers -> $DEMOS"
    (
        echo 1000 > /proc/self/oom_score_adj  # if RAM runs out, the kernel kills collectors first
        exec python -m examples.collect_new_tasks_demos -e "$ENV_ID" -n "$N_DEMOS" --workers "$WORKERS" \
            --start-seed "$START_SEED" --max-attempts "$MAX_ATTEMPTS" --domain-randomization --fast \
            --action-noise "$NOISE" --miss-prob "$MISS" -o "$DEMOS"
    ) 2>&1 | tee "logs/collect_$EXP_NAME.log"
fi

# 2) static checks (schema, lengths, success at the end). No replay: DR episodes don't replay bit-exactly.
say "step 2: checking $DEMOS"
python -m examples.verify_demos "$DEMOS" --no-replay 2>&1 | tail -5 | tee -a "logs/collect_$EXP_NAME.log"

[ "$TRAIN" = 1 ] || { say "TRAIN=0, stopping after collection"; exit 0; }

# 3) train. DR + color jitter are the train_squint.py defaults (env_domain_randomization, apply_jitter).
# --no-cudagraphs: the CUDA-graph capture of the QC update hasn't been tested on the GPU yet (log 2026-09-27 16:10).
say "step 3: training $EXP_NAME ($OFFLINE_STEPS offline, $ONLINE_STEPS online) -> runs/$EXP_NAME, log logs/$EXP_NAME.log"
# shellcheck disable=SC2086
python train_squint_qc.py --env_id "$ENV_ID" --demo_path "$DEMOS" --max_demo_trajs "$N_DEMOS" \
    --env_domain_randomization --apply_jitter --no-cudagraphs \
    --gamma "$GAMMA" --horizon 5 --hidden_dim "$HIDDEN_DIM" --num_layers "$NUM_LAYERS" \
    --offline_steps "$OFFLINE_STEPS" --total_timesteps "$ONLINE_STEPS" --eval_seed "$EVAL_SEED" \
    --exp_name "$EXP_NAME" $TRAIN_ARGS 2>&1 | tee "logs/$EXP_NAME.log"
say "done. Deploy with:"
say "  python deploy_qc.py --env_id $ENV_ID --checkpoint runs/$EXP_NAME/ckpt_best.pt"
