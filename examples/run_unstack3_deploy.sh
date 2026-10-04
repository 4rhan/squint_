#!/usr/bin/env bash
# QC-FQL on SO101Unstack3Cube-v1 for real-robot deployment (deploy_qc.py).
# Strictly one job at a time: 1) collect scripted demos (skipped if $DEMOS exists), 2) check them, 3) train.
#
# Real setup (envs/unstack3.py, Unstack3RandomizationConfig): tower red (top) / black (middle) / red (base) on a
# photo of the real table (envs/lift_overlay.png). Learning settings are the best Unstack3 run so far,
# unstack3_500_qmin (best eval 0.89): 500 recovery demos (noise 0.2, miss 0.3, FAST solver), 200k offline, 2M
# online, gamma 0.99, 512x4, h=5, --q_agg min, eval seed 100 (16 eval episodes here; that run used 64). Unlike that run, domain
# randomization is ON for demos, training and eval (camera pose/FOV noise, 5 deg joint-reading noise, cube
# sizes, friction, colour jitter, colour-jittered images), as in run_lift_qc.sh.
#
#   bash examples/run_unstack3_deploy.sh                 # from the repo root, inside tmux
#   N_DEMOS=300 ONLINE_STEPS=3000000 bash examples/run_unstack3_deploy.sh
#   TRAIN=0 bash examples/run_unstack3_deploy.sh         # collect + check only
#
# Logs go to logs/ (git-ignored): logs/collect_$EXP_NAME.log, logs/$EXP_NAME.log. Run dir: runs/$EXP_NAME/
# (ckpt_best.pt = best eval, for deploy_qc.py).
set -euo pipefail
export PYTHONNOUSERSITE=1  # see collect_tp3_recovery.sh: a torch in ~/.local breaks the env's torchvision

ENV_ID=SO101Unstack3Cube-v1
N_DEMOS=${N_DEMOS:-500}
WORKERS=${WORKERS:-8}
START_SEED=${START_SEED:-1000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-$((N_DEMOS * 4))}  # local test with DR: ~55% of recovery attempts saved
NOISE=${NOISE:-0.2}
MISS=${MISS:-0.3}
# "rbr-table": red/black/red cubes on the table photo
DEMOS=${DEMOS:-demos/qc/SO101Unstack3Cube-rbr-table-dr-recovery${N_DEMOS}.h5}

TRAIN=${TRAIN:-1}
EXP_NAME=${EXP_NAME:-unstack3_rbr_table_dr${N_DEMOS}}
OFFLINE_STEPS=${OFFLINE_STEPS:-200000}
ONLINE_STEPS=${ONLINE_STEPS:-2000000}
GAMMA=${GAMMA:-0.99}                     # 150-step task (gamma 0.9 hides the second cube from the critic)
HIDDEN_DIM=${HIDDEN_DIM:-512}
NUM_LAYERS=${NUM_LAYERS:-4}
EVAL_SEED=${EVAL_SEED:-100}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-16}       # eval episodes per evaluation (all from eval seed 100)
TRAIN_ARGS=${TRAIN_ARGS:-}

cd "$(dirname "$0")/.."
mkdir -p logs "$(dirname "$DEMOS")"
say() { echo "[$(date +%F\ %T)] $*"; }

# 1) recovery demos with the FAST solver so they fit the 150-step limit; too-long attempts are dropped
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

# 3) train. DR + colour jitter are the train_squint.py defaults (env_domain_randomization, apply_jitter).
# --no-cudagraphs: as in run_lift_qc.sh (the CUDA-graph QC update hasn't been tested on the GPU).
say "step 3: training $EXP_NAME ($OFFLINE_STEPS offline, $ONLINE_STEPS online) -> runs/$EXP_NAME, log logs/$EXP_NAME.log"
# shellcheck disable=SC2086
python train_squint_qc.py --env_id "$ENV_ID" --demo_path "$DEMOS" --max_demo_trajs "$N_DEMOS" \
    --env_domain_randomization --apply_jitter --no-cudagraphs \
    --gamma "$GAMMA" --horizon 5 --hidden_dim "$HIDDEN_DIM" --num_layers "$NUM_LAYERS" --q_agg min \
    --offline_steps "$OFFLINE_STEPS" --total_timesteps "$ONLINE_STEPS" \
    --eval_seed "$EVAL_SEED" --num_eval_envs "$NUM_EVAL_ENVS" \
    --exp_name "$EXP_NAME" $TRAIN_ARGS 2>&1 | tee "logs/$EXP_NAME.log"
say "done. Check in sim, then deploy:"
say "  python -m examples.check_deploy_qc --env_id $ENV_ID --checkpoint runs/$EXP_NAME/ckpt_best.pt --max_episode_steps 150"
say "  python deploy_qc.py --env_id $ENV_ID --checkpoint runs/$EXP_NAME/ckpt_best.pt --qc_exec_steps 1"
