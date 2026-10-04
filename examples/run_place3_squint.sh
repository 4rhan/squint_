#!/usr/bin/env bash
# Squint SAC baseline (train_squint.py: no demos, no action chunking) on SO101Place3Cube-v1, for comparison
# with QC-FQL (examples/run_place3_qc.sh). Same env and eval protocol as the QC run: no domain randomization,
# 64 eval episodes on fixed seed 100, 2M env steps (= QC's online budget; QC also gets 500 demos + offline
# pretraining). RUNS picks the variants (run one after the other):
#   g09  (default) <prefix>_g09:  the Squint baseline, Squint's own settings (gamma 0.9, C51 support [-20, 20])
#   g099 (optional) <prefix>_g099: gamma 0.99 like QC; support widened to [-60, 100] (normalized reward is <= 1 per
#        step, so returns reach 1/(1-0.99) = 100; the worst penalties are ~ -0.4 per step)
#
#   bash examples/run_place3_squint.sh                  # from the repo root, inside tmux, conda env active
#   RUNS="g09 g099" bash examples/run_place3_squint.sh  # also the gamma-0.99 variant
#
# Logs: logs/<exp_name>.log. Runs: runs/<exp_name>/ (metrics.jsonl, plots/, ckpt.pt).
set -uo pipefail
export PYTHONNOUSERSITE=1  # see collect_tp3_recovery.sh: a torch in ~/.local breaks the env's torchvision

PREFIX=${PREFIX:-place3_squint}
STEPS=${STEPS:-2000000}
EVAL_SEED=${EVAL_SEED:-100}
RUNS=${RUNS:-g09}
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "$(dirname "$0")/.."
mkdir -p logs
say() { echo "[$(date +%F\ %T)] $*"; }
status=0
for r in $RUNS; do
    case $r in
        g09)  gargs="--gamma 0.9 --v_min -20 --v_max 20" ;;
        g099) gargs="--gamma 0.99 --v_min -60 --v_max 100" ;;
        *) echo "unknown run $r (use g09, g099)" >&2; exit 1 ;;
    esac
    name="${PREFIX}_$r"
    if [ -e "runs/$name" ]; then echo "runs/$name exists; set PREFIX=..." >&2; exit 1; fi
    say "start $name: $STEPS steps, $gargs, eval seed $EVAL_SEED"
    # shellcheck disable=SC2086
    python train_squint.py --env_id SO101Place3Cube-v1 --exp_name "$name" --total_timesteps "$STEPS" \
        --no-env_domain_randomization --num_eval_envs 64 --eval_seed "$EVAL_SEED" $gargs $EXTRA_ARGS \
        2>&1 | tee "logs/$name.log"
    s=${PIPESTATUS[0]}
    say "finished $name with exit code $s"
    [ "$s" = 0 ] || status=$s
done
exit "$status"
