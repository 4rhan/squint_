#!/bin/bash
# Frozen SQUINT pilot launcher (one-seed, 15-min budget each).
# Geometry/camera/rewards/eval frozen by TASK_SPEC.md + config.json per run.
# Usage:
#   bash examples/launch_squint_pilots.sh <env_id> <seed> [timesteps]
#   bash examples/launch_squint_pilots.sh all 1        # prints (does NOT launch sweep)
# Pilot order: StackCube (repro) -> Pack1 -> Pack3 -> Tower2 -> Tower3 -> Rearrange2/3 (if validated).
# Each run: 16x16 baseline; 32/64 only as separate labelled runs.
set -u
BUDGET=900  # 15-min training budget (excludes eval); total_timesteps is a cap, time rules
STEPS=10000000
EVAL_SEED_OFFSET=1000

run_one() {
  local env_id=$1 seed=$2 steps=${3:-$STEPS}
  local eval_seed=$((seed + EVAL_SEED_OFFSET))
  local exp="pilot16_${env_id}__s${seed}"
  echo "python train_squint.py --env_id=${env_id} --seed=${seed} --eval_seed=${eval_seed} \\"
  echo "  --exp_name=${exp} --time_budget=${BUDGET} --total_timesteps=${steps} \\"
  echo "  --num_envs=1024 --num_eval_envs=16 --image_size=16"
  if [ "${LAUNCH:-0}" = "1" ]; then
    python train_squint.py --env_id="${env_id}" --seed="${seed}" --eval_seed="${eval_seed}" \
      --exp_name="${exp}" --time_budget="${BUDGET}" --total_timesteps="${steps}" \
      --num_envs=1024 --num_eval_envs=16 --image_size=16 "$@"
  fi
}

if [ "${1:-}" = "all" ]; then
  seed=${2:-1}
  for e in SO101StackCube-v1 SO101TrayPack1-v1 SO101TrayPack3-v1 SO101Tower2Cube-v1 SO101Tower3Cube-v1 SO101Rearrange2-v1 SO101Rearrange3-v1; do
    run_one "$e" "$seed"
    echo ""
  done
  echo "# Dry run (LAUNCH=1 to execute one at a time in the order above)."
else
  run_one "$@"
fi
