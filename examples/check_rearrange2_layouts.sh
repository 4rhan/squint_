#!/usr/bin/env bash
# Scripted-solver success on SO101Rearrange2-v1 for 4 pocket layouts (one worker per layout).
#   bash examples/check_rearrange2_layouts.sh            # 20 seeds each, from the repo root inside tmux
#   SEEDS=40 bash examples/check_rearrange2_layouts.sh
# Log: logs/check_rearrange2_layouts.log. Don't run next to a training run.
set -euo pipefail
export PYTHONNOUSERSITE=1
SEEDS=${SEEDS:-20}
cd "$(dirname "$0")/.."
mkdir -p logs
python -m examples.check_rearrange2_layouts --seeds "$SEEDS" 2>&1 | grep -v -i warn | tee logs/check_rearrange2_layouts.log
