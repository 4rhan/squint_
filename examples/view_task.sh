#!/usr/bin/env bash
# Scripted solver on a task, live or recorded.
#   bash examples/view_task.sh                                   # live SAPIEN viewer (box desktop / DISPLAY=:1)
#   VIDEO=1 bash examples/view_task.sh                           # smooth 50 fps mp4s -> task_videos/<env_id>/, headless
#   VIDEO=1 FAST=1 SEEDS="0 1 2 3" bash examples/view_task.sh    # FAST solver profile (fewer settle pauses)
#   ENV_ID=SO101Rearrange3-v1 SEEDS="0 5" bash examples/view_task.sh
set -euo pipefail
export PYTHONNOUSERSITE=1
ENV_ID=${ENV_ID:-SO101Rearrange2-v1}
SEEDS=${SEEDS:-0 1 2}
VIDEO=${VIDEO:-0}
FAST=${FAST:-0}
VIDEO_DIR=${VIDEO_DIR:-task_videos/$ENV_ID}
cd "$(dirname "$0")/.."
EXTRA=()
if [ "$VIDEO" = 1 ]; then EXTRA+=(--video "$VIDEO_DIR"); mkdir -p logs; fi
if [ "$FAST" = 1 ]; then EXTRA+=(--fast); fi
# shellcheck disable=SC2086
python -m examples.view_task -e "$ENV_ID" --seeds $SEEDS "${EXTRA[@]}" 2>&1 | grep -v -i warn \
    | if [ "$VIDEO" = 1 ]; then tee "logs/view_task_$ENV_ID.log"; else cat; fi
