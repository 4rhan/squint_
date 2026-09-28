#!/usr/bin/env bash
# Replay the first N demos of a demo file and save one mp4 each (scene camera + recorded 16x16 wrist
# image, with step / reward / success overlaid), so the demos can be checked by eye on the box.
#
#   OUT=demos/qc/SO101StackCube-fast500.h5 bash examples/demo_videos.sh      # first 10 -> demo_videos/<file>/
#   OUT=demos/qc/SO101StackCube-fast500.h5 N=20 bash examples/demo_videos.sh
#
# Run from the repo root inside tmux, not next to a training run (a second GPU sim has crashed PhysX).
set -euo pipefail
export PYTHONNOUSERSITE=1

OUT=${OUT:?set OUT=path/to/demos.h5}
N=${N:-10}
VIDEO_DIR=${VIDEO_DIR:-demo_videos/$(basename "$OUT" .h5)}
LOG=${LOG:-logs/demo_videos_$(basename "$OUT" .h5).log}

cd "$(dirname "$0")/.."
mkdir -p "$VIDEO_DIR" logs
echo "replaying the first $N demos of $OUT -> $VIDEO_DIR"
python -m examples.verify_demos "$OUT" --limit "$N" --video "$VIDEO_DIR" 2>&1 | tail -8 | tee "$LOG"
ls -1 "$VIDEO_DIR" | tee -a "$LOG"
