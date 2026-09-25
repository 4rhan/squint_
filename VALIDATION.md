# Validation report — run vs assumed (phase 2)

Hardware: 1× RTX 4090 24 GB, 69 GB free disk. CPU sim for logic checks, GPU sim
for batched/training. Branch `tasks-and-data-loaders`. Sim feasibility is NOT
hardware performance. No success check weakened; no favourable seeds selected
(all seeds declared; failures reported with categories).

## 1. Geometry / task correctness (run)
- Registration/reset/step for all 7 IDs in `rgb+segmentation`, CPU single + GPU
  batched (8 envs): rgb (N,128,128,3) uint8, state (N,12), action Box(-1,1,(6,)).
  Partial reset `env_idx=[0,3]` resets dwell/latch — PASS.
- Policy obs: `extra={}` in deployment mode; flattened state exactly 12
  (`noisy_qpos`6+`target_qpos`6, asserted in `train_squint.py` — aborts otherwise).
  No poses/flags/stages/identities — PASS.
- Controller: `pd_joint_target_delta_pos`, ±0.1 arm / ±0.2 gripper @10 Hz, sim
  100 Hz — PASS.
- Footprint-aware containment (new): Tray valid 0°/30°/45° → complete True (45°
  max span 33.9 mm < 42 mm inner); +12 mm wall overlap → False; divider-wall rim
  (immediate evaluate) → per [F,T,T]; stacked → [T,F,T]; swapped compartments →
  num 1/False — PASS.
- Tower: valid (medium yaw 0.3) complete+stable True; 12 mm overhang →
  `medium_supported` False; small 45° centred → True (corner check) — PASS.
- Rearrange: valid B,C,A + buffer empty → complete True num 3; 45° in pocket →
  True; +12 mm wall overlap → False; initial perm 0/False; A-in-buffer →
  `buffer_empty` False — PASS.
- Dwell guard: valid tower, 1 env step (count 1), then two `evaluate()` calls
  without stepping → counts stay 1,1 — PASS (no double advance).
- Rejection fallback: 8/8 resets per task with no fallback warnings after
  rectangular tray clearance + moved tray (0.33,-0.055); fallback path with
  warning implemented and tested — PASS.
- Fixtures kinematic (`build_kinematic` trays/pockets/marker); tray goals from
  live pose, pocket goals fixed — PASS.
- Reward modes sparse/dense/normalized_dense step OK — PASS.

## 2. Rearrangement demos under training controller (run)
- Declared 20-seed sets (seeds 100-119), training controller, CPU:
  Rearrange2 12/20 (60%, mean T 165, horizon 300, too_long 0); fails:
  7× pick_grasp (pick0×2, pick1×1, pick0b×4), 1× success_false.
  Rearrange3 4/20 (20%, mean T 250, horizon 400, too_long 0); fails:
  15× pick_grasp, 1× success_false(num_correct=2).
- Root-caused jaw-vs-wall jam: 37 mm open jaws vs 35-36 mm walls gave 3.7 N wall
  contact with gripper stuck open (target 0.047, actual 0.299); pocket picks now
  `open_extra=0.008` (30-32 mm) → wall force 0.0, grasp True.
- Planner fixes: flush `speed_scale` effective in delta (removed 0.1 floor),
  delta settle/servo 3/4, verified servo substeps (≤8 mrad, 6 tries) + hold(2)
  with clipping/placement-error logs, fail stage/reason (`LAST_FAIL`).
- Tower/Pack rechecked after shared changes (new geometry/horizons): Pack1 3/3
  (mean 70), Pack2 3/4 (mean 121, 1× pick_ik), Pack3 3/6 (mean 185),
  Tower2 3/8 (mean 175), Tower3 3/8 (mean 228); too_long 0 everywhere.
- Collector now takes `--domain-randomization` / `--control-mode` explicitly and
  records actual settings in meta (prior runs: DR off, default delta controller).
- Horizons extended with justification (demo mean + dwell 10 + ~25% margin):
  Pack1 150, Pack2 200, Pack3 300, Tower2 200, Tower3 300, Rearrange2 300,
  Rearrange3 400. Old-horizon demos replaced (stale geometry).

## 3. Wrist setup + resolution (run)
- New live capture (`examples/capture_wrist_rollout.py`) runs the training
  controller (no teleports for manipulation frames): Tower3 seed 1 full 8 stages,
  Pack3 partial (far-pick flake), Rearrange2 seed 100 full 8 stages in
  `validation/wrist_live/` (128 + area 16/32/64 actual inputs + per-stage stats).
- Measured extents: grasp/carry frames show cubes 39-128 px wide at 128
  (→ 5-16 px at 16, clearly visible); initial far objects ~56-90 px at 128
  (→ 7-11 px at 16, color-visible). Corrected scaling: W px at 128 → W/8 at 16,
  W/4 at 32, W/2 at 64 (8 px → 1, 2, 4).
- Findings: near-field manipulation well inside FOV; initial far objects small
  but color-distinguishable (out-of-view ≠ resolution — more pixels can't fix
  framing); Tower3 post-retreat final frame nearly empty (tower leaves FOV after
  release — policy has no memory by design). Camera mount/FOV unchanged from
  original; no second camera, no privileged obs, no scripted camera moves.
- Old teleport stills kept in `validation/wrist_vis/` (static checks).

## 4. SQUINT baselines (run: pilot 1; pilots 2-5 in progress)
- Pilot 1 `SO101StackCube-v1` seed 1 / eval 1001, 16×16, 1024 envs, 15-min
  training budget (wall excl. eval; total incl. eval logged): 2.79M steps,
  ~3300 sps, setup 1.7 s, buffer 1.55 GB. Final eval: success_at_end 0.94,
  success_once 1.00, return 38.3; grasp/on-top diagnostics 1.00.
  Reproduction check PASSES. Artifacts: `runs/pilot16_SO101StackCube-v1__s1/`
  (ckpt.pt, config.json, metrics.csv, curves.png, eval videos).
- Time-budget stop + final eval + guaranteed save verified (timed_out=True path).
- Launcher: `examples/launch_squint_pilots.sh` (frozen order, seeds, 16×16;
  32/64 as separate labels only). No multi-seed sweep launched.
- Pending this phase: Pack1, Pack3, Tower2, Tower3 pilots (queued), Rearrange
  pilots after demo validation (env validated; demos 60%/20%).

## Findings table
| class | finding | evidence |
|---|---|---|
| environment (fixed) | rotated 26 mm cubes clipped 35/36 mm walls; centre+tol counted protrusions | overlap/rim near-miss FAIL→ now PASS with 42 mm + footprint checks |
| controller (fixed) | delta speed floor ignored slow descents; settle/servo cut; clipped servo unchecked | clip logs, wall-force 3.7 N → fixed, Tower/Pack re-pass |
| observation | 16 px marginal for far objects (7-11 px color blobs); near-field fine; final tower leaves FOV | live stats; no memory by design; 32/64 separate-label option |
| runtime | old horizons too short after slower descents (Tower3 6× too_long at 200) | horizons extended with demo-mean justification; too_long 0 |
| learning | StackCube reproduces (0.94/1.00 in 15 min) | pilot 1 curves + eval |
| learning (pending) | Pack/Tower/Rearrange pilots queued, not yet run | — |

## Remaining blockers / unrun
1. Pilots 2-6 (Pack1/Pack3/Tower2/Tower3 + Rearrange once through): queued after
   pilot 1; multi-seed repeats not launched (launcher provided).
2. Rearrange3 demo yield 20% (pick_grasp in pockets dominates): bulk collection
   needs ~100 attempts/20 demos; further gains via pocket-pick alignment or
   slightly larger pockets (would refreeze geometry — not done here).
3. DR-on demo coverage + disjoint demo/eval seed accounting at scale: collector
   supports it, bulk DR run not executed.
4. Hardware: printed-part tolerances, gripper fit on 36 mm large cube, tower
   stability, camera/lighting calibration, 10 Hz delta tracking — all untested.
5. `train_squint_qc.py` absent on branch (out of scope for baseline phase).
