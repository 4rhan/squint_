# Validation report (tasks-and-data-loaders) — run vs assumed

Hardware: 1× RTX 4090, CPU sim for checks unless noted, GPU for batched.
Code: this branch (`tasks-and-data-loaders`), commit at validation time.
Sim feasibility is NOT hardware performance.

## Actually run
- Registration/reset/step for all 7 IDs (`SO101Tower3/2Cube`, `TrayPack1/2/3`,
  `Rearrange3/2`) in `rgb+segmentation`, CPU single + GPU batched (8 envs):
  rgb (N,128,128,3) uint8, state (N,12) float32, action Box(-1,1,(6,)).
  5 zero-action steps, partial reset `env_idx=[0,3]`, dwell counters reset — PASS.
- Flattened policy obs: `extra={}` in deployment mode, `agent={noisy_qpos,
  controller.target_qpos}` only; `rgb+segmentation+state` exposes privileged
  `state` for critics. No poses/flags/stages in policy obs — PASS (see TASK_SPEC).
- Controller explicit: `pd_joint_target_delta_pos`, ±0.1 arm / ±0.2 gripper per
  10 Hz step, sim 100 Hz — PASS (inspected `so101.py` + spaces).
- Tower valid tower teleported → `tower_complete True`, `stable True`, `success
  True` after 12 zero steps (dwell 10); wrong order (S middle/M top), base offset
  3 cm (>12 mm tol), medium overhang 15 mm (>~9 mm allow) → correctly False — PASS.
- Pack valid (3 in assigned compartments) → `num_correct 3`, `complete True`,
  `success True` after dwell; swapped compartments → `num_correct 1`, `complete
  False`; rim (divider wall, immediate evaluate) → per [F,T,T]; stacked (one on
  another) → per [T,F,T], `complete False` — PASS. Note: teleported-high cube
  settles into compartment after 3 physics steps (correctly becomes True); rim
  must be checked immediately or placed on wall.
- Rearrange valid final (B,C,A, buffer empty) → `num_correct 3`, `buffer_empty
  True`, `success True` after dwell; initial perm → 0/False; buffer-occupied
  (A in buffer) → `buffer_empty False`, `complete False` — PASS. Tolerances
  relaxed to xy 0.012/z 0.006 to match planner accuracy (~13–15 mm); rim/stack
  still fail via z (`dz 12 mm > 6 mm`) and `too_high` guards.
- Reward modes `sparse/dense/normalized_dense` step without error; sparse 0
  when unsolved, dense/normalized >0 — PASS.
- Demos: Tower3 (192 steps/20 s horizon 200), Tower2 (113/150), Pack1 (45/100),
  Pack2 (99/150), Pack3 (156/200) in `demos/qc/`, schema `obs/rgb T+1`,
  `obs/state T+1`, `actions T` in [-1,1], `rewards T normalized_dense`,
  `terminated/truncated`, meta JSON. `T+1` asserts + chunk-boundary (per-traj
  groups, `ep_end` via term) checks — PASS. Too-long 0 (all within horizons).
- Wrist visibility: `examples/check_wrist_visibility.py` → 
  `validation/wrist_vis/<task>/{stage}_{128,16,32,64}.png`. 128px non-black
  14–24% (arm+cubes on black overlay); cubes ~8px at 128 → ~1px at 16 (color
  only), ~4px at 32, ~8px at 64. Colors distinguishable at 128; 16 marginal
  (documented, 32/64 configurable). Trays shallow, layout compact; wrist-only
  has no memory — far objects leave view, documented below. No second view added
  to baseline.

## Not run / unverified assumptions (do NOT claim)
- Full RL training (`train_squint.py`/`train_squint_qc.py` to convergence),
  co-training algorithm changes — explicitly out of scope; only registration +
  wrapper compatibility checked, no training curves.
- Large-scale demo collection with domain randomization + disjoint demo/eval
  seeds (current demos `reconfiguration_freq=1`, mostly default sizes, CPU;
  DR coverage not quantified).
- Rearrange delta-control scripted demos: 0/12 (Rearrange3) and 0/5
  (Rearrange2) in training control mode (placements 20–40 mm off vs 18 mm
  allow; third pick from pocket fails). Pos-control demos succeed (Rearrange3
  seed0 True, Rearrange2 2/3 True) proving sim feasibility, but delta-compatible
  demos need further tuning (slower trajectories, larger pockets/clearance,
  better servo/gripper timing). Treated as blocker, not success.
- Real hardware, sim-to-real transfer, camera alignment (`tune_camera.py`),
  lighting/overlay match, motor calibration, printed-part tolerances, friction.
- Representative demo videos (mp4): h5 + stills provided; `RecordEpisode` videos
  not rendered in this pass.
- `train_squint_qc.py`/`qc_data.py` not on this branch; compatibility is by
  schema (matches `collect_qc_demos`/`qc_data` docs from `feat/scripted-demos`)
  plus wrapper parity, not by running that script.

## Blockers for physical replication
1. Print trays/pockets/marker to spec sizes (±0.5 mm), verify gripper fit
   (35–36 mm compartments vs ~30–34 mm occupied jaws) and colors under real light.
2. Validate largest-cube grasp (36 mm), 82 mm tower stability, tray/pocket wall
   heights vs occlusion, marker visibility at 16px.
3. Tune wrist camera to sim (`deploy_utils/tune_camera.py`, black overlay) and
   calibrate motors; confirm 10 Hz delta tracking.
4. Fix rearrange delta demo solver (accuracy/clearance) and collect disjoint
   DR demo/eval sets; render eval videos.
5. Decide explicit optional second view (if wrist-only inadequate for far
   targets/occlusions) and use it in ALL compared methods; current baseline is
   wrist-only by design.
