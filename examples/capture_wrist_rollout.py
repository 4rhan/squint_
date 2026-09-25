"""Live wrist-camera capture along controller-executed demonstrations.

Unlike check_wrist_visibility.py (teleports with arm at rest), this runs the
IK motion planner in the training controller (pd_joint_target_delta_pos,
normalized delta @10Hz) and saves fresh 128px wrist frames at manipulation
stages: initial, approach, grasp, carry, pre-place, placed, next-selection.
Each stage saves the 128 render plus the ACTUAL policy inputs at 16/32/64 via
area downsampling (same op as DownsampleObsWrapper), with measured visibility
stats (non-black fraction, per-color bbox widths in px).

No second camera, no scripted camera moves, no privileged obs. Original wrist
mount/FOV untouched.

Usage:
  python -m examples.capture_wrist_rollout -e SO101Tower3Cube-v1 --seed 1 --outdir validation/wrist_live
"""
import argparse
import json
import os

import envs  # noqa
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


def area_down(rgb_np, size):
    x = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(x, size=(size, size), mode="area").round().clamp(0, 255).to(torch.uint8)
    return y.squeeze(0).permute(1, 2, 0).numpy()


def wrist_frame(env):
    unw = env.unwrapped
    imgs = unw.get_sensor_images()
    return imgs["base_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)


def color_stats(rgb):
    """Non-black fraction + bbox widths for R/G/B/Y masks (px at 128)."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    masks = {
        "red": (r > 150) & (g < 100) & (b < 100),
        "green": (g > 150) & (r < 100) & (b < 100),
        "blue": (b > 150) & (r < 100) & (g < 100),
        "yellow": (r > 150) & (g > 150) & (b < 100),
        "white": (r > 180) & (g > 180) & (b > 180),
    }
    out = {"nonblack_frac": float((rgb.max(axis=2) > 30).mean())}
    for name, m in masks.items():
        if m.any():
            ys, xs = np.where(m)
            out[f"{name}_frac"] = float(m.mean())
            out[f"{name}_bbox_w"] = int(xs.max() - xs.min() + 1)
            out[f"{name}_bbox_h"] = int(ys.max() - ys.min() + 1)
        else:
            out[f"{name}_frac"] = 0.0
            out[f"{name}_bbox_w"] = 0
            out[f"{name}_bbox_h"] = 0
    return out


def save_stage(outdir, stage, rgb128):
    os.makedirs(outdir, exist_ok=True)
    Image.fromarray(rgb128).save(os.path.join(outdir, f"{stage}_128.png"))
    stats = {"stage": stage, "at128": color_stats(rgb128)}
    for s in (16, 32, 64):
        small = area_down(rgb128, s)
        Image.fromarray(small).save(os.path.join(outdir, f"{stage}_{s}.png"))
        Image.fromarray(small).resize((128, 128), Image.NEAREST).save(
            os.path.join(outdir, f"{stage}_{s}_nearest128.png"))
        # Correct scaling reference: W px at 128 -> W*s/128 at s.
        stats[f"at{s}"] = color_stats(
            np.array(Image.fromarray(small).resize((128, 128), Image.NEAREST)))
    with open(os.path.join(outdir, f"{stage}_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    nb = stats["at128"]["nonblack_frac"]
    print(f"  [{stage}] nonblack {nb:.3f} "
          f"R{stats['at128']['red_bbox_w']} G{stats['at128']['green_bbox_w']} "
          f"B{stats['at128']['blue_bbox_w']} Y{stats['at128']['yellow_bbox_w']} px@128")
    return stats


def run_staged(env_id, seed, outdir, stages_fn):
    """stages_fn(solver, e, save) runs the task calling save(name) at each stage."""
    env = gym.make(env_id, num_envs=1, obs_mode="rgb+segmentation", render_mode="rgb_array",
                   sim_backend="cpu", sensor_configs=dict(width=128, height=128),
                   reward_mode="normalized_dense", reconfiguration_freq=1)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    from examples.motionplanning.so101.motionplanner import SO101GraspSolver
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=False)
    e = env.unwrapped
    os.makedirs(outdir, exist_ok=True)
    all_stats = {}

    def save(name):
        all_stats[name] = save_stage(outdir, name, wrist_frame(env))

    all_stats["seed"] = seed
    stages_fn(solver, e, save)
    meta = dict(env_id=env_id, seed=seed, controller=e.agent.control_mode,
                horizon=int(__import__("mani_skill.utils.gym_utils", fromlist=["find_max_episode_steps_value"]).find_max_episode_steps_value(env)),
                note="live training-controller rollout; 128 render + area 16/32/64 actual policy inputs; "
                     "8px@128 ~= 1px@16, 2px@32, 4px@64")
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    env.close()
    return all_stats


def tower_stages(solver, e, save):
    import numpy as np
    hl, hm, hs = float(e.large_half[0]), float(e.medium_half[0]), float(e.small_half[0])
    tx, ty = float(e.tower_xy[0]), float(e.tower_xy[1])
    save("initial")
    assert solver.pick(e.cubeL, hl)
    save("grasp_large")
    target_L = np.array([tx, ty, hl])
    assert solver.place(e.cubeL, target_L, solver.horizontal_axes(e.cubeL),
                        approach_height=0.05, retreat_height=0.07)
    save("placed_base")
    assert solver.pick(e.cubeM, hm, approach_height=0.07)
    save("grasp_medium")
    base_p = e.cubeL.pose.sp.p.astype(np.float64)
    assert solver.place(e.cubeM, base_p + np.array([0, 0, hl + hm]),
                        solver.horizontal_axes(e.cubeL), approach_height=0.05, retreat_height=0.07)
    save("placed_medium")
    assert solver.pick(e.cubeS, hs, approach_height=0.07)
    save("grasp_small")
    mid_p = e.cubeM.pose.sp.p.astype(np.float64)
    assert solver.place(e.cubeS, mid_p + np.array([0, 0, hm + hs]),
                        solver.horizontal_axes(e.cubeM), approach_height=0.05, retreat_height=0.07)
    save("placed_small")
    solver.hold(12)
    save("final")


def pack_stages(solver, e, save):
    import numpy as np
    half = float(e.cube_half[0])
    comp_x = [float(e.tray.pose.sp.p[0]) + float(x) for x in e.comp_local_x.cpu().numpy()]
    tray_y, floor_z = float(e.tray.pose.sp.p[1]), float(e.tray.pose.sp.p[2]) + e.floor_t
    row = np.array([1.0, 0.0, 0.0])
    xa, ya = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    save("initial")
    base = e.agent.robot.pose.sp.p
    order = sorted(range(3), key=lambda k: np.linalg.norm(e.cubes[k].pose.sp.p[:2] - base[:2]))
    for k in order:
        assert solver.pick(e.cubes[k], half, open_extra=0.015)
        save(f"grasp_cube{k}")
        assert solver.place(e.cubes[k], np.array([comp_x[k], tray_y, floor_z + half]),
                            (xa, ya), jaw_perp=row, release_gap=0.008)
        save(f"placed_cube{k}")
    solver.hold(12)
    save("final")


def rearr2_stages(solver, e, save):
    import numpy as np
    half = float(e.cube_half[0])
    xa, ya = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    row = np.array([0.0, 1.0, 0.0])
    save("initial")

    def place(obj, pock):
        target = np.array([e.pocket_x, e.pocket_ys[pock], e.pocket_floor_t + half])
        return solver.place(e.cubes[obj], target, (xa, ya), jaw_perp=row, release_gap=0.008)
    assert solver.pick(e.cubes[0], half, open_extra=0.008)
    save("grasp_a")
    assert place(0, 2)
    save("a_in_buffer")
    assert solver.pick(e.cubes[1], half, open_extra=0.008)
    save("grasp_b")
    assert place(1, 0)
    save("b_placed")
    assert solver.pick(e.cubes[0], half, open_extra=0.008)
    save("grasp_a2")
    assert place(0, 1)
    save("a_placed")
    solver.hold(12)
    save("final")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--env-id", default="SO101Tower3Cube-v1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", default="validation/wrist_live")
    args = p.parse_args()
    fns = {"SO101Tower3Cube-v1": tower_stages, "SO101TrayPack3-v1": pack_stages,
           "SO101Rearrange2-v1": rearr2_stages}
    if args.env_id not in fns:
        raise SystemExit(f"live capture implemented for {sorted(fns)}")
    out = os.path.join(args.outdir, args.env_id)
    try:
        run_staged(args.env_id, args.seed, out, fns[args.env_id])
        print(f"OK -> {out}")
    except AssertionError:
        print(f"INCOMPLETE (solver stage failed) -> partial frames in {out}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
