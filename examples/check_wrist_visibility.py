"""Wrist-camera visibility check: full-res + policy-res images at task stages.

Saves PNGs to validation/wrist_vis/<task>/ showing 128px wrist view plus
16/32/64 area-downsampled policy views. Run after envs are implemented.

Usage: python -m examples.check_wrist_visibility --outdir validation/wrist_vis
"""
import argparse
import os
import gymnasium as gym
import envs  # noqa
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from mani_skill.utils.structs.pose import Pose


def area_down(rgb, size):
    # rgb: (H,W,C) uint8 numpy
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(x, size=(size, size), mode="area").round().clamp(0, 255).to(torch.uint8)
    return y.squeeze(0).permute(1, 2, 0).numpy()


def get_wrist(env):
    unw = env.unwrapped
    imgs = unw.get_sensor_images()  # dict camera -> dict rgb...
    # base_camera is wrist
    rgb = imgs["base_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)
    return rgb


def save_stage(outdir, task, stage, rgb128):
    d = os.path.join(outdir, task)
    os.makedirs(d, exist_ok=True)
    Image.fromarray(rgb128).save(os.path.join(d, f"{stage}_128.png"))
    for s in (16, 32, 64):
        Image.fromarray(area_down(rgb128, s)).resize((128, 128), Image.NEAREST).save(
            os.path.join(d, f"{stage}_{s}_nearest128.png"))
        Image.fromarray(area_down(rgb128, s)).save(os.path.join(d, f"{stage}_{s}.png"))
    print(f"[{task}/{stage}] saved 128 + 16/32/64 (mean {rgb128.mean():.1f}, std {rgb128.std():.1f})")


def tower_stages(env):
    unw = env.unwrapped
    tx, ty = float(unw.tower_xy[0]), float(unw.tower_xy[1])
    hl, hm, hs = float(unw.large_half[0]), float(unw.medium_half[0]), float(unw.small_half[0])
    env.reset(seed=0)
    yield "initial", get_wrist(env)
    # base placed
    unw.cubeL.set_pose(Pose.create_from_pq(torch.tensor([[tx, ty, hl]], device=unw.device)))
    env.step(torch.zeros((1, 6)))
    yield "base_placed", get_wrist(env)
    # medium on large
    unw.cubeM.set_pose(Pose.create_from_pq(torch.tensor([[tx, ty, 2 * hl + hm]], device=unw.device)))
    env.step(torch.zeros((1, 6)))
    yield "medium_placed", get_wrist(env)
    # full tower
    unw.cubeS.set_pose(Pose.create_from_pq(torch.tensor([[tx, ty, 2 * hl + 2 * hm + hs]], device=unw.device)))
    env.step(torch.zeros((1, 6)))
    yield "final", get_wrist(env)


def pack_stages(env):
    unw = env.unwrapped
    env.reset(seed=0)
    yield "initial", get_wrist(env)
    comp = unw._comp_world()[0]
    half = float(unw.cube_half[0])
    for k in range(3):
        p = torch.tensor([[comp[k, 0].item(), comp[k, 1].item(), comp[k, 2].item() + half]], device=unw.device)
        unw.cubes[k].set_pose(Pose.create_from_pq(p))
        env.step(torch.zeros((1, 6)))
        yield f"placed_{k+1}", get_wrist(env)


def rearr_stages(env):
    unw = env.unwrapped
    env.reset(seed=0)
    yield "initial", get_wrist(env)
    half = float(unw.cube_half[0])
    # A->buffer
    unw.cubes[0].set_pose(Pose.create_from_pq(torch.tensor([[unw.pocket_x, unw.pocket_ys[3], unw.pocket_floor_t + half]], device=unw.device)))
    env.step(torch.zeros((1, 6)))
    yield "a_to_buffer", get_wrist(env)
    # B->P0, C->P1
    unw.cubes[1].set_pose(Pose.create_from_pq(torch.tensor([[unw.pocket_x, unw.pocket_ys[0], unw.pocket_floor_t + half]], device=unw.device)))
    env.step(torch.zeros((1, 6)))
    yield "b_placed", get_wrist(env)
    unw.cubes[2].set_pose(Pose.create_from_pq(torch.tensor([[unw.pocket_x, unw.pocket_ys[1], unw.pocket_floor_t + half]], device=unw.device)))
    env.step(torch.zeros((1, 6)))
    yield "c_placed", get_wrist(env)
    unw.cubes[0].set_pose(Pose.create_from_pq(torch.tensor([[unw.pocket_x, unw.pocket_ys[2], unw.pocket_floor_t + half]], device=unw.device)))
    env.step(torch.zeros((1, 6)))
    yield "final", get_wrist(env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="validation/wrist_vis")
    args = p.parse_args()
    tasks = {
        "SO101Tower3Cube-v1": tower_stages,
        "SO101TrayPack3-v1": pack_stages,
        "SO101Rearrange3-v1": rearr_stages,
        "SO101Tower2Cube-v1": None, "SO101TrayPack1-v1": None, "SO101Rearrange2-v1": None,
    }
    for task, fn in tasks.items():
        if fn is None:
            continue
        env = gym.make(task, num_envs=1, obs_mode="rgb+segmentation", render_mode="rgb_array",
                       sim_backend="cpu", sensor_configs=dict(width=128, height=128))
        for stage, rgb in fn(env):
            save_stage(args.outdir, task, stage, rgb)
        env.close()
    print(f"Done -> {args.outdir}")


if __name__ == "__main__":
    main()
