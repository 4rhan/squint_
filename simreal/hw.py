"""Shared hardware setup for the simreal/ scripts.

- Runs in the `squint` conda env with the LeRobot deploy.py uses (pip lerobot 0.4.3):  conda activate squint
- Calibration files live in the repo, in simreal/calibration/{follower,leader}/<id>.json, not in ~/.cache.
- real <-> sim joint conversion is the one deploy_utils/manipulator.py uses at deployment: arm joints are
  LeRobot degrees -> radians, the gripper goes through the measured servo <-> sim degree mapping.
"""
import os
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
# lerobot 0.4.3: importing lerobot.teleoperators before lerobot.robots hits a circular import; keep this order
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader

CALIB_DIR = Path(__file__).resolve().parent / "calibration"
FOLLOWER_ID = "so101_follower_arm"
LEADER_ID = "so101_leader_arm"
FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0")
LEADER_PORT = os.environ.get("LEADER_PORT", "/dev/ttyACM1")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]  # sim order

# Gripper: sim joint degrees <-> servo degrees (LeRobot DEGREES mode), measured for deploy_utils/manipulator.py
SIM_GRIPPER_DEG = (-10.0, 120.0)  # (closed, open) in sim
SERVO_GRIPPER_DEG = (-62.5, 64.62)  # matching servo readings

# envs/robot/so101.py keyframes, radians, sim joint order
KEYFRAMES = {
    "zero": [0, 0, 0, 0, 0, 0],
    "rest": [0, -1.5708, 1.5708, 0.66, -np.pi, np.deg2rad(-10)],
    "start": [0, 0, 0, np.pi / 2, -np.pi / 2, np.deg2rad(60)],
    "extended": [0, -0.7854, 0.7854, 0, 0, np.deg2rad(100)],
}

# training action limits per 10 Hz control step (pd_joint_target_delta_pos): arm 0.1 rad, gripper 0.2 rad
SIM_CONTROL_FREQ = 10
MAX_DELTA = np.array([0.1] * 5 + [0.2])


def make_follower(port=FOLLOWER_PORT, camera=None, max_relative_target=None) -> SOFollower:
    """camera: None, or an OpenCV index / device path for the wrist camera (named base_camera, as in the envs)."""
    cameras = {}
    if camera is not None:
        cameras["base_camera"] = OpenCVCameraConfig(index_or_path=camera, fps=30, width=640, height=480)
    cfg = SO101FollowerConfig(port=port, id=FOLLOWER_ID, calibration_dir=CALIB_DIR / "follower", use_degrees=True,
                              cameras=cameras, max_relative_target=max_relative_target)
    return SOFollower(cfg)


def connect_follower(follower: SOFollower, retries=3):
    """follower.connect(), but torque comes on holding the present pose.

    LeRobot's configure() re-enables torque without touching Goal_Position. After a (re)calibration the homing
    offsets change, so the old goal register points somewhere else and the arm jumps on torque-on; a gripper
    driven into its closed stop this way stopped answering ("Failed to write 'Lock' on id_=6 ... no status
    packet", 2026-09-30). So: calibrate if needed, set Goal_Position = Present_Position with torque off, then
    configure (which enables torque), with retries on the bus writes.
    """
    import time
    follower.bus.connect()
    if not follower.is_calibrated:
        follower.calibrate()
    for attempt in range(retries):
        try:
            follower.bus.disable_torque(num_retry=2)
            present = follower.bus.sync_read("Present_Position", normalize=False, num_retry=2)
            follower.bus.sync_write("Goal_Position", present, normalize=False, num_retry=2)
            follower.configure()
            break
        except ConnectionError as e:
            if attempt == retries - 1:
                raise ConnectionError(f"{e}\nA motor stopped answering: power-cycle the follower (unplug its power "
                                      "supply for ~5 s), check the cable to that motor id, then retry.") from e
            print(f"[connect] {e}; retrying in 1 s")
            time.sleep(1.0)
    for cam in follower.cameras.values():
        cam.connect()


def make_leader(port=LEADER_PORT) -> SOLeader:
    return SOLeader(SO101LeaderConfig(port=port, id=LEADER_ID, calibration_dir=CALIB_DIR / "leader",
                                      use_degrees=True))


def _res(bus, motor):
    return bus.model_resolution_table[bus.motors[motor].model] - 1


def gripper_pct_to_servo_deg(bus, pct):
    """LeRobot RANGE_0_100 gripper value -> the value the same position reads in DEGREES mode."""
    c = bus.calibration["gripper"]
    raw = pct / 100 * (c.range_max - c.range_min) + c.range_min
    return (raw - (c.range_min + c.range_max) / 2) * 360 / _res(bus, "gripper")


def gripper_servo_deg_to_pct(bus, deg):
    c = bus.calibration["gripper"]
    raw = deg * _res(bus, "gripper") / 360 + (c.range_min + c.range_max) / 2
    return float(np.clip((raw - c.range_min) / (c.range_max - c.range_min) * 100, 0, 100))


def real_to_sim(pos: dict, bus) -> np.ndarray:
    """LeRobot joint dict ('<joint>.pos': deg, gripper 0-100) -> sim qpos (6,) in radians."""
    q = np.deg2rad([pos[f"{j}.pos"] for j in JOINTS[:5]])
    servo = gripper_pct_to_servo_deg(bus, pos["gripper.pos"])
    sim_g = (servo - SERVO_GRIPPER_DEG[0]) / (SERVO_GRIPPER_DEG[1] - SERVO_GRIPPER_DEG[0]) \
        * (SIM_GRIPPER_DEG[1] - SIM_GRIPPER_DEG[0]) + SIM_GRIPPER_DEG[0]
    return np.append(q, np.deg2rad(sim_g))


def sim_to_real(qpos, bus) -> dict:
    """Inverse of real_to_sim: sim qpos (rad) -> LeRobot action dict."""
    qpos = np.asarray(qpos, dtype=np.float64)
    act = {f"{j}.pos": float(np.rad2deg(qpos[i])) for i, j in enumerate(JOINTS[:5])}
    sim_g = np.rad2deg(qpos[5])
    servo = (sim_g - SIM_GRIPPER_DEG[0]) / (SIM_GRIPPER_DEG[1] - SIM_GRIPPER_DEG[0]) \
        * (SERVO_GRIPPER_DEG[1] - SERVO_GRIPPER_DEG[0]) + SERVO_GRIPPER_DEG[0]
    act["gripper.pos"] = gripper_servo_deg_to_pct(bus, servo)
    return act


def read_joints(robot_or_bus) -> dict:
    """Present joint positions as a LeRobot action-style dict (works for the follower and the leader)."""
    bus = getattr(robot_or_bus, "bus", robot_or_bus)
    return {f"{m}.pos": v for m, v in bus.sync_read("Present_Position").items()}


def move_smoothly(follower, target: dict, max_deg_per_step=1.5, fps=30):
    """Interpolate the follower from where it is to `target` (LeRobot units), at most max_deg_per_step per tick."""
    import time
    cur = read_joints(follower)
    keys = list(target)
    a, b = np.array([cur[k] for k in keys]), np.array([target[k] for k in keys])
    n = max(1, int(np.ceil(np.abs(b - a).max() / max_deg_per_step)))
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        follower.send_action(dict(zip(keys, a + (b - a) * i / n)))
        time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))


class SimMirror:
    """SAPIEN viewer showing the sim SO101 at the real robot's joint positions (no physics stepping)."""

    def __init__(self, env_id="SO101LiftCube-v1"):
        import gymnasium as gym
        import envs  # noqa: F401  (registers the SO101 tasks; run from the repo root with python -m simreal.X)
        self.env = gym.make(env_id, num_envs=1, sim_backend="cpu", obs_mode="state", render_mode="human")
        self.env.reset(seed=0)
        self.base = self.env.unwrapped

    def show(self, qpos):
        import torch
        self.base.agent.robot.set_qpos(torch.as_tensor(np.asarray(qpos), dtype=torch.float32)[None])
        self.base.scene.update_render()
        self.base.render_human()

    def close(self):
        self.env.close()
