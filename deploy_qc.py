"""
Deploy a QC-FQL policy (train_squint_qc.py) on the real SO101 robot.

This runs Squint's original deploy.py (LeRobot robot, Sim2RealEnv, wrist-camera preprocessing, keyboard
controls, recording, --debug overlay) unchanged. Only two of its pieces are swapped:
- the agent: qc_agent.QCDeployAgent instead of train_squint.DeployAgent (plays each action chunk open loop
  and asks the policy again after --qc_exec_steps actions)
- the real-robot reset: also clears the agent's chunk, so a new episode never finishes the old one

- the real wrist roll: shifted by --wrist_roll_offset_deg (default 90) between sim and real, because our arm's
  wrist roll is mounted 90 deg off the sim; training is unchanged

Usage (same flags as deploy.py, plus --qc_exec_steps and --wrist_roll_offset_deg):
    python deploy_qc.py --checkpoint runs/lift_qc_dr200/ckpt_best.pt --env_id SO101LiftCube-v1
    python deploy_qc.py --checkpoint runs/lift_qc_dr200/ckpt_best.pt --env_id SO101LiftCube-v1 --qc_exec_steps 2

Keyboard Controls:
    's' - Skip current episode
    'q' - Quit evaluation
"""

from dataclasses import dataclass
from typing import Optional

import tyro

import deploy
from qc_agent import QCDeployAgent


@dataclass
class QCArgs(deploy.Args):
    qc_exec_steps: Optional[int] = None
    """actions played from each chunk before asking the policy again (default: the whole chunk, as in
    training; smaller = more closed-loop)"""
    wrist_roll_offset_deg: float = 90.0
    """real wrist roll = sim wrist roll + this (our arm is mounted 90 deg off the sim). Training keeps the sim
    start pose (wrist_roll -90); with 90 the real arm starts at real 0 and the policy still reads -90. 0 = no offset.
    Check with --debug (sim/real overlay) before running a policy."""


def main(args: QCArgs):
    assert args.checkpoint and args.checkpoint != "wandb", "deploy_qc.py needs a local --checkpoint path"
    agents = []

    def make_agent(sim_env, sample_obs):
        agent = QCDeployAgent(sim_env, sample_obs, args.checkpoint, exec_steps=args.qc_exec_steps)
        agents.append(agent)
        return agent

    original_reset = deploy.silent_reset

    def reset_with_agent(env, seed=None, options=None):
        for agent in agents:
            agent.reset()
        original_reset(env, seed=seed, options=options)

    deploy.LeRobotRealAgent.WRIST_ROLL_OFFSET_DEG = args.wrist_roll_offset_deg
    print(f"wrist roll offset: real = sim + {args.wrist_roll_offset_deg} deg")

    # deploy.main() looks both names up in its module when it runs
    deploy.DeployAgent = make_agent
    deploy.silent_reset = reset_with_agent
    deploy.main(args)


if __name__ == "__main__":
    main(tyro.cli(QCArgs))
