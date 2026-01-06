# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# Argument parsing
parser = argparse.ArgumentParser(description="Minimal Isaac Lab task runner.")
parser.add_argument("--task", type=str, required=True, help="Task name registered in Gym.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

# Isaac Sim / AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports AFTER Isaac Sim launch
import gymnasium as gym
import torch

import isaaclab_tasks  # registers built-in tasks
import Galaxea_Lab_External.tasks  # registers external tasks
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Task          : {args_cli.task}")
    print(f"[INFO] Obs space     : {env.observation_space}")
    print(f"[INFO] Action space  : {env.action_space}")

    env.reset()

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.from_numpy(env.action_space.sample()).to(env.unwrapped.device)
            env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
