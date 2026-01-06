# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Galaxea reach environment using Differential IK control."""

from __future__ import annotations

import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from Galaxea_Lab_External.robots import GALAXEA_R1_CHALLENGE_CFG


@configclass
class GalaxeaReachDiffIKEnvCfg(DirectRLEnvCfg):
    """Configuration for the Galaxea Reach environment using Differential IK."""
    
    # ===== Environment settings =====
    episode_length_s = 5.0
    decimation = 4
    action_scale = 1.0
    action_space = 6  # 3 pos delta + 3 rot delta
    observation_space = 19  # 3 + 4 + 3 + 3 + 6
    state_space = 0
    
    # ===== Simulation settings =====
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=2**26,
            gpu_total_aggregate_pairs_capacity=2**26,
        ),
    )
    
    # ===== Scene settings =====
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )
    
    # ===== Robot configuration =====
    # Using GALAXEA_R1_CHALLENGE_CFG for position control (high stiffness/damping)
    robot_cfg = GALAXEA_R1_CHALLENGE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
    
    # ===== Joint configuration =====
    left_arm_joint_dof_name = [
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
        "left_arm_joint6",
    ]
    right_arm_joint_dof_name = [
        "right_arm_joint1",
        "right_arm_joint2",
        "right_arm_joint3",
        "right_arm_joint4",
        "right_arm_joint5",
        "right_arm_joint6",
    ]
    left_gripper_dof_name = ["left_gripper_axis1", "left_gripper_axis2"]
    right_gripper_dof_name = ["right_gripper_axis1", "right_gripper_axis2"]
    torso_joint_dof_name = ["torso_joint1", "torso_joint2", "torso_joint3"]
    
    # ===== IK settings =====
    ik_method: str = "dls"  # Damped Least Squares
    
    # ===== Action thresholds =====
    pos_action_threshold = (0.05, 0.05, 0.05)  # max position delta per step
    rot_action_threshold = (0.1, 0.1, 0.1)     # max rotation delta per step (rad)
    
    # ===== Target position range (for randomization) =====
    target_position_range = (
        (0.25, 0.55),  # x range (front of robot)
        (0.1, 0.4),    # y range (left side, positive Y)
        (0.7, 1.1),    # z range (reachable height)
    )
    
    # ===== Reward settings =====
    # Task rewards
    reward_position_tracking_weight: float = 2.0
    reward_position_tracking_fine_weight: float = 0.5
    reward_position_tracking_fine_std: float = 0.05
    reward_orientation_tracking_weight: float = 0.5
    
    # Regularization rewards
    reward_action_rate_weight: float = 0.02
    reward_joint_vel_weight: float = 0.001
    
    # ===== Success thresholds =====
    success_position_threshold: float = 0.02  # 2cm
    success_orientation_threshold: float = 0.2  # ~11.5 degrees