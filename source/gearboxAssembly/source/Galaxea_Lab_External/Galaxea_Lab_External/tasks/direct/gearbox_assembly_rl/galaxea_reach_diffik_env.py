# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Galaxea reach environment using Differential IK control."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from .galaxea_reach_diffik_env_cfg import GalaxeaReachDiffIKEnvCfg

from isaaclab.sim.spawners.materials import physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material

import isaacsim.core.utils.torch as torch_utils


class GalaxeaReachDiffIKEnv(DirectRLEnv):
    """Galaxea reach environment using Differential IK control."""
    
    cfg: GalaxeaReachDiffIKEnvCfg

    def __init__(self, cfg: GalaxeaReachDiffIKEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices
        self._left_arm_joint_idx, _ = self.robot.find_joints(self.cfg.left_arm_joint_dof_name)
        self._right_arm_joint_idx, _ = self.robot.find_joints(self.cfg.right_arm_joint_dof_name)
        self._left_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.left_gripper_dof_name)
        self._right_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.right_gripper_dof_name)
        self._torso_joint_idx, _ = self.robot.find_joints(self.cfg.torso_joint_dof_name)

        print(f"Left arm joints: {self._left_arm_joint_idx}")
        print(f"Right arm joints: {self._right_arm_joint_idx}")

        # Get body indices for end-effectors
        self.left_ee_body_idx = self.robot.body_names.index("left_arm_link6")
        self.right_ee_body_idx = self.robot.body_names.index("right_arm_link6")
        
        print(f"Left EE body index: {self.left_ee_body_idx}")
        print(f"Right EE body index: {self.right_ee_body_idx}")
        
        # Compute Jacobian index (for fixed base, frame index is body index - 1)
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.left_ee_body_idx - 1
        else:
            self.ee_jacobi_idx = self.left_ee_body_idx
        
        print(f"EE Jacobian index: {self.ee_jacobi_idx}")
        print(f"Robot is fixed base: {self.robot.is_fixed_base}")

        # Initialize Differential IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=self.cfg.ik_method,
        )
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        # Initialize tensors
        self._init_tensors()
        
        # Initialize visualization markers
        self._setup_markers()

    def _init_tensors(self):
        """Initialize tensors."""
        # Control targets
        self.joint_pos_des = torch.zeros((self.num_envs, len(self._left_arm_joint_idx)), device=self.device)
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        
        # IK command buffer (7-dim: position + quaternion)
        self.ik_commands = torch.zeros((self.num_envs, self.diff_ik_controller.action_dim), device=self.device)
        
        # Initialize target position and orientation
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos[:, 0] = 0.4  # x
        self.target_pos[:, 1] = 0.2  # y (left side)
        self.target_pos[:, 2] = 0.9  # z
        
        # Target orientation (z-axis pointing down toward ground)
        # 180 degree rotation around X-axis: quat = (w=0, x=1, y=0, z=0)
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 0] = 0.0  # w
        self.target_quat[:, 1] = 1.0  # x (180 deg rotation around X-axis)
        
        # EE state tensors
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Joint states
        self.joint_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        
        # Store previous actions for observation
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # Action thresholds
        self.pos_threshold = torch.tensor(self.cfg.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._initialize_scene()

    def _setup_markers(self):
        """Setup visualization markers for EE and target."""
        # Frame marker for current end-effector pose
        ee_marker_cfg = FRAME_MARKER_CFG.copy()
        ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        
        # Frame marker for goal/target pose
        goal_marker_cfg = FRAME_MARKER_CFG.copy()
        goal_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        self.goal_marker = VisualizationMarkers(goal_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    def _initialize_scene(self):
        """Initialize physics materials for gripper."""
        gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gripper_material", gripper_mat_cfg)
        
        num_envs = self.scene.num_envs
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(
                f"/World/envs/env_{env_idx}/Robot/left_gripper_link1/collisions", 
                "/World/Materials/gripper_material"
            )
            sim_utils.bind_physics_material(
                f"/World/envs/env_{env_idx}/Robot/left_gripper_link2/collisions", 
                "/World/Materials/gripper_material"
            )

    def _compute_intermediate_values(self):
        """Compute intermediate values needed for control."""
        # Get end-effector pose in world frame
        ee_pose_w = self.robot.data.body_pose_w[:, self.left_ee_body_idx]
        self.ee_pos = ee_pose_w[:, 0:3] - self.scene.env_origins
        self.ee_quat = ee_pose_w[:, 3:7]
        
        # Get root pose
        root_pose_w = self.robot.data.root_pose_w
        
        # Compute EE pose in body (root) frame for IK
        self.ee_pos_b, self.ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # Get EE velocities
        self.ee_linvel = self.robot.data.body_lin_vel_w[:, self.left_ee_body_idx]
        self.ee_angvel = self.robot.data.body_ang_vel_w[:, self.left_ee_body_idx]
        
        # Get joint states
        self.joint_pos = self.robot.data.joint_pos.clone()
        self.joint_vel = self.robot.data.joint_vel.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store previous actions and current actions."""
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions using Differential IK."""
        # Compute current EE state
        self._compute_intermediate_values()
        
        # Interpret actions as target pose deltas
        pos_actions = self.actions[:, 0:3] * self.pos_threshold
        rot_actions = self.actions[:, 3:6] * self.rot_threshold
        
        # Compute target position (current EE pos + delta)
        ctrl_target_ee_pos = self.ee_pos + pos_actions
        
        # Convert rotation actions to quaternion and apply to current orientation
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / (angle.unsqueeze(-1) + 1e-8)
        
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_ee_quat = torch_utils.quat_mul(rot_actions_quat, self.ee_quat)
        
        # Set IK command (in body frame for DifferentialIKController)
        # Convert target from world to body frame
        root_pose_w = self.robot.data.root_pose_w
        ctrl_target_ee_pos_w = ctrl_target_ee_pos + self.scene.env_origins
        ctrl_target_ee_pos_b, ctrl_target_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ctrl_target_ee_pos_w, ctrl_target_ee_quat
        )
        
        self.ik_commands[:, 0:3] = ctrl_target_ee_pos_b
        self.ik_commands[:, 3:7] = ctrl_target_ee_quat_b
        
        # Set command for IK controller
        self.diff_ik_controller.set_command(self.ik_commands)
        
        # Get Jacobian
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self._left_arm_joint_idx]
        
        # Compute joint position targets using IK
        self.joint_pos_des = self.diff_ik_controller.compute(
            self.ee_pos_b, self.ee_quat_b, jacobian, self.joint_pos[:, self._left_arm_joint_idx]
        )
        
        # Set joint position targets
        self.ctrl_target_joint_pos[:, self._left_arm_joint_idx] = self.joint_pos_des
        
        # Keep gripper open
        self.ctrl_target_joint_pos[:, self._left_gripper_dof_idx] = 0.04
        self.ctrl_target_joint_pos[:, self._right_gripper_dof_idx] = 0.04
        
        # Apply control
        self.robot.set_joint_position_target(self.ctrl_target_joint_pos)
        
        # Update visualization markers
        self._update_markers()

    def _update_markers(self):
        """Update visualization markers for EE and target poses."""
        # Visualize current end-effector pose (world frame)
        ee_pos_w = self.ee_pos + self.scene.env_origins
        self.ee_marker.visualize(ee_pos_w, self.ee_quat)
        
        # Visualize target pose (world frame)
        target_pos_w = self.target_pos + self.scene.env_origins
        self.goal_marker.visualize(target_pos_w, self.target_quat)

    def _get_observations(self) -> dict:
        """Get observations."""
        # Position relative to target
        ee_pos_to_target = self.target_pos - self.ee_pos
        
        obs = torch.cat(
            [
                ee_pos_to_target,      # 3: position error to target
                self.ee_quat,          # 4: current orientation
                self.ee_linvel,        # 3: linear velocity
                self.ee_angvel,        # 3: angular velocity
                self.prev_actions,     # 6: previous actions
            ],
            dim=-1,
        )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for reach task."""
        # Compute position error
        pos_error = torch.norm(self.target_pos - self.ee_pos, dim=-1)
        
        # Compute orientation error
        quat_diff = torch_utils.quat_mul(
            self.target_quat, 
            torch_utils.quat_conjugate(self.ee_quat)
        )
        rot_error = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0].abs(), 0.0, 1.0))
        
        # --- Task rewards ---
        # 1. Position tracking reward (linear penalty)
        pos_tracking_reward = -self.cfg.reward_position_tracking_weight * pos_error
        
        # 2. Position tracking fine-grained reward (tanh for smooth gradient near target)
        pos_tracking_fine_reward = self.cfg.reward_position_tracking_fine_weight * (
            1.0 - torch.tanh(pos_error / self.cfg.reward_position_tracking_fine_std)
        )
        
        # 3. Orientation tracking reward
        orientation_tracking_reward = -self.cfg.reward_orientation_tracking_weight * rot_error
        
        # --- Action penalty rewards ---
        # 4. Action rate penalty (penalize rapid changes in actions)
        action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
        action_rate_reward = -self.cfg.reward_action_rate_weight * action_rate
        
        # 5. Joint velocity penalty (penalize high joint velocities)
        joint_vel_left_arm = self.joint_vel[:, self._left_arm_joint_idx]
        joint_vel_l2 = torch.sum(torch.square(joint_vel_left_arm), dim=-1)
        joint_vel_reward = -self.cfg.reward_joint_vel_weight * joint_vel_l2
        
        # Total reward
        reward = (
            pos_tracking_reward
            + pos_tracking_fine_reward
            + orientation_tracking_reward
            + action_rate_reward
            + joint_vel_reward
        )
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Compute success
        pos_error = torch.norm(self.target_pos - self.ee_pos, dim=-1)
        quat_diff = torch_utils.quat_mul(
            self.target_quat, 
            torch_utils.quat_conjugate(self.ee_quat)
        )
        rot_error = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0].abs(), 0.0, 1.0))
        
        success = (pos_error < self.cfg.success_position_threshold) & (rot_error < self.cfg.success_orientation_threshold)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default joint positions
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        
        # Set robot pose
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset controller
        self.diff_ik_controller.reset()
        
        # Reset joint position targets
        self.joint_pos_des[env_ids] = joint_pos[env_ids][:, self._left_arm_joint_idx] if len(env_ids) > 0 else self.joint_pos_des[env_ids]
        
        # Reset buffers
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        
        # Randomize target position within configured ranges
        x_range, y_range, z_range = self.cfg.target_position_range
        self.target_pos[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
        self.target_pos[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (y_range[1] - y_range[0]) + y_range[0]
        self.target_pos[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (z_range[1] - z_range[0]) + z_range[0]