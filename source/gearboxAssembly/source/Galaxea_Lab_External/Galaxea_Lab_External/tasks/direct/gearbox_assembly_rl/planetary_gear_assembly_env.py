from __future__ import annotations

import torch
from collections.abc import Sequence

import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.sim.spawners.materials import physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

import isaacsim.core.utils.torch as torch_utils

from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from .planetary_gear_assembly_env_cfg import PlanetaryGearAssemblyEnvCfg
from . import gearbox_assembly_utils

class PlanetaryGearAssemblyEnv(DirectRLEnv):
    cfg: PlanetaryGearAssemblyEnvCfg

    def __init__(self, cfg: PlanetaryGearAssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_arm_joint_idx, _ = self.robot.find_joints(self.cfg.left_arm_joint_dof_name)
        self._right_arm_joint_idx, _ = self.robot.find_joints(self.cfg.right_arm_joint_dof_name)
        self._left_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.left_gripper_dof_name)
        self._right_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.right_gripper_dof_name)

        self._torso_joint_idx, _ = self.robot.find_joints(self.cfg.torso_joint_dof_name)

        print(f"_left_arm_joint_idx: {self._left_arm_joint_idx}")
        print(f"_right_arm_joint_idx: {self._right_arm_joint_idx}")
        print(f"_left_gripper_dof_idx: {self._left_gripper_dof_idx}")
        print(f"_right_gripper_dof_idx: {self._right_gripper_dof_idx}")

        self._joint_idx = self._left_arm_joint_idx + self._right_arm_joint_idx + self._left_gripper_dof_idx + self._right_gripper_dof_idx

        self.left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_idx]
        self.left_gripper_joint_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_pos = self.robot.data.joint_pos[:, self._right_gripper_dof_idx]
        
        print(f"left_arm_joint_pos: {self.left_arm_joint_pos}")
        print(f"right_arm_joint_pos: {self.right_arm_joint_pos}")
        print(f"left_gripper_joint_pos: {self.left_gripper_joint_pos}")
        print(f"right_gripper_joint_pos: {self.right_gripper_joint_pos}")

        self.joint_pos = self.robot.data.joint_pos[:, self._joint_idx]

        # self.rule_policy = GalaxeaRulePolicy(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
        self.initial_root_state = None

        # Intermediate values for reward computation
        self.unmounted_pin_positions = []

        # Left arm entity configuration for observation computation
        self.left_arm_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["left_arm_joint.*"],
            body_names=["left_arm_link6"]
        )
        self.left_arm_entity_cfg.resolve(self.scene)

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Differential IK Controller Setup ----------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        # Get body index for left end-effector
        self.left_ee_body_idx = self.robot.body_names.index("left_arm_link6")
        
        # Compute Jacobian index (for fixed base, frame index is body index - 1)
        if self.robot.is_fixed_base:
            self.left_ee_jacobi_idx = self.left_ee_body_idx - 1
        else:
            self.left_ee_jacobi_idx = self.left_ee_body_idx
        
        # Initialize Differential IK controller for left arm
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=self.cfg.ik_method,
        )
        self.left_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        
        # Initialize Diff IK related tensors
        self._init_diff_ik_tensors()
        
        # Initialize visualization markers for pins
        self._setup_markers()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # self.table = Articulation(self.cfg.table_cfg)
        # self.cfg.table_cfg.func("/World/envs/env_.*/Table", self.cfg.table_cfg)

        self.table = sim_utils.spawn_from_usd("/World/envs/env_.*/Table", self.cfg.table_cfg.spawn,
            translation=self.cfg.table_cfg.init_state.pos, 
            orientation=self.cfg.table_cfg.init_state.rot
        )

        # Only spawn objects needed for the task
        self.sun_planetary_gear_4 = RigidObject(self.cfg.sun_planetary_gear_4_cfg)
        self.planetary_carrier = RigidObject(self.cfg.planetary_carrier_cfg)

        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2
        ]

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.obj_dict = {
            "planetary_carrier": self.planetary_carrier,
            "sun_planetary_gear_4": self.sun_planetary_gear_4,
        }

        self._initialize_scene()

    def _init_diff_ik_tensors(self):
        """Initialize tensors for Differential IK control."""
        # Control targets
        self.left_joint_pos_des = torch.zeros((self.num_envs, len(self._left_arm_joint_idx)), device=self.device)
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        
        # IK command buffer (7-dim: position + quaternion)
        self.left_ik_commands = torch.zeros((self.num_envs, self.left_diff_ik_controller.action_dim), device=self.device)
        
        # EE state tensors in body frame (for IK)
        self.left_ee_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Full joint states
        self.full_joint_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.full_joint_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        
        # Store previous actions for observation
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # Action scale factors
        self.pos_action_scale = torch.tensor(self.cfg.pos_action_scale, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_action_scale = torch.tensor(self.cfg.rot_action_scale, device=self.device).repeat(
            (self.num_envs, 1)
        )
        
        # Action smoothing and bounds (Factory-style)
        self.ema_factor = self.cfg.ema_factor
        self.pos_action_bounds = torch.tensor(self.cfg.pos_action_bounds, device=self.device)
        self.rot_action_bounds = self.cfg.rot_action_bounds
        
        # Tensors for finite-differencing EE velocity
        self.prev_left_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_left_ee_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def _setup_markers(self):
        """Setup visualization markers for pin positions, left EEF, and gear4."""
        # Create marker only for the first pin (target pin)
        pin_marker_cfg = FRAME_MARKER_CFG.copy()
        pin_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.pin_marker = VisualizationMarkers(pin_marker_cfg.replace(prim_path="/Visuals/pin_0"))
        
        # Create marker for left arm end-effector
        left_eef_marker_cfg = FRAME_MARKER_CFG.copy()
        left_eef_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.left_eef_marker = VisualizationMarkers(left_eef_marker_cfg.replace(prim_path="/Visuals/left_eef"))
        
        # Create marker for sun_planetary_gear_4 (held gear)
        gear4_marker_cfg = FRAME_MARKER_CFG.copy()
        gear4_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        self.gear4_marker = VisualizationMarkers(gear4_marker_cfg.replace(prim_path="/Visuals/gear4"))

    def _update_markers(self):
        """Update visualization markers for pin positions and left EEF."""
        # Get pin world positions and orientations
        pin_world_positions, pin_world_quats, _, _ = self.get_key_points()
        
        # Update marker only for the first pin (target pin)
        self.pin_marker.visualize(pin_world_positions[0], pin_world_quats[0])
        
        # Update marker for left arm end-effector (world frame) with Z-axis offset
        # Offset the EEF marker by 0.07m in the EEF's local Z direction
        left_eef_pos_w = self.left_ee_pos_e + self.scene.env_origins
        
        # Local Z-axis offset (0.07m)
        z_offset_local = torch.tensor([0.0, 0.0, 0.07], device=self.device).repeat(self.num_envs, 1)
        
        # Rotate the local offset to world frame using the EEF quaternion
        # quaternion: (w, x, y, z)
        z_offset_world = torch_utils.quat_rotate(self.left_ee_quat_w, z_offset_local)
        
        # Apply offset to EEF position
        left_eef_pos_w_offset = left_eef_pos_w + z_offset_world
        
        self.left_eef_marker.visualize(left_eef_pos_w_offset, self.left_ee_quat_w)
        
        # Update marker for sun_planetary_gear_4
        gear4_pos_w = self.sun_planetary_gear_4.data.root_pos_w
        gear4_quat_w = self.sun_planetary_gear_4.data.root_quat_w
        self.gear4_marker.visualize(gear4_pos_w, gear4_quat_w)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        # Apply EMA smoothing (Factory-style)
        self.actions = self.ema_factor * actions.clone() + (1 - self.ema_factor) * self.actions
        # print(f"_pre_physics_step actions: {self.actions}")

    def get_key_points(self):
        # Used member variables
        num_envs = self.scene.num_envs
        num_pins = len(self.pin_local_positions)

        # Pin positions
        # Calculate world positions of all pins (vectorized)
        planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        # Stack all pin local positions: (num_pins, 3) -> (num_envs, num_pins, 3)
        pin_local_pos_stacked = torch.stack(self.pin_local_positions, dim=0)  # (num_pins, 3)
        pin_local_pos_batch = pin_local_pos_stacked.unsqueeze(0).expand(num_envs, -1, -1)  # (num_envs, num_pins, 3)
        
        # Identity quaternion for all pins: (num_envs, num_pins, 4)
        pin_quat_batch = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).view(1, 1, 4).expand(num_envs, num_pins, -1)
        
        # Expand carrier pose for all pins: (num_envs, num_pins, 3/4)
        carrier_pos_expanded = planetary_carrier_pos.unsqueeze(1).expand(-1, num_pins, -1)  # (num_envs, num_pins, 3)
        carrier_quat_expanded = planetary_carrier_quat.unsqueeze(1).expand(-1, num_pins, -1)  # (num_envs, num_pins, 4)
        
        # Reshape for batch tf_combine: (num_envs * num_pins, 3/4)
        pin_world_quat_flat, pin_world_pos_flat = torch_utils.tf_combine(
            carrier_quat_expanded.reshape(-1, 4),
            carrier_pos_expanded.reshape(-1, 3),
            pin_quat_batch.reshape(-1, 4),
            pin_local_pos_batch.reshape(-1, 3)
        )
        
        # Reshape back and split into list: (num_envs, num_pins, 3/4) -> list of (num_envs, 3/4)
        pin_world_pos_all = pin_world_pos_flat.view(num_envs, num_pins, 3)
        pin_world_pos_all[:, :, 2] += 0.014  # Add 14mm offset in z-axis
        pin_world_quat_all = pin_world_quat_flat.view(num_envs, num_pins, 4)
        pin_world_positions = [pin_world_pos_all[:, i, :] for i in range(num_pins)]
        pin_world_quats = [pin_world_quat_all[:, i, :] for i in range(num_pins)]

        return pin_world_positions, pin_world_quats, planetary_carrier_pos, planetary_carrier_quat

    def evaluate_score(self):
        """Evaluate task success - only checks if gear4 is mounted on first pin."""
        pin_world_positions, pin_world_quats, planetary_carrier_pos, planetary_carrier_quat = self.get_key_points()
        score_batch = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # Get gear4 position
        gear4_pos = self.sun_planetary_gear_4.data.root_state_w[:, :3].clone()
        gear4_quat = self.sun_planetary_gear_4.data.root_state_w[:, 3:7].clone()

        # Check if gear4 is mounted on the first pin
        pin_world_pos = pin_world_positions[0]
        pin_world_quat = pin_world_quats[0]

        distance = torch.norm(gear4_pos[:, :2] - pin_world_pos[:, :2], dim=1)
        height_diff = gear4_pos[:, 2] - pin_world_pos[:, 2]

        dot_product = (gear4_quat * pin_world_quat).sum(dim=-1)
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))

        mounted_mask = (distance < 0.002) & (angle < 1.1) & (height_diff < 0.012)
        score_batch += mounted_mask.float()

        time_cost = 0
        return score_batch, time_cost

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        score_batch, _ = self.evaluate_score()
        finish_task = score_batch >= 1.0  # Task complete when gear4 is mounted
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return finish_task, time_out

    def _initialize_scene(self):
        gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
            # (optional) combination modes if you need them:
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gripper_material", gripper_mat_cfg)

        num_envs = self.scene.num_envs
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link1/collisions", "/World/Materials/gripper_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link2/collisions", "/World/Materials/gripper_material")  
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link1/collisions", "/World/Materials/gripper_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link2/collisions", "/World/Materials/gripper_material")

        gear_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=0.12,
            dynamic_friction=0.12,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gear_material", gear_mat_cfg)
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_4/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_carrier/node_/mesh_", "/World/Materials/gear_material")
        
        table_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/table_material", table_mat_cfg)
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Table/table/body_whiteLarge", "/World/Materials/table_material")
        
    def _randomize_object_positions(self, object_list: list, object_names: list,
                              safety_margin: float = 0.02, max_attempts: int = 1000):
        """Randomize positions of objects on table without overlapping

        This function places objects on the table surface ensuring they don't overlap by checking
        their bounding radii. Each object type has a defined approximate radius that represents
        its circular footprint on the table.

        Args:
            sim: Simulation context
            scene: Interactive scene containing the objects
            object_names: List of object names to randomize
            safety_margin: Additional safety distance to add between objects (in meters)
            max_attempts: Maximum attempts to find a non-overlapping position per object per environment

        Note:
            Adjust the OBJECT_RADII dictionary below if your objects have different sizes.
            To measure the radius of an object, observe it in the simulation and estimate
            the distance from its center to its furthest edge in the XY plane.
        """
        # Define approximate radii for each object type (in meters)
        # These values represent the circular bounding area of each object on the table surface
        OBJECT_RADII = {
            'sun_planetary_gear_4': 0.035,  # Held gear
            'planetary_carrier': 0.07,      # Target carrier with pins
        }

        x_offset = 0.2

        initial_root_state = {obj_name: torch.zeros((self.scene.num_envs, 7), device=self.device) for obj_name in object_names}

        num_envs = self.scene.num_envs

        # Store positions and object names of already placed objects for each environment
        # Each entry is a tuple: (position_tensor, object_name)
        placed_objects = [[] for _ in range(num_envs)]

        # for obj_name in object_names:
        for obj_idx, obj in enumerate(object_list):
            obj_name = object_names[obj_idx]
            # obj_cfg = SceneEntityCfg(obj_name, body_names=['node_'])
            # print(f"obj_cfg: {obj_cfg}")
            # obj_cfg.resolve(self.scene)

            # obj = self.scene[obj_name]
            # func = getattr(self, obj_name)
            # func = globals().get(obj_name)
            root_state = obj.data.default_root_state.clone()

            # Get radius for current object
            current_radius = OBJECT_RADII.get(obj_name, 0.05)  # Default to 0.05m if not specified

            # Generate non-overlapping positions for each environment
            for env_idx in range(num_envs):
                position_found = False

                for attempt in range(max_attempts):
                    # Generate random position
                    x = torch.rand(1, device=self.device).item() * 0.2 + 0.3 + x_offset  # range [0.3, 0.6]
                    y = torch.rand(1, device=self.device).item() * 0.6 - 0.3  # range [-0.3, 0.3]
                    z = 0.92

                    if obj_name == "planetary_carrier":
                        x = 0.42 + x_offset 
                        y = 0.0
                    elif obj_name == "sun_planetary_gear_4":
                        y = -torch.rand(1, device=self.device).item() * 0.3

                    pos = torch.tensor([x, y, z], device=self.device)

                    # Check for overlaps with already placed objects in this environment
                    is_valid = True
                    for placed_pos, placed_obj_name in placed_objects[env_idx]:
                        # Get radius of the already placed object
                        placed_radius = OBJECT_RADII.get(placed_obj_name, 0.05)

                        # Calculate minimum required distance (sum of radii + safety margin)
                        min_distance = current_radius + placed_radius + safety_margin

                        # Check only x, y distance (ignore z for table surface)
                        distance = torch.norm(pos[:2] - placed_pos[:2]).item()
                        if distance < min_distance:
                            is_valid = False
                            break

                    if is_valid:
                        # Position is valid, use it
                        world_pos = pos + self.scene.env_origins[env_idx]
                        root_state[env_idx, :3] = world_pos
                        
                        placed_objects[env_idx].append((pos, obj_name))
                        position_found = True
                        break

                if not position_found:
                    # Max attempts reached, use the last generated position anyway with a warning
                    print(f"[WARN] Could not find non-overlapping position for {obj_name} in env {env_idx} after {max_attempts} attempts.")
                    print(f"       This may indicate the table area is too crowded. Consider reducing the number of objects")
                    print(f"       or increasing the table area (x: [0.2, 0.5], y: [-0.3, 0.3]).")
                    world_pos = pos + self.scene.env_origins[env_idx]
                    root_state[env_idx, :3] = world_pos
                    
                    placed_objects[env_idx].append((pos, obj_name))

            # Write the state to simulation
            obj.write_root_state_to_sim(root_state)
            initial_root_state[obj_name] = root_state.clone()

        return initial_root_state

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Randomize object positions (only planetary_carrier and sun_planetary_gear_4)
        self.initial_root_state = self._randomize_object_positions([
            self.planetary_carrier,
            self.sun_planetary_gear_4
        ], [
            'planetary_carrier',
            'sun_planetary_gear_4'
        ])

        # Set robot to default pose first
        joint_pos = self.robot.data.default_joint_pos[env_ids][:, self._joint_idx]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, self._joint_idx, env_ids)
        self.robot.set_joint_position_target(joint_pos, self._joint_idx, env_ids)
        
        # Initialize gripper to open position
        gearbox_assembly_utils.set_gripper_open(
            self.robot, self.ctrl_target_joint_pos, self._left_gripper_dof_idx, env_ids
        )
        self._step_sim_no_action()
        
        # Perform grasp initialization (Factory-style)
        gearbox_assembly_utils.initialize_grasp(
            self, env_ids, self.sun_planetary_gear_4,
            grasp_height_offset=0.15, held_offset_z=0.07, grasp_duration=0.25
        )
        
        # Align EEF with first pin position at episode start
        self._align_eef_with_pin(env_ids)
        
        # Compute EE state after alignment to get accurate positions
        self._compute_ee_state_for_ik()
        
        # Initialize previous EE state for velocity computation (must be after alignment)
        self.prev_left_ee_pos[env_ids] = self.left_ee_pos_e[env_ids].clone()
        self.prev_left_ee_quat[env_ids] = self.left_ee_quat_w[env_ids].clone()
    
    def _step_sim_no_action(self):
        """Step the simulation without an action. Used for resets only."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_ee_state_for_ik()

    def _align_eef_with_pin(self, env_ids):
        """Align EEF with first pin position at episode start."""
        # Get first pin position (with 14mm offset)
        pin_world_positions, _, _, _ = self.get_key_points()
        target_pin_pos = pin_world_positions[0][env_ids] - self.scene.env_origins[env_ids]
        
        # Add vertical offset to hover above pin (e.g., 5cm above)
        hover_offset = torch.tensor([0.0, 0.0, 0.05], device=self.device).repeat(len(env_ids), 1)
        target_eef_pos = target_pin_pos + hover_offset
        
        # Use downward-facing orientation
        target_eef_quat = gearbox_assembly_utils.constrain_quat_to_downward(
            self.left_ee_quat_w[env_ids], self.device
        )
        
        # Convert target from environment to world frame
        target_eef_pos_w = target_eef_pos + self.scene.env_origins[env_ids]
        
        # Apply IK to move EEF to target position
        root_pose_w = self.robot.data.root_pose_w[env_ids]
        target_eef_pos_b, target_eef_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_eef_pos_w, target_eef_quat
        )
        
        # Create a temporary IK controller with the correct batch size
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=self.cfg.ik_method,
        )
        temp_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=len(env_ids), device=self.device)
        
        # Set IK command
        ik_commands = torch.zeros((len(env_ids), temp_ik_controller.action_dim), device=self.device)
        ik_commands[:, 0:3] = target_eef_pos_b
        ik_commands[:, 3:7] = target_eef_quat_b
        
        # Get current EE state for selected envs
        self._compute_ee_state_for_ik()
        left_ee_pos_b_selected = self.left_ee_pos_b[env_ids]
        left_ee_quat_b_selected = self.left_ee_quat_b[env_ids]
        
        # Get Jacobian for left arm
        left_jacobian = self.robot.root_physx_view.get_jacobians()[env_ids, self.left_ee_jacobi_idx, :, :][:, :, self._left_arm_joint_idx]
        
        # Compute joint position targets
        joint_pos_current = self.full_joint_pos[env_ids][:, self._left_arm_joint_idx]
        
        # Use the temporary controller for this batch
        for _ in range(50):  # Iterate to converge to target
            temp_ik_controller.set_command(ik_commands)
            joint_pos_des = temp_ik_controller.compute(
                left_ee_pos_b_selected, left_ee_quat_b_selected, left_jacobian, joint_pos_current
            )
            
            # Update joint positions
            self.ctrl_target_joint_pos[env_ids][:, self._left_arm_joint_idx] = joint_pos_des
            self.robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
            
            # Step simulation
            self._step_sim_no_action()
            
            # Update for next iteration
            self._compute_ee_state_for_ik()
            left_ee_pos_b_selected = self.left_ee_pos_b[env_ids]
            left_ee_quat_b_selected = self.left_ee_quat_b[env_ids]
            joint_pos_current = self.full_joint_pos[env_ids][:, self._left_arm_joint_idx]
            left_jacobian = self.robot.root_physx_view.get_jacobians()[env_ids, self.left_ee_jacobi_idx, :, :][:, :, self._left_arm_joint_idx]

    # ----------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------- #
    # Action ------------------------------------------------------------------------------------------------------------------- # 
    # -------------------------------------------------------------------------------------------------------------------------- #
    def _apply_action(self) -> None:
        """Apply actions using Differential IK for left arm only."""
        # Compute current EE state for IK
        self._compute_ee_state_for_ik()
        
        # Action space: 4-dim (constrained to downward-facing EEF, gripper always closed)
        # [0:3] - left arm position delta (x, y, z)
        # [3:4] - yaw rotation only (pitch and roll are fixed)
        
        # --- Left Arm IK ---
        left_pos_actions = self.actions[:, 0:3] * self.pos_action_scale
        left_yaw_action = self.actions[:, 3:4] * self.rot_action_scale[:, 2:3]  # Use only Z component of rot_action_scale
        
        # Compute target position (current EE pos + delta)
        ctrl_target_left_ee_pos = self.left_ee_pos_e + left_pos_actions
        
        # Apply position bounds clipping (Factory-style)
        # Get first pin position as reference frame
        pin_world_positions, _, _, _ = self.get_key_points()
        first_pin_pos = pin_world_positions[0] - self.scene.env_origins
        delta_pos = ctrl_target_left_ee_pos - first_pin_pos
        # Clip XY and Z separately
        delta_pos_xy_clipped = torch.clamp(delta_pos[:, 0:2], -self.pos_action_bounds[0], self.pos_action_bounds[0])
        delta_pos_z_clipped = torch.clamp(delta_pos[:, 2:3], -self.pos_action_bounds[1], self.pos_action_bounds[1])
        delta_pos_clipped = torch.cat([delta_pos_xy_clipped, delta_pos_z_clipped], dim=-1)
        ctrl_target_left_ee_pos = first_pin_pos + delta_pos_clipped
        
        # Apply rotation bounds clipping (Factory-style)
        left_yaw_action_clipped = torch.clamp(left_yaw_action, -self.rot_action_bounds, self.rot_action_bounds)
        
        # Convert yaw rotation action to quaternion and apply to current orientation
        # Only apply yaw rotation, keeping roll and pitch fixed to face downward
        yaw_angle = left_yaw_action_clipped.squeeze(-1)
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        yaw_rot_quat = torch_utils.quat_from_angle_axis(yaw_angle, yaw_axis)
        yaw_rot_quat = torch.where(
            (yaw_angle.unsqueeze(-1).abs().repeat(1, 4)) > 1e-6,
            yaw_rot_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        # Apply yaw rotation to current orientation
        ctrl_target_left_ee_quat_temp = torch_utils.quat_mul(yaw_rot_quat, self.left_ee_quat_w)
        
        # Constrain EEF orientation to always face downward (use helper function)
        ctrl_target_left_ee_quat = gearbox_assembly_utils.constrain_quat_to_downward(ctrl_target_left_ee_quat_temp, self.device)
        
        # Convert target from world to body frame for IK
        root_pose_w = self.robot.data.root_pose_w
        ctrl_target_left_ee_pos_w = ctrl_target_left_ee_pos + self.scene.env_origins
        ctrl_target_left_ee_pos_b, ctrl_target_left_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ctrl_target_left_ee_pos_w, ctrl_target_left_ee_quat
        )
        
        self.left_ik_commands[:, 0:3] = ctrl_target_left_ee_pos_b
        self.left_ik_commands[:, 3:7] = ctrl_target_left_ee_quat_b
        self.left_diff_ik_controller.set_command(self.left_ik_commands)
        
        # Get Jacobian for left arm
        left_jacobian = self.robot.root_physx_view.get_jacobians()[:, self.left_ee_jacobi_idx, :, self._left_arm_joint_idx]
        
        # Compute joint position targets using IK
        self.left_joint_pos_des = self.left_diff_ik_controller.compute(
            self.left_ee_pos_b, self.left_ee_quat_b, left_jacobian, self.full_joint_pos[:, self._left_arm_joint_idx]
        )
        
        # --- Set joint position targets ---
        self.ctrl_target_joint_pos[:, self._left_arm_joint_idx] = self.left_joint_pos_des
        
        # Gripper control - always closed at 0.0 (Factory-style)
        self.ctrl_target_joint_pos[:, self._left_gripper_dof_idx[0]] = 0.0
        self.ctrl_target_joint_pos[:, self._left_gripper_dof_idx[1]] = 0.0
        
        # Apply control
        self.robot.set_joint_position_target(self.ctrl_target_joint_pos)
        
        # Update visualization markers
        self._update_markers()
        
        # Update rigid objects
        sim_dt = self.sim.get_physics_dt()
        for obj_name, obj in self.obj_dict.items():
            obj.update(sim_dt)
    
    def _compute_ee_state_for_ik(self):
        """Compute left EE states in both world and body frames for IK."""
        # Get root pose
        root_pose_w = self.robot.data.root_pose_w
        
        # Get left end-effector pose in world frame
        left_ee_pose_w = self.robot.data.body_pose_w[:, self.left_ee_body_idx]
        self.left_ee_pos_e = left_ee_pose_w[:, 0:3] - self.scene.env_origins
        self.left_ee_quat_w = left_ee_pose_w[:, 3:7]
        
        # Compute left EE pose in body (root) frame for IK
        self.left_ee_pos_b, self.left_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7]
        )
        
        # Get joint states
        self.full_joint_pos = self.robot.data.joint_pos.clone()
        self.full_joint_vel = self.robot.data.joint_vel.clone()
    
    def _compute_ee_velocity(self):
        """Compute EE linear and angular velocity using finite differencing."""
        dt = self.physics_dt * self.cfg.decimation
        
        # Linear velocity with clipping
        ee_linvel = (self.left_ee_pos_e - self.prev_left_ee_pos) / dt
        ee_linvel = torch.clamp(ee_linvel, -10.0, 10.0)  # Clip to [-10, 10] m/s
        
        # Angular velocity from quaternion difference
        ee_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        quat_diff = torch_utils.quat_mul(self.left_ee_quat_w, torch_utils.quat_conjugate(self.prev_left_ee_quat))
        
        # Convert quaternion difference to axis-angle
        angle = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))
        axis = quat_diff[:, 1:4]
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        
        # Avoid division by zero
        mask = axis_norm.squeeze() > 1e-6
        ee_angvel[mask] = (axis[mask] / axis_norm[mask]) * (angle[mask].unsqueeze(-1) / dt)
        ee_angvel = torch.clamp(ee_angvel, -20.0, 20.0)  # Clip to [-20, 20] rad/s
        
        # Check for NaN/Inf and replace with zeros
        ee_linvel = torch.where(torch.isfinite(ee_linvel), ee_linvel, torch.zeros_like(ee_linvel))
        ee_angvel = torch.where(torch.isfinite(ee_angvel), ee_angvel, torch.zeros_like(ee_angvel))
        
        # Update previous values
        self.prev_left_ee_pos = self.left_ee_pos_e.clone()
        self.prev_left_ee_quat = self.left_ee_quat_w.clone()
        
        return ee_linvel, ee_angvel

    # -------------------------------------------------------------------------------------------------------------------------- #
    # Observation -------------------------------------------------------------------------------------------------------------- # 
    # -------------------------------------------------------------------------------------------------------------------------- #
    def _compute_intermediate_values(self):
        """Compute assembly state values needed for reward computation."""
        # No intermediate computation needed - always use first pin
        pass

    def _get_planetary_gear_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        # Get first pin position (with 14mm offset)
        pin_world_positions, _, _, _ = self.get_key_points()
        first_pin_pos = pin_world_positions[0] - self.scene.env_origins  # Environment frame
        
        # Compute EE velocity
        ee_linvel, ee_angvel = self._compute_ee_velocity()
        
        # Get gear4 pose
        gear4_pos = self.sun_planetary_gear_4.data.root_pos_w - self.scene.env_origins
        gear4_quat = self.sun_planetary_gear_4.data.root_quat_w
        
        # Get joint states
        left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        left_gripper_joint_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx]
        
        # Previous actions
        prev_actions = self.prev_actions.clone()
        
        # Observation dict (for policy - partial observability)
        obs_dict = {
            "ee_pos": self.left_ee_pos_e,
            "ee_pos_rel_pin": self.left_ee_pos_e - first_pin_pos,
            "ee_quat": self.left_ee_quat_w,
            "ee_linvel": ee_linvel,
            "ee_angvel": ee_angvel,
            "prev_actions": prev_actions,
        }
        
        # State dict (for critic - full state information)
        state_dict = {
            "ee_pos": self.left_ee_pos_e,
            "ee_pos_rel_pin": self.left_ee_pos_e - first_pin_pos,
            "ee_quat": self.left_ee_quat_w,
            "ee_linvel": ee_linvel,
            "ee_angvel": ee_angvel,
            "prev_actions": prev_actions,
            "left_arm_joint_pos": left_arm_joint_pos,
            "left_gripper_joint_pos": left_gripper_joint_pos,
            "gear4_pos": gear4_pos,
            "gear4_pos_rel_pin": gear4_pos - first_pin_pos,
            "gear4_quat": gear4_quat,
            "pin_pos": first_pin_pos,
        }
        
        return obs_dict, state_dict

    def _get_observations(self) -> dict:
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_planetary_gear_obs_state_dict()
        
        # Collapse observation dict to tensor following obs_order
        obs_tensors = torch.cat(
            [obs_dict[key] for key in self.cfg.obs_order],
            dim=-1,
        )
        
        # Collapse state dict to tensor following state_order
        state_tensors = torch.cat(
            [state_dict[key] for key in self.cfg.state_order],
            dim=-1,
        )
        
        # Safety check: replace NaN/Inf with zeros
        obs_tensors = torch.where(torch.isfinite(obs_tensors), obs_tensors, torch.zeros_like(obs_tensors))
        state_tensors = torch.where(torch.isfinite(state_tensors), state_tensors, torch.zeros_like(state_tensors))
        
        return {"policy": obs_tensors, "critic": state_tensors}
    
    # -------------------------------------------------------------------------------------------------------------------------- #
    # Reward ------------------------------------------------------------------------------------------------------------------- # 
    # -------------------------------------------------------------------------------------------------------------------------- #
    def _get_curr_successes(self, success_threshold: float) -> torch.Tensor:
        """Check if gear4 is successfully placed on the first pin."""
        # Get gear4 position
        gear4_pos = self.sun_planetary_gear_4.data.root_pos_w - self.scene.env_origins
        
        # Always use first pin as target
        pin_world_positions, _, _, _ = self.get_key_points()
        target_pin_pos = pin_world_positions[0] - self.scene.env_origins
        
        # Compute XY distance to target pin
        dist = torch.norm(gear4_pos[:, :2] - target_pin_pos[:, :2], dim=1)
        
        return dist < success_threshold
    
    def _get_rew_dict(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Compute reward terms at current timestep (Factory-style)."""
        # Get gear4 (held asset) position
        gear4_pos = self.sun_planetary_gear_4.data.root_pos_w - self.scene.env_origins
        gear4_quat = self.sun_planetary_gear_4.data.root_quat_w
        
        # Always use first pin as target
        pin_world_positions, pin_world_quats, _, _ = self.get_key_points()
        target_pin_pos = pin_world_positions[0] - self.scene.env_origins
        target_pin_quat = pin_world_quats[0]
        
        # Compute keypoints on gear4 and target positions (vectorized)
        num_keypoints = self.cfg.num_keypoints
        keypoint_offsets = gearbox_assembly_utils.get_keypoint_offsets(num_keypoints, self.device) * self.cfg.keypoint_scale
        
        # Expand for batch processing: (num_keypoints, 3) -> (num_envs, num_keypoints, 3)
        offsets_batch = keypoint_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        
        # Expand quaternions: (num_envs, 4) -> (num_envs, num_keypoints, 4)
        gear4_quat_expanded = gear4_quat.unsqueeze(1).expand(-1, num_keypoints, -1)
        target_pin_quat_expanded = target_pin_quat.unsqueeze(1).expand(-1, num_keypoints, -1)
        
        # Rotate all offsets at once (vectorized)
        rotated_offsets_held = torch_utils.quat_rotate(
            gear4_quat_expanded.reshape(-1, 4), 
            offsets_batch.reshape(-1, 3)
        ).reshape(self.num_envs, num_keypoints, 3)
        
        rotated_offsets_target = torch_utils.quat_rotate(
            target_pin_quat_expanded.reshape(-1, 4),
            offsets_batch.reshape(-1, 3)
        ).reshape(self.num_envs, num_keypoints, 3)
        
        # Compute keypoint positions
        keypoints_held = gear4_pos.unsqueeze(1) + rotated_offsets_held
        keypoints_target = target_pin_pos.unsqueeze(1) + rotated_offsets_target
        
        # Compute mean keypoint distance
        keypoint_dist = torch.norm(keypoints_held - keypoints_target, p=2, dim=-1).mean(dim=-1)
        
        # Get coefficients from config
        a0, b0 = self.cfg.keypoint_coef_baseline
        a1, b1 = self.cfg.keypoint_coef_coarse
        a2, b2 = self.cfg.keypoint_coef_fine
        
        # Action penalties
        action_penalty_ee = torch.norm(self.actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        
        # Success checks
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg.engage_threshold)
        curr_success = self._get_curr_successes(success_threshold=self.cfg.success_threshold)
        
        rew_dict = {
            "kp_baseline": gearbox_assembly_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": gearbox_assembly_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": gearbox_assembly_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_success.float(),
        }
        
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }

        return rew_dict, rew_scales
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute total reward from reward dictionary."""
        # Update intermediate values (unmounted_pin_positions)
        self._compute_intermediate_values()
        
        rew_dict, rew_scales = self._get_rew_dict()
        
        # Sum all reward components with their scales
        rew_buf = torch.zeros((self.num_envs,), device=self.device)
        for rew_name, rew in rew_dict.items():
            rew_buf += rew * rew_scales[rew_name]
        
        return rew_buf