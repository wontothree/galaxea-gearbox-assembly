from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms


from pxr import Usd, Sdf, UsdPhysics, UsdGeom
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

import isaacsim.core.utils.torch as torch_utils

from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera

from isaaclab.managers import SceneEntityCfg

# from Galaxea_Lab_External.robots import GalaxeaRulePolicy

from .planetary_gear_assembly_rl_env_cfg import PlanetaryGearAssemblyRLEnvCfg
from . import gearbox_assembly_utils
from ....jensen_lovers_agent.agent import GalaxeaGearboxAssemblyAgent
from ....jensen_lovers_agent.finite_state_machine import StateMachine, Context, InitializationState

class PlanetaryGearAssemblyRLEnv(DirectRLEnv):
    cfg: PlanetaryGearAssemblyRLEnvCfg

    def __init__(self, cfg: PlanetaryGearAssemblyRLEnvCfg, render_mode: str | None = None, **kwargs):
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

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] _compute_intermediate_values ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        self.left_ee_pos_e       = None
        self.left_ee_quat_w      = None
        self.left_ee_linvel_w    = None
        self.left_ee_angvel_w    = None
        self.right_ee_pos_e      = None
        self.right_ee_quat_w     = None
        self.right_ee_linvel_w   = None
        self.right_ee_angvel_w   = None
        # self.left_arm_joint_pos  = None   # for preventing error
        # self.right_arm_joint_pos = None
        self.sun_planetary_gear_positions           = None
        self.sun_planetary_gear_quats               = None
        self.ring_gear_pos                          = None
        self.ring_gear_quat                         = None
        self.planetary_reducer_pos                  = None
        self.planetary_reducer_quat                 = None
        self.planetary_carrier_pos                  = None
        self.planetary_carrier_quat                 = None
        self.pin_positions                          = None
        self.pin_quats                              = None
        self.num_mounted_planetary_gears            = 0
        self.is_sun_gear_mounted                    = False
        self.is_ring_gear_mounted                   = False
        self.is_planetary_reducer_mounted           = False
        self.unmounted_sun_planetary_gear_positions = []
        self.unmounted_sun_planetary_gear_quats     = []
        self.unmounted_pin_positions                = []

        # ------------------------------------------------------
        self.agent = GalaxeaGearboxAssemblyAgent(
            sim=sim_utils.SimulationContext.instance(),
            scene=self.scene,
            obj_dict=self.obj_dict
        )
        self.context = Context(sim_utils.SimulationContext.instance(), self.agent)
        initial_state = InitializationState()
        fsm = StateMachine(initial_state, self.context)
        self.context.fsm = fsm
        # ------------------------------------------------------
		
        self.left_arm_entity_cfg = SceneEntityCfg(
            "robot",                            # robot entity name
            joint_names=["left_arm_joint.*"],   # joint entity set
            body_names=["left_arm_link6"]       # body entity set`
		)
        self.right_arm_entity_cfg = SceneEntityCfg(
				"robot",
				joint_names=["right_arm_joint.*"],
				body_names=["right_arm_link6"]
		)
        self.left_arm_entity_cfg.resolve(self.scene) # Resolving the scene entities
        self.right_arm_entity_cfg.resolve(self.scene)

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Differential IK Controller Setup ----------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        # Get body indices for end-effectors
        self.left_ee_body_idx = self.robot.body_names.index("left_arm_link6")
        self.right_ee_body_idx = self.robot.body_names.index("right_arm_link6")
        
        # Compute Jacobian index (for fixed base, frame index is body index - 1)
        if self.robot.is_fixed_base:
            self.left_ee_jacobi_idx = self.left_ee_body_idx - 1
            self.right_ee_jacobi_idx = self.right_ee_body_idx - 1
        else:
            self.left_ee_jacobi_idx = self.left_ee_body_idx
            self.right_ee_jacobi_idx = self.right_ee_body_idx
        
        print(f"Left EE body index: {self.left_ee_body_idx}, Jacobian index: {self.left_ee_jacobi_idx}")
        print(f"Right EE body index: {self.right_ee_body_idx}, Jacobian index: {self.right_ee_jacobi_idx}")
        
        # Initialize Differential IK controller for left arm
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=self.cfg.ik_method,
        )
        self.left_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self.right_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        
        # Initialize Diff IK related tensors
        self._init_diff_ik_tensors()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # self.table = Articulation(self.cfg.table_cfg)
        # self.cfg.table_cfg.func("/World/envs/env_.*/Table", self.cfg.table_cfg)

        # Disable cameras for now
        # self.head_camera = Camera(self.cfg.head_camera_cfg)
        # self.left_hand_camera = Camera(self.cfg.left_hand_camera_cfg)
        # self.right_hand_camera = Camera(self.cfg.right_hand_camera_cfg)

        self.table = sim_utils.spawn_from_usd("/World/envs/env_.*/Table", self.cfg.table_cfg.spawn,
            translation=self.cfg.table_cfg.init_state.pos, 
            orientation=self.cfg.table_cfg.init_state.rot
        )

        self.ring_gear = RigidObject(self.cfg.ring_gear_cfg)
        self.sun_planetary_gear_1 = RigidObject(self.cfg.sun_planetary_gear_1_cfg)
        self.sun_planetary_gear_2 = RigidObject(self.cfg.sun_planetary_gear_2_cfg)
        self.sun_planetary_gear_3 = RigidObject(self.cfg.sun_planetary_gear_3_cfg)
        self.sun_planetary_gear_4 = RigidObject(self.cfg.sun_planetary_gear_4_cfg)
        self.planetary_carrier = RigidObject(self.cfg.planetary_carrier_cfg)
        self.planetary_reducer = RigidObject(self.cfg.planetary_reducer_cfg)

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
            "ring_gear": self.ring_gear,
            "planetary_carrier": self.planetary_carrier,
            "sun_planetary_gear_1": self.sun_planetary_gear_1,
            "sun_planetary_gear_2": self.sun_planetary_gear_2,
            "sun_planetary_gear_3": self.sun_planetary_gear_3,
            "sun_planetary_gear_4": self.sun_planetary_gear_4,
            "planetary_reducer": self.planetary_reducer
        }

        self._initialize_scene()

    def _init_diff_ik_tensors(self):
        """Initialize tensors for Differential IK control."""
        # Control targets
        self.left_joint_pos_des = torch.zeros((self.num_envs, len(self._left_arm_joint_idx)), device=self.device)
        self.right_joint_pos_des = torch.zeros((self.num_envs, len(self._right_arm_joint_idx)), device=self.device)
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        
        # IK command buffer (7-dim: position + quaternion)
        self.left_ik_commands = torch.zeros((self.num_envs, self.left_diff_ik_controller.action_dim), device=self.device)
        self.right_ik_commands = torch.zeros((self.num_envs, self.right_diff_ik_controller.action_dim), device=self.device)
        
        # EE state tensors in body frame (for IK)
        self.left_ee_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.right_ee_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Full joint states
        self.full_joint_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.full_joint_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()
        # print(f"_pre_physics_step actions: {self.actions}")

    def get_key_points(self):
        # Used member variables
        num_envs = self.scene.num_envs

        # Pin positions
        # Calculate world positions of all pins
        planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        pin_world_positions = []
        pin_world_quats = []
        for pin_local_pos in self.pin_local_positions:
            pin_quat_batch = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
            pin_local_pos_batch = pin_local_pos.repeat(num_envs, 1)

            pin_world_quat, pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, 
                planetary_carrier_pos, 
                pin_quat_batch, 
                pin_local_pos_batch
            )

            pin_world_positions.append(pin_world_pos)
            pin_world_quats.append(pin_world_quat)

        gear_world_positions = []
        gear_world_quats = []
        
        gear_names = [
            'sun_planetary_gear_1', 
            'sun_planetary_gear_2',
            'sun_planetary_gear_3', 
            'sun_planetary_gear_4'
        ]
        for gear_name in gear_names:
            gear_obj = self.obj_dict[gear_name]
            gear_pos = gear_obj.data.root_state_w[:, :3].clone()
            gear_quat = gear_obj.data.root_state_w[:, 3:7].clone()

            gear_world_positions.append(gear_pos)
            gear_world_quats.append(gear_quat)
        
        # carrier_world_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        # carrier_world_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        ring_gear_world_pos = self.ring_gear.data.root_state_w[:, :3].clone()
        ring_gear_world_quat = self.ring_gear.data.root_state_w[:, 3:7].clone()

        reducer_world_pos = self.planetary_reducer.data.root_state_w[:, :3].clone()
        reducer_world_quat = self.planetary_reducer.data.root_state_w[:, 3:7].clone()

        return pin_world_positions, pin_world_quats, gear_world_positions, gear_world_quats, planetary_carrier_pos, planetary_carrier_quat, ring_gear_world_pos, ring_gear_world_quat, reducer_world_pos, reducer_world_quat

    def evaluate_score(self):
        pin_world_positions, pin_world_quats, gear_world_positions, gear_world_quats, planetary_carrier_pos, planetary_carrier_quat, ring_gear_world_pos, ring_gear_world_quat, reducer_world_pos, reducer_world_quat = self.get_key_points()
        score_batch = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]

            # Search how many gears are mounted to the planetary carrier
            for pin_idx in range(len(pin_world_positions)):
                pin_world_pos = pin_world_positions[pin_idx]
                pin_world_quat = pin_world_quats[pin_idx]

                distance = torch.norm(gear_world_pos[:, :2] - pin_world_pos[:, :2], dim=1)
                height_diff = gear_world_pos[:, 2] - pin_world_pos[:, 2]

                # Evaluate the angle between gear_world_quat and pin_world_quat
                dot_product = (gear_world_quat * pin_world_quat).sum(dim=-1)
                angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))

                # if distance < 0.002 and angle < 0.1 + 1 and height_diff < 0.012: # dismiss angle
                #     num_mounted_gears += 1
                mounted_mask = (distance < 0.002) & (angle < 1.1) & (height_diff < 0.012)
                score_batch += mounted_mask.float()

            # score += num_mounted_gears

        # Check whether the planetary carrier is mounted to the ring gear
        # distance = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_world_pos[:, :2])
        # height_diff = planetary_carrier_pos[:, 2] - ring_gear_world_pos[:, 2]
        # angle = torch.acos(torch.dot(planetary_carrier_quat.squeeze(0), ring_gear_world_quat.squeeze(0)))
        # if distance < 0.005 and angle < 0.1 and height_diff < 0.004:
        #     score += 1

        # Check whether the planetary carrier is mounted to the ring gear
        distance = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_world_pos[:, :2], dim=1)
        height_diff = planetary_carrier_pos[:, 2] - ring_gear_world_pos[:, 2]

        dot_product = (planetary_carrier_quat * ring_gear_world_quat).sum(dim=-1)
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        
        carrier_on_ring_mask = (distance < 0.005) & (angle < 0.1) & (height_diff < 0.004)
        score_batch += carrier_on_ring_mask.float() #

        # # Check whehter the gear is mount in the middle
        # for gear_idx in range(len(gear_world_positions)):
        #     gear_world_pos = gear_world_positions[gear_idx]
        #     gear_world_quat = gear_world_quats[gear_idx]
        #     distance = torch.norm(gear_world_pos[:, :2] - ring_gear_world_pos[:, :2])
        #     height_diff = gear_world_pos[:, 2] - ring_gear_world_pos[:, 2]
        #     angle = torch.acos(torch.dot(gear_world_quat.squeeze(0), ring_gear_world_quat.squeeze(0)))
        #     if distance < 0.005 and angle < 0.1 and height_diff < 0.004:
        #         score += 1
        # Check whether the gear is mount in the middle
        for gear_idx in range(len(gear_world_positions)):
            gear_pos = gear_world_positions[gear_idx]
            gear_quat = gear_world_quats[gear_idx]
            
            distance = torch.norm(gear_pos[:, :2] - ring_gear_world_pos[:, :2], dim=1)
            height_diff = gear_pos[:, 2] - ring_gear_world_pos[:, 2]
            dot_product = (gear_quat * ring_gear_world_quat).sum(dim=-1)
            angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            
            sun_gear_mask = (distance < 0.005) & (angle < 0.1) & (height_diff < 0.004)
            score_batch += sun_gear_mask.float()

        # # Check whether the reducer is mounted to the gear
        # for gear_idx in range(len(gear_world_positions)):
        #     gear_world_pos = gear_world_positions[gear_idx]
        #     gear_world_quat = gear_world_quats[gear_idx]
        #     distance = torch.norm(gear_world_pos[:, :2] - reducer_world_pos[:, :2])
        #     height_diff = gear_world_pos[:, 2] - reducer_world_pos[:, 2]
        #     angle = torch.acos(torch.dot(gear_world_quat.squeeze(0), reducer_world_quat.squeeze(0)))
        #     if distance < 0.005 and angle < 0.1 and height_diff < 0.002:
        #         score += 1
        # Check whether the reducer is mounted to the gear
        for gear_idx in range(len(gear_world_positions)):
            gear_pos = gear_world_positions[gear_idx]
            gear_quat = gear_world_quats[gear_idx]
            
            distance = torch.norm(gear_pos[:, :2] - reducer_world_pos[:, :2], dim=1)
            height_diff = gear_pos[:, 2] - reducer_world_pos[:, 2]
            dot_product = (gear_quat * reducer_world_quat).sum(dim=-1)
            angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            
            reducer_mask = (distance < 0.005) & (angle < 0.1) & (height_diff < 0.002)
            score_batch += reducer_mask.float()

        # time_cost = self.rule_policy.count * self.sim.get_physics_dt()
        time_cost = 0

        return score_batch, time_cost

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        score_batch, _ = self.evaluate_score()
        finish_task = score_batch >= 3.0
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
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/ring_gear/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_1/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_2/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_3/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_4/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_carrier/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_reducer/node_/mesh_", "/World/Materials/gear_material")
        
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
        # Adjust these based on your actual object sizes
        OBJECT_RADII = {
            'ring_gear': 0.1,              # Largest gear
            'sun_planetary_gear_1': 0.035,  # Small planetary gears
            'sun_planetary_gear_2': 0.035,
            'sun_planetary_gear_3': 0.035,
            'sun_planetary_gear_4': 0.035,
            'planetary_carrier': 0.07,     # Medium-large carrier
            'planetary_reducer': 0.04,     # Medium reducer
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

                    if obj_name == "ring_gear":
                        x = 0.24 + x_offset
                        y = 0.0
                    elif obj_name == "planetary_carrier":
                        x = 0.42 + x_offset 
                        y = 0.0
                    elif obj_name == "sun_planetary_gear_1":
                        y = torch.rand(1, device=self.device).item() * 0.3
                    elif obj_name == "sun_planetary_gear_2":
                        y = torch.rand(1, device=self.device).item() * 0.3
                    elif obj_name == "sun_planetary_gear_3":
                        y = -torch.rand(1, device=self.device).item() * 0.3
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

        self.initial_root_state = self._randomize_object_positions([
            self.ring_gear, self.planetary_carrier,
            self.sun_planetary_gear_1, self.sun_planetary_gear_2,
            self.sun_planetary_gear_3, self.sun_planetary_gear_4,
            self.planetary_reducer], ['ring_gear', 'planetary_carrier',
            'sun_planetary_gear_1', 'sun_planetary_gear_2',
            'sun_planetary_gear_3', 'sun_planetary_gear_4',
            'planetary_reducer'
        ])

        # self.rule_policy.set_initial_root_state(self.initial_root_state)
        # self.rule_policy.prepare_mounting_plan()

        # joint_pos = self.robot.data.default_joint_pos[env_ids, self._joint_idx]
        joint_pos = self.robot.data.default_joint_pos[env_ids][:, self._joint_idx]
        # joint_pos[:, self._joint_idx] += sample_uniform(
        #     self.cfg.initial_joint_angle_range[0] * math.pi,
        #     self.cfg.initial_joint_angle_range[1] * math.pi,
        #     joint_pos[:, self._joint_idx].shape,
        #     joint_pos.device,
        # )

        # print(f"shape of joint_pos: {joint_pos.shape}")

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        # print(f"default_root_state: {default_root_state}")

        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # print(f"default_root_state: {default_root_state}")
        # print(f"self.scene.env_origins[env_ids]: {self.scene.env_origins[env_ids]}")

        self.joint_pos[env_ids] = joint_pos
        # self.arm_joint_vel[env_ids] = arm_joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, None, self._joint_idx, env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, self._joint_idx, env_ids)
        self.robot.set_joint_position_target(joint_pos, self._joint_idx, env_ids)

        # self.robot.write_joint_position_to_sim(torch.tensor([28.6479 / 180.0 * math.pi, -45.8366 / 180.0 * math.pi, 28.6479 / 180.0 * math.pi], device=self.device), self._torso_joint_idx, env_ids)

    # ----------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------- #
    # Action ------------------------------------------------------------------------------------------------------------------- # 
    # -------------------------------------------------------------------------------------------------------------------------- #
    def _apply_action(self) -> None:
        """Apply actions using Differential IK for left arm only."""
        # Compute current EE state for IK
        self._compute_ee_state_for_ik()
        
        # Action space: 7-dim (left arm only)
        # [0:3] - left arm position delta
        # [3:6] - left arm rotation delta (axis-angle)
        # [6:7] - left gripper
        
        # --- Left Arm IK ---
        left_pos_actions = self.actions[:, 0:3] * self.pos_threshold
        left_rot_actions = self.actions[:, 3:6] * self.rot_threshold
        left_gripper_action = self.actions[:, 6:7]
        
        # Compute target position (current EE pos + delta)
        ctrl_target_left_ee_pos = self.left_ee_pos_e + left_pos_actions
        
        # Convert rotation actions to quaternion and apply to current orientation
        left_angle = torch.norm(left_rot_actions, p=2, dim=-1)
        left_axis = left_rot_actions / (left_angle.unsqueeze(-1) + 1e-8)
        left_rot_actions_quat = torch_utils.quat_from_angle_axis(left_angle, left_axis)
        left_rot_actions_quat = torch.where(
            left_angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            left_rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_left_ee_quat = torch_utils.quat_mul(left_rot_actions_quat, self.left_ee_quat_w)
        
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
        
        # Gripper control (scale from [-1, 1] to [0, 0.04])
        left_gripper_pos = (left_gripper_action.squeeze(-1) + 1.0) * 0.02  # [0, 0.04]
        self.ctrl_target_joint_pos[:, self._left_gripper_dof_idx[0]] = left_gripper_pos
        self.ctrl_target_joint_pos[:, self._left_gripper_dof_idx[1]] = left_gripper_pos
        
        # Apply control
        self.robot.set_joint_position_target(self.ctrl_target_joint_pos)
        
        # Update rigid objects
        sim_dt = self.sim.get_physics_dt()
        for obj_name, obj in self.obj_dict.items():
            obj.update(sim_dt)
    
    # def _apply_action_fsm(self) -> None:
    #     """Original FSM-based action application (deprecated)."""
    #     self.context.fsm.update()
    #     joint_command = self.agent.joint_position_command # (num_envs, n_joints)
    #     joint_ids = self.agent.joint_command_ids
    #     if joint_command is not None:
    #         self.robot.set_joint_position_target(
    #             joint_command, 
    #             joint_ids=joint_ids,
    #             env_ids=self.robot._ALL_INDICES
    #         )
    #
    #     sim_dt = self.sim.get_physics_dt()
    #     for obj_name, obj in self.obj_dict.items():
    #         obj.update(sim_dt)
    
    def _compute_ee_state_for_ik(self):
        """Compute EE states in both world and body frames for IK."""
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
        
        # Get right end-effector pose in world frame
        right_ee_pose_w = self.robot.data.body_pose_w[:, self.right_ee_body_idx]
        self.right_ee_pos_e = right_ee_pose_w[:, 0:3] - self.scene.env_origins
        self.right_ee_quat_w = right_ee_pose_w[:, 3:7]
        
        # Compute right EE pose in body (root) frame for IK
        self.right_ee_pos_b, self.right_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            right_ee_pose_w[:, 0:3], right_ee_pose_w[:, 3:7]
        )
        
        # Get joint states
        self.full_joint_pos = self.robot.data.joint_pos.clone()
        self.full_joint_vel = self.robot.data.joint_vel.clone()

    # -------------------------------------------------------------------------------------------------------------------------- #
    # Observation -------------------------------------------------------------------------------------------------------------- # 
    # -------------------------------------------------------------------------------------------------------------------------- #
    def _compute_intermediate_values(self):
        # Simulation ground truth data
        # Used member variables
        env_origins              = self.scene.env_origins
        num_envs                 = self.scene.num_envs
        left_arm_body_ids        = self.left_arm_entity_cfg.body_ids
        left_arm_joint_ids       = self.left_arm_entity_cfg.joint_ids
        right_arm_body_ids       = self.right_arm_entity_cfg.body_ids
        right_arm_joint_ids      = self.right_arm_entity_cfg.joint_ids

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Robot states ------------------------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        # End effector states
        self.left_ee_pos_e       = self.robot.data.body_state_w[:, left_arm_body_ids[0], 0:3] - env_origins
        self.left_ee_quat_w      = self.robot.data.body_state_w[:, left_arm_body_ids[0], 3:7]
        self.left_ee_linvel_w    = self.robot.data.body_state_w[:, left_arm_body_ids[0], 7:10]
        self.left_ee_angvel_w    = self.robot.data.body_state_w[:, left_arm_body_ids[0], 10:13]
        self.right_ee_pos_e      = self.robot.data.body_state_w[:, right_arm_body_ids[0], 0:3] - env_origins
        self.right_ee_quat_w     = self.robot.data.body_state_w[:, right_arm_body_ids[0], 3:7]
        self.right_ee_linvel_w   = self.robot.data.body_state_w[:, right_arm_body_ids[0], 7:10]
        self.right_ee_angvel_w   = self.robot.data.body_state_w[:, right_arm_body_ids[0], 10:13]

        # Arm joint positions
        self.left_arm_joint_pos  = self.robot.data.joint_pos[:, left_arm_joint_ids]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, right_arm_joint_ids]

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Object states ------------------------------------------------------------------------------------------------------------ #
        # -------------------------------------------------------------------------------------------------------------------------- #
        # Sun gear and planetary gears
        self.sun_planetary_gear_positions = []
        self.sun_planetary_gear_quats = []
        sun_planetary_gear_names = [
            'sun_planetary_gear_1', 
            'sun_planetary_gear_2', 
            'sun_planetary_gear_3', 
            'sun_planetary_gear_4'
        ]
        for sun_planetary_gear_name in sun_planetary_gear_names:
            gear_obj = self.obj_dict[sun_planetary_gear_name]
            gear_pos = gear_obj.data.root_state_w[:, :3].clone()
            gear_quat = gear_obj.data.root_state_w[:, 3:7].clone()

            self.sun_planetary_gear_positions.append(gear_pos)
            self.sun_planetary_gear_quats.append(gear_quat)

        # Ring gear
        self.ring_gear_pos = self.ring_gear.data.root_state_w[:, :3].clone()
        self.ring_gear_quat = self.ring_gear.data.root_state_w[:, 3:7].clone()

        # Planetary reducer
        self.planetary_reducer_pos = self.planetary_reducer.data.root_state_w[:, :3].clone()
        self.planetary_reducer_quat = self.planetary_reducer.data.root_state_w[:, 3:7].clone()

        # Planetary carrier
        self.planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        self.planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        # Pin in planetary carrier
        self.pin_positions = []
        self.pin_quats = []
        for pin_local_pos in self.pin_local_positions:
            pin_quat_repeated = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
            pin_local_pos_repeated = pin_local_pos.repeat(num_envs, 1)

            pin_quat, pin_pos = torch_utils.tf_combine(
                self.planetary_carrier_quat, 
                self.planetary_carrier_pos, 
                pin_quat_repeated, 
                pin_local_pos_repeated
            )

            self.pin_positions.append(pin_pos)
            self.pin_quats.append(pin_quat)

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Assembly states ---------------------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        # Used member variables
        pin_positions = self.pin_positions
        pin_quats = self.pin_quats
        sun_planetary_gear_positions = self.sun_planetary_gear_positions
        sun_planetary_gear_quats = self.sun_planetary_gear_quats
        planetary_carrier_pos = self.planetary_carrier_pos
        planetary_carrier_quat = self.planetary_carrier_quat
        ring_gear_pos = self.ring_gear_pos
        ring_gear_quat = self.ring_gear_quat
        planetary_reducer_pos = self.planetary_reducer_pos
        planetary_reducer_quat = self.planetary_reducer_quat

        # initialize
        self.num_mounted_planetary_gears = 0
        self.is_sun_gear_mounted = False
        self.is_ring_gear_mounted = False
        self.is_planetary_reducer_mounted = False
        self.unmounted_sun_planetary_gear_positions = []
        self.unmounted_sun_planetary_gear_quats = []
        self.unmounted_pin_positions = []
        
        # How many planetary gear mounted on planetary carrier?
        pin_occupied = [False] * len(pin_positions)
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions)):
            sun_planetary_gear_pos = sun_planetary_gear_positions[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats[sun_planetary_gear_idx]

            is_mounted = False
            for pin_idx in range(len(pin_positions)):
                pin_pos = pin_positions[pin_idx]
                pin_quat = pin_quats[pin_idx]

                horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - pin_pos[:, :2])
                vertical_error = sun_planetary_gear_pos[:, 2] - pin_pos[:, 2]
                # orientation_error = torch.acos(torch.dot(sun_planetary_gear_quat.squeeze(0), pin_quat.squeeze(0)))
                orientation_error = torch.acos(torch.clamp((sun_planetary_gear_quat * pin_quat).sum(dim=-1), -1.0, 1.0))

                th = self.mounting_thresholds["planetary_gear"]
                if (horizontal_error < th["horizontal"] and
                    vertical_error < th["vertical"] and
                    orientation_error < th["orientation"]):
                    self.num_mounted_planetary_gears += 1
                    is_mounted = True
                    pin_occupied[pin_idx] = True

            if not is_mounted:
                self.unmounted_sun_planetary_gear_positions.append(self.sun_planetary_gear_positions[sun_planetary_gear_idx])
                self.unmounted_sun_planetary_gear_quats.append(self.sun_planetary_gear_quats[sun_planetary_gear_idx])

        self.unmounted_pin_positions = [pin_positions[i] for i in range(len(pin_positions)) if not pin_occupied[i]]

        if len(self.unmounted_pin_positions) > 0 and len(self.unmounted_sun_planetary_gear_positions) > 0:
            gear_positions_batch = torch.stack(self.unmounted_sun_planetary_gear_positions, dim=0).squeeze(2) # [N, num_envs, 3]
            gear_quats_batch = torch.stack(self.unmounted_sun_planetary_gear_quats, dim=0).squeeze(2)         # [N, num_envs, 4]

            gear_xy = gear_positions_batch[..., :2]
            carrier_xy = self.planetary_carrier_pos[:, :2].unsqueeze(0) 
            
            gear_distances = torch.norm(gear_xy - carrier_xy, dim=2)
            
            sorted_gear_indices = torch.argsort(gear_distances, dim=0) # [N, num_envs]

            new_gear_pos_list = []
            new_gear_quat_list = []
            for i in range(sorted_gear_indices.shape[0]): 
                row_pos = torch.stack([gear_positions_batch[sorted_gear_indices[i, e], e] for e in range(num_envs)], dim=0)
                row_quat = torch.stack([gear_quats_batch[sorted_gear_indices[i, e], e] for e in range(num_envs)], dim=0)
                new_gear_pos_list.append(row_pos.unsqueeze(1)) # (num_envs, 1, 3) 유지
                new_gear_quat_list.append(row_quat.unsqueeze(1))

            self.unmounted_sun_planetary_gear_positions = new_gear_pos_list
            self.unmounted_sun_planetary_gear_quats = new_gear_quat_list

            pin_positions_tensor = torch.stack(self.unmounted_pin_positions, dim=0).squeeze(2) # [M, num_envs, 3]
            reference_gear_pos = self.unmounted_sun_planetary_gear_positions[0].squeeze(1)     # [num_envs, 3]
            
            pin_xy = pin_positions_tensor[..., :2]
            ref_gear_xy = reference_gear_pos[:, :2].unsqueeze(0)
            
            pin_distances = torch.norm(pin_xy - ref_gear_xy, dim=2) # [M, num_envs]
            sorted_pin_indices = torch.argsort(pin_distances, dim=0)

            new_pin_pos_list = []
            for i in range(sorted_pin_indices.shape[0]): # M개 핀만큼
                row_pin = torch.stack([pin_positions_tensor[sorted_pin_indices[i, e], e] for e in range(num_envs)], dim=0)
                new_pin_pos_list.append(row_pin.unsqueeze(1))
            
            self.unmounted_pin_positions = new_pin_pos_list

        # Is the sun gear mounted?
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions)):
            sun_planetary_gear_pos = sun_planetary_gear_positions[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats[sun_planetary_gear_idx]

            horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - ring_gear_pos[:, :2])
            vertical_error = sun_planetary_gear_pos[:, 2] - ring_gear_pos[:, 2]
            # orientation_error = torch.acos(torch.dot(sun_planetary_gear_quat.squeeze(0), ring_gear_quat.squeeze(0)))
            orientation_error = torch.acos(torch.clamp((sun_planetary_gear_quat * ring_gear_quat).sum(dim=-1), -1.0, 1.0))

            th = self.mounting_thresholds["sun_gear"]
            if (horizontal_error < th["horizontal"] and
                vertical_error < th["vertical"] and
                orientation_error < th["orientation"]):
                self.is_sun_gear_mounted = True

        # Is the ring gear mounted?
        horizontal_error = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_pos[:, :2])
        vertical_error = planetary_carrier_pos[:, 2] - ring_gear_pos[:, 2]
        # orientation_error = torch.acos(torch.dot(planetary_carrier_quat.squeeze(0), ring_gear_quat.squeeze(0)))
        orientation_error = torch.acos(torch.clamp((planetary_carrier_quat * ring_gear_quat).sum(dim=-1), -1.0, 1.0))

        th = self.mounting_thresholds["ring_gear"]
        if (horizontal_error < th["horizontal"] and
            vertical_error < th["vertical"] and
            orientation_error < th["orientation"]):
            self.is_ring_gear_mounted = True

        # Is the planetary reducer mounted?
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions)):
            sun_planetary_gear_pos = sun_planetary_gear_positions[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats[sun_planetary_gear_idx]

            horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - planetary_reducer_pos[:, :2])
            vertical_error = sun_planetary_gear_pos[:, 2] - planetary_reducer_pos[:, 2]
            # orientation_error = torch.acos(torch.dot(sun_planetary_gear_quat.squeeze(0), planetary_reducer_quat.squeeze(0)))
            orientation_error = torch.acos(torch.clamp((sun_planetary_gear_quat * planetary_reducer_quat).sum(dim=-1), -1.0, 1.0))

            th = self.mounting_thresholds["planetary_reducer"]
            if (horizontal_error < th["horizontal"] and
                vertical_error < th["vertical"] and
                orientation_error < th["orientation"]):
                self.is_planetary_reducer_mounted = True

    def _get_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        # Used member variables
        # prev_actions = None
        # gear_pos_e   = self.agent.target_pick_pos_w - self.scene.env_origins
        # gear_quat_w  = self.agent.target_pick_quat_w
        # pin_pos_e    = self.agent.target_place_pos_w - self.scene.env_origins

        # # Decide arm
        # arm_name = self.agent.active_arm_name
        # if arm_name == "left":
        #     ee_pos_e      = self.left_ee_pos_e
        #     ee_quat_w     = self.left_ee_quat_w
        #     ee_linevel_w  = self.left_ee_linvel_w
        #     ee_angvel_w   = self.left_ee_angvel_w
        #     arm_joint_pos = self.left_arm_joint_pos
        # elif arm_name == "right":
        #     ee_pos_e      = self.right_ee_pos_e
        #     ee_quat_w     = self.right_ee_quat_w
        #     ee_linevel_w  = self.right_ee_linvel_w
        #     ee_angvel_w   = self.right_ee_angvel_w
        #     arm_joint_pos = self.right_arm_joint_pos

        # obs_dict = {
        #     "fingertip_pos": ee_pos_e,
        #     "fingertip_pos_rel_fixed": ee_pos_e - pin_pos_e,
        #     "fingertip_quat": ee_quat_w,
        #     "ee_linvel": ee_linevel_w,
        #     "ee_angvel": ee_angvel_w,
        #     "prev_actions": prev_actions
        # }
        # state_dict = {
        #     "fingertip_pos": ee_pos_e,
        #     "fingertip_pos_rel_fixed": ee_pos_e - pin_pos_e,
        #     "fingertip_quat": ee_quat_w,
        #     "ee_linvel": ee_linevel_w,
        #     "ee_angvel": ee_angvel_w,
        #     "joint_ps": arm_joint_pos,
        #     "held_pos": gear_pos_e,
        #     "held_pos_rel_fixed": gear_pos_e - pin_pos_e,
        #     "held_quat": gear_quat_w,
        #     "fixed_pos": pin_pos_e,
        #     "fixed_quat": 11,
        #     "task_prop_gains": 12,
        #     "pos_threshold": 13,
        #     "rot_threshold": 14,
        #     "prev_actions": prev_actions
        # }

        obs_dict = {}
        state_dict = {}
        return obs_dict, state_dict

    def _get_observations(self) -> dict:
        """Get actor/critic inputs using left arm only."""
        obs_dict, state_dict = self._get_obs_state_dict()
        # obs_tensors = gearbox_assembly_utils.collapse_obs_dict(
        #    obs_dict=obs_dict,
        #    obs_order=self.cfg.obs_order + ["prev_actions"]
        #)
        #state_tensors = gearbox_assembly_utils.collapse_obs_dict(
        #    obs_dict=state_dict,
        #    obs_order=self.cfg.state_order + ["prev_actions"]
        #)

        # Left arm only observation
        obs_tensors = torch.cat(
            (
                self.left_arm_joint_pos,
                self.left_gripper_joint_pos
            ),
            dim=-1,
        )
            
        observations = {"policy": obs_tensors}
        return observations
    
    # -------------------------------------------------------------------------------------------------------------------------- #
    # Reward ------------------------------------------------------------------------------------------------------------------- # 
    # -------------------------------------------------------------------------------------------------------------------------- #
    def _get_rew_dict(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        rew_dict = {
            "kp_baseline"        : 1.0,
            "kp_coarse"          : 1.0,
            "kp_fine"            : 1.0,
            "action_penalty_ee"  : 1,
            "action_grad_penalty": 1,
            "curr_engaged"       : 1.0,
            "curr_success"       : 1.0
        }
        rew_scales = {
            "kp_baseline"        : 1.0,
            "kp_coarse"          : 1.0,
            "kp_fine"            : 1.0,
            "action_penalty_ee"  : 0.0,
            "action_grad_penalty": 0.0,
            "curr_engaged"       : 1.0,
            "curr_success"       : 1.0
        }

        return rew_dict, rew_scales
    
    def _get_rewards(self) -> torch.Tensor:
        scores_batch, _ = self.evaluate_score()
        reward_tensor = scores_batch.clone()

        return reward_tensor