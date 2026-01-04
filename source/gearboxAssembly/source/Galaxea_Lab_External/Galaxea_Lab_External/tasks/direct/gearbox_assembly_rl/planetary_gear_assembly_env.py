from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform


from pxr import Usd, Sdf, UsdPhysics, UsdGeom
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.managers import SceneEntityCfg

import isaacsim.core.utils.torch as torch_utils

from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera

from isaaclab.managers import SceneEntityCfg

# from Galaxea_Lab_External.robots import GalaxeaRulePolicy

from .planetary_gear_assembly_env_cfg import PlanetaryGearAssemblyEnvCfg
from . import gearbox_assembly_utils
from ....jensen_lovers_agent.agent import GalaxeaGearboxAssemblyAgent
from ....jensen_lovers_agent.finite_state_machine import StateMachine, Context, InitializationState

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
        # self.left_arm_joint_pos  = None
        # self.right_arm_joint_pos = None
                







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


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # self.table = Articulation(self.cfg.table_cfg)
        # self.cfg.table_cfg.func("/World/envs/env_.*/Table", self.cfg.table_cfg)
        self.head_camera = Camera(self.cfg.head_camera_cfg)
        self.left_hand_camera = Camera(self.cfg.left_hand_camera_cfg)
        self.right_hand_camera = Camera(self.cfg.right_hand_camera_cfg)

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # print(f"_pre_physics_step actions: {self.actions}")

    def _apply_action(self) -> None:
        self.context.fsm.update()
        
        joint_command = self.agent.joint_position_command # (num_envs, n_joints)
        joint_ids = self.agent.joint_command_ids
        
        if joint_command is not None:
            self.robot.set_joint_position_target(
                joint_command, 
                joint_ids=joint_ids,
                env_ids=self.robot._ALL_INDICES
            )

        sim_dt = self.sim.get_physics_dt()
        for obj_name, obj in self.obj_dict.items():
            obj.update(sim_dt)

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

    def _get_rewards(self) -> torch.Tensor:
        scores_batch, _ = self.evaluate_score()
        reward_tensor = scores_batch.clone()

        return reward_tensor

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

    def _compute_intermediate_values(self):
        # Simulation ground truth data
        # Used member variables
        env_origins = self.scene.env_origins
        
        # Robot states
        left_arm_body_ids        = self.left_arm_entity_cfg.body_ids
        left_arm_joint_ids       = self.left_arm_entity_cfg.joint_ids
        right_arm_body_ids       = self.right_arm_entity_cfg.body_ids
        right_arm_joint_ids      = self.right_arm_entity_cfg.joint_ids

        self.left_ee_pos_e       = self.robot.data.body_state_w[:, left_arm_body_ids[0], 0:3] - env_origins
        self.left_ee_quat_w      = self.robot.data.body_state_w[:, left_arm_body_ids[0], 3:7]
        self.left_ee_linvel_w    = self.robot.data.body_state_w[:, left_arm_body_ids[0], 7:10]
        self.left_ee_angvel_w    = self.robot.data.body_state_w[:, left_arm_body_ids[0], 10:13]
        self.right_ee_pos_e      = self.robot.data.body_state_w[:, right_arm_body_ids[0], 0:3] - env_origins
        self.right_ee_quat_w     = self.robot.data.body_state_w[:, right_arm_body_ids[0], 3:7]
        self.right_ee_linvel_w   = self.robot.data.body_state_w[:, right_arm_body_ids[0], 7:10]
        self.right_ee_angvel_w   = self.robot.data.body_state_w[:, right_arm_body_ids[0], 10:13]

        self.left_arm_joint_pos  = self.robot.data.joint_pos[:, left_arm_joint_ids]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, right_arm_joint_ids]

    def _get_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        # Used member variables
        prev_actions = None

        obs_dict = {
            "fingertip_pos": 1,
            "fingertip_pos_rel_fixed": 2,
            "fingertip_quat": 3,
            "ee_linvel": 4,
            "ee_angvel": 5,
            "prev_actions": 6
        }
        state_dict = {
            "fingertip_pos": 1,
            "fingertip_pos_rel, fixed": 2,
            "fingertip_quat": 3,
            "ee_linvel": 4,
            "ee_angvel": 5,
            "joint_ps": 6,
            "held_pos": 7,
            "held_pos_rel_fixed": 8,
            "held_quat": 9,
            "fixed_pos": 10,
            "fixed_quat": 11,
            "task_prop_gains": 12,
            "pos_threshold": 13,
            "rot_threshold": 14,
            "prev_actions": 15
        }
        return obs_dict, state_dict

    def _get_observations(self) -> dict:
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_obs_state_dict()
        # obs_tensors = gearbox_assembly_utils.collapse_obs_dict(
        #    obs_dict=obs_dict,
        #    obs_order=self.cfg.obs_order + ["prev_actions"]
        #)
        #state_tensors = gearbox_assembly_utils.collapse_obs_dict(
        #    obs_dict=state_dict,
        #    obs_order=self.cfg.state_order + ["prev_actions"]
        #)

        # tempt keep for debugging
        obs_tensors = torch.cat(
            (
                self.left_arm_joint_pos,
                self.right_arm_joint_pos,
                self.left_gripper_joint_pos,
                self.right_gripper_joint_pos
            ),
            dim=-1,
        )
            
        observations = {"policy": obs_tensors}
        return observations