import torch
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

import carb.input
from carb.input import KeyboardEventType
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils


from pxr import Usd, Sdf, UsdPhysics, UsdGeom
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material

from isaaclab.sim import SimulationContext

class GalaxeaRulePolicy:
    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene, obj_dict: dict):
        self.sim = sim
        self.scene = scene
        # self.arm_name = arm_name
        self.device = sim.device

        # Object state
        self.obj_dict = obj_dict
        self.planetary_carrier = obj_dict["planetary_carrier"]
        self.ring_gear = obj_dict["ring_gear"]
        self.sun_planetary_gear_1 = obj_dict["sun_planetary_gear_1"]
        self.sun_planetary_gear_2 = obj_dict["sun_planetary_gear_2"]
        self.sun_planetary_gear_3 = obj_dict["sun_planetary_gear_3"]
        self.sun_planetary_gear_4 = obj_dict["sun_planetary_gear_4"]
        self.planetary_reducer = obj_dict["planetary_reducer"]

        # Define pin positions in local coordinates relative to planetary carrier
        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2
        ]

        self.TCP_offset_z = 1.1475 - 1.05661
        self.TCP_offset_x = 0.3864 - 0.3785
        self.table_height = 0.9
        self.grasping_height = -0.003

        self.diff_ik_controller, self.left_arm_entity_cfg, self.left_gripper_entity_cfg = self.get_config("left")
        self.diff_ik_controller, self.right_arm_entity_cfg, self.right_gripper_entity_cfg = self.get_config("right")

        self.right_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids
        self.left_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids

        # was commented
        self.target_position_left = torch.tensor([0.3864, 0.5237, 1.1475], device=self.device)
        self.target_orientation_left = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        self.target_position_right = torch.tensor([0.3864, -0.5237, 1.1475], device=self.device)
        self.target_orientation_right = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)

        self.num_gripper_joints = None

        self.gear_to_pin_map = None
        
        self.current_target_position = None
        self.current_target_orientation = None

        self.sim_dt = sim.get_physics_dt()
        print(f"sim_dt: {self.sim_dt}")
        self.count = 0

        # Time for intital stabilization
        self.time_step_0 = 0.2
        self.count_step_0 = int(self.time_step_0 / self.sim_dt)
        print(f"count_step_0: {self.count_step_0}")

        # Times for each step
        # 1. Move the arm to the target position above the gear and keep the orientation
        # 2. Move the arm to the target position and keep the orientation
        # 3. Close the gripper
        # 4. Move the arm to the target position above the gear and keep the orientation
        # time_step_1 = torch.tensor([0.0, 5.0, 1.0, 2.0, 1.0, 2.0], device=sim.device)
        self.time_step_1 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_1 = torch.cumsum(self.time_step_1, dim=0) + self.time_step_0
        self.count_step_1 = self.time_step_1 / self.sim_dt
        self.count_step_1 = self.count_step_1.int()
        print(f"count_step_1: {self.count_step_1}")

        # Mount the gear to the planetary_carrier
        self.time_step_2 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_2 = torch.cumsum(self.time_step_2, dim=0) + self.time_step_1[-1]
        self.count_step_2 = self.time_step_2 / self.sim_dt
        self.count_step_2 = self.count_step_2.int()
        print(f"count_step_2: {self.count_step_2}")

        # Pick up the 2nd gear
        self.time_step_3 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_3 = torch.cumsum(self.time_step_3, dim=0) + self.time_step_2[-1]
        self.count_step_3 = self.time_step_3 / self.sim_dt
        self.count_step_3 = self.count_step_3.int()
        print(f"count_step_3: {self.count_step_3}")

        # Mount the 2nd gear to the planetary_carrier
        self.time_step_4 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_4 = torch.cumsum(self.time_step_4, dim=0) + self.time_step_3[-1]
        self.count_step_4 = self.time_step_4 / self.sim_dt
        self.count_step_4 = self.count_step_4.int()
        print(f"count_step_4: {self.count_step_4}")

        # Pick up the 3rd gear
        self.time_step_5 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_5 = torch.cumsum(self.time_step_5, dim=0) + self.time_step_4[-1]
        self.count_step_5 = self.time_step_5 / self.sim_dt
        self.count_step_5 = self.count_step_5.int()
        print(f"count_step_5: {self.count_step_5}")

        # Mount the 3rd gear to the planetary_carrier
        self.time_step_6 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_6 = torch.cumsum(self.time_step_6, dim=0) + self.time_step_5[-1]
        self.count_step_6 = self.time_step_6 / self.sim_dt
        self.count_step_6 = self.count_step_6.int()
        print(f"count_step_6: {self.count_step_6}")

        # Reset right arm
        self.time_step_7 = torch.tensor([0.0, 0.5], device=sim.device)
        self.time_step_7 = torch.cumsum(self.time_step_7, dim=0) + self.time_step_6[-1]
        self.count_step_7 = self.time_step_7 / self.sim_dt
        self.count_step_7 = self.count_step_7.int()
        print(f"count_step_7: {self.count_step_7}")

        # Pick up the carrier
        self.time_step_8 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_8 = torch.cumsum(self.time_step_8, dim=0) + self.time_step_7[-1]
        self.count_step_8 = self.time_step_8 / self.sim_dt
        self.count_step_8 = self.count_step_8.int()
        print(f"count_step_8: {self.count_step_8}")

        # Mount the carrier on the ring gear
        self.time_step_9 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_9 = torch.cumsum(self.time_step_9, dim=0) + self.time_step_8[-1]
        self.count_step_9 = self.time_step_9 / self.sim_dt
        self.count_step_9 = self.count_step_9.int()
        print(f"count_step_9: {self.count_step_9}")

        # Pick up the 4th gear
        self.time_step_10 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_10 = torch.cumsum(self.time_step_10, dim=0) + self.time_step_9[-1]
        self.count_step_10 = self.time_step_10 / self.sim_dt
        self.count_step_10 = self.count_step_10.int()
        print(f"count_step_10: {self.count_step_10}")

        # Mount the 4th gear to the planetary_carrier
        self.time_step_11 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_11 = torch.cumsum(self.time_step_11, dim=0) + self.time_step_10[-1]
        self.count_step_11 = self.time_step_11 / self.sim_dt
        self.count_step_11 = self.count_step_11.int()
        print(f"count_step_11: {self.count_step_11}")

        # Pick up the reducer
        self.time_step_12 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_12 = torch.cumsum(self.time_step_12, dim=0) + self.time_step_11[-1]
        self.count_step_12 = self.time_step_12 / self.sim_dt
        self.count_step_12 = self.count_step_12.int()
        print(f"count_step_12: {self.count_step_12}")

        # Mount the reducer to the gear
        self.time_step_13 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_13 = torch.cumsum(self.time_step_13, dim=0) + self.time_step_12[-1]
        self.count_step_13 = self.time_step_13 / self.sim_dt
        self.count_step_13 = self.count_step_13.int()
        print(f"count_step_13: {self.count_step_13}")

        left_init_pos = [0.3864, 0.5237, 1.1475]
        left_init_rot = [0.0, -1.0, 0.0, 0.0]
        right_init_pos = [0.3864, -0.5237, 1.1475]
        right_init_rot = [0.0, -1.0, 0.0, 0.0]

        self.initial_root_state = None



        # add
        self.agent = Galaxear1GearboxAssemblyAgent(
            sim=sim,
            scene=scene,
            obj_dict=obj_dict
        )
        self.context = Context(sim, self.agent)
        initial_state = InitializationState()
        fsm = StateMachine(initial_state, self.context)
        self.context.fsm = fsm

    def set_initial_root_state(self, initial_root_state: dict):
        self.initial_root_state = initial_root_state.copy()

    def get_config(self, arm_name: str):
        # arm_name: left or right

        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device
        )

        # Specify robot-specific parameters
        arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{arm_name}_arm_joint.*"], body_names=[f"{arm_name}_arm_link6"]
        )
        gripper_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{arm_name}_gripper_axis.*"]
        )

        # Resolving the scene entities
        arm_entity_cfg.resolve(self.scene)
        gripper_entity_cfg.resolve(self.scene)
        
        # gripper_entity_cfg = SceneEntityCfg("robot", joint_names=[f"{arm_name}_gripper_.*"], body_names=[f"{arm_name}_gripper_link1"])
        # gripper_entity_cfg.resolve(self.scene)
        
        return diff_ik_controller, arm_entity_cfg, gripper_entity_cfg

    # End Effector Control by Differential Inverse Kinematics
    def move_robot_to_position(self,
                            arm_entity_cfg: SceneEntityCfg,
                            gripper_entity_cfg: SceneEntityCfg,
                            diff_ik_controller: DifferentialIKController,
                            target_position: torch.Tensor, 
                            target_orientation: torch.Tensor,
                            target_marker: VisualizationMarkers):
        robot = self.scene["robot"]

        arm_joint_ids = arm_entity_cfg.joint_ids
        arm_body_ids = arm_entity_cfg.body_ids
        num_arm_joints = len(arm_joint_ids)

        gripper_joint_ids = gripper_entity_cfg.joint_ids
        gripper_body_ids = gripper_entity_cfg.body_ids
        self.num_gripper_joints = len(gripper_joint_ids)

        if robot.is_fixed_base:
            ee_jacobi_idx = arm_body_ids[0] - 1
        else:
            ee_jacobi_idx = arm_body_ids[0]

        # Get the target position and orientation of the arm
        # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
        ik_commands = torch.cat([target_position, target_orientation], dim=-1)
        diff_ik_controller.set_command(ik_commands)

        # IK solver
        # obtain quantities from simulation
        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, arm_entity_cfg.joint_ids
        ]
        ee_pose_w = robot.data.body_state_w[
            :, arm_body_ids[0], 0:7
        ]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        # compute the joint commands
        joint_pos_des = diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )

        # print(f"ee_pos_b: {ee_pos_b}, ee_quat_b: {ee_quat_b}")
        # print(f"joint_pos_des: {joint_pos_des}")

        # Apply
        # print(f"joint_pos_des: {joint_pos_des}")
        # robot.set_joint_position_target(
        #     joint_pos_des, joint_ids=arm_entity_cfg.joint_ids
        # )

        return joint_pos_des, arm_entity_cfg.joint_ids
        


    def prepare_mounting_plan(self, gear_names: list = None):
        """
        Plan which gear mounts to which arm and which pin by finding the nearest arm and pin for each gear.
        Each pin can only be used once.

        Args:
            sim: Simulation context
            scene: Interactive scene containing the robot and objects
            left_arm_entity_cfg: Configuration for left arm
            right_arm_entity_cfg: Configuration for right arm
            initial_root_state: Dictionary containing initial states of all objects
            gear_names: List of gear names to plan for (e.g., ['sun_planetary_gear_1', 'sun_planetary_gear_2'])
                    If None, defaults to all 4 sun planetary gears

        Returns:
            gear_to_pin_map: Dictionary mapping gear_name -> {'arm': 'left'/'right', 'pin': pin_index,
                                                                'pin_world_pos': tensor, 'pin_world_quat': tensor}
        """

        # Default to all 4 gears if not specified
        if gear_names is None:
            gear_names = ['sun_planetary_gear_1', 'sun_planetary_gear_2',
                        'sun_planetary_gear_3', 'sun_planetary_gear_4']

        # Get the planetary carrier positions and orientations
        root_state = self.initial_root_state["planetary_carrier"]
        planetary_carrier_pos = root_state[:, :3].clone()
        planetary_carrier_quat = root_state[:, 3:7].clone()
        num_envs = planetary_carrier_pos.shape[0]

        # Calculate world positions of all pins in planetary carrier
        pin_world_positions = []
        pin_world_quats = []
        for pin_local_pos in self.pin_local_positions:
            pin_local_pos_batch = pin_local_pos.unsqueeze(0).expand(num_envs, -1)
            pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(num_envs, -1)

            pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, pin_quat, pin_local_pos_batch)[1]

            pin_world_positions.append(pin_world_pos)
            pin_world_quats.append(pin_quat)

        # Stack all pin positions: shape (num_envs, num_pins, 3)
        pin_world_positions = torch.stack(pin_world_positions, dim=1)
        pin_world_quats = torch.stack(pin_world_quats, dim=1)

        # Get end-effector positions for both arms
        left_ee_pos = self.scene["robot"].data.body_state_w[:, self.left_arm_entity_cfg.body_ids[0], 0:3]
        right_ee_pos = self.scene["robot"].data.body_state_w[:, self.right_arm_entity_cfg.body_ids[0], 0:3]

        # Track which pins are occupied
        occupied_pins = set()

        # Result mapping
        self.gear_to_pin_map = {}

        # For each gear, find nearest arm and nearest available pin
        for gear_name in gear_names:
            if gear_name not in self.initial_root_state:
                print(f"[WARN] Gear {gear_name} not found in initial_root_state, skipping")
                continue

            # Get gear position
            gear_pos = self.initial_root_state[gear_name][:, :3].clone()  # shape: (num_envs, 3)

            # For the first environment (env_idx=0), calculate distances to both arms
            env_idx = 0
            gear_pos_env = gear_pos[env_idx]

            # Calculate distance to both arms
            left_dist = torch.norm(gear_pos_env - left_ee_pos[env_idx])
            right_dist = torch.norm(gear_pos_env - right_ee_pos[env_idx])

            # Choose the nearest arm
            if left_dist < right_dist:
                chosen_arm = 'left'
                chosen_arm_pos = left_ee_pos[env_idx]
            else:
                chosen_arm = 'right'
                chosen_arm_pos = right_ee_pos[env_idx]

            # Find the nearest available pin
            min_pin_dist = float('inf')
            nearest_pin_idx = None

            for pin_idx in range(len(self.pin_local_positions)):
                if pin_idx in occupied_pins:
                    continue  # Skip occupied pins

                pin_pos = pin_world_positions[env_idx, pin_idx]
                pin_dist = torch.norm(gear_pos_env - pin_pos)

                if pin_dist < min_pin_dist:
                    min_pin_dist = pin_dist
                    nearest_pin_idx = pin_idx

            if nearest_pin_idx is None:
                print(f"[WARN] No available pins for gear {gear_name}, all pins occupied!")
                continue

            # Mark this pin as occupied
            occupied_pins.add(nearest_pin_idx)

            # Store the mapping
            self.gear_to_pin_map[gear_name] = {
                'arm': chosen_arm,
                'pin': nearest_pin_idx,
                'pin_local_pos': self.pin_local_positions[nearest_pin_idx],
                'pin_world_pos': pin_world_positions[:, nearest_pin_idx],  # All environments
                'pin_world_quat': pin_world_quats[:, nearest_pin_idx],
            }

            print(f"[INFO] {gear_name} -> {chosen_arm} arm, pin_{nearest_pin_idx}")
            print(f"       Gear pos: {gear_pos_env}, Pin pos: {pin_world_positions[env_idx, nearest_pin_idx]}")
            print(f"       Distance: {min_pin_dist:.4f}m")

        return self.gear_to_pin_map


    # Step 1: Pick up the sun_planetary_gear_1
    # grasping_state = 1
    # joint_pos = None
    def pick_up_sun_planetary_gear(self,
                                    gear_id: int,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg,
                                    diff_ik_controller: DifferentialIKController):
        
        sim_dt = self.sim.get_physics_dt()

        obj_height_offset = 0.0

        if gear_id == 5:
            # root_state = initial_root_state["planetary_carrier"]
            planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
            planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

            # print(f"planetary_carrier_pos: {planetary_carrier_pos}")
            # print(f"planetary_carrier_quat: {planetary_carrier_quat}")

            # planetary_carrier_pos = root_state[:, :3].clone()
            # planetary_carrier_quat = root_state[:, 3:7].clone()
            local_pos = torch.tensor([0.0, 0.054, 0], device=self.sim.device).unsqueeze(0)

            target_orientation, target_position = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, 
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), local_pos
            )
            root_state = torch.cat([target_position, target_orientation], dim=-1)

        elif gear_id == 6: # Reducer
            root_state = self.initial_root_state["planetary_reducer"]
            # target_position = root_state[:, :3].clone()
            obj_height_offset = 0.03

        else:
            root_state = self.initial_root_state[f"sun_planetary_gear_{gear_id}"]
        # print(f"obj: {obj}")
        # target_position, target_orientation = target_frame.get_local_pose()
        # target_position, target_orientation = target_frame.get_world_poses()
        target_position = root_state[:, :3].clone()
        target_position[:, 2] = self.table_height + self.grasping_height + obj_height_offset
        target_position = target_position + torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        # target_orientation = obj.data.default_root_state[:, 3:7].clone()
        # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
        # Step 1.1: Move the arm to the target position above the gear and keep the orientation
        target_position_h = target_position + torch.tensor([0.0, 0.0, 0.1], device=self.sim.device)
        
        # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
        target_orientation = root_state[:, 3:7].clone()
        # Rotate the target orientation 180 degrees around the y-axis
        target_orientation, target_position = torch_utils.tf_combine(
            target_orientation, target_position, 
            torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.sim.device), torch.tensor([[0.0, 0.0, 0.0]], device=self.sim.device)
        )

        # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
        # print(f"target_position_h: {target_position_h}, target_orientation: {target_orientation}")



        # ik_commands = torch.cat([target_position, target_orientation], dim=-1,)

        # print("scene['robot'].data.joint_pos: ", scene["robot"].data.joint_pos)
        if self.count >= count_step[0] and self.count < count_step[1]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                    target_position_h, target_orientation, None)
        
        # Step 1.2: Open the gripper
        gripper_joint_ids = gripper_entity_cfg.joint_ids
        gripper_body_ids = gripper_entity_cfg.body_ids
        num_gripper_joints = len(gripper_joint_ids)

        # Step 1.3: Move the arm to the target position and keep the orientation
        # target_position_2 = target_position
        # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
        if self.count >= count_step[1] and self.count < count_step[2]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
            #                         target_position, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                    target_position, target_orientation, None)
            # target_marker.visualize(target_position, target_orientation)

        # Step 1.4: Close the gripper
        
        if self.count >= count_step[2] and self.count < count_step[3]:
            # gripper_joint_pos_des = torch.full(
            #         (num_gripper_joints,), 0.0, device=self.sim.device
            #     )
            # self.scene["robot"].set_joint_position_target(
            #         gripper_joint_pos_des, joint_ids=gripper_joint_ids
            #     )
            action = torch.tensor([[0.0, 0.0]], device=self.sim.device)
            joint_ids = gripper_joint_ids


        # Step 1.5: Move the arm to the target position above the gear and keep the orientation
        # target_position = target_position + torch.tensor([0.0, 0.0, 0.1 + TCP_offset], device=sim.device)
        # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
        if self.count >= count_step[3] and self.count < count_step[4]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # gripper_joint_pos_des = torch.full(
            #         (num_gripper_joints,), 0.0, device=self.sim.device
            #     )
            # self.scene["robot"].set_joint_position_target(
            #         gripper_joint_pos_des, joint_ids=gripper_joint_ids
            #     )
            # target_marker.visualize(target_position_h, target_orientation)

        return action, joint_ids


    def mount_gear_to_planetary_carrier(self,
                                    gear_id: int,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg):

        obj_height_offset = 0.0

        if gear_id == 5: # Place the carrier on the ring gear
            root_state = self.initial_root_state["ring_gear"]
            ring_gear_pos = root_state[:, :3].clone()
            ring_gear_quat = root_state[:, 3:7].clone()
            local_pos = torch.tensor([0.0, 0.054, 0.0], device=self.sim.device).unsqueeze(0)
            # local_pos = torch.tensor([0.054, 0.0, 0.0], device=sim.device).unsqueeze(0)

            if self.count == count_step[0]:
                self.current_target_orientation, self.current_target_position = torch_utils.tf_combine(
                    ring_gear_quat, ring_gear_pos, 
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), local_pos
                )
            # root_state = target_position
            # target_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)
        elif gear_id == 4:
            root_state = self.planetary_carrier.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
            # planetary_carrier_quat = root_state[:, 3:7].clone()

        elif gear_id == 6: # Reducer
            root_state = self.sun_planetary_gear_4.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
            obj_height_offset = 0.02


        else: # Mount the gear on the planetary carrier
            # root_state = initial_root_state[f"sun_planetary_gear_{gear_id}"]

            planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
            planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()
            original_planetary_carrier_pos = self.initial_root_state["planetary_carrier"][:, :3].clone()
            original_planetary_carrier_quat = self.initial_root_state["planetary_carrier"][:, 3:7].clone()

            # Local pose of the pin
            pin_local_pos = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['pin_local_pos'].clone()
            # Transfer the local pose of the pin to the world frame after the planetary carrier is moved
            # target_orientation = planetary_carrier_quat.clone()
            target_orientation, pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, 
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), pin_local_pos.unsqueeze(0)
            )
            _, original_pin_world_pos = torch_utils.tf_combine(
                original_planetary_carrier_quat, original_planetary_carrier_pos, 
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), pin_local_pos.unsqueeze(0)
            )
            if self.count == count_step[0]:
                self.current_target_position = pin_world_pos.clone()
            # target_orientation = planetary_carrier_quat.clone()
        
        # target_marker.visualize(target_position, target_orientation)

        # print(f"self.current_target_position: {self.current_target_position}")

        target_position = self.current_target_position.clone()


        target_position[:, 2] = self.table_height + self.grasping_height
        target_position[:, 2] += obj_height_offset

        target_position += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        
        target_position_h = target_position + torch.tensor([0.0, 0.0, 0.1], device=self.sim.device)
        target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.sim.device)

        target_position_h_down = target_position + torch.tensor([0.0, 0.0, 0.02], device=self.sim.device)

        if self.count >= count_step[0] and self.count < count_step[1]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)

        if self.count >= count_step[1] and self.count < count_step[2]:

            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h_down, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h_down, target_orientation, None)
            # target_marker.visualize(target_position_h_down, target_orientation)

        if self.count >= count_step[2] and self.count < count_step[3]:
            gripper_joint_ids = gripper_entity_cfg.joint_ids
            gripper_body_ids = gripper_entity_cfg.body_ids
            num_gripper_joints = len(gripper_joint_ids)

            gripper_joint_pos_des = torch.full(
                    (num_gripper_joints,), 0.04, device=self.device
                )

            if gear_id == 5:
                gripper_joint_pos_des = torch.full(
                    (num_gripper_joints,), 0.017, device=self.device
                )

            # self.scene["robot"].set_joint_position_target(
            #         gripper_joint_pos_des, joint_ids=gripper_joint_ids
            #     )
            action = gripper_joint_pos_des.unsqueeze(0)
            joint_ids = gripper_joint_ids
            
        if self.count >= count_step[3] and self.count < count_step[4]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)

        return action, joint_ids

    def get_action(self):
        if self.count < self.count_step_0:
            # left arm
            # left_action, left_arm_joint_ids = self.move_robot_to_position(self.left_arm_entity_cfg, 
            #                         self.left_gripper_entity_cfg, self.diff_ik_controller, 
            #                         self.target_position_left, self.target_orientation_left, None)
            # # right arm
            # right_action, right_arm_joint_ids = self.move_robot_to_position(self.right_arm_entity_cfg, self.right_gripper_entity_cfg, self.diff_ik_controller, 
            #                         self.target_position_right, self.target_orientation_right, None)
            
            # # gripper_joint_pos_des = torch.full(
            # #     (self.num_gripper_joints,), 0.04, device=self.sim.device
            # # )
            # gripper_joint_pos_des = torch.tensor([[0.04, 0.04]], device=self.sim.device)

            # # self.scene["robot"].set_joint_position_target(
            # #     gripper_joint_pos_des, joint_ids=self.right_gripper_joint_ids
            # # )
            # # self.scene["robot"].set_joint_position_target(
            # #     gripper_joint_pos_des, joint_ids=self.left_gripper_joint_ids
            # # )

            # # print(f"left_action: {left_action}")
            # # print(f"right_action: {right_action}")
            # # print(f"gripper_joint_pos_des: {gripper_joint_pos_des}")

            # action = torch.cat([left_action, right_action, gripper_joint_pos_des, gripper_joint_pos_des], dim=-1)
            # joint_ids = left_arm_joint_ids + right_arm_joint_ids + self.right_gripper_joint_ids + self.left_gripper_joint_ids

            action = None
            joint_ids = None

        # Pick up the 1st gear
        if self.count >= self.count_step_1[0] and self.count < self.count_step_1[-1]:
            gear_id = 1
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg

            action, joint_ids = self.pick_up_sun_planetary_gear(gear_id, self.count_step_1, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 1st gear to the planetary_carrier
        if self.count >= self.count_step_2[0] and self.count < self.count_step_2[-1]:
            gear_id = 1
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
                
            action, joint_ids = self.mount_gear_to_planetary_carrier(gear_id, self.count_step_2, current_arm, current_gripper)

        # Pick up the 2nd gear
        if self.count >= self.count_step_3[0] and self.count < self.count_step_3[-1]:
            gear_id = 2
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_sun_planetary_gear(gear_id, self.count_step_3, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 2nd gear to the planetary_carrier
        if self.count >= self.count_step_4[0] and self.count < self.count_step_4[-1]:
            gear_id = 2
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_planetary_carrier(gear_id, self.count_step_4, current_arm, current_gripper)

        # Pick up the 3rd gear
        if self.count >= self.count_step_5[0] and self.count < self.count_step_5[-1]:
            gear_id = 3
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_sun_planetary_gear(gear_id, self.count_step_5, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 3rd gear to the planetary_carrier
        if self.count >= self.count_step_6[0] and self.count < self.count_step_6[-1]:
            gear_id = 3
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_planetary_carrier(gear_id, self.count_step_6, current_arm, current_gripper)

        # Reset right arm
        if self.count >= self.count_step_7[0] and self.count < self.count_step_7[-1]:
            action, joint_ids = self.move_robot_to_position(self.right_arm_entity_cfg, self.right_gripper_entity_cfg, self.diff_ik_controller, 
                                    self.target_position_right, self.target_orientation_right, None)

        # Pick up the carrier
        if self.count >= self.count_step_8[0] and self.count < self.count_step_8[-1]:
            gear_id = 5
            current_arm = self.left_arm_entity_cfg
            current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.pick_up_sun_planetary_gear(gear_id, self.count_step_8, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the carrier on the ring gear
        if self.count >= self.count_step_9[0] and self.count < self.count_step_9[-1]:
            gear_id = 5
            current_arm = self.left_arm_entity_cfg
            current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_planetary_carrier(gear_id, self.count_step_9, current_arm, current_gripper)

        # Pick up the 4th gear
        if self.count >= self.count_step_10[0] and self.count < self.count_step_10[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_sun_planetary_gear(gear_id, self.count_step_10, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 4th gear to the planetary_carrier
        if self.count >= self.count_step_11[0] and self.count < self.count_step_11[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_planetary_carrier(gear_id, self.count_step_11, current_arm, current_gripper)

        # Pick up the reducer
        if self.count >= self.count_step_12[0] and self.count < self.count_step_12[-1]:
            gear_id = 6
            # Reducer location
            pos = self.initial_root_state["planetary_reducer"][:, :3].clone()
            if pos[0, 1] < 0.0:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.pick_up_sun_planetary_gear(gear_id, self.count_step_12, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the reducer to the gear
        if self.count >= self.count_step_13[0] and self.count < self.count_step_13[-1]:
            gear_id = 6
            # Reducer location
            pos = self.initial_root_state["planetary_reducer"][:, :3].clone()
            if pos[0, 1] < 0.0:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_planetary_carrier(gear_id, self.count_step_13, current_arm, current_gripper)

        # self.print_inner_state()
        self.context.fsm.update()
        action, joint_ids = self.agent.joint_position_command, self.agent.joint_command_ids

        return action, joint_ids

    # ...
    def print_inner_state(self):
        # Initial object state [x, y, z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, w_x, w_y, w_z]
        # self.initial_root_state

        # Current object state
        # self.planetary_carrier
        # self.ring_gear
        # self.sun_planetary_gear_1
        # self.sun_planetary_gear_2
        # self.sun_planetary_gear_3
        # self.sun_planetary_gear_4
        # self.planetary_reducer

        # Object state
        # planetary_carrier state: position and quaternion
        planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()          # [x, y, z]
        planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()        # [q_w, q_x, q_y, q_z]
        pin_locals = torch.stack(self.pin_local_positions)
        q_w = planetary_carrier_quat[:, 0].view(-1, 1, 1)
        q_vec = planetary_carrier_quat[:, 1:].unsqueeze(1)
        v = pin_locals.unsqueeze(0)
        t = 2.0 * torch.cross(q_vec, v, dim=-1)
        rotated_pins = v + q_w * t + torch.cross(q_vec, t, dim=-1)  
        final_pin_positions = planetary_carrier_pos.unsqueeze(1) + rotated_pins

        # sun_planetary_gear state: position
        sun_planetary_gear_1_pos = self.sun_planetary_gear_1.data.root_state_w[:, :3].clone()
        sun_planetary_gear_2_pos = self.sun_planetary_gear_2.data.root_state_w[:, :3].clone()
        sun_planetary_gear_3_pos = self.sun_planetary_gear_3.data.root_state_w[:, :3].clone()
        sun_planetary_gear_4_pos = self.sun_planetary_gear_4.data.root_state_w[:, :3].clone()
        # print(f"final_pin_positions: {final_pin_positions}")
        # print(f"sun_planetary_gear_1: {sun_planetary_gear_1_pos}")
        # print(f"sun_planetary_gear_2: {sun_planetary_gear_2_pos}")
        # print(f"sun_planetary_gear_3: {sun_planetary_gear_3_pos}")
        # print(f"sun_planetary_gear_4: {sun_planetary_gear_4_pos}")
        self.context.fsm.update()


        # print(f"left_diff_ik_controller: {self.robot_agent.left_diff_ik_controller}")
        # print(f"left_arm_entity_cf.body_ids: {self.robot_agent.left_arm_entity_cfg.body_ids[0] -1}")
        # print(f"left_gripper_entity_cfg: {self.robot_agent.left_gripper_entity_cfg}")
        # print(self.scene["robot"].is_fixed_base)




from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg
class Galaxear1GearboxAssemblyAgent:
    def __init__(self,
            sim: sim_utils.SimulationContext,
            scene: InteractiveScene,
            obj_dict: dict
        ):
        self.sim = sim
        self.device = sim.device
        self.scene = scene
        self.robot = scene["robot"]

        # Object state
        self.obj_dict = obj_dict
        self.planetary_carrier = obj_dict["planetary_carrier"]
        self.ring_gear = obj_dict["ring_gear"]
        self.sun_planetary_gear_1 = obj_dict["sun_planetary_gear_1"]
        self.sun_planetary_gear_2 = obj_dict["sun_planetary_gear_2"]
        self.sun_planetary_gear_3 = obj_dict["sun_planetary_gear_3"]
        self.sun_planetary_gear_4 = obj_dict["sun_planetary_gear_4"]
        self.planetary_reducer = obj_dict["planetary_reducer"]

        # Define pin positions in local coordinates relative to planetary carrier
        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2
        ]

        # Constants for pick and place
        self.TCP_offset_z = 1.1475 - 1.05661
        self.TCP_offset_x = 0.3864 - 0.3785
        self.table_height = 0.9
        self.grasping_height = -0.003

        # Initial target pose and orientation
        self.target_position_left = torch.tensor([0.3864, 0.5237, 1.1475], device=self.device)
        self.target_orientation_left = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        self.target_position_right = torch.tensor([0.3864, -0.5237, 1.1475], device=self.device)
        self.target_orientation_right = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)

        # Initialize arm controller
        self.left_diff_ik_controller, self.left_arm_entity_cfg, self.left_gripper_entity_cfg = self.initialize_arm_controller("left")
        self.right_diff_ik_controller, self.right_arm_entity_cfg, self.right_gripper_entity_cfg = self.initialize_arm_controller("right")
        # self.left_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
        # self.right_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids

        # Action
        self.joint_position_command = None
        self.joint_command_ids = None

        # pick and place
        self.pick_and_place_fsm_state = "PICK_READY"
        self.pick_and_place_fsm_timer = 0

        # State
        self.current_ee_position_world = None
        self.current_left_gripper_state = None
        self.sun_planetary_gear_1_state = None

        # observe_object_state
        self.sun_planetary_gear_positions = []
        self.sun_planetary_gear_quats = []
        self.planetary_carrier_pos = None
        self.planetary_carrier_quat = None
        self.ring_gear_pos = None
        self.ring_gear_quat = None
        self.planetary_reducer_pos = None
        self.planetary_reducer_quat = None
        self.pin_positions = []
        self.pin_quats = []

        # observe_assembly_state
        self.num_mounted_planetary_gears = 0
        self.is_sun_gear_mounted = False
        self.is_ring_gear_mounted = False
        self.is_planetary_reducer_mounted = False
        self.unmounted_sun_planetary_gear_positions = []
        self.unmounted_pin_positions = []
    
    def observe_robot_state(self):
        """
        update robot state: simulation ground truth
        """
        # end effector
        left_arm_body_ids = self.left_arm_entity_cfg.body_ids
        self.current_ee_position_world = self.robot.data.body_state_w[                    # simulation ground truth
            :,
            left_arm_body_ids[0],
            0:3
        ]

        # gripper
        left_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
        self.current_left_gripper_state = self.robot.data.joint_pos[:, left_gripper_joint_ids]   # 2-DOF

    def observe_object_state(self):
        """
        update object state: simulation ground truth
        """
        # Sun planetary gears
        self.sun_planetary_gear_positions = []
        self.sun_planetary_gear_quats = []
        sun_planetary_gear_names = ['sun_planetary_gear_1', 'sun_planetary_gear_2', 'sun_planetary_gear_3', 'sun_planetary_gear_4']
        for sun_planetary_gear_name in sun_planetary_gear_names:
            gear_obj = self.obj_dict[sun_planetary_gear_name]
            gear_pos = gear_obj.data.root_state_w[:, :3].clone()
            gear_quat = gear_obj.data.root_state_w[:, 3:7].clone()

            self.sun_planetary_gear_positions.append(gear_pos)
            self.sun_planetary_gear_quats.append(gear_quat)

        # Planetary carrier
        self.planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        self.planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        # Ring gear
        self.ring_gear_pos = self.ring_gear.data.root_state_w[:, :3].clone()
        self.ring_gear_quat = self.ring_gear.data.root_state_w[:, 3:7].clone()

        # Planetary reducer
        self.planetary_reducer_pos = self.planetary_reducer.data.root_state_w[:, :3].clone()
        self.planetary_reducer_quat = self.planetary_reducer.data.root_state_w[:, 3:7].clone()

        # Pin in planetary carrier
        self.pin_positions = []
        self.pin_quats = []
        for pin_local_pos in self.pin_local_positions:
            pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

            pin_quat, pin_pos = torch_utils.tf_combine(
                self.planetary_carrier_quat, 
                self.planetary_carrier_pos, 
                pin_quat.unsqueeze(0), 
                pin_local_pos.unsqueeze(0)
            )

            self.pin_positions.append(pin_pos)
            self.pin_quats.append(pin_quat)

        # tempt
        self.sun_planetary_gear_1_state = self.sun_planetary_gear_1.data.root_state_w.clone()    # 13-DOF

        # Planetary carrier pins
        planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()          # [x, y, z]
        planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()        # [q_w, q_x, q_y, q_z]
        pin_locals = torch.stack(self.pin_local_positions)
        q_w = planetary_carrier_quat[:, 0].view(-1, 1, 1)
        q_vec = planetary_carrier_quat[:, 1:].unsqueeze(1)
        v = pin_locals.unsqueeze(0)
        t = 2.0 * torch.cross(q_vec, v, dim=-1)
        rotated_pins = v + q_w * t + torch.cross(q_vec, t, dim=-1)  
        self.planetary_carrier_pin_positions = planetary_carrier_pos.unsqueeze(1) + rotated_pins

    def observe_assembly_state(self):
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

        # Constants
        PLANETARY_GEAR_HORIZONTAL_THRESHOLD = 0.002
        PLANETARY_GEAR_VERTICAL_THRESHOLD = 0.012
        PLANETARY_GEAR_ORIENTATION_THRESHOLD = 0.1
        SUN_GEAR_HORIZONTAL_THRESHOLD = 0.005
        SUN_GEAR_VERTICAL_THRESHOLD = 0.004
        SUN_GEAR_ORIENTATION_THRESHOLD = 0.1
        RING_GEAR_HORIZONTAL_THRESHOLD = 0.005
        RING_GEAR_VERTICAL_THRESHOLD = 0.004
        RING_GEAR_ORIENTATION_THRESHOLD = 0.1
        PLANETARY_REDUCER_HORIZONTAL_THRESHOLD = 0.005
        PLANETARY_REDUCER_THRESHOLD = 0.002
        PLANETARY_REDUCER_ORIENTATION_THRESHOLD = 0.1

        # initialize flags
        self.num_mounted_planetary_gears = 0
        self.is_sun_gear_mounted = False
        self.is_ring_gear_mounted = False
        self.is_planetary_reducer_mounted = False
        self.unmounted_sun_planetary_gear_positions = []
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
                orientation_error = torch.acos(torch.dot(sun_planetary_gear_quat.squeeze(0), pin_quat.squeeze(0)))

                if (horizontal_error < PLANETARY_GEAR_HORIZONTAL_THRESHOLD and 
                    vertical_error < PLANETARY_GEAR_VERTICAL_THRESHOLD and 
                    orientation_error < PLANETARY_GEAR_ORIENTATION_THRESHOLD):
                    self.num_mounted_planetary_gears += 1
                    is_mounted = True
                    pin_occupied[pin_idx] = True
            
            if not is_mounted:
                self.unmounted_sun_planetary_gear_positions.append(self.sun_planetary_gear_positions[sun_planetary_gear_idx])

        self.unmounted_pin_positions = [pin_positions[i] for i in range(len(pin_positions)) if not pin_occupied[i]]

        # Is the sun gear mounted?
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions)):
            sun_planetary_gear_pos = sun_planetary_gear_positions[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats[sun_planetary_gear_idx]

            horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - ring_gear_pos[:, :2])
            vertical_error = sun_planetary_gear_pos[:, 2] - ring_gear_pos[:, 2]
            orientation_error = torch.acos(torch.dot(sun_planetary_gear_quat.squeeze(0), ring_gear_quat.squeeze(0)))

            if (horizontal_error < SUN_GEAR_HORIZONTAL_THRESHOLD and 
                vertical_error < SUN_GEAR_VERTICAL_THRESHOLD and 
                orientation_error < SUN_GEAR_ORIENTATION_THRESHOLD):
                self.is_sun_gear_mounted = True

        # Is the ring gear mounted?
        horizontal_error = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_pos[:, :2])
        vertical_error = planetary_carrier_pos[:, 2] - ring_gear_pos[:, 2]
        orientation_error = torch.acos(torch.dot(planetary_carrier_quat.squeeze(0), ring_gear_quat.squeeze(0)))
        if (horizontal_error < RING_GEAR_HORIZONTAL_THRESHOLD and 
            vertical_error < RING_GEAR_VERTICAL_THRESHOLD and 
            orientation_error < RING_GEAR_ORIENTATION_THRESHOLD):
            self.is_ring_gear_mounted = True

        # Is the planetary reducer mounted?
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions)):
            sun_planetary_gear_pos = sun_planetary_gear_positions[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats[sun_planetary_gear_idx]

            horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - planetary_reducer_pos[:, :2])
            vertical_error = sun_planetary_gear_pos[:, 2] - planetary_reducer_pos[:, 2]
            orientation_error = torch.acos(torch.dot(sun_planetary_gear_quat.squeeze(0), planetary_reducer_quat.squeeze(0)))

            if (horizontal_error < PLANETARY_REDUCER_HORIZONTAL_THRESHOLD and 
                vertical_error < PLANETARY_REDUCER_THRESHOLD and 
                orientation_error < PLANETARY_REDUCER_ORIENTATION_THRESHOLD):
                self.is_planetary_reducer_mounted = True

        print("| Aseembly State                       |")
        print("----------------------------------------")
        print(f"| # of mounted planetary gears | {self.num_mounted_planetary_gears}     |")
        print(f"| sun gear                     | {self.is_sun_gear_mounted} |")
        print(f"| ring gear                    | {self.is_ring_gear_mounted} |")
        print(f"| planetary reducer            | {self.is_planetary_reducer_mounted} |")
        print("----------------------------------------")

    def initialize_arm_controller(self, arm_name: str):
        """
        arm_name: left or right
        """
        # Create differential inverse kinematics controller calculating joint angle for target end effect pose
        differential_inverse_kinematics_cfg = DifferentialIKControllerCfg(
            command_type="pose",                      # position and orientation
            use_relative_mode=False,                  # global coordinate (True: relative coordinate)
            ik_method="dls"                           # damped least squares: inverse kinematics standard solver in assembly task 
        )
        differential_inverse_kinematics_controller = DifferentialIKController(
            differential_inverse_kinematics_cfg,
            num_envs=self.scene.num_envs,             # number of parallel environment in Isaac Sim, vectorizated simulation
            device=self.device                        # "cuda": gpu / "cpu": cpu
        )

        # Robot parameter
        arm_entity_cfg = SceneEntityCfg(
            "robot",                                  # robot entity name
            joint_names=[f"{arm_name}_arm_joint.*"],  # joint entity set
            body_names=[f"{arm_name}_arm_link6"]      # body entity set
        )
        gripper_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[f"{arm_name}_gripper_axis.*"]
        )
        # Resolving the scene entities
        arm_entity_cfg.resolve(self.scene)
        gripper_entity_cfg.resolve(self.scene)

        return differential_inverse_kinematics_controller, arm_entity_cfg, gripper_entity_cfg

    # End effector control by differential inverse kinematics (need to remove simulation ground truth dependency)
    def solve_inverse_kinematics(self,
            arm_name: str,
            target_ee_position_base: torch.Tensor,
            target_ee_orientation_base: torch.Tensor
        ):
        """
        target_ee_pose, current_joint_pose -> inverse kinematics -> desired_joint_pose
        """
        # Arm selection and configuration
        if arm_name == "left":
            arm_joint_ids = self.left_arm_entity_cfg.joint_ids
            arm_body_ids = self.left_arm_entity_cfg.body_ids
            diff_ik_controller = self.left_diff_ik_controller
        elif arm_name == "right":
            arm_joint_ids = self.right_arm_entity_cfg.joint_ids
            arm_body_ids = self.right_arm_entity_cfg.body_ids
            diff_ik_controller = self.right_diff_ik_controller

        if self.robot.is_fixed_base:                                             # True
            ee_jacobi_idx = arm_body_ids[0] - 1                                  # index of end effector jacobian
        else:
            ee_jacobi_idx = arm_body_ids[0]

        # Get the target position and orientation of the arm
        ik_commands = torch.cat(
            [target_ee_position_base, target_ee_orientation_base], 
            dim=-1
        )
        diff_ik_controller.set_command(ik_commands)

        # Inverse kinematics solver
        current_ee_pose_world = self.robot.data.body_state_w[                    # simulation ground truth
            :,
            arm_body_ids[0],
            0:7
        ]
        current_base_pose_world = self.robot.data.root_state_w[:, 0:7]           # constant if is_fixed_base
        current_ee_position_base, current_ee_quaternion_base = subtract_frame_transforms(
            current_base_pose_world[:, 0:3],
            current_base_pose_world[:, 3:7],
            current_ee_pose_world[:, 0:3],
            current_ee_pose_world[:, 3:7],
        )

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, 
            ee_jacobi_idx, 
            :,
            arm_joint_ids
        ]
        current_arm_joint_pose = self.robot.data.joint_pos[                          # state space to be able to measured        
            :,
            arm_joint_ids
        ]
        # compute the joint commands
        desired_arm_joint_position = diff_ik_controller.compute(
            current_ee_position_base, 
            current_ee_quaternion_base,
            jacobian, 
            current_arm_joint_pose
        )

        return desired_arm_joint_position, arm_joint_ids

    def pick_and_place(self,
            arm_name: str,   # left or right
            # pick_pose,
            # place_pose,
            object_name: str # planetary_gear, sun_gear, planetary_carrier, ring_gear, planetary_reducer
        ) -> None:
        FSM_INITIALIZATION_STATE  = "INITIALIZATION"
        FSM_PICK_READY_STATE      = "PICK_READY"
        FSM_PICK_APPROACH_STATE   = "PICK_APPROACH"
        FSM_PICK_EXECUTION_STATE  = "PICK_EXECUTION"
        FSM_PICK_COMPLETE_STATE   = "PICK_COMPLETE"
        FSM_PLACE_READY_STATE     = "PLACE_READY"
        FSM_PLACE_APPROACH_STATE  = "PLACE_APPROACH"
        FSM_PLACE_EXECUTION_STATE = "PLACE_EXECUTION"
        FSM_PLACE_COMPLETE_STATE  = "PLACE_COMPLETE"
        FSM_FINALIZATION_STATE    = "FINALIZATION"

        # Gripper selection and configuration
        if arm_name == "left":
            gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
        elif arm_name == "right":
            gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids

        # Observe robot state and object state
        self.observe_robot_state()
        self.observe_object_state()
        self.observe_assembly_state()

        if object_name == "planetary_carrier":
            pass
        elif object_name == "ring_gear":
            pass
        elif object_name == "planetary_reducer":
            pass
        elif object_name == "sun_gear":
            pass
        elif object_name == "planetary_gear":
            object_height_offset = 0.0
            object_state = self.sun_planetary_gear_1_state

        target_position = object_state[:, :3].clone()
        target_position[:, 2] = self.table_height + self.grasping_height + object_height_offset
        target_position = target_position + torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)

        target_orientation = object_state[:, 3:7].clone()
        # Rotate the target orientation 180 degrees around the y-axis
        target_orientation, target_position = torch_utils.tf_combine(
            target_orientation, 
            target_position,
            torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device), 
            torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        )

        # approeach position (above 10cm)
        target_position_h = target_position + torch.tensor([0.0, 0.0, 0.1], device=self.device)

        # place
        target_position2 = self.planetary_carrier_pin_positions[:, 1, :].clone()
        target_position2[:, 2] = self.table_height + self.grasping_height
        target_position2[:, 2] += object_height_offset
        target_position2 += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)
        
        target_position2_h = target_position2 + torch.tensor([0.0, 0.0, 0.1], device=self.device)
        target_position2_h_down = target_position2 + torch.tensor([0.0, 0.0, 0.02], device=self.device)
        target_orientation2 = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)

        print(f"[PICK & PLACE FSM] {self.pick_and_place_fsm_state}")
        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Start State] PICK_READY --------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        if self.pick_and_place_fsm_state == FSM_INITIALIZATION_STATE:
            # [State Transition] INITIALIZATION -> PICK_READY
            self.pick_and_place_fsm_state = FSM_PICK_READY_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_READY -------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PICK_READY_STATE:
            desired_joint_position, joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_position_base=target_position_h, 
                target_ee_orientation_base=target_orientation
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = joint_ids

            # [State Transition] PICK_READY -> PICK_APPROACH
            if self.position_reached(self.current_ee_position_world, target_position_h):
                self.pick_and_place_fsm_state = FSM_PICK_APPROACH_STATE
            
        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_APPROACH ----------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PICK_APPROACH_STATE:
            desired_joint_position, joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_position_base=target_position, 
                target_ee_orientation_base=target_orientation
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = joint_ids

            # [State Transition] PICK_APPROACH -> PICK_EXECUTION
            if self.position_reached(self.current_ee_position_world, target_position):
                self.pick_and_place_fsm_state = FSM_PICK_EXECUTION_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_EXECUTION ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PICK_EXECUTION_STATE:
            desired_joint_position = torch.tensor([[0.0, 0.0]], device=self.device)
            joint_ids = gripper_joint_ids
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = joint_ids

            # [State Transition] PICK_EXECUTION -> FSM_PICK_COMPLETE_STATE
            self.pick_and_place_fsm_timer += 1
            if self.pick_and_place_fsm_timer > 50:
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = FSM_PICK_COMPLETE_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] FSM_PICK_COMPLETE_STATE ------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PICK_COMPLETE_STATE:
            self.pick_and_place_fsm_state = FSM_PLACE_READY_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] FSM_PLACE_READY_STATE --------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PLACE_READY_STATE:
            desired_joint_position, joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_position_base=target_position2_h, 
                target_ee_orientation_base=target_orientation2
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = joint_ids

            # [State Transition] FSM_PLACE_READY_STATE -> FSM_PLACE_APPROACH_STATE
            if self.position_reached(self.current_ee_position_world, target_position2_h):
                self.pick_and_place_fsm_state = FSM_PLACE_APPROACH_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] FSM_PLACE_APPROACH_STATE ------------------------------------------------------------------------ #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PLACE_APPROACH_STATE:
            desired_joint_position, joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_position_base=target_position2_h_down, 
                target_ee_orientation_base=target_orientation2
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = joint_ids

            # [State Transition] FSM_PLACE_APPROACH_STATE -> FSM_PLACE_EXECUTION_STATE
            self.pick_and_place_fsm_timer += 1
            if self.position_reached(self.current_ee_position_world, target_position2_h_down) and self.pick_and_place_fsm_timer > 30: # 30
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = FSM_PLACE_EXECUTION_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] FSM_PLACE_EXECUTION_STATE ----------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PLACE_EXECUTION_STATE:
            joint_ids = gripper_joint_ids
            num_gripper_joints = len(joint_ids)
            desired_joint_position = torch.full(
                (num_gripper_joints,), 0.4, device=self.device
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = joint_ids

            # [State Transition] FSM_PLACE_EXECUTION_STATE -> FSM_PLACE_COMPLETE_STATE
            self.pick_and_place_fsm_timer += 1
            if self.pick_and_place_fsm_timer > 100:
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = FSM_PLACE_COMPLETE_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] FSM_PLACE_COMPLETE_STATE ------------------------------------------------------------------------ #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_PLACE_COMPLETE_STATE:
            desired_joint_position, joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_position_base=target_position2_h, 
                target_ee_orientation_base=target_orientation2
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = joint_ids

            # [State Transition] FSM_PLACE_COMPLETE_STATE -> FSM_FINALIZATION_STATE
            if self.position_reached(self.current_ee_position_world, target_position2_h):
                self.pick_and_place_fsm_state = FSM_FINALIZATION_STATE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM End State] FSM_FINALIZATION_STATE ----------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == FSM_FINALIZATION_STATE:
            pass


    def plan_planetary_gear(self):
        # Used member variables
        planetary_carrier_pos = self.planetary_carrier_pos
        sun_planetary_gear_positions = self.sun_planetary_gear_positions
        pin_positions = self.pin_positions

        # How many planetary gear mounted on planetary carrier?
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions)):
            sun_planetary_gear_pos = sun_planetary_gear_positions[sun_planetary_gear_idx]

            distance = torch.norm(sun_planetary_gear_pos[:, :2] - planetary_carrier_pos[:, :2])

        gear_with_distance = []
        for gear_pos in sun_planetary_gear_positions:
            distance = torch.norm(sun_planetary_gear_pos[:, :2] - planetary_carrier_pos[:, :2]).item()
            gear_with_distance.append((distance, gear_pos))
        
        sorted_gear_with_distance = sorted(gear_with_distance, key=lambda x: x[0])
        self.sun_planetary_gear_positions = [gear[1] for gear in sorted_gear_with_distance]




            # for pin_idx in range(len(pin_positions)):
            #     pin_pos = pin_positions[pin_idx]
            #     pin_quat = pin_quats[pin_idx]

            #     horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - pin_pos[:, :2])
            #     vertical_error = sun_planetary_gear_pos[:, 2] - pin_pos[:, 2]
            #     orientation_error = torch.acos(torch.dot(sun_planetary_gear_quat.squeeze(0), pin_quat.squeeze(0)))

            #     if horizontal_error < PLANETARY_GEAR_HORIZONTAL_THRESHOLD and vertical_error < PLANETARY_GEAR_VERTICAL_THRESHOLD and orientation_error < PLANETARY_GEAR_ORIENTATION_THRESHOLD:
            #         self.num_mounted_planetary_gears += 1
            #         break


    # Utility functions
    def position_reached(self, current_pos, target_pos, tol=0.02):
        return torch.norm(current_pos - target_pos, dim=1).item() < tol

from abc import ABC, abstractmethod

# State's life cycle (enter -> update -> exit) interface
class State(ABC):
    def enter(self, context):
        pass

    @abstractmethod
    def update(self, context):
        pass

    def exit(self, context):
        pass

# State controller
class StateMachine:
    def __init__(self, initial_state, context):
        self.state = initial_state
        self.context = context
        self.state.enter(self.context)

    def transition_to(self, next_state):
        self.state.exit(self.context)
        self.state = next_state
        self.state.enter(self.context)

    def update(self):
        self.state.update(self.context)

class Context:
    NUM_PLANETARY_GEARS = 3
    def __init__(self, sim, agent):
        self.sim = sim
        self.agent = agent
        self.fsm = None            # object of StateMachine
    
    @property
    def is_all_planetary_gear_mounted(self):
        return self.agent.num_mounted_planetary_gears >= self.NUM_PLANETARY_GEARS
        
    @property
    def is_sun_gear_mounted(self):
        return self.agent.is_sun_gear_mounted
    
    @property
    def is_ring_gear_mounted(self):
        return self.agent.is_ring_gear_mounted
    
    @property
    def is_planetary_reducer_mounted(self):
        return self.agent.is_planetary_reducer_mounted
    
# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Start State] Initialization ----------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class InitializationState(State):
    def enter(self, context):
        print("[FSM Start State] Initialization: enter")
        # context.reset()

    def update(self, context):
        # [State Transition] Initialization -> Planetary Gear Mounting
        if context.fsm is not None:
            context.fsm.transition_to(PlanetaryGearMountingState())

    def exit(self, context):
        print("[FSM Start State] Initialization: exit")

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Planetary Gear Mounting ------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class PlanetaryGearMountingState(State):
    def enter(self, context):
        print("[FSM Intermediate State] Planetary Gear Mounting: enter")

    def update(self, context):
        context.agent.pick_and_place(
            arm_name="left",
            object_name="planetary_gear"
        )
        
        if context.agent.pick_and_place_fsm_state == "FINALIZATION":
            # [State Transition] Planetary Gear Mounting -> Sun Gear Mounting
            if context.is_all_planetary_gear_mounted:
                context.fsm.transition_to(SunGearMountingState())

    def exit(self, context):
        print("[FSM Intermediate State] Planetary Gear Mounting: exit")

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Sun Gear Mounting ------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class SunGearMountingState(State):
    pass

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Ring Gear Mounting ------------------------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------------- #
class RingGearMountingState(State):
    pass

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Planetary Reducer Mounting ---------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class PlanetaryReducerMountingState(State):
    pass

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM End State] Finalization --------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class FinalizationState(State):
    def enter(self, context):
        print("[FSM Start State] FINALIZATION: enter")

    def update(self, context):
        pass