import torch
import isaacsim.core.utils.torch as torch_utils

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils

from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms      # Transformation from world to base coordinate

from enum import Enum, auto
class PickAndPlaceState(Enum):
    INITIALIZATION  = auto()
    PICK_READY      = auto()
    PICK_APPROACH   = auto()
    PICK_EXECUTION  = auto()
    PICK_COMPLETE   = auto()
    PLACE_READY     = auto()
    PLACE_APPROACH  = auto()
    PLACE_EXECUTION = auto()
    PLACE_COMPLETE  = auto()
    FINALIZATION    = auto()

class GalaxeaGearboxAssemblyAgent:
    def __init__(self,
            sim: sim_utils.SimulationContext,
            scene: InteractiveScene,
            obj_dict: dict
        ):
        self.sim = sim
        self.device = sim.device
        self.scene = scene
        self.robot = scene["robot"]
        self.num_envs = self.scene.num_envs

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

        # Initial target pose and orientation
        self.initial_left_ee_pos_w = torch.tensor([[0.3864, 0.5237, 1.1475]], device=self.device)
        self.initial_left_ee_quat_w = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)
        self.initial_right_ee_pos_w = torch.tensor([[0.3864, -0.5237, 1.1475]], device=self.device)
        self.initial_right_ee_quat_w = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)

        # Initialize arm controller
        self.left_diff_ik_controller, self.left_arm_entity_cfg, self.left_gripper_entity_cfg = self.initialize_arm_controller("left")
        self.right_diff_ik_controller, self.right_arm_entity_cfg, self.right_gripper_entity_cfg = self.initialize_arm_controller("right")

        # Action
        self.joint_position_command = None
        self.joint_command_ids = None

        # [Function] observe_robot_state
        self.left_ee_pos_w = None
        self.left_ee_quat_w = None
        self.right_ee_pos_w = None
        self.right_ee_quat_w = None
        self.base_pos_w = None
        self.base_quat_w = None
        self.current_left_gripper_state = None

        # [Function] observe_object_state
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

        # [Function] observe_assembly_state
        self.mounting_thresholds = { # relaxed
            "planetary_gear": {
                "horizontal": 0.002 + 0.008,
                "vertical": 0.012,
                "orientation": 0.1 + float("inf"),
            },
            "sun_gear": {
                "horizontal": 0.005,
                "vertical": 0.004,
                "orientation": 0.1 + float("inf"),
            },
            "ring_gear": {
                "horizontal": 0.005,
                "vertical": 0.004,
                "orientation": 0.1 + float("inf"),
            },
            "planetary_reducer": {
                "horizontal": 0.005,
                "vertical": 0.002,
                "orientation": 0.1 + float("inf"),
            },
        }
        self.num_mounted_planetary_gears = 0
        self.is_sun_gear_mounted = False
        self.is_ring_gear_mounted = False
        self.is_planetary_reducer_mounted = False
        self.unmounted_sun_planetary_gear_positions = []
        self.unmounted_sun_planetary_gear_quats = []
        self.unmounted_pin_positions = []

        # [Function] pick_and_place
        self.TCP_offset_z = 1.1475 - 1.05661
        self.TCP_offset_x = 0.3864 - 0.3785
        self.table_height = 0.9
        self.grasping_height = -0.003
        self.pick_and_place_fsm_state = PickAndPlaceState.INITIALIZATION
        self.pick_and_place_fsm_timer = 0
        self.pick_and_place_fsm_states = torch.full(
            (self.num_envs,), 
            PickAndPlaceState.INITIALIZATION.value, 
            dtype=torch.long, 
            device=self.device
        )
        self.pick_and_place_fsm_timers = torch.zeros(
            self.num_envs, 
            dtype=torch.long, 
            device=self.device
        )

        self.active_arm_name = None
        self.active_gripper_joint_ids = None
    
    def observe_robot_state(self):
        """
        update robot state: simulation ground truth
        """
        # end effector (simulation ground truth)
        left_arm_body_ids = self.left_arm_entity_cfg.body_ids
        right_arm_body_ids = self.right_arm_entity_cfg.body_ids
        self.left_ee_pos_w = self.robot.data.body_state_w[          
            :,
            left_arm_body_ids[0],
            0:3
        ]
        self.left_ee_quat_w = self.robot.data.body_state_w[
            :,
            left_arm_body_ids[0],
            3:7
        ]
        self.right_ee_pos_w = self.robot.data.body_state_w[        
            :,
            right_arm_body_ids[0],
            0:3
        ]
        self.right_ee_quat_w = self.robot.data.body_state_w[
            :,
            right_arm_body_ids[0],
            3:7
        ]

        # Base
        self.base_pos_w  = self.robot.data.root_state_w[:, 0:3]
        self.base_quat_w = self.robot.data.root_state_w[:, 3:7]

        # gripper
        left_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
        self.current_left_gripper_state = self.robot.data.joint_pos[:, left_gripper_joint_ids]   # 2-DOF

    def observe_object_state(self):
        """
        update object state: simulation ground truth
        """
        # Used member variables
        num_envs = self.num_envs

        # Sun planetary gears
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

    def observe_assembly_state(self):
        # Observe object state
        self.observe_object_state()

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
        num_envs = self.num_envs

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
            num_envs=self.num_envs,                   # number of parallel environment in Isaac Sim, vectorizated simulation
            device=self.device                        # "cuda": gpu / "cpu": cpu
        )

        # Robot parameter
        arm_entity_cfg = SceneEntityCfg(
            "robot",                                  # robot entity name
            joint_names=[f"{arm_name}_arm_joint.*"],  # joint entity set
            body_names=[f"{arm_name}_arm_link6"]      # body entity set (ee)
        )
        gripper_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[f"{arm_name}_gripper_axis.*"] #################
        )
        # Resolving the scene entities
        arm_entity_cfg.resolve(self.scene)
        gripper_entity_cfg.resolve(self.scene)

        return differential_inverse_kinematics_controller, arm_entity_cfg, gripper_entity_cfg

    # End effector control by differential inverse kinematics (need to remove simulation ground truth dependency)
    def solve_inverse_kinematics(self,
            arm_name: str,
            target_ee_pos_b: torch.Tensor,
            target_ee_quat_b: torch.Tensor
        ):
        """
        target_ee_pose, current_joint_pose -> inverse kinematics -> desired_joint_pose
        """
        left_ee_pos_w = self.left_ee_pos_w
        left_ee_quat_w = self.left_ee_quat_w
        right_ee_pos_w = self.right_ee_pos_w
        right_ee_quat_w = self.right_ee_quat_w
        base_pos_w = self.base_pos_w
        base_quat_w = self.base_quat_w

        # Arm selection and configuration
        if arm_name == "left":
            arm_joint_ids = self.left_arm_entity_cfg.joint_ids
            arm_body_ids = self.left_arm_entity_cfg.body_ids
            diff_ik_controller = self.left_diff_ik_controller

            ee_pos_w = left_ee_pos_w
            ee_quat_w = left_ee_quat_w
        elif arm_name == "right":
            arm_joint_ids = self.right_arm_entity_cfg.joint_ids
            arm_body_ids = self.right_arm_entity_cfg.body_ids
            diff_ik_controller = self.right_diff_ik_controller

            ee_pos_w = right_ee_pos_w
            ee_quat_w = right_ee_quat_w

        if self.robot.is_fixed_base:                                             # True
            ee_jacobi_idx = arm_body_ids[0] - 1                                  # index of end effector jacobian
        else:
            ee_jacobi_idx = arm_body_ids[0]

        # Get the target position and orientation of the arm
        ik_commands = torch.cat(
            [target_ee_pos_b, target_ee_quat_b], 
            dim=-1
        )
        diff_ik_controller.set_command(ik_commands)

        # Inverse kinematics solver
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pos_w,
            base_quat_w,
            ee_pos_w,
            ee_quat_w
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
            ee_pos_b, 
            ee_quat_b,
            jacobian, 
            current_arm_joint_pose
        )
        desired_arm_joint_ids = arm_joint_ids

        return desired_arm_joint_position, desired_arm_joint_ids

    def pick_and_place(self,
            object_name: str # planetary_gear, sun_gear, planetary_carrier, ring_gear, planetary_reducer
        ) -> None:
        # Observe robot state and object state
        self.observe_robot_state()
        self.observe_object_state()
        self.observe_assembly_state()

        # Used member variables
        num_envs = self.num_envs
        unmounted_sun_planetary_gear_positions = self.unmounted_sun_planetary_gear_positions
        unmounted_sun_planetary_gear_quats = self.unmounted_sun_planetary_gear_quats
        unmounted_pin_positions = self.unmounted_pin_positions
        initial_left_ee_pos_w = self.initial_left_ee_pos_w
        initial_left_ee_quat_w = self.initial_left_ee_quat_w
        initial_right_ee_pos_w = self.initial_right_ee_pos_w
        initial_right_ee_quat_w = self.initial_right_ee_quat_w
        initial_left_ee_pos_w_batch = self.initial_left_ee_pos_w.repeat(num_envs, 1)
        initial_left_ee_quat_w_batch = self.initial_left_ee_quat_w.repeat(num_envs, 1)
        initial_right_ee_pos_w_batch = self.initial_right_ee_pos_w.repeat(num_envs, 1)
        initial_right_ee_quat_w_batch = self.initial_right_ee_quat_w.repeat(num_envs, 1)
        base_pos_w = self.base_pos_w
        base_quat_w = self.base_quat_w

        left_ee_pos_w = self.left_ee_pos_w
        right_ee_pos_w = self.right_ee_pos_w

        TIME_CONSTANT_50 = 50
        TIME_CONSTANT_100 = 100 
        TIME_CONSTANT_150 = 150 

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

        if len(self.unmounted_pin_positions) >= 1:
            # Pick
            target_pick_pos_w = unmounted_sun_planetary_gear_positions[0].squeeze(1).clone()
            target_pick_quat_w = unmounted_sun_planetary_gear_quats[0].squeeze(1).clone()
            
            target_pick_pos_w[:, 2] = self.table_height + self.grasping_height + object_height_offset
            target_pick_pos_w = target_pick_pos_w + torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)

            rotate_y_180_batch = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
            zero_pos_batch = torch.tensor([[0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)

            target_pick_quat_w, target_pick_pos_w = torch_utils.tf_combine( # Rotate the target orientation 180 degrees around the y-axis
                target_pick_quat_w, 
                target_pick_pos_w,
                rotate_y_180_batch, # (num_envs, 4)
                zero_pos_batch      # (num_envs, 3)
            )
            target_pick_ready_pos_w = target_pick_pos_w + torch.tensor([0.0, 0.0, 0.1], device=self.device)

            # Transform coordinate from world to base
            target_pick_pos_b, target_pick_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                target_pick_pos_w,
                target_pick_quat_w   
            )
            target_pick_ready_pos_b, _ = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                target_pick_ready_pos_w,
                target_pick_quat_w  
            )

            # Place
            target_place_pos_w = unmounted_pin_positions[0].squeeze(1).clone()
            target_place_pos_w[:, 2] = self.table_height + self.grasping_height
            target_place_pos_w[:, 2] += object_height_offset
            target_place_pos_w += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)
            target_place_quat_w_batch = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
            target_place_ready_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, 0.1], device=self.device)
            target_place_approach_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, 0.02], device=self.device)

            _, target_place_quat_b = subtract_frame_transforms(
                base_pos_w,                 # (num_envs, 3)
                base_quat_w,                # (num_envs, 4)
                target_place_pos_w,         # (num_envs, 3)
                target_place_quat_w_batch   # (num_envs, 4)
            )
            target_place_ready_pos_b, _ = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                target_place_ready_pos_w,
                target_place_quat_w_batch   
            )
            target_place_approach_pos_b, _ = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                target_place_approach_pos_w,
                target_place_quat_w_batch   
            )

        print(f"[PICK & PLACE FSM] {self.pick_and_place_fsm_state}")
        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Start State] INITIALIZATION ----------------------------------------------------------------------------------------- # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        if self.pick_and_place_fsm_state == PickAndPlaceState.INITIALIZATION:
            initial_left_ee_pos_b, initial_left_ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                initial_left_ee_pos_w_batch,
                initial_left_ee_quat_w_batch
            )
            initial_right_ee_pos_b, initial_right_ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                initial_right_ee_pos_w_batch,
                initial_right_ee_quat_w_batch   
            )
            desired_left_joint_position, desired_left_joint_ids = self.solve_inverse_kinematics( 
                arm_name="left",
                target_ee_pos_b=initial_left_ee_pos_b, 
                target_ee_quat_b=initial_left_ee_quat_b
            )
            desired_right_joint_position, desired_right_joint_ids = self.solve_inverse_kinematics( 
                arm_name="right",
                target_ee_pos_b=initial_right_ee_pos_b, 
                target_ee_quat_b=initial_right_ee_quat_b
            )

            desired_left_gripper_ids = self.left_gripper_entity_cfg.joint_ids
            desired_left_gripper_position = torch.tensor([[0.05, 0.05]], device=self.device).repeat(num_envs, 1)
            desired_right_gripper_ids = self.right_gripper_entity_cfg.joint_ids
            desired_right_gripper_position = torch.tensor([[0.05, 0.05]], device=self.device).repeat(num_envs, 1)

            self.joint_position_command = torch.cat(
                [desired_left_joint_position, 
                 desired_right_joint_position, 
                 desired_left_gripper_position, 
                 desired_right_gripper_position], 
                dim=1
            )
            self.joint_command_ids = torch.tensor(
                desired_left_joint_ids + desired_right_joint_ids + desired_left_gripper_ids + desired_right_gripper_ids, 
                device=self.device
            )

            # Decide arm and gripper configuration
            dist_left = torch.norm(target_pick_pos_w - initial_left_ee_pos_w)
            dist_right = torch.norm(target_pick_pos_w - initial_right_ee_pos_w)
            if dist_left <= dist_right:
                self.active_arm_name = "left"
                self.active_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
            else:
                self.active_arm_name = "right"
                self.active_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids

            # [State Transition] INITIALIZATION -> PICK_READY
            self.pick_and_place_fsm_timer += 1
            if self.pick_and_place_fsm_timer > TIME_CONSTANT_50:
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PICK_READY

        arm_name = self.active_arm_name
        gripper_joint_ids = self.active_gripper_joint_ids
        if arm_name == "left":
            ee_pos_w = left_ee_pos_w
        else:
            ee_pos_w = right_ee_pos_w

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_READY -------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        if self.pick_and_place_fsm_state == PickAndPlaceState.PICK_READY:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_pick_ready_pos_b, 
                target_ee_quat_b=target_pick_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_READY -> PICK_APPROACH
            self.pick_and_place_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_pick_ready_pos_w) and 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_50 or 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PICK_APPROACH

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_APPROACH ----------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.PICK_APPROACH:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_pick_pos_b, 
                target_ee_quat_b=target_pick_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_APPROACH -> PICK_EXECUTION
            self.pick_and_place_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_pick_pos_w) and 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_100 or 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PICK_EXECUTION

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_EXECUTION ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.PICK_EXECUTION:

            desired_joint_position = torch.tensor([[0.0, 0.0]], device=self.device)
            # desired_joint_position = torch.tensor([[0.0]], device=self.device)
            desired_joint_ids = gripper_joint_ids
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_EXECUTION -> PICK_COMPLETE
            self.pick_and_place_fsm_timer += 1
            if self.pick_and_place_fsm_timer > TIME_CONSTANT_100:
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PICK_COMPLETE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_COMPLETE ----------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.PICK_COMPLETE:
            # [State Transition] PICK_COMPLETE -> PLACE_READY
            self.pick_and_place_fsm_timer += 1
            if self.pick_and_place_fsm_timer > TIME_CONSTANT_50:
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PLACE_READY

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_READY ------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.PLACE_READY:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_place_ready_pos_b, 
                target_ee_quat_b=target_place_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PLACE_READY -> PLACE_APPROACH
            self.pick_and_place_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_place_ready_pos_w) and 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_50 or 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PLACE_APPROACH

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_APPROACH ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.PLACE_APPROACH:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_place_approach_pos_b, 
                target_ee_quat_b=target_place_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PLACE_APPROACH -> PLACE_EXECUTION
            self.pick_and_place_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_place_approach_pos_w) and 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_100 or
                self.pick_and_place_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PLACE_EXECUTION

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_EXECUTION --------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.PLACE_EXECUTION:
            desired_joint_ids = gripper_joint_ids
            num_gripper_joints = len(desired_joint_ids)
            desired_joint_position = torch.full(
                (num_gripper_joints,), 0.1, device=self.device
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PLACE_EXECUTION -> PLACE_COMPLETE
            self.pick_and_place_fsm_timer += 1
            if self.pick_and_place_fsm_timer > TIME_CONSTANT_100:
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.PLACE_COMPLETE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_COMPLETE ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.PLACE_COMPLETE:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_place_ready_pos_b, 
                target_ee_quat_b=target_place_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PLACE_COMPLETE -> FINALIZATION
            self.pick_and_place_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_place_ready_pos_w) and 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_50 or
                self.pick_and_place_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place_fsm_timer = 0
                self.pick_and_place_fsm_state = PickAndPlaceState.FINALIZATION

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM End State] FINALIZATION --------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place_fsm_state == PickAndPlaceState.FINALIZATION:
            # [State Transition] FINALIZATION -> INITIALIZATION
            self.pick_and_place_fsm_timer += 1
            if self.pick_and_place_fsm_timer > TIME_CONSTANT_50:
                self.pick_and_place_fsm_timer = 0
                self.reset_pick_and_place()

    def reset_pick_and_place(self):
        self.pick_and_place_fsm_state = PickAndPlaceState.INITIALIZATION
        self.pick_and_place_fsm_timer = 0

    # Utility functions
    def position_reached(self, current_pos, target_pos, tol=0.01):
        error = torch.norm(current_pos - target_pos, dim=1)
        return torch.all(error < tol)
        
    # def position_reached(
    #         self,
    #         current_pos: torch.Tensor,
    #         target_pos: torch.Tensor,
    #         tol: float = 0.01
    #     ) -> torch.Tensor:
    #     error = torch.norm(current_pos - target_pos, dim=1)
    #     return error < tol # True/False tensor of [num_envs] size