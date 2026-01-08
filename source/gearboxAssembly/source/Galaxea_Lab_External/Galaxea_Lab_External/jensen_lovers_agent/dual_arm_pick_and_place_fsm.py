from isaaclab.scene import InteractiveScene

from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms      # Transformation from world to base coordinate

import isaacsim.core.utils.torch as torch_utils
import torch

from enum import Enum, auto
class PickAndPlaceState(Enum):
    INITIALIZATION  = auto()
    PLANNING        = auto()
    STAGING         = auto()
    PICK_READY      = auto()
    PICK_APPROACH   = auto()
    PICK_EXECUTION  = auto()
    PICK_COMPLETE   = auto()
    PLACE_READY     = auto()
    PLACE_APPROACH  = auto()
    PLACE_EXECUTION = auto()
    PLACE_COMPLETE  = auto()
    FINALIZATION    = auto()

class DualArmPickAndPlaceFSM:
    # Timer constant
    TIME_CONSTANT_50  = 50
    TIME_CONSTANT_100 = 100
    TIME_CONSTANT_150 = 150

    STAGING_DURATION  = 30
    PICK_HOLD_TIME    = 20
    PLACE_HOLD_TIME   = 20

    def __init__(self, scene: InteractiveScene, device):
        self.scene           = scene
        self.robot           = scene["robot"]
        self.device          = device        

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",                      # position and orientation
            use_relative_mode=False,                  # global coordinate (True: relative coordinate)
            ik_method="dls"                           # damped least squares: inverse kinematics standard solver in assembly task 
        )
        self.left_diff_ik_controller = DifferentialIKController(
            diff_ik_cfg,
            num_envs=self.scene.num_envs,             # number of parallel environment in Isaac Sim, vectorizated simulation
            device=self.device                        # "cuda": gpu / "cpu": cpu
        )
        self.right_diff_ik_controller = DifferentialIKController(
            diff_ik_cfg,
            num_envs=self.scene.num_envs,        
            device=self.device                    
        )

        # Robot parameter
        self.left_arm_entity_cfg = SceneEntityCfg(
            "robot",                          # robot entity name
            joint_names=["left_arm_joint.*"], # joint entity set
            body_names=["left_arm_link6"]     # body entity set (ee)
        )
        self.left_gripper_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["left_gripper_axis.*"]
        )
        self.right_arm_entity_cfg = SceneEntityCfg(
            "robot",                         
            joint_names=["right_arm_joint.*"],
            body_names=["right_arm_link6"]    
        )
        self.right_gripper_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["right_gripper_axis.*"]
        )
        self.left_arm_entity_cfg.resolve(self.scene)
        self.left_gripper_entity_cfg.resolve(self.scene)
        self.right_arm_entity_cfg.resolve(self.scene)
        self.right_gripper_entity_cfg.resolve(self.scene)

        # Member variables
        # Constants
        self.initial_left_ee_pos_e         = torch.tensor([[0.3864, 0.5237, 1.1475]], device=self.device) - self.scene.env_origins
        self.initial_left_ee_quat_w        = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)
        self.initial_right_ee_pos_e        = torch.tensor([[0.3864, -0.5237, 1.1475]], device=self.device) - self.scene.env_origins
        self.initial_right_ee_quat_w       = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)
        self.initial_left_ee_pos_e_batch   = self.initial_left_ee_pos_e
        self.initial_left_ee_quat_w_batch  = self.initial_left_ee_quat_w.repeat(self.scene.num_envs, 1)
        self.initial_right_ee_pos_e_batch  = self.initial_right_ee_pos_e
        self.initial_right_ee_quat_w_batch = self.initial_right_ee_quat_w.repeat(self.scene.num_envs, 1)
        self.TCP_offset_z                  = 1.1475 - 1.05661
        self.TCP_offset_x                  = 0.3864 - 0.3785
        self.table_height                  = 0.9
        self.grasping_height               = -0.003
        self.state_dispatch = {
            PickAndPlaceState.INITIALIZATION  : self._state_initialization,
            PickAndPlaceState.PLANNING        : self._state_planning,
            PickAndPlaceState.STAGING         : self._state_staging,
            PickAndPlaceState.PICK_READY      : self._state_pick_ready,
            PickAndPlaceState.PICK_APPROACH   : self._state_pick_approach,
            PickAndPlaceState.PICK_EXECUTION  : self._state_pick_execution,
            PickAndPlaceState.PICK_COMPLETE   : self._state_pick_complete,
            PickAndPlaceState.PLACE_READY     : self._state_place_ready,
            PickAndPlaceState.PLACE_APPROACH  : self._state_place_approach,
            PickAndPlaceState.PLACE_EXECUTION : self._state_place_execution,
            PickAndPlaceState.PLACE_COMPLETE  : self._state_place_complete,
            PickAndPlaceState.FINALIZATION    : self._state_finalization,
        }

        self.state                         = PickAndPlaceState.INITIALIZATION
        self.timer                         = 0
        self.joint_pos_command             = None
        self.joint_pos_command_ids         = None
        self.active_arm_name               = None
        self.active_gripper_joint_ids      = None
        self.target_pick_pos_b             = None
        self.target_pick_quat_b            = None
        self.target_pick_ready_pos_b       = None
        self.target_place_quat_b           = None
        self.target_place_ready_pos_b      = None
        self.target_place_approach_pos_b   = None
        self.target_pick_pos_w             = None # *
        self.target_pick_quat_w            = None # *
        self.target_pick_ready_pos_w       = None
        self.target_place_pos_w            = None # *
        self.target_place_ready_pos_w      = None
        self.target_place_approach_pos_w   = None

    def step(self):
        self.state_dispatch[self.state]()
        return self.joint_pos_command, self.joint_pos_command_ids

    def update_observation(self,
            left_ee_pos_w,
            left_ee_quat_w,
            right_ee_pos_w,
            right_ee_quat_w,
            base_pos_w,
            base_quat_w,        
            ring_gear_pos_w,
            ring_gear_quat_w,
            planetary_reducer_pos_w,
            planetary_reducer_quat_w,

            target_sun_planetary_gear_pos_w,
            target_sun_planetary_gear_quat_w,
            target_pin_pos_w          
        ):
        # Robot states
        self.left_ee_pos_w   = left_ee_pos_w
        self.left_ee_quat_w  = left_ee_quat_w
        self.right_ee_pos_w  = right_ee_pos_w
        self.right_ee_quat_w = right_ee_quat_w
        self.base_pos_w      = base_pos_w
        self.base_quat_w     = base_quat_w

        # Object states
        self.ring_gear_pos_w = ring_gear_pos_w
        self.ring_gear_quat_w = ring_gear_quat_w
        self.planetary_reducer_pos_w = planetary_reducer_pos_w
        self.planetary_reducer_quat_w = planetary_reducer_quat_w
        self.target_sun_planetary_gear_pos_w = target_sun_planetary_gear_pos_w
        self.target_sun_planetary_gear_quat_w = target_sun_planetary_gear_quat_w
        self.target_pin_pos_w = target_pin_pos_w

    def reset(self):
        self.timer                       = 0
        self.active_arm_name             = None
        self.active_gripper_joint_ids    = None
        self.target_pick_pos_b           = None
        self.target_pick_quat_b          = None
        self.target_pick_ready_pos_b     = None
        self.target_place_quat_b         = None
        self.target_place_ready_pos_b    = None
        self.target_place_approach_pos_b = None
        self.target_pick_pos_w           = None
        self.target_pick_quat_w          = None 
        self.target_pick_ready_pos_w     = None
        self.target_place_pos_w          = None 
        self.target_place_ready_pos_w    = None
        self.target_place_approach_pos_w = None

    def solve_inverse_kinematics(self,
            arm_name        : str,
            target_ee_pos_b : torch.Tensor,
            target_ee_quat_b: torch.Tensor
        ):
        """
        target_ee_pose, current_joint_pose -> desired_joint_pose
        """
        left_ee_pos_w = self.left_ee_pos_w
        left_ee_quat_w = self.left_ee_quat_w
        right_ee_pos_w = self.right_ee_pos_w
        right_ee_quat_w = self.right_ee_quat_w
        left_arm_entity_cfg = self.left_arm_entity_cfg
        right_arm_entity_cfg = self.right_arm_entity_cfg
        left_diff_ik_controller = self.left_diff_ik_controller
        right_diff_ik_controller = self.right_diff_ik_controller
        robot = self.robot

        # Arm selection and configuration
        if arm_name == "left":
            arm_joint_ids = left_arm_entity_cfg.joint_ids
            arm_body_ids = left_arm_entity_cfg.body_ids
            diff_ik_controller = left_diff_ik_controller

            ee_pos_w = left_ee_pos_w
            ee_quat_w = left_ee_quat_w
        elif arm_name == "right":
            arm_joint_ids = right_arm_entity_cfg.joint_ids
            arm_body_ids = right_arm_entity_cfg.body_ids
            diff_ik_controller = right_diff_ik_controller

            ee_pos_w = right_ee_pos_w
            ee_quat_w = right_ee_quat_w

        if robot.is_fixed_base:                  # True
            ee_jacobi_idx = arm_body_ids[0] - 1  # index of end effector jacobian
        else:
            ee_jacobi_idx = arm_body_ids[0]

        # Get the target position and orientation of the arm
        ik_commands = torch.cat(
            [target_ee_pos_b, target_ee_quat_b], 
            dim=-1
        )
        diff_ik_controller.set_command(ik_commands)

        # Inverse kinematics solver
        ee_pos_b, ee_quat_b = self.transform_world_to_base(
            pos_w =ee_pos_w,
            quat_w=ee_quat_w
        )

        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, arm_joint_ids
        ]
        current_arm_joint_pose = robot.data.joint_pos[    # state space to be able to measured        
            :, arm_joint_ids
        ]
        # compute the joint commands
        desired_arm_joint_pos = diff_ik_controller.compute(
            ee_pos_b, 
            ee_quat_b,
            jacobian, 
            current_arm_joint_pose
        )
        desired_arm_joint_ids = arm_joint_ids

        return desired_arm_joint_pos, desired_arm_joint_ids

    def transform_world_to_base(self, 
            pos_w: torch.Tensor, 
            quat_w: torch.Tensor
        ):
        """
        World frame to base frame
        """
        base_pos_w = self.base_pos_w
        base_quat_w = self.base_quat_w
        pos_b, quat_b = subtract_frame_transforms(
            base_pos_w,
            base_quat_w,
            pos_w,
            quat_w
        )
        return pos_b, quat_b
    
    def position_reached(self, current_pos, target_pos, tol=0.01):
        error = torch.norm(current_pos - target_pos, dim=1)
        return error < tol
    
    def transition_to(self, next_state):
        self.timer = 0
        self.state = next_state

    def get_active_ee_pos_w(self):
        if self.active_arm_name == "left":
            return self.left_ee_pos_w
        elif self.active_arm_name == "right":
            return self.right_ee_pos_w
        
    def is_ee_reached_to(self, target_pos, tolerance=0.01):
        current_ee_pos_w = self.get_active_ee_pos_w()
        error = torch.norm(current_ee_pos_w - target_pos, dim=1)
        return error < tolerance
    
    # -------------------------------------------------------------------------------------------------------------------------- #
    # FSM states --------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------------------- #
    def _state_initialization(self):
        self.reset()
        self.state = PickAndPlaceState.PLANNING

    def _state_planning(self):
        # Used member variables
        target_sun_planetary_gear_pos_w = self.target_sun_planetary_gear_pos_w
        target_sun_planetary_gear_quat_w = self.target_sun_planetary_gear_quat_w
        target_pin_pos_w = self.target_pin_pos_w
        num_envs = self.scene.num_envs
        initial_left_ee_pos_e = self.initial_left_ee_pos_e
        initial_right_ee_pos_e = self.initial_right_ee_pos_e

        # 임시
        object_height_offset = 0.0

        # Pick
        target_pick_pos_w = target_sun_planetary_gear_pos_w.clone()
        target_pick_quat_w = target_sun_planetary_gear_quat_w.clone()

        target_pick_pos_w[:, 2] = self.table_height + self.grasping_height + object_height_offset
        target_pick_pos_w = target_pick_pos_w + torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)

        rotate_y_180_batch = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
        zero_pos_batch = torch.tensor([[0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
        target_pick_quat_w, target_pick_pos_w = torch_utils.tf_combine( # Rotate the target orientation 180 degrees around the y-axis
            target_pick_quat_w, 
            target_pick_pos_w,
            rotate_y_180_batch, 
            zero_pos_batch      
        )
        target_pick_ready_pos_w = target_pick_pos_w + torch.tensor([0.0, 0.0, 0.1], device=self.device)

        # Place
        target_place_pos_w = target_pin_pos_w
        target_place_pos_w[:, 2] = self.table_height + self.grasping_height
        target_place_pos_w[:, 2] += object_height_offset
        target_place_pos_w += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)
        target_place_quat_w_batch = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
        target_place_ready_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, 0.1], device=self.device)
        target_place_approach_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, 0.02], device=self.device)

        # Transform coordinate from world to base
        target_pick_pos_b, target_pick_quat_b = self.transform_world_to_base(
            pos_w =target_pick_pos_w,
            quat_w=target_pick_quat_w   
        )
        target_pick_ready_pos_b, _ = self.transform_world_to_base(
            pos_w =target_pick_ready_pos_w,
            quat_w=target_pick_quat_w   
        )
        _, target_place_quat_b = self.transform_world_to_base(
            pos_w =target_place_pos_w,       
            quat_w=target_place_quat_w_batch 
        )
        target_place_ready_pos_b, _ = self.transform_world_to_base(
            pos_w =target_place_ready_pos_w,       
            quat_w=target_place_quat_w_batch 
        )
        target_place_approach_pos_b, _ = self.transform_world_to_base(
            pos_w =target_place_approach_pos_w,       
            quat_w=target_place_quat_w_batch 
        )

        # Update member variables
        self.target_pick_pos_w           = target_pick_pos_w
        self.target_pick_quat_w          = target_pick_quat_w
        self.target_pick_ready_pos_w     = target_pick_ready_pos_w
        self.target_place_pos_w          = target_place_pos_w
        self.target_place_ready_pos_w    = target_place_ready_pos_w
        self.target_place_approach_pos_w = target_place_approach_pos_w
        self.target_pick_pos_b           = target_pick_pos_b
        self.target_pick_quat_b          = target_pick_quat_b
        self.target_pick_ready_pos_b     = target_pick_ready_pos_b
        self.target_place_quat_b         = target_place_quat_b
        self.target_place_ready_pos_b    = target_place_ready_pos_b
        self.target_place_approach_pos_b = target_place_approach_pos_b

        # Decide arm and gripper configuration
        dist_left = torch.norm(target_pick_pos_w - initial_left_ee_pos_e, dim=1).mean()
        dist_right = torch.norm(target_pick_pos_w - initial_right_ee_pos_e, dim=1).mean()
        if dist_left <= dist_right:
            self.active_arm_name = "left"
            self.active_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
        else:
            self.active_arm_name = "right"
            self.active_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids

        # [State Transition] PLANNING -> STAGING
        self.timer += 1
        if self.timer > self.TIME_CONSTANT_50:
            self.timer = 0
            self.state = PickAndPlaceState.STAGING

    def _state_staging(self):
        # Used member variables
        num_envs = self.scene.num_envs
        initial_left_ee_pos_e_batch = self.initial_left_ee_pos_e_batch
        initial_left_ee_quat_w_batch = self.initial_left_ee_quat_w_batch
        initial_right_ee_pos_e_batch = self.initial_right_ee_pos_e_batch
        initial_right_ee_quat_w_batch = self.initial_right_ee_quat_w_batch

        initial_left_ee_pos_b, initial_left_ee_quat_b = self.transform_world_to_base(
            pos_w =initial_left_ee_pos_e_batch,
            quat_w=initial_left_ee_quat_w_batch   
        )
        initial_right_ee_pos_b, initial_right_ee_quat_b = self.transform_world_to_base(
            pos_w =initial_right_ee_pos_e_batch,
            quat_w=initial_right_ee_quat_w_batch   
        )
        desired_left_joint_position, desired_left_joint_ids = self.solve_inverse_kinematics( 
            arm_name="left",
            target_ee_pos_b =initial_left_ee_pos_b, 
            target_ee_quat_b=initial_left_ee_quat_b
        )
        desired_right_joint_position, desired_right_joint_ids = self.solve_inverse_kinematics( 
            arm_name="right",
            target_ee_pos_b =initial_right_ee_pos_b, 
            target_ee_quat_b=initial_right_ee_quat_b
        )
        desired_left_gripper_ids = self.left_gripper_entity_cfg.joint_ids
        desired_right_gripper_ids = self.right_gripper_entity_cfg.joint_ids
        desired_left_gripper_position = torch.tensor([[0.035, 0.035]], device=self.device).repeat(num_envs, 1)
        desired_right_gripper_position = torch.tensor([[0.035, 0.035]], device=self.device).repeat(num_envs, 1)
        desired_joint_position = torch.cat(
            [
                desired_left_joint_position, 
                desired_right_joint_position, 
                desired_left_gripper_position, 
                desired_right_gripper_position
            ], 
            dim=1
        )
        desired_joint_ids = torch.tensor(
            desired_left_joint_ids + desired_right_joint_ids + desired_left_gripper_ids + desired_right_gripper_ids, 
            device=self.device
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] STAGING -> PICK_READY
        self.timer += 1
        if self.timer > self.TIME_CONSTANT_50:
            self.timer = 0
            self.state = PickAndPlaceState.PICK_READY

    def _state_pick_ready(self):
        arm_name = self.active_arm_name
        target_pick_ready_pos_b = self.target_pick_ready_pos_b
        target_pick_quat_b = self.target_pick_quat_b
        ee_pos_w = self.get_active_ee_pos_w()
        target_pick_ready_pos_w = self.target_pick_ready_pos_w

        desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
            arm_name=arm_name,
            target_ee_pos_b=target_pick_ready_pos_b, 
            target_ee_quat_b=target_pick_quat_b
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PICK_READY -> PICK_APPROACH
        self.timer += 1
        if (self.position_reached(ee_pos_w, target_pick_ready_pos_w) and 
            self.timer > self.TIME_CONSTANT_50 or 
            self.timer > self.TIME_CONSTANT_150):
            self.timer = 0
            self.state = PickAndPlaceState.PICK_APPROACH

    def _state_pick_approach(self):
        arm_name = self.active_arm_name
        target_pick_pos_b = self.target_pick_pos_b
        target_pick_quat_b = self.target_pick_quat_b
        ee_pos_w = self.get_active_ee_pos_w()
        target_pick_pos_w = self.target_pick_pos_w


        desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
            arm_name=arm_name,
            target_ee_pos_b=target_pick_pos_b, 
            target_ee_quat_b=target_pick_quat_b
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PICK_APPROACH -> PICK_EXECUTION
        self.timer += 1
        if (self.position_reached(ee_pos_w, target_pick_pos_w) and 
            self.timer > self.TIME_CONSTANT_100 or 
            self.timer > self.TIME_CONSTANT_150):
            self.timer = 0
            self.state = PickAndPlaceState.PICK_EXECUTION

    def _state_pick_execution(self):
        num_envs = self.scene.num_envs
        gripper_joint_ids = self.active_gripper_joint_ids

        desired_joint_position = torch.tensor([[0.03, 0.03]], device=self.device).repeat(num_envs, 1)
        desired_joint_ids = gripper_joint_ids
        desired_joint_ids = gripper_joint_ids
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PICK_EXECUTION -> PICK_COMPLETE
        self.timer += 1
        if self.timer > self.TIME_CONSTANT_100:
            self.timer = 0
            self.state = PickAndPlaceState.PICK_COMPLETE

    def _state_pick_complete(self):
        arm_name = self.active_arm_name
        target_pick_ready_pos_b = self.target_pick_ready_pos_b
        target_pick_quat_b = self.target_pick_quat_b
        ee_pos_w = self.get_active_ee_pos_w()
        target_pick_ready_pos_w = self.target_pick_ready_pos_w

        desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics(
            arm_name        =arm_name,
            target_ee_pos_b =target_pick_ready_pos_b,
            target_ee_quat_b=target_pick_quat_b
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PICK_COMPLETE -> PLACE_READY
        self.timer += 1
        if (self.position_reached(ee_pos_w, target_pick_ready_pos_w) and 
            self.timer > self.TIME_CONSTANT_50 or
            self.timer > self.TIME_CONSTANT_150):
            self.timer = 0
            self.state = PickAndPlaceState.PLACE_READY

    def _state_place_ready(self):
        arm_name = self.active_arm_name
        target_place_ready_pos_b = self.target_place_ready_pos_b
        target_place_quat_b = self.target_place_quat_b
        ee_pos_w = self.get_active_ee_pos_w()
        target_place_ready_pos_w = self.target_place_ready_pos_w

        desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
            arm_name=arm_name,
            target_ee_pos_b=target_place_ready_pos_b, 
            target_ee_quat_b=target_place_quat_b
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PLACE_READY -> PLACE_APPROACH
        self.timer += 1
        if (self.position_reached(ee_pos_w, target_place_ready_pos_w) and 
            self.timer > self.TIME_CONSTANT_50 or 
            self.timer > self.TIME_CONSTANT_150):
            self.timer = 0
            self.state = PickAndPlaceState.PLACE_APPROACH

    def _state_place_approach(self):
        arm_name = self.active_arm_name
        target_place_approach_pos_b = self.target_place_approach_pos_b
        target_place_quat_b = self.target_place_quat_b
        ee_pos_w = self.get_active_ee_pos_w()
        target_place_approach_pos_w = self.target_place_approach_pos_w

        desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
            arm_name=arm_name,
            target_ee_pos_b=target_place_approach_pos_b, 
            target_ee_quat_b=target_place_quat_b
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PLACE_APPROACH -> PLACE_EXECUTION
        self.timer += 1
        if (self.position_reached(ee_pos_w, target_place_approach_pos_w) and 
            self.timer > self.TIME_CONSTANT_100 or
            self.timer > self.TIME_CONSTANT_150):
            self.timer = 0
            self.state = PickAndPlaceState.PLACE_EXECUTION

    def _state_place_execution(self):
        num_envs = self.scene.num_envs
        gripper_joint_ids = self.active_gripper_joint_ids

        desired_joint_ids = gripper_joint_ids
        num_gripper_joints = len(desired_joint_ids)         
        desired_joint_position = torch.full(
            (num_envs, num_gripper_joints), 0.035, device=self.device
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PLACE_EXECUTION -> PLACE_COMPLETE
        self.timer += 1
        if self.timer > self.TIME_CONSTANT_100:
            self.timer = 0
            self.state = PickAndPlaceState.PLACE_COMPLETE

    def _state_place_complete(self):
        arm_name = self.active_arm_name
        target_place_ready_pos_b = self.target_place_ready_pos_b
        target_place_quat_b = self.target_place_quat_b
        ee_pos_w = self.get_active_ee_pos_w()
        target_place_ready_pos_w = self.target_place_ready_pos_w

        desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
            arm_name=arm_name,
            target_ee_pos_b=target_place_ready_pos_b, 
            target_ee_quat_b=target_place_quat_b
        )
        self.joint_pos_command = desired_joint_position
        self.joint_pos_command_ids = desired_joint_ids

        # [State Transition] PLACE_COMPLETE -> FINALIZATION
        self.timer += 1
        if (self.position_reached(ee_pos_w, target_place_ready_pos_w) and 
            self.timer > self.TIME_CONSTANT_50 or
            self.timer > self.TIME_CONSTANT_150):
            self.timer = 0
            self.state = PickAndPlaceState.FINALIZATION

    def _state_finalization(self):
        self.timer += 1
        if self.timer > self.TIME_CONSTANT_50:
            self.reset()
            self.state = PickAndPlaceState.INITIALIZATION

