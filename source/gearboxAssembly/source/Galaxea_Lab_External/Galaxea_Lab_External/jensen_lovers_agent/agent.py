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

        # Object state
        self.obj_dict             = obj_dict
        self.planetary_carrier    = obj_dict["planetary_carrier"]
        self.ring_gear            = obj_dict["ring_gear"]
        self.sun_planetary_gear_1 = obj_dict["sun_planetary_gear_1"]
        self.sun_planetary_gear_2 = obj_dict["sun_planetary_gear_2"]
        self.sun_planetary_gear_3 = obj_dict["sun_planetary_gear_3"]
        self.sun_planetary_gear_4 = obj_dict["sun_planetary_gear_4"]
        self.planetary_reducer    = obj_dict["planetary_reducer"]

        # Define pin positions in local coordinates relative to planetary carrier
        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2
        ]

        # Initialize arm controller
        self.left_diff_ik_controller, self.left_arm_entity_cfg, self.left_gripper_entity_cfg = self.initialize_arm_controller("left")
        self.right_diff_ik_controller, self.right_arm_entity_cfg, self.right_gripper_entity_cfg = self.initialize_arm_controller("right")

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] observe_robot_state ------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        self.left_ee_pos_w = None
        self.left_ee_quat_w = None
        self.right_ee_pos_w = None
        self.right_ee_quat_w = None
        self.base_pos_w = None
        self.base_quat_w = None
        self.current_left_gripper_state = None

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] observe_object_state ------------------------------------------------------------------------------------------ #
        # -------------------------------------------------------------------------------------------------------------------------- #
        self.sun_planetary_gear_positions_w = []
        self.sun_planetary_gear_quats_w = []
        self.planetary_carrier_pos_w = None
        self.planetary_carrier_quat_w = None
        self.ring_gear_pos_w = None
        self.ring_gear_quat_w = None
        self.planetary_reducer_pos_w = None
        self.planetary_reducer_quat_w = None
        self.pin_positions_w = []
        self.pin_quats_w = []

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] observe_assembly_state ---------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        self.mounting_thresholds = { # relaxed
            "planetary_gear": {
                "horizontal": 0.002 + 0.008,
                "vertical": 0.012,
                "orientation": 0.1 + float("inf"),
            },
            "sun_gear": {
                "horizontal": 0.005 - 0.001,
                "vertical": 0.004 + 0.006,
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
                "orientation": 0.1,
            },
        }
        self.num_mounted_planetary_gears = 0
        self.is_sun_gear_mounted = False
        self.is_ring_gear_mounted = False
        self.is_planetary_reducer_mounted = False
        self.unmounted_sun_planetary_gear_positions_w = []
        self.unmounted_sun_planetary_gear_quats_w = []
        self.unmounted_pin_positions_w = []
        self.target_sun_planetary_gear_pos = None
        self.target_sun_planetary_gear_quat = None
        self.target_pin_pos = None

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] pick_and_place ------------------------------------------------------------------------------------------------ #
        # -------------------------------------------------------------------------------------------------------------------------- #
        self.joint_position_command = None
        self.joint_command_ids = None

        self.initial_left_ee_pos_e         = torch.tensor([[0.3864, 0.5237, 1.1475]], device=self.device) - self.scene.env_origins
        self.initial_left_ee_quat_w        = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)
        self.initial_right_ee_pos_e        = torch.tensor([[0.3864, -0.5237, 1.1475]], device=self.device) - self.scene.env_origins
        self.initial_right_ee_quat_w       = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)
        self.initial_left_ee_pos_e_batch   = self.initial_left_ee_pos_e
        self.initial_left_ee_quat_w_batch  = self.initial_left_ee_quat_w.repeat(self.scene.num_envs, 1)
        self.initial_right_ee_pos_e_batch  = self.initial_right_ee_pos_e
        self.initial_right_ee_quat_w_batch = self.initial_right_ee_quat_w.repeat(self.scene.num_envs, 1)

        self.TCP_offset_z = 1.1475 - 1.05661
        self.TCP_offset_x = 0.3864 - 0.3785
        self.table_height = 0.9
        self.grasping_height = -0.003
        self.pick_and_place_fsm_state = PickAndPlaceState.INITIALIZATION
        self.pick_and_place_fsm_timer = 0
        self.pick_and_place_fsm_states = torch.full(
            (self.scene.num_envs,), 
            PickAndPlaceState.INITIALIZATION.value, 
            dtype=torch.long, 
            device=self.device
        )
        self.pick_and_place_fsm_timers = torch.zeros(
            self.scene.num_envs, 
            dtype=torch.long, 
            device=self.device
        )

        self.active_arm_name             = None
        self.active_gripper_joint_ids    = None

        self.target_pick_pos_b           = None
        self.target_pick_quat_b          = None
        self.target_pick_ready_pos_b     = None
        self.target_place_quat_b         = None
        self.target_place_ready_pos_b    = None
        self.target_place_approach_pos_b = None
        self.target_pick_pos_w           = None # *
        self.target_pick_quat_w          = None # *
        self.target_pick_ready_pos_w     = None
        self.target_place_pos_w          = None # *
        self.target_place_ready_pos_w    = None
        self.target_place_approach_pos_w = None

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] pick_and_place2 ----------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        self.pick_and_place2_fsm_state = PickAndPlaceState.INITIALIZATION
        self.pick_and_place2_fsm_timer = 0
    
    def observe_robot_state(self):
        """
        update robot state: simulation ground truth
        """
        # Used member variables
        env_origins = self.scene.env_origins

        # end effector (simulation ground truth)
        left_arm_body_ids    = self.left_arm_entity_cfg.body_ids
        right_arm_body_ids   = self.right_arm_entity_cfg.body_ids
        self.left_ee_pos_w   = self.robot.data.body_state_w[:, left_arm_body_ids[0], 0:3] - env_origins
        self.left_ee_quat_w  = self.robot.data.body_state_w[:, left_arm_body_ids[0], 3:7]
        self.right_ee_pos_w  = self.robot.data.body_state_w[:, right_arm_body_ids[0], 0:3] - env_origins
        self.right_ee_quat_w = self.robot.data.body_state_w[:, right_arm_body_ids[0], 3:7]

        # Base
        self.base_pos_w  = self.robot.data.root_state_w[:, 0:3]
        self.base_quat_w = self.robot.data.root_state_w[:, 3:7]

    def observe_object_state(self):
        """
        update object state: simulation ground truth
        """
        # Used member variables
        num_envs = self.scene.num_envs

        # Sun planetary gears
        self.sun_planetary_gear_positions_w = []
        self.sun_planetary_gear_quats_w     = []
        sun_planetary_gear_names            = [
            'sun_planetary_gear_1', 
            'sun_planetary_gear_2', 
            'sun_planetary_gear_3', 
            'sun_planetary_gear_4'
        ]
        for sun_planetary_gear_name in sun_planetary_gear_names:
            gear_obj  = self.obj_dict[sun_planetary_gear_name]
            gear_pos  = gear_obj.data.root_state_w[:, :3].clone()
            gear_quat = gear_obj.data.root_state_w[:, 3:7].clone()

            self.sun_planetary_gear_positions_w.append(gear_pos)
            self.sun_planetary_gear_quats_w.append(gear_quat)

        # Planetary carrier
        self.planetary_carrier_pos_w  = self.planetary_carrier.data.root_state_w[:, :3].clone()
        self.planetary_carrier_quat_w = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        # Ring gear
        self.ring_gear_pos_w          = self.ring_gear.data.root_state_w[:, :3].clone()
        self.ring_gear_quat_w         = self.ring_gear.data.root_state_w[:, 3:7].clone()

        # Planetary reducer
        self.planetary_reducer_pos_w  = self.planetary_reducer.data.root_state_w[:, :3].clone()
        self.planetary_reducer_quat_w = self.planetary_reducer.data.root_state_w[:, 3:7].clone()

        # Pin in planetary carrier
        self.pin_positions_w = []
        self.pin_quats_w     = []
        for pin_local_pos in self.pin_local_positions:
            pin_quat_repeated = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
            pin_local_pos_repeated = pin_local_pos.repeat(num_envs, 1)

            pin_quat, pin_pos = torch_utils.tf_combine(
                self.planetary_carrier_quat_w, 
                self.planetary_carrier_pos_w, 
                pin_quat_repeated, 
                pin_local_pos_repeated
            )

            self.pin_positions_w.append(pin_pos)
            self.pin_quats_w.append(pin_quat)

    def observe_assembly_state(self):
        # Observe object state
        self.observe_object_state()

        # Used member variables
        pin_positions_w = self.pin_positions_w
        pin_quats_w = self.pin_quats_w
        sun_planetary_gear_positions_w = self.sun_planetary_gear_positions_w
        sun_planetary_gear_quats_w = self.sun_planetary_gear_quats_w
        planetary_carrier_pos_w = self.planetary_carrier_pos_w
        planetary_carrier_quat_w = self.planetary_carrier_quat_w
        ring_gear_pos_w = self.ring_gear_pos_w
        ring_gear_quat_w = self.ring_gear_quat_w
        planetary_reducer_pos_w = self.planetary_reducer_pos_w
        planetary_reducer_quat_w = self.planetary_reducer_quat_w
        num_envs = self.scene.num_envs
        
        base_pos_w = self.base_pos_w

        # initialize
        self.num_mounted_planetary_gears              = 0
        self.is_sun_gear_mounted                      = False
        self.is_ring_gear_mounted                     = False
        self.is_planetary_reducer_mounted             = False
        self.unmounted_sun_planetary_gear_positions_w = []
        self.unmounted_sun_planetary_gear_quats_w     = []
        self.unmounted_pin_positions_w                = []
        
        # -------------------------------------------------------------------------------------------------------------------------- #
        # How many planetary gear mounted on planetary carrier? -------------------------------------------------------------------- # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        pin_occupied = [False] * len(pin_positions_w)
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions_w)):
            sun_planetary_gear_pos = sun_planetary_gear_positions_w[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats_w[sun_planetary_gear_idx]

            is_mounted = False
            for pin_idx in range(len(pin_positions_w)):
                pin_pos = pin_positions_w[pin_idx]
                pin_quat = pin_quats_w[pin_idx]

                horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - pin_pos[:, :2])
                vertical_error = sun_planetary_gear_pos[:, 2] - pin_pos[:, 2]
                orientation_error = torch.acos(torch.clamp((sun_planetary_gear_quat * pin_quat).sum(dim=-1), -1.0, 1.0))

                th = self.mounting_thresholds["planetary_gear"]
                if (horizontal_error < th["horizontal"] and
                    vertical_error < th["vertical"] and
                    orientation_error < th["orientation"]):
                    self.num_mounted_planetary_gears += 1
                    is_mounted = True
                    pin_occupied[pin_idx] = True

            if not is_mounted:
                self.unmounted_sun_planetary_gear_positions_w.append(self.sun_planetary_gear_positions_w[sun_planetary_gear_idx])
                self.unmounted_sun_planetary_gear_quats_w.append(self.sun_planetary_gear_quats_w[sun_planetary_gear_idx])
        self.unmounted_pin_positions_w = [pin_positions_w[i] for i in range(len(pin_positions_w)) if not pin_occupied[i]]

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Sorting ------------------------------------------------------------------------------------------------------------------ # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        if len(self.unmounted_pin_positions_w) > 0 and len(self.unmounted_sun_planetary_gear_positions_w) > 0:
            num_envs = self.scene.num_envs
            
            # 1. 핀 정렬: Base 기준 먼 순서 (기존과 동일)
            pins_batch = torch.stack(self.unmounted_pin_positions_w, dim=0).squeeze(2)
            pin_to_base_dist = torch.norm(pins_batch[..., :2] - self.base_pos_w[:, :2].unsqueeze(0), dim=2)
            sorted_pin_idx = torch.argsort(pin_to_base_dist, dim=0, descending=True)
            
            self.unmounted_pin_positions_w = [
                torch.stack([pins_batch[sorted_pin_idx[i, e], e] for e in range(num_envs)], dim=0).unsqueeze(1) 
                for i in range(sorted_pin_idx.shape[0])
            ]
            
            # 2. 기어 정렬: 타겟 핀과 Y축 부호가 같은 기어를 우선순위로 정렬
            gears_batch = torch.stack(self.unmounted_sun_planetary_gear_positions_w, dim=0).squeeze(2)
            target_pin_pos = self.unmounted_pin_positions_w[0].squeeze(1) # [num_envs, 3]
            
            # 실제 거리 계산
            gear_to_pin_dist = torch.norm(gears_batch[..., :2] - target_pin_pos[:, :2].unsqueeze(0), dim=2)
            
            # --- [추가] 부호 판별 및 패널티 부여 ---
            # 핀의 Y 부호와 기어들의 Y 부호가 다른지 확인 (다르면 True)
            # 곱했을 때 음수가 나오면 부호가 다른 것임
            pin_y_sign = target_pin_pos[:, 1].unsqueeze(0) # [1, num_envs]
            gear_y_signs = gears_batch[:, :, 1]            # [N개기어, num_envs]
            
            different_side_mask = (pin_y_sign * gear_y_signs) < 0
            
            # 부호가 다른 기어들에게 100m라는 큰 거리를 더해 리스트 뒤로 보냄
            penalty = different_side_mask.float() * 100.0
            adjusted_dist = gear_to_pin_dist + penalty
            # --------------------------------------

            # 수정된 거리(adjusted_dist) 기준으로 정렬
            sorted_gear_idx = torch.argsort(adjusted_dist, dim=0)
            
            self.unmounted_sun_planetary_gear_positions_w = [
                torch.stack([gears_batch[sorted_gear_idx[i, e], e] for e in range(num_envs)], dim=0).unsqueeze(1) 
                for i in range(sorted_gear_idx.shape[0])
            ]

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Is the sun gear mounted? ------------------------------------------------------------------------------------------------- # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions_w)):
            sun_planetary_gear_pos = sun_planetary_gear_positions_w[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats_w[sun_planetary_gear_idx]

            horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - planetary_carrier_pos_w[:, :2])
            vertical_error = sun_planetary_gear_pos[:, 2] - planetary_carrier_pos_w[:, 2]
            orientation_error = torch.acos(torch.clamp((sun_planetary_gear_quat * planetary_carrier_quat_w).sum(dim=-1), -1.0, 1.0))

            th = self.mounting_thresholds["sun_gear"]
            if (horizontal_error < th["horizontal"] and
                vertical_error < th["vertical"] and
                orientation_error < th["orientation"]):
                self.is_sun_gear_mounted = True

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Is the ring gear mounted? ------------------------------------------------------------------------------------------------ # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        horizontal_error = torch.norm(planetary_carrier_pos_w[:, :2] - ring_gear_pos_w[:, :2])
        vertical_error = planetary_carrier_pos_w[:, 2] - ring_gear_pos_w[:, 2]
        # orientation_error = torch.acos(torch.dot(planetary_carrier_quat_w.squeeze(0), ring_gear_quat_w.squeeze(0)))
        orientation_error = torch.acos(torch.clamp((planetary_carrier_quat_w * ring_gear_quat_w).sum(dim=-1), -1.0, 1.0))

        th = self.mounting_thresholds["ring_gear"]
        if (horizontal_error < th["horizontal"] and
            vertical_error < th["vertical"] and
            orientation_error < th["orientation"]):
            self.is_ring_gear_mounted = True

        # -------------------------------------------------------------------------------------------------------------------------- #
        # Is the planetary reducer mounted? ---------------------------------------------------------------------------------------- # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        for sun_planetary_gear_idx in range(len(sun_planetary_gear_positions_w)):
            sun_planetary_gear_pos = sun_planetary_gear_positions_w[sun_planetary_gear_idx]
            sun_planetary_gear_quat = sun_planetary_gear_quats_w[sun_planetary_gear_idx]

            horizontal_error = torch.norm(sun_planetary_gear_pos[:, :2] - planetary_reducer_pos_w[:, :2])
            # vertical_error = sun_planetary_gear_pos[:, 2] - planetary_reducer_pos_w[:, 2]
            vertical_error = planetary_reducer_pos_w[:, 2] - sun_planetary_gear_pos[:, 2]
            # vertical_error = torch.abs(sun_planetary_gear_pos[:, 2] - planetary_reducer_pos_w[:, 2])
            orientation_error = torch.acos(torch.clamp((sun_planetary_gear_quat * planetary_reducer_quat_w).sum(dim=-1), -1.0, 1.0))

            th = self.mounting_thresholds["planetary_reducer"]
            if (horizontal_error < th["horizontal"] and
                vertical_error < th["vertical"] and
                orientation_error < th["orientation"]):
                self.is_planetary_reducer_mounted = True

        if self.unmounted_sun_planetary_gear_positions_w:
            self.target_sun_planetary_gear_pos  = self.unmounted_sun_planetary_gear_positions_w[0].view(num_envs, -1).clone()
            self.target_sun_planetary_gear_quat = self.unmounted_sun_planetary_gear_quats_w[0].view(num_envs, -1).clone()
        if self.unmounted_pin_positions_w:
            self.target_pin_pos = self.unmounted_pin_positions_w[0].view(num_envs, -1).clone()

        print("| Aseembly State                       |")
        print("----------------------------------------")
        print(f"| # of mounted planetary gears | {self.num_mounted_planetary_gears}     |")
        print(f"| sun gear                     | {self.is_sun_gear_mounted} |")
        print(f"| ring gear                    | {self.is_ring_gear_mounted} |")
        print(f"| planetary reducer            | {self.is_planetary_reducer_mounted} |")
        print("----------------------------------------")

        score = self.num_mounted_planetary_gears + int(self.is_sun_gear_mounted) + int(self.is_ring_gear_mounted) + int(self.is_planetary_reducer_mounted)
        print(f"score: {score}")

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
        ee_pos_b, ee_quat_b = self.transform_world_to_base(
            pos_w =ee_pos_w,
            quat_w=ee_quat_w
        )

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, arm_joint_ids
        ]
        current_arm_joint_pose = self.robot.data.joint_pos[                          # state space to be able to measured        
            :, arm_joint_ids
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
            object_name: str # planetary_gear
        ) -> None:
        # Observe robot state and object state
        self.observe_robot_state()
        self.observe_object_state()
        self.observe_assembly_state()

        # Used member variables
        num_envs = self.scene.num_envs
        initial_left_ee_pos_e = self.initial_left_ee_pos_e
        initial_left_ee_quat_w = self.initial_left_ee_quat_w
        initial_right_ee_pos_e = self.initial_right_ee_pos_e
        initial_right_ee_quat_w = self.initial_right_ee_quat_w
        initial_left_ee_pos_e_batch = self.initial_left_ee_pos_e_batch
        initial_left_ee_quat_w_batch = self.initial_left_ee_quat_w_batch
        initial_right_ee_pos_e_batch = self.initial_right_ee_pos_e_batch
        initial_right_ee_quat_w_batch = self.initial_right_ee_quat_w_batch
        target_sun_planetary_gear_pos = self.target_sun_planetary_gear_pos
        target_sun_planetary_gear_quat = self.target_sun_planetary_gear_quat
        target_pin_pos = self.target_pin_pos

        left_ee_pos_w = self.left_ee_pos_w
        right_ee_pos_w = self.right_ee_pos_w

        TIME_CONSTANT_50 = 50
        TIME_CONSTANT_100 = 100 
        TIME_CONSTANT_150 = 150 


        object_height_offset = 0.0

        print(f"[PICK & PLACE FSM] {self.pick_and_place_fsm_state}")
        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Start State] INITIALIZATION ----------------------------------------------------------------------------------------- # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        if self.pick_and_place_fsm_state == PickAndPlaceState.INITIALIZATION:
            # -------------------------------------------------------------------------------------------------------------------------- #
            # Planetary gear ----------------------------------------------------------------------------------------------------------- # 
            # -------------------------------------------------------------------------------------------------------------------------- #
            if len(self.unmounted_pin_positions_w) >= 1:
                # Pick
                target_pick_pos_w = target_sun_planetary_gear_pos
                target_pick_quat_w = target_sun_planetary_gear_quat

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
                target_place_pos_w = target_pin_pos
                target_place_pos_w[:, 2] = self.table_height + self.grasping_height
                target_place_pos_w[:, 2] += object_height_offset
                target_place_pos_w += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)
                target_place_quat_w_batch = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
                target_place_ready_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, 0.1], device=self.device)
                target_place_approach_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, 0.02], device=self.device)

                # Update member variables
                self.target_pick_pos_w       = target_pick_pos_w
                self.target_pick_quat_w      = target_pick_quat_w
                self.target_pick_ready_pos_w = target_pick_ready_pos_w
                self.target_place_pos_w = target_place_pos_w
                self.target_place_ready_pos_w = target_place_ready_pos_w
                self.target_place_approach_pos_w = target_place_approach_pos_w

                # Transform coordinate from world to base
                self.target_pick_pos_b, self.target_pick_quat_b = self.transform_world_to_base(
                    pos_w =target_pick_pos_w,
                    quat_w=target_pick_quat_w   
                )
                self.target_pick_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_pick_ready_pos_w,
                    quat_w=target_pick_quat_w   
                )
                _, self.target_place_quat_b = self.transform_world_to_base(
                    pos_w =target_place_pos_w,       
                    quat_w=target_place_quat_w_batch 
                )
                self.target_place_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_ready_pos_w,       
                    quat_w=target_place_quat_w_batch 
                )
                self.target_place_approach_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_approach_pos_w,       
                    quat_w=target_place_quat_w_batch 
                )

            # -------------------------------------------------------------------------------------------------------------------------- #
            # Initial end effector and gripper state  ---------------------------------------------------------------------------------- # 
            # -------------------------------------------------------------------------------------------------------------------------- #
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
                target_ee_pos_b=initial_left_ee_pos_b, 
                target_ee_quat_b=initial_left_ee_quat_b
            )
            desired_right_joint_position, desired_right_joint_ids = self.solve_inverse_kinematics( 
                arm_name="right",
                target_ee_pos_b=initial_right_ee_pos_b, 
                target_ee_quat_b=initial_right_ee_quat_b
            )
            desired_left_gripper_ids = self.left_gripper_entity_cfg.joint_ids
            desired_left_gripper_position = torch.tensor([[0.035, 0.035]], device=self.device).repeat(num_envs, 1)
            desired_right_gripper_ids = self.right_gripper_entity_cfg.joint_ids
            desired_right_gripper_position = torch.tensor([[0.035, 0.035]], device=self.device).repeat(num_envs, 1)
            self.joint_position_command = torch.cat(
                [
                    desired_left_joint_position, 
                    desired_right_joint_position, 
                    desired_left_gripper_position, 
                    desired_right_gripper_position
                ], 
                dim=1
            )
            self.joint_command_ids = torch.tensor(
                desired_left_joint_ids + desired_right_joint_ids + desired_left_gripper_ids + desired_right_gripper_ids, 
                device=self.device
            )

            # Decide arm and gripper configuration
            dist_left = torch.norm(target_pick_pos_w - initial_left_ee_pos_e, dim=1).mean()
            dist_right = torch.norm(target_pick_pos_w - initial_right_ee_pos_e, dim=1).mean()
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

        # Used member variables
        target_pick_pos_b           = self.target_pick_pos_b
        target_pick_quat_b          = self.target_pick_quat_b
        target_pick_ready_pos_b     = self.target_pick_ready_pos_b
        target_place_quat_b         = self.target_place_quat_b
        target_place_ready_pos_b    = self.target_place_ready_pos_b
        target_place_approach_pos_b = self.target_place_approach_pos_b
        target_pick_pos_w           = self.target_pick_pos_w
        target_pick_ready_pos_w     = self.target_pick_ready_pos_w
        target_place_ready_pos_w    = self.target_place_ready_pos_w
        target_place_approach_pos_w = self.target_place_approach_pos_w

        gripper_joint_ids           = self.active_gripper_joint_ids
        arm_name                    = self.active_arm_name
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
            desired_joint_position = torch.tensor([[0.03, 0.03]], device=self.device)
            desired_joint_ids = gripper_joint_ids
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
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics(
                arm_name        =arm_name,
                target_ee_pos_b =target_pick_ready_pos_b,
                target_ee_quat_b=target_pick_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_COMPLETE -> PLACE_READY
            self.pick_and_place_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_pick_ready_pos_w) and 
                self.pick_and_place_fsm_timer > TIME_CONSTANT_50 or
                self.pick_and_place_fsm_timer > TIME_CONSTANT_150):
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
                (num_gripper_joints,), 0.035, device=self.device
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

    def pick_and_place2(self,
            object_name: str # sun_gear, ring_gear, planetary_reducer
        ) -> None:
        # Observe robot state and object state
        self.observe_robot_state()
        self.observe_object_state()
        self.observe_assembly_state()

        # Used member variables
        num_envs = self.scene.num_envs
        initial_left_ee_pos_e = self.initial_left_ee_pos_e
        initial_left_ee_quat_w = self.initial_left_ee_quat_w
        initial_right_ee_pos_e = self.initial_right_ee_pos_e
        initial_right_ee_quat_w = self.initial_right_ee_quat_w
        initial_left_ee_pos_e_batch = self.initial_left_ee_pos_e_batch
        initial_left_ee_quat_w_batch = self.initial_left_ee_quat_w_batch
        initial_right_ee_pos_e_batch = self.initial_right_ee_pos_e_batch
        initial_right_ee_quat_w_batch = self.initial_right_ee_quat_w_batch
        target_sun_planetary_gear_pos = self.target_sun_planetary_gear_pos
        target_sun_planetary_gear_quat = self.target_sun_planetary_gear_quat
        target_pin_pos = self.target_pin_pos
        ring_gear_pos_w = self.ring_gear_pos_w
        ring_gear_quat_w = self.ring_gear_quat_w
        planetary_reducer_pos_w = self.planetary_reducer_pos_w
        planetary_reducer_quat_w = self.planetary_reducer_quat_w

        left_ee_pos_w = self.left_ee_pos_w
        right_ee_pos_w = self.right_ee_pos_w

        TIME_CONSTANT_50 = 50
        TIME_CONSTANT_100 = 100 
        TIME_CONSTANT_150 = 150 

        if object_name == "planetary_gear":
            object_height_offset = 0.0
        elif object_name == "sun_gear":
            object_height_offset = 0.005
            mount_height_offset  = 0.04
        elif object_name == "ring_gear":
            object_height_offset = 0.030
            mount_height_offset  = 0.04
        elif object_name == "planetary_reducer":
            object_height_offset = 0.05
            mount_height_offset = 0.025

        print(f"[PICK & PLACE FSM] {self.pick_and_place2_fsm_state}")
        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Start State] INITIALIZATION ----------------------------------------------------------------------------------------- # 
        # -------------------------------------------------------------------------------------------------------------------------- #
        if self.pick_and_place2_fsm_state == PickAndPlaceState.INITIALIZATION:
            # -------------------------------------------------------------------------------------------------------------------------- #
            # Sun gear ----------------------------------------------------------------------------------------------------------------- # 
            # -------------------------------------------------------------------------------------------------------------------------- #
            if object_name == "sun_gear":
                # Pick
                target_pick_pos_w  = target_sun_planetary_gear_pos
                target_pick_quat_w = target_sun_planetary_gear_quat

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
                target_pick_ready_pos_w = target_pick_pos_w + torch.tensor([0.0, 0.0, 0.07], device=self.device)

                # Place
                target_place_pos_w = self.planetary_carrier_pos_w
                target_place_pos_w[:, 2] = self.table_height + self.grasping_height
                target_place_pos_w[:, 2] += object_height_offset
                target_place_pos_w += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
                target_place_quat_w = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.sim.device).repeat(num_envs, 1)

                self.lifting_height = 0.2
                target_place_ready_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, self.lifting_height], device=self.device)
                target_place_approach_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, mount_height_offset], device=self.device)

                # Update member variables
                self.target_pick_pos_w       = target_pick_pos_w
                self.target_pick_quat_w      = target_pick_quat_w
                self.target_pick_ready_pos_w = target_pick_ready_pos_w
                self.target_place_pos_w = target_place_pos_w
                self.target_place_ready_pos_w = target_place_ready_pos_w
                self.target_place_approach_pos_w = target_place_approach_pos_w

                # Transform coordinate from world to base
                self.target_pick_pos_b, self.target_pick_quat_b = self.transform_world_to_base(
                    pos_w =target_pick_pos_w,
                    quat_w=target_pick_quat_w   
                )
                self.target_pick_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_pick_ready_pos_w,
                    quat_w=target_pick_quat_w   
                )
                _, self.target_place_quat_b = self.transform_world_to_base(
                    pos_w =target_place_pos_w,       
                    quat_w=target_place_quat_w
                )
                self.target_place_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_ready_pos_w,       
                    quat_w=target_place_quat_w
                )
                self.target_place_approach_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_approach_pos_w,       
                    quat_w=target_place_quat_w
                )
            # -------------------------------------------------------------------------------------------------------------------------- #
            # Ring gear ---------------------------------------------------------------------------------------------------------------- # 
            # -------------------------------------------------------------------------------------------------------------------------- #
            elif object_name == "ring_gear":
                # Pick
                target_pick_pos_w  = ring_gear_pos_w
                target_pick_quat_w = ring_gear_quat_w

                target_pick_pos_w[:, 2] = self.table_height + self.grasping_height + object_height_offset
                target_pick_pos_w = target_pick_pos_w + torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)

                rotate_y_180_batch = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
                zero_pos_batch = torch.tensor([[0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
                target_pick_quat_w, target_pick_pos_w = torch_utils.tf_combine(
                    target_pick_quat_w, 
                    target_pick_pos_w,
                    rotate_y_180_batch, 
                    zero_pos_batch      
                )
                target_pick_ready_pos_w = target_pick_pos_w + torch.tensor([0.0, 0.0, 0.07], device=self.device)

                # Place
                target_place_pos_w = self.planetary_carrier_pos_w
                target_place_pos_w[:, 2] = self.table_height + self.grasping_height
                target_place_pos_w[:, 2] += object_height_offset
                target_place_pos_w += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
                target_place_quat_w = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.sim.device).repeat(num_envs, 1)

                self.lifting_height = 0.2
                target_place_ready_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, self.lifting_height], device=self.device)
                target_place_approach_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, mount_height_offset], device=self.device)

                # Update member variables
                self.target_pick_pos_w       = target_pick_pos_w
                self.target_pick_quat_w      = target_pick_quat_w
                self.target_pick_ready_pos_w = target_pick_ready_pos_w
                self.target_place_pos_w = target_place_pos_w
                self.target_place_ready_pos_w = target_place_ready_pos_w
                self.target_place_approach_pos_w = target_place_approach_pos_w

                # Transform coordinate from world to base
                self.target_pick_pos_b, self.target_pick_quat_b = self.transform_world_to_base(
                    pos_w =target_pick_pos_w,
                    quat_w=target_pick_quat_w   
                )
                self.target_pick_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_pick_ready_pos_w,
                    quat_w=target_pick_quat_w   
                )
                _, self.target_place_quat_b = self.transform_world_to_base(
                    pos_w =target_place_pos_w,       
                    quat_w=target_place_quat_w
                )
                self.target_place_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_ready_pos_w,       
                    quat_w=target_place_quat_w
                )
                self.target_place_approach_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_approach_pos_w,       
                    quat_w=target_place_quat_w
                )

            # -------------------------------------------------------------------------------------------------------------------------- #
            # Planetary reducer -------------------------------------------------------------------------------------------------------- # 
            # -------------------------------------------------------------------------------------------------------------------------- #
            elif object_name == "planetary_reducer":
                # pick
                target_pick_pos_w = planetary_reducer_pos_w
                target_pick_quat_w = planetary_reducer_quat_w

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
                target_place_pos_w = target_sun_planetary_gear_pos
                target_place_pos_w[:, 2] = self.table_height + self.grasping_height
                target_place_pos_w[:, 2] += object_height_offset
                target_place_pos_w += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)
                # target_place_quat_w_batch = target_sun_planetary_gear_quat
                base_quat = target_sun_planetary_gear_quat 
                # 2. 위에서 아래로 꽂기 위해 그리퍼를 180도 뒤집는 회전 생성 (Y축 기준 180도)
                rotate_y_180 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
                zero_pos = torch.tensor([[0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
                # 3. 두 회전을 결합하여 '뒤집힌 상태로 홈을 맞추는' 쿼터니언 생성
                target_place_quat_w_batch, _ = torch_utils.tf_combine(
                    base_quat, zero_pos,
                    rotate_y_180, zero_pos
                )

                target_place_ready_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, 0.1], device=self.device)
                target_place_approach_pos_w = target_place_pos_w + torch.tensor([0.0, 0.0, mount_height_offset], device=self.device)

                # Update member variables
                self.target_pick_pos_w       = target_pick_pos_w
                self.target_pick_quat_w      = target_pick_quat_w
                self.target_pick_ready_pos_w = target_pick_ready_pos_w
                self.target_place_pos_w = target_place_pos_w
                self.target_place_ready_pos_w = target_place_ready_pos_w
                self.target_place_approach_pos_w = target_place_approach_pos_w

                # Transform coordinate from world to base
                self.target_pick_pos_b, self.target_pick_quat_b = self.transform_world_to_base(
                    pos_w =target_pick_pos_w,
                    quat_w=target_pick_quat_w   
                )
                self.target_pick_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_pick_ready_pos_w,
                    quat_w=target_pick_quat_w   
                )
                _, self.target_place_quat_b = self.transform_world_to_base(
                    pos_w =target_place_pos_w,       
                    quat_w=target_place_quat_w_batch 
                )
                self.target_place_ready_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_ready_pos_w,       
                    quat_w=target_place_quat_w_batch 
                )
                self.target_place_approach_pos_b, _ = self.transform_world_to_base(
                    pos_w =target_place_approach_pos_w,       
                    quat_w=target_place_quat_w_batch 
                )

            # -------------------------------------------------------------------------------------------------------------------------- #
            # Initial end effector and gripper state  ---------------------------------------------------------------------------------- # 
            # -------------------------------------------------------------------------------------------------------------------------- #
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
                target_ee_pos_b=initial_left_ee_pos_b, 
                target_ee_quat_b=initial_left_ee_quat_b
            )
            desired_right_joint_position, desired_right_joint_ids = self.solve_inverse_kinematics( 
                arm_name="right",
                target_ee_pos_b=initial_right_ee_pos_b, 
                target_ee_quat_b=initial_right_ee_quat_b
            )
            desired_left_gripper_ids = self.left_gripper_entity_cfg.joint_ids
            desired_left_gripper_position = torch.tensor([[0.035, 0.035]], device=self.device).repeat(num_envs, 1)
            desired_right_gripper_ids = self.right_gripper_entity_cfg.joint_ids
            desired_right_gripper_position = torch.tensor([[0.035, 0.035]], device=self.device).repeat(num_envs, 1)

            if object_name == "planetary_reducer":
                desired_left_gripper_position = torch.tensor([[0.01, 0.01]], device=self.device).repeat(num_envs, 1)
                desired_right_gripper_position = torch.tensor([[0.01, 0.01]], device=self.device).repeat(num_envs, 1)

            self.joint_position_command = torch.cat(
                [
                    desired_left_joint_position, 
                    desired_right_joint_position, 
                    desired_left_gripper_position, 
                    desired_right_gripper_position
                ], 
                dim=1
            )
            self.joint_command_ids = torch.tensor(
                desired_left_joint_ids + desired_right_joint_ids + desired_left_gripper_ids + desired_right_gripper_ids, 
                device=self.device
            )

            # Decide arm and gripper configuration
            dist_left = torch.norm(target_pick_pos_w - initial_left_ee_pos_e, dim=1).mean()
            dist_right = torch.norm(target_pick_pos_w - initial_right_ee_pos_e, dim=1).mean()
            if dist_left <= dist_right:
                self.active_arm_name = "left"
                self.active_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
            else:
                self.active_arm_name = "right"
                self.active_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids

            # [State Transition] INITIALIZATION -> PICK_READY
            self.pick_and_place2_fsm_timer += 1
            if self.pick_and_place2_fsm_timer > TIME_CONSTANT_50:
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PICK_READY

        # Used member variables
        target_pick_pos_b           = self.target_pick_pos_b
        target_pick_quat_b          = self.target_pick_quat_b
        target_pick_ready_pos_b     = self.target_pick_ready_pos_b
        target_place_quat_b         = self.target_place_quat_b
        target_place_ready_pos_b    = self.target_place_ready_pos_b
        target_place_approach_pos_b = self.target_place_approach_pos_b
        target_pick_pos_w           = self.target_pick_pos_w
        target_pick_ready_pos_w     = self.target_pick_ready_pos_w
        target_place_ready_pos_w    = self.target_place_ready_pos_w
        target_place_approach_pos_w = self.target_place_approach_pos_w

        gripper_joint_ids           = self.active_gripper_joint_ids
        arm_name                    = self.active_arm_name
        if arm_name == "left":
            ee_pos_w = left_ee_pos_w
        else:
            ee_pos_w = right_ee_pos_w

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_READY -------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        if self.pick_and_place2_fsm_state == PickAndPlaceState.PICK_READY:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_pick_ready_pos_b, 
                target_ee_quat_b=target_pick_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_READY -> PICK_APPROACH
            self.pick_and_place2_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_pick_ready_pos_w) and 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_50 or 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PICK_APPROACH

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_APPROACH ----------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == PickAndPlaceState.PICK_APPROACH:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_pick_pos_b, 
                target_ee_quat_b=target_pick_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_APPROACH -> PICK_EXECUTION
            self.pick_and_place2_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_pick_pos_w) and 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_100 or 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PICK_EXECUTION

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_EXECUTION ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == PickAndPlaceState.PICK_EXECUTION:
            if object_name == "planetary_reducer":
                desired_joint_position = torch.tensor([[0.0, 0.0]], device=self.device)             # 오브젝트에 따라 그립 정도를 구분해야 한다.
            else:
                desired_joint_position = torch.tensor([[0.02, 0.02]], device=self.device)
            desired_joint_ids = gripper_joint_ids
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_EXECUTION -> PICK_COMPLETE
            self.pick_and_place2_fsm_timer += 1
            if self.pick_and_place2_fsm_timer > TIME_CONSTANT_100:
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PICK_COMPLETE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PICK_COMPLETE ----------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == PickAndPlaceState.PICK_COMPLETE:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics(
                arm_name        =arm_name,
                target_ee_pos_b =target_pick_ready_pos_b,
                target_ee_quat_b=target_pick_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PICK_COMPLETE -> PLACE_READY
            self.pick_and_place2_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_pick_ready_pos_w) and 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_50 or
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PLACE_READY

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_READY ------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == PickAndPlaceState.PLACE_READY:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_place_ready_pos_b, 
                target_ee_quat_b=target_place_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PLACE_READY -> PLACE_APPROACH
            self.pick_and_place2_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_place_ready_pos_w) and 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_50 or 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PLACE_APPROACH

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_APPROACH ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == PickAndPlaceState.PLACE_APPROACH:
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name=arm_name,
                target_ee_pos_b=target_place_approach_pos_b, 
                target_ee_quat_b=target_place_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PLACE_APPROACH -> TWIST_INSERTION
            self.pick_and_place2_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_place_approach_pos_w) and 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_100 or
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = "TWIST_INSERTION"

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] TWIST_INSERTION --------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == "TWIST_INSERTION":
            rot_deg = 120.0                     # 총 회전 목표 각도
            insertion_depth = 0.0055            # 추가로 내려갈 깊이 (mount_height_offset 이후)
            TWIST_TIME = 300                    # 회전 삽입에 소요될 총 스텝 수

            if object_name == "planetary_reducer":
                rot_deg = 30
                TWIST_TIME = 80
            elif object_name == "planetary_gear":
                rot_deg = 30
                TWIST_TIME = 80

            # 2. 초기 상태 저장 (상태 진입 첫 프레임)
            if self.pick_and_place2_fsm_timer == 0:
                # 현재 팔의 모든 관절 각도 저장
                arm_joint_ids = self.left_arm_entity_cfg.joint_ids if arm_name == "left" else self.right_arm_entity_cfg.joint_ids
                self.step_initial_joint_pos = self.robot.data.joint_pos[:, arm_joint_ids].clone()
                
                # 현재 End-Effector의 위치를 기준으로 삽입 목표 위치 설정
                self.twist_initial_ee_pos_b, self.twist_initial_ee_quat_b = self.transform_world_to_base(
                    ee_pos_w, 
                    self.left_ee_quat_w if arm_name == "left" else self.right_ee_quat_w
                )
                self.twist_target_ee_pos_b = self.twist_initial_ee_pos_b.clone()
                self.twist_target_ee_pos_b[:, 2] -= insertion_depth # 아래로 더 삽입

            # 3. 보간(Interpolation) 비율 계산 (0.0 ~ 1.0)
            alpha = min(self.pick_and_place2_fsm_timer / TWIST_TIME, 1.0)

            # 4. 하이브리드 제어: IK(위치) + Joint(회전)
            # A. 위치 제어 (천천히 아래로 삽입)
            current_ee_pos_b = torch.lerp(self.twist_initial_ee_pos_b, self.twist_target_ee_pos_b, alpha)
            
            # B. IK를 통한 기본 관절 각도 계산
            desired_arm_jpos, desired_arm_ids = self.solve_inverse_kinematics(
                arm_name=arm_name,
                target_ee_pos_b=current_ee_pos_b,
                target_ee_quat_b=target_place_quat_b
            )

            # C. 회전 추가 (마지막 축인 인덱스 5번에 회전각 누적)
            delta_rot_rad = (rot_deg * torch.pi / 180.0) * alpha

            wiggle_amplitude = 15.0 * torch.pi / 180.0  # 5도 정도의 진동
            wiggle_freq = 10.0                          # 진동 속도
            wiggle_offset = wiggle_amplitude * torch.sin(torch.tensor(self.pick_and_place2_fsm_timer * wiggle_freq))

            desired_arm_jpos[:, 5] += (delta_rot_rad + wiggle_offset)

            self.joint_position_command = desired_arm_jpos
            self.joint_command_ids = desired_arm_ids

            # [State Transition] TWIST_INSERTION -> PLACE_EXECUTION
            self.pick_and_place2_fsm_timer += 1
            if self.pick_and_place2_fsm_timer >= TWIST_TIME:
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PLACE_EXECUTION

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_EXECUTION --------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == PickAndPlaceState.PLACE_EXECUTION:
            desired_joint_ids = gripper_joint_ids
            num_gripper_joints = len(desired_joint_ids)
            desired_joint_position = torch.full(
                (num_gripper_joints,), 0.035, device=self.device
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids = desired_joint_ids

            # [State Transition] PLACE_EXECUTION -> PLACE_COMPLETE
            self.pick_and_place2_fsm_timer += 1
            if self.pick_and_place2_fsm_timer > TIME_CONSTANT_150:
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.PLACE_COMPLETE

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [FSM Intermediate State] PLACE_COMPLETE ---------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        elif self.pick_and_place2_fsm_state == PickAndPlaceState.PLACE_COMPLETE:
            if self.pick_and_place2_fsm_timer == 0:
                self.fixed_ee_pos_b, self.fixed_ee_quat_b = self.transform_world_to_base(
                    ee_pos_w, 
                    self.left_ee_quat_w if arm_name == "left" else self.right_ee_quat_w
                )
                self.fixed_ee_pos_b[:, 2] += 0.1
            desired_joint_position, desired_joint_ids = self.solve_inverse_kinematics( 
                arm_name        =arm_name,
                target_ee_pos_b =self.fixed_ee_pos_b, 
                target_ee_quat_b=self.fixed_ee_quat_b
            )
            self.joint_position_command = desired_joint_position
            self.joint_command_ids      = desired_joint_ids

            # [State Transition] PLACE_COMPLETE -> FINALIZATION
            self.pick_and_place2_fsm_timer += 1
            if (self.position_reached(ee_pos_w, target_place_ready_pos_w) and 
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_50 or
                self.pick_and_place2_fsm_timer > TIME_CONSTANT_150):
                self.pick_and_place2_fsm_timer = 0
                self.pick_and_place2_fsm_state = PickAndPlaceState.FINALIZATION

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

    def reset_pick_and_place2(self):
        self.pick_and_place2_fsm_state = PickAndPlaceState.INITIALIZATION
        self.pick_and_place2_fsm_timer = 0

    # Utility functions
    def position_reached(self, current_pos, target_pos, tol=0.01):
        error = torch.norm(current_pos - target_pos, dim=1)
        return error < tol
            
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