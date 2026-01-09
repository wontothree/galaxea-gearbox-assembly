from .dual_arm_pick_and_place_fsm import DualArmPickAndPlaceFSM

from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils

from isaaclab.managers import SceneEntityCfg

import isaacsim.core.utils.torch as torch_utils
import torch

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

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] observe_robot_state ------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        # Robot parameter
        self.left_arm_entity_cfg = SceneEntityCfg(
            "robot",                            # robot entity name
            joint_names=[f"left_arm_joint.*"],  # joint entity set
            body_names=[f"left_arm_link6"]      # body entity set (ee)
        )
        self.right_arm_entity_cfg = SceneEntityCfg(
            "robot",                           
            joint_names=[f"right_arm_joint.*"],
            body_names=[f"right_arm_link6"]    
        )
        self.left_arm_entity_cfg.resolve(self.scene)
        self.right_arm_entity_cfg.resolve(self.scene)
        
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
        self.target_sun_planetary_gear_pos_w = None
        self.target_sun_planetary_gear_quat_w = None
        self.target_pin_pos_w = None

        # -------------------------------------------------------------------------------------------------------------------------- #
        # [Function] DualArmPickAndPlaceFSM ----------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------- #
        self.joint_pos_command           = None
        self.joint_pos_command_ids       = None
        self.dual_arm_pick_and_place_fsm = DualArmPickAndPlaceFSM(scene=scene, device=self.device)
        self.state = None

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
        pin_positions_w                = self.pin_positions_w
        pin_quats_w                    = self.pin_quats_w
        sun_planetary_gear_positions_w = self.sun_planetary_gear_positions_w
        sun_planetary_gear_quats_w     = self.sun_planetary_gear_quats_w
        planetary_carrier_pos_w        = self.planetary_carrier_pos_w
        planetary_carrier_quat_w       = self.planetary_carrier_quat_w
        ring_gear_pos_w                = self.ring_gear_pos_w
        ring_gear_quat_w               = self.ring_gear_quat_w
        planetary_reducer_pos_w        = self.planetary_reducer_pos_w
        planetary_reducer_quat_w       = self.planetary_reducer_quat_w
        num_envs                       = self.scene.num_envs
        
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
            pin_to_base_dist = torch.norm(pins_batch[..., :2] - base_pos_w[:, :2].unsqueeze(0), dim=2)
            sorted_pin_idx = torch.argsort(pin_to_base_dist, dim=0, descending=True)
            
            self.unmounted_pin_positions_w = [
                torch.stack([pins_batch[sorted_pin_idx[i, e], e] for e in range(num_envs)], dim=0).unsqueeze(1) 
                for i in range(sorted_pin_idx.shape[0])
            ]
            
            gears_batch = torch.stack(self.unmounted_sun_planetary_gear_positions_w, dim=0).squeeze(2)
            target_pin_pos_w = self.unmounted_pin_positions_w[0].squeeze(1) # [num_envs, 3]
            
            # 실제 거리 계산
            gear_to_pin_dist = torch.norm(gears_batch[..., :2] - target_pin_pos_w[:, :2].unsqueeze(0), dim=2)
            
            pin_y_sign = target_pin_pos_w[:, 1].unsqueeze(0) # [1, num_envs]
            gear_y_signs = gears_batch[:, :, 1]              # [N, num_envs]
            
            different_side_mask = (pin_y_sign * gear_y_signs) < 0
            
            penalty = different_side_mask.float() * 100.0
            adjusted_dist = gear_to_pin_dist + penalty

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
            vertical_error = planetary_reducer_pos_w[:, 2] - sun_planetary_gear_pos[:, 2]
            orientation_error = torch.acos(torch.clamp((sun_planetary_gear_quat * planetary_reducer_quat_w).sum(dim=-1), -1.0, 1.0))

            th = self.mounting_thresholds["planetary_reducer"]
            if (horizontal_error < th["horizontal"] and
                vertical_error < th["vertical"] and
                orientation_error < th["orientation"]):
                self.is_planetary_reducer_mounted = True

        if self.unmounted_sun_planetary_gear_positions_w:
            self.target_sun_planetary_gear_pos_w  = self.unmounted_sun_planetary_gear_positions_w[0].view(num_envs, -1).clone()
            self.target_sun_planetary_gear_quat_w = self.unmounted_sun_planetary_gear_quats_w[0].view(num_envs, -1).clone()
        if self.unmounted_pin_positions_w:
            self.target_pin_pos_w = self.unmounted_pin_positions_w[0].view(num_envs, -1).clone()

    def pick_and_place(self, target_object_name):
        self.observe_robot_state()
        self.observe_assembly_state()

        self.dual_arm_pick_and_place_fsm.update_observation(
            left_ee_pos_w                    = self.left_ee_pos_w,
            left_ee_quat_w                   = self.left_ee_quat_w,
            right_ee_pos_w                   = self.right_ee_pos_w,
            right_ee_quat_w                  = self.right_ee_quat_w,
            base_pos_w                       = self.base_pos_w,
            base_quat_w                      = self.base_quat_w,        
            planetary_carrier_pos_w          = self.planetary_carrier_pos_w,
            planetary_carrier_quat_w         = self.planetary_carrier_quat_w,
            ring_gear_pos_w                  = self.ring_gear_pos_w,
            ring_gear_quat_w                 = self.ring_gear_quat_w,
            planetary_reducer_pos_w          = self.planetary_reducer_pos_w,
            planetary_reducer_quat_w         = self.planetary_reducer_quat_w,

            target_sun_planetary_gear_pos_w  = self.target_sun_planetary_gear_pos_w,
            target_sun_planetary_gear_quat_w = self.target_sun_planetary_gear_quat_w,
            target_pin_pos_w                 = self.target_pin_pos_w   
        )

        self.dual_arm_pick_and_place_fsm.set_target_object(target_object_name=target_object_name)
        self.joint_pos_command, self.joint_pos_command_ids = self.dual_arm_pick_and_place_fsm.step()
        self.state = self.dual_arm_pick_and_place_fsm.state

        self.log()

    def log(self):
        print(f"[Low Level FSM State] {self.dual_arm_pick_and_place_fsm.state}")
        print("| Aseembly State                       |")
        print("----------------------------------------")
        print(f"| # of mounted planetary gears | {self.num_mounted_planetary_gears:<5} |")
        print(f"| sun gear                     | {str(self.is_sun_gear_mounted):<5} |")
        print(f"| ring gear                    | {str(self.is_ring_gear_mounted):<5} |")
        print(f"| planetary reducer            | {str(self.is_planetary_reducer_mounted):<5} |")
        print("----------------------------------------")

        score = self.num_mounted_planetary_gears + int(self.is_sun_gear_mounted) + int(self.is_ring_gear_mounted) + int(self.is_planetary_reducer_mounted)
        print(f"score: {score}")
