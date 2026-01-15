from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

from Galaxea_Lab_External.robots import (
    GALAXEA_R1_CHALLENGE_CFG,
    GALAXEA_HEAD_CAMERA_CFG,
    GALAXEA_HAND_CAMERA_CFG,
    TABLE_CFG,
    RING_GEAR_CFG,
    SUN_PLANETARY_GEAR_CFG,
    PLANETARY_CARRIER_CFG,
    PLANETARY_REDUCER_CFG,
)

@configclass
class GearboxAssemblyBaseEnvCfg(DirectRLEnvCfg):
    # Record data
    record_data = True
    record_freq = 5

    # env
    sim_dt = 0.01
    decimation = 2
    episode_length_s = 60.0

    # spaces definition
    action_space = 14
    observation_space = 16
    state_space = 0
    num_rerenders_on_reset = 5

    obs_order: list = [
        "fingertip_pos_rel_fixed", 
        "fingertip_quat", 
        "ee_linvel", 
        "ee_angvel"
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=2**26,
            gpu_total_aggregate_pairs_capacity=2**26,
            gpu_max_rigid_contact_count=2**23,
            gpu_found_lost_pairs_capacity=2**22,
            gpu_heap_capacity=2**26,
            gpu_temp_buffer_capacity=2**24,
            gpu_max_num_partitions=8,
            gpu_max_soft_body_contacts=2**20,
            gpu_max_particle_contacts=2**20,
            gpu_collision_stack_size=768 * 1024 * 1024,  # 768 MB (increased from 512 MB)
            gpu_max_rigid_patch_count = 262144 * 2
        ),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = GALAXEA_R1_CHALLENGE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # table_cfg: AssetBaseCfg = TABLE_CFG.copy()
    table_cfg: RigidObjectCfg = TABLE_CFG.replace(prim_path="/World/envs/env_.*/Table")

    ring_gear_cfg: RigidObjectCfg = RING_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/ring_gear",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    sun_planetary_gear_1_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.2, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    sun_planetary_gear_2_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.25, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    sun_planetary_gear_3_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.15, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    sun_planetary_gear_4_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, -0.3, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    planetary_carrier_cfg: RigidObjectCfg = PLANETARY_CARRIER_CFG.replace(
        prim_path="/World/envs/env_.*/planetary_carrier",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.25, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    planetary_reducer_cfg: RigidObjectCfg = PLANETARY_REDUCER_CFG.replace(
        prim_path="/World/envs/env_.*/planetary_reducer",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.1, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    # Camera
    head_camera_cfg: CameraCfg = GALAXEA_HEAD_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam"
    )
    left_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam"
    )
    right_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/right_realsense_link/right_hand_cam/right_hand_cam"
    )

    # Physics
    table_friction_coefficient = 0.4
    gears_friction_coefficient = 0.01
    gripper_friction_coefficient = 2.0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # controllable joint
    left_arm_joint_dof_name = "left_arm_joint.*"
    right_arm_joint_dof_name = "right_arm_joint.*"
    # left_gripper_dof_name = "left_gripper_axis1"
    # right_gripper_dof_name = "right_gripper_axis1"
    left_gripper_dof_name = "left_gripper_axis.*"
    right_gripper_dof_name = "right_gripper_axis.*"

    torso_joint_dof_name = "torso_joint[1-3]" # Since in current task, torso_joint4 will always be fixed at 0.0
    torso_joint1_dof_name = "torso_joint1"
    torso_joint2_dof_name = "torso_joint2"
    torso_joint3_dof_name = "torso_joint3"
    torso_joint4_dof_name = "torso_joint4"

    # Robot initial torso joint position
    initial_torso_joint1_pos = 0.5
    initial_torso_joint2_pos = -0.8
    initial_torso_joint3_pos = 0.5

    x_offset = 0.2

    # ===== IK settings =====
    ik_method: str = "dls"  # Damped Least Squares
    
    # ===== Action thresholds =====
    pos_action_threshold = (0.05, 0.05, 0.05)  # max position delta per step
    rot_action_threshold = (0.1, 0.1, 0.1)     # max rotation delta per step (rad)