import torch
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg

def create_dual_arm_control_config(scene, device):
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",              # position and orientation
        use_relative_mode=False,          # global coordinate (True: relative coordinate)
        ik_method="dls"                   # damped least squares: inverse kinematics standard solver in assembly task 
    )
    left_diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=scene.num_envs,     # number of parallel environment in Isaac Sim, vectorizated simulation
        device=device                # "cuda": gpu / "cpu": cpu
    )
    right_diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=scene.num_envs,        
        device=device                    
    )

    # Robot parameter
    left_arm_entity_cfg = SceneEntityCfg(
        "robot",                          # robot entity name
        joint_names=["left_arm_joint.*"], # joint entity set
        body_names=["left_arm_link6"]     # body entity set (ee)
    )
    right_arm_entity_cfg = SceneEntityCfg(
        "robot",                         
        joint_names=["right_arm_joint.*"],
        body_names=["right_arm_link6"]    
    )
    left_gripper_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["left_gripper_axis.*"]
    )
    right_gripper_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["right_gripper_axis.*"]
    )
    left_arm_entity_cfg.resolve(scene)
    left_gripper_entity_cfg.resolve(scene)
    right_arm_entity_cfg.resolve(scene)
    right_gripper_entity_cfg.resolve(scene)

    return left_diff_ik_controller, right_diff_ik_controller, left_arm_entity_cfg, right_arm_entity_cfg, left_gripper_entity_cfg, right_gripper_entity_cfg

def solve_inverse_kinematics(
        target_ee_pos_b,
        target_ee_quat_b,
        arm_joint_ids,
        arm_joint_pos,
        arm_jacobian,
        diff_ik_controller,
        ee_pos_b,
        ee_quat_b,
    ):
    """
    target_ee_pose, current_joint_pose -> desired_joint_pose
    """
    # Get the target position and orientation of the arm
    ik_commands = torch.cat(
        [target_ee_pos_b, target_ee_quat_b], 
        dim=-1
    )
    diff_ik_controller.set_command(ik_commands)

    desired_arm_joint_pos = diff_ik_controller.compute(
        ee_pos_b, 
        ee_quat_b,
        arm_jacobian, 
        arm_joint_pos
    )
    desired_arm_joint_ids = arm_joint_ids

    return desired_arm_joint_pos, desired_arm_joint_ids