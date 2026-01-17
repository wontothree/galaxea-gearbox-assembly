import torch
import isaacsim.core.utils.torch as torch_utils


def squashing_fn(error: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Squashing function for multi-scale keypoint rewards.
    
    Returns reward that decreases as distance error increases.
    Different (alpha, beta) parameters create different reward scales:
    - Low alpha, high beta: gentle slope for encouraging approach from far
    - High alpha, low beta: steep slope for fine alignment
    
    Args:
        error: Distance tensor
        alpha: Steepness parameter
        beta: Offset parameter
        
    Returns:
        Reward tensor in range (0, 1/(2+beta)]
    """
    return 1.0 / (torch.exp(alpha * error) + beta + torch.exp(-alpha * error))


def collapse_obs_dict(obs_dict, obs_order) -> torch.Tensor:
    """Stack observations in given order."""
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors


def get_keypoint_offsets(num_keypoints: int, device: torch.device) -> torch.Tensor:
    """Get keypoint offsets for reward computation.
    
    Creates offsets arranged in a pattern around the object center.
    
    Args:
        num_keypoints: Number of keypoints (1, 4, or 8)
        device: Torch device
        
    Returns:
        Keypoint offsets tensor (num_keypoints, 3)
    """
    if num_keypoints == 1:
        return torch.zeros((1, 3), device=device)
    elif num_keypoints == 4:
        # 4 corners pattern
        offsets = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], device=device)
        return offsets
    elif num_keypoints == 8:
        # 8 corners of a cube
        offsets = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ], device=device)
        return offsets
    else:
        return torch.zeros((num_keypoints, 3), device=device)


def create_downward_quat(batch_size: int, device: torch.device, yaw: torch.Tensor = None) -> torch.Tensor:
    """Create quaternion for downward-facing EEF orientation.
    
    Args:
        batch_size: Number of quaternions to create
        device: Torch device
        yaw: Optional yaw angles (batch_size,). If None, yaw=0 for all.
        
    Returns:
        Quaternion tensor (batch_size, 4) with roll=π, pitch=0, yaw=specified
    """
    roll = torch.full((batch_size,), 3.14159, device=device)
    pitch = torch.zeros(batch_size, device=device)
    if yaw is None:
        yaw = torch.zeros(batch_size, device=device)
    return torch_utils.quat_from_euler_xyz(roll=roll, pitch=pitch, yaw=yaw)


def constrain_quat_to_downward(quat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Constrain quaternion to downward-facing orientation, preserving only yaw.
    
    Args:
        quat: Input quaternion (batch_size, 4)
        device: Torch device
        
    Returns:
        Constrained quaternion (batch_size, 4) with roll=π, pitch=0, yaw preserved
    """
    # Extract euler angles and keep only yaw
    euler_xyz = torch.stack(torch_utils.get_euler_xyz(quat), dim=1)
    return create_downward_quat(quat.shape[0], device, yaw=euler_xyz[:, 2])


def set_gripper_open(robot, ctrl_target_joint_pos, left_gripper_dof_idx, env_ids):
    """Set gripper to open position.
    
    Args:
        robot: Robot articulation object
        ctrl_target_joint_pos: Joint position control target tensor
        left_gripper_dof_idx: Gripper DOF indices
        env_ids: Environment indices to update
    """
    gripper_open_pos = 0.04  # Maximum open position
    ctrl_target_joint_pos[env_ids, left_gripper_dof_idx[0]] = gripper_open_pos
    ctrl_target_joint_pos[env_ids, left_gripper_dof_idx[1]] = gripper_open_pos
    robot.set_joint_position_target(
        ctrl_target_joint_pos[env_ids][:, left_gripper_dof_idx],
        left_gripper_dof_idx, env_ids
    )


def set_gripper_close(robot, ctrl_target_joint_pos, left_gripper_dof_idx, env_ids):
    """Set gripper to closed position.
    
    Args:
        robot: Robot articulation object
        ctrl_target_joint_pos: Joint position control target tensor
        left_gripper_dof_idx: Gripper DOF indices
        env_ids: Environment indices to update
    """
    gripper_close_pos = 0.0  # Closed position
    ctrl_target_joint_pos[env_ids, left_gripper_dof_idx[0]] = gripper_close_pos
    ctrl_target_joint_pos[env_ids, left_gripper_dof_idx[1]] = gripper_close_pos
    robot.set_joint_position_target(
        ctrl_target_joint_pos[env_ids][:, left_gripper_dof_idx],
        left_gripper_dof_idx, env_ids
    )


def initialize_grasp(env, env_ids, held_asset, grasp_height_offset=0.15, held_offset_z=0.07, grasp_duration=0.25):
    """Initialize gripper to grasp held asset (Factory-style).
    
    Steps:
    1. Disable gravity
    2. Move gripper above held asset using IK
    3. Move held asset to gripper position
    4. Close gripper
    5. Enable gravity
    
    Args:
        env: Environment object containing robot, scene, etc.
        env_ids: Environment indices to initialize
        held_asset: Asset to grasp (RigidObject)
        grasp_height_offset: Height offset above asset for initial EE positioning
        held_offset_z: Z offset for placing asset relative to gripper
        grasp_duration: Time duration for gripper closing
    """
    import carb
    import isaaclab.sim as sim_utils
    
    # 1. Disable gravity
    physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
    
    # 2. Get held asset position
    held_pos_w = held_asset.data.root_pos_w.clone()  # World position
    held_quat_w = held_asset.data.root_quat_w.clone()  # World quaternion
    
    # Calculate target EE position above the gear
    target_ee_pos_w = held_pos_w.clone()
    target_ee_pos_w[:, 2] += grasp_height_offset
    
    # Target orientation: downward-facing
    target_ee_quat = create_downward_quat(env.num_envs, env.device)
    
    # Move gripper to target position using IK
    move_gripper_to_pose(env, env_ids, target_ee_pos_w, target_ee_quat)
    
    # 3. Move held asset to gripper position
    # Get current fingertip position after IK movement
    env._compute_ee_state_for_ik()
    fingertip_pos_w = env.left_ee_pos_e + env.scene.env_origins
    fingertip_quat_w = env.left_ee_quat_w
    
    # Calculate held asset position relative to fingertip
    held_offset_local = torch.tensor([0.0, 0.0, held_offset_z], device=env.device).repeat(env.num_envs, 1)
    held_offset_world = torch_utils.quat_rotate(fingertip_quat_w, held_offset_local)
    
    new_held_pos_w = fingertip_pos_w + held_offset_world
    new_held_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    # Write held asset to new position
    held_state = held_asset.data.default_root_state.clone()
    held_state[:, 0:3] = new_held_pos_w
    held_state[:, 3:7] = new_held_quat_w
    held_state[:, 7:] = 0.0  # Zero velocity
    held_asset.write_root_pose_to_sim(held_state[:, 0:7])
    held_asset.write_root_velocity_to_sim(held_state[:, 7:])
    held_asset.reset()
    
    env._step_sim_no_action()
    
    # 4. Close gripper
    grasp_time = 0.0
    while grasp_time < grasp_duration:
        set_gripper_close(env.robot, env.ctrl_target_joint_pos, env._left_gripper_dof_idx, env_ids)
        env._step_sim_no_action()
        grasp_time += env.sim.get_physics_dt()
    
    # 5. Enable gravity
    physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
    
    # Final simulation step
    env._step_sim_no_action()
    
    # Reset action buffers
    env.actions = torch.zeros_like(env.actions)
    env.prev_actions = torch.zeros_like(env.actions)


def move_gripper_to_pose(env, env_ids, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor, max_ik_time: float = 0.5, pos_tolerance: float = 0.005):
    """Move gripper to target pose using iterative IK.
    
    Args:
        env: Environment object containing robot, IK controller, etc.
        env_ids: Environment indices (not currently used but kept for API consistency)
        target_pos_w: Target position in world frame (num_envs, 3)
        target_quat_w: Target quaternion in world frame (num_envs, 4)
        max_ik_time: Maximum time for IK convergence
        pos_tolerance: Position error tolerance (meters)
    """
    from isaaclab.utils.math import subtract_frame_transforms
    
    ik_time = 0.0
    
    while ik_time < max_ik_time:
        env._compute_ee_state_for_ik()
        
        # Convert target from world to body frame for IK
        root_pose_w = env.robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos_w, target_quat_w
        )
        
        # Set IK command
        env.left_ik_commands[:, 0:3] = target_pos_b
        env.left_ik_commands[:, 3:7] = target_quat_b
        env.left_diff_ik_controller.set_command(env.left_ik_commands)
        
        # Get Jacobian and compute IK
        left_jacobian = env.robot.root_physx_view.get_jacobians()[:, env.left_ee_jacobi_idx, :, env._left_arm_joint_idx]
        joint_pos_des = env.left_diff_ik_controller.compute(
            env.left_ee_pos_b, env.left_ee_quat_b, left_jacobian,
            env.robot.data.joint_pos[:, env._left_arm_joint_idx]
        )
        
        # Update joint targets
        env.ctrl_target_joint_pos[:, env._left_arm_joint_idx] = joint_pos_des
        env.robot.set_joint_position_target(env.ctrl_target_joint_pos)
        
        env._step_sim_no_action()
        ik_time += env.physics_dt
        
        # Check convergence
        current_pos_w = env.left_ee_pos_e + env.scene.env_origins
        pos_error = torch.norm(current_pos_w - target_pos_w, dim=1).max()
        if pos_error < pos_tolerance:
            break