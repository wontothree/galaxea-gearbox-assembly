from .gearbox_assembly_base_env_cfg import GearboxAssemblyBaseEnvCfg, OBS_DIM_CFG, STATE_DIM_CFG

class PlanetaryGearAssemblyEnvCfg(GearboxAssemblyBaseEnvCfg):
    # Disable cameras
    head_camera_cfg = None
    left_hand_camera_cfg = None
    right_hand_camera_cfg = None
    
    # Reward configuration (Factory-style)
    # Keypoint coefficients for squashing function: [a, b]
    # squashing_fn(x, a, b) = 1 / (exp(a*x) + b + exp(-a*x))
    keypoint_coef_baseline: tuple[float, float] = (5.0, 4.0)    # Gentle slope for distant rewards
    keypoint_coef_coarse: tuple[float, float] = (50.0, 2.0)     # Steeper for medium distances
    keypoint_coef_fine: tuple[float, float] = (100.0, 0.0)      # Very steep for fine alignment
    
    # Reward scales
    action_penalty_ee_scale: float = 0.0       # Penalty for large actions (disabled for precise manipulation)
    action_grad_penalty_scale: float = 0.0     # Penalty for action changes (disabled)
    
    # Success/engagement thresholds
    success_threshold: float = 0.01           # 10mm for success (gear on pin)
    engage_threshold: float = 0.02             # 20mm for engagement bonus
    
    # Number of keypoints for reward computation
    num_keypoints: int = 4
    keypoint_scale: float = 0.02               # Scale for keypoint offsets
    
    def __post_init__(self):
        # Override values from GearboxAssemblyBaseEnvCfg
        self.sim_dt = 1/120
        self.decimation = 8
        self.episode_length_s = 10.0
        
        # spaces definition (6DOF robot arm)
        # action: 3 position + 1 yaw rotation = 4 (gripper always closed)
        # observation: ee_pos(3) + ee_pos_rel_pin(3) + ee_quat(4) + ee_linvel(3) + ee_angvel(3) + prev_actions(4) = 20
        # state: observation + left_arm_joint_pos(6) + left_gripper_joint_pos(2) + gear4_pos(3) + gear4_pos_rel_pin(3) + gear4_quat(4) + pin_pos(3) = 41
        self.action_space = 4
        self.observation_space = sum([OBS_DIM_CFG[obs] for obs in self.obs_order])
        self.state_space = sum([STATE_DIM_CFG[state] for state in self.state_order])
        
        # Call parent's __post_init__
        super().__post_init__()
