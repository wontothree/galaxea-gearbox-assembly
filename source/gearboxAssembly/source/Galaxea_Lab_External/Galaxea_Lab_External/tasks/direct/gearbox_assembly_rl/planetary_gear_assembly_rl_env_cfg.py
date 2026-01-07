from .gearbox_assembly_base_env_cfg import GearboxAssemblyBaseEnvCfg

class PlanetaryGearAssemblyRLEnvCfg(GearboxAssemblyBaseEnvCfg):
    # Disable cameras
    head_camera_cfg = None
    left_hand_camera_cfg = None
    right_hand_camera_cfg = None
    
    def __post_init__(self):
        # Override values from GearboxAssemblyBaseEnvCfg
        self.sim_dt = 0.01
        self.decimation = 2
        self.episode_length_s = 60.0
        
        # spaces definition (6DOF robot arm)
        # action: 3 position + 1 yaw rotation + 1 gripper = 5
        # observation: 6 arm joints + 2 gripper + 5 prev_actions = 13
        self.action_space = 5
        self.observation_space = 13

        
        # Call parent's __post_init__
        super().__post_init__()
