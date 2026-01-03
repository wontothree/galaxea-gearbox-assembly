import gymnasium as gym
from . import agents

gym.register(
    id="Galaxea-Planetary-Gear-Assembly-v0",
    entry_point=f"{__name__}.planetary_gear_assembly_env:PlanetaryGearAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.planetary_gear_assembly_env_cfg:PlanetaryGearAssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Galaxea-Sun-Gear-Assembly-v0",
    entry_point=f"{__name__}.sun_gear_assembly_env:SunGearAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sun_gear_assembly_env_cfg:SunGearAssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Galaxea-Ring-Gear-Assembly-v0",
    entry_point=f"{__name__}.ring_gear_assembly_env:RingGearAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ring_gear_assembly_env_cfg:RingGearAssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Galaxea-Planetary-Reducer-Assembly-v0",
    entry_point=f"{__name__}.planetary_reducer_assembly_env:PlanetaryReducerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.planetary_reducer_assembly_env_cfg:PlanetaryReducerAssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)