import gymnasium as gym
from . import agents

gym.register(
    id="Galaxea-Planetary-Gear",
    entry_point=f"{__name__}.planetary_gear_env:PlanetaryGearEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.planetary_gear_env_cfg:PlanetaryGearEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)