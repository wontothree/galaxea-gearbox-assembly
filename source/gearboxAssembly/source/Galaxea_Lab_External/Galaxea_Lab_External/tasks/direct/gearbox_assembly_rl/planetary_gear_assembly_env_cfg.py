from .gearbox_assembly_base_env_cfg import GearboxAssemblyBaseEnvCfg

class PlanetaryGearAssemblyEnvCfg(GearboxAssemblyBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
