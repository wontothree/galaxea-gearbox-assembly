    gearboxAssembly
    ├── scripts                                
    │   ├── basic         
    │	│   ├── assembly_scene_teleop_demo.py  # (*)
    │	│   └── create_scene_rule_based.py     # (*)
    │	│   
    │   ├── rl_games
    │	│   ├── play.py                        # (*)
    │	│   └── train.py                       # (*)
    │	│   
    │   ├── list_envs.py                       # file to print envs
    │   ├── random_agent.py                    # file to test
    │   ├── rule_based_agent.py                # file to test
    │   └── zero_agent.py                      # file to test 
    │
    └── source/Galaxea_Lab_External            # env
        ├── Galaxea_Lab_External
        │   ├── robots
        │	│   ├── galaxea_assets.py
        │	│   ├── galaxea_robots.py 		
        │	│   ├── galaxea_rule_policy.py 			
        │	│   └── gears_assets.py 		
        │	│
        │   └── tasks
        │		    └── direct
        │				└── galaxea_lab_external
        │				    ├── agents
        │					│   └── rl_games_ppo_cfg.yaml
        │                   │
        │				    ├── galaxea_lab_external_env.py      # (*)
        │				    └── galaxea_lab_external_env_cfg.py  # (*)
        │   
        ├── assets
        │   ├── play.py
        │   └── train.py
        │   
        ├── config
        │   ├── play.py
        │   └── train.py
        │
        ├── docs
        │   └── CHANGELOG.rst
        │
        ├── pyproject.toml
        └── setup.py       

# Action (16-DOF)

$$
\boldsymbol{u}_t = 
\begin{bmatrix}
q_{t}^{l, \text{arm}, 1} \\
q_{t}^{r, \text{arm}, 1} \\
q_{t}^{l, \text{arm}, 2} \\
q_{t}^{r, \text{arm}, 2} \\
q_{t}^{l, \text{arm}, 3} \\
q_{t}^{r, \text{arm}, 3} \\
q_{t}^{l, \text{arm}, 4} \\
q_{t}^{r, \text{arm}, 4} \\
q_{t}^{l, \text{arm}, 5} \\
q_{t}^{r, \text{arm}, 5} \\
q_{t}^{l, \text{arm}, 6} \\
q_{t}^{r, \text{arm}, 6} \\
q_{t}^{l, \text{grip}, 1} \\
q_{t}^{l, \text{grip}, 2} \\
q_{t}^{r, \text{grip}, 1} \\
q_{t}^{r, \text{grip}, 2} \\
\end{bmatrix}
$$

|Role|Left Arm|Right Arm|
|---|---|---|
|Base Rotation|3|4|
|Shoulder Pitch|5|6|
|Elbow Pitch|7|8|
|Wrist|9|10|
|Wrist|11|12|
|Wrist|13|14|
|Gripper|15, 16|17, 18|

# Classes

|Class||Function|
|---|---|---|
|`Galaxear1GearboxAssemblyAgent`||initialize_arm_controller|
|||observe_robot_state|
|||observe_object_state|
|||observe_assembly_progress|
|||initialize_arm_controller|
|||solve_inverse_kinematics|
|||pick_and_place|
|`State`|abstract class for states|
|`StateMachine`|managing states|
|`Context`||
|`InitializationState`||
|`PlanetaryGearInsertionState`||
|`SunGearInsertionState`||
|`RingGearInsertionState`||
|`PlanetaryReducerInsertionState`||
|`FinalizationState`||

FSM → Context → Robot

StateMachine / State에서는 solve_inverse_kinematics를 직접 호출하지 않고, context.robot.solve_inverse_kinematics(...) 형태로 호출한다.

```
context.robot.solve_inverse_kinematics(...)
```

FSM은 행동을 지시하기만 하고 Context는 의존성 주입만 한다.

```py
robot_agent = Galaxear1GearboxAssemblyAgent(
    sim=sim,
    scene=scene,
    obj_dict=obj_dict
)
self.context = Context(sim, robot_agent)
initial_state = InitializationState()
fsm = StateMachine(initial_state, self.context)
self.context.fsm = fsm
```