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
