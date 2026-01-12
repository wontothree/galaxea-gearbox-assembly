<div align="center">

  # Gearbox Assembly by Galaxea R1
  
  We participated in Robotic Collaborative Assembling Challenge (RoCo Challenge) - HMI Workshop @ AAAI 2026 

  [![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
  [![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/index.html)
  [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)

</div>

![RoCo Challenge Poster](docs/images/poster.png)
[RoCo Challenge@AAAI 2026](https://rocochallenge.github.io/RoCo2026/doc.html)

--- 

## 🚀 Getting Started on Docker

0. Install [Docker](https://docs.docker.com/desktop/setup/install/linux/), [Docker Compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository), and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. Git clone

```bash
git lfs install
git lfs pull

git clone https://github.com/wontothree/galaxea-gearbox-assembly.git
```

2. Create the Docker container and run the container

```bash
cd galaxea-gearbox-assembly
./docker/container.py start base
./docker/container.py enter base
```

3. Install dependencies

```bash
cd /workspace/isaaclab/source/gearboxAssembly
python -m pip install -e source/Galaxea_Lab_External
```

4. Run the Policy in the IsaacLab

```bash
cd /workspace/isaaclab/source/gearboxAssembly
python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras --device cpu
```

## Getting Started

### [Docker 1] Isaac Lab (Ubuntu 24.04, ROS2 Jazzy): Simulation Environment and Agent

1. Create the container or run the container

```bash
cd galaxe1-gearbox-assembly
./docker/container.py start ros2 # --suffix [container_name]
# or
./docker/container.py enter ros2 # --suffix [container_name]
```

Check out the ROS

```bash
ros2 topic list
```

2. Install dependencies

```bash
cd /galaxea-gearbox-assembly/source/gearboxAssembly
python -m pip install -e source/Galaxea_Lab_External
```

3. Run the simulation

```bash
cd source/gearboxAssembly
python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras --device cpu
```

4. Set `Extensions` up in Isaac Sim

`Window` -> `Extensions`

- OMNIGRAPH ACTION GRAPH EDITOR: `ENABLED` & `AUTOLOAD`
- OMNIGRAPH ACTION GRAPH: `ENABLED` & `AUTOLOAD`
- ROS 2 BRIDGE: `ENABLED` & `AUTOLOAD`

5. Connect the Action Graph

`Window` -> `Graph Editors` -> `Action Graph`

Node `On Playback Trick` -> Node `Isaac Create Render Product` -> Node `ROS2 Camera Helper
`
![Action Graph](docs/images/action-graph.png)

6. Publish the topics

- Click the node `Isaac Create Render Product`
- `Inputs` > `cameraPrim` > click `/World/envs/env_0/Robot/zed_link/head_cam/head_cam` in the drag drop
- Play

7. Check the topic list out

In the other terminal,

```bash
cd galaxea-gearbox-assembly
./docker/container.py enter ros2 

source /opt/ros/jazzy/setup.bash
ros2 topic list
```

### [Docker 2] Isaac ROS (Ubuntu 24.04, ROS2 Jazzy): Foundation Pose

1. Build the Docker image of Isaac ROS and run the container

```bash
xhost +local:docker
cd galaxea-gearbox-assembly/isaac_ros_docker
docker build -t isaac_ros .
./run.sh
```

2. Open the additional window on the same container


```bash
./run.sh
```

3. Install the dependencies

```bash
export ISAAC_ROS_WS=/workspace/isaac_ros_ws
echo $ISAAC_ROS_WS
sudo apt-get update
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_pose_estimation/isaac_ros_foundationpose --ignore-src -y

rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection/isaac_ros_rtdetr --ignore-src -y
```

3. Build the packages

```bash
cd ${ISAAC_ROS_WS}/ && \
   colcon build --symlink-install --packages-up-to isaac_ros_foundationpose --base-paths ${ISAAC_ROS_WS}/src/isaac_ros_pose_estimation/isaac_ros_foundationpose

cd ${ISAAC_ROS_WS} && \
   colcon build --symlink-install --packages-up-to isaac_ros_rtdetr --base-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection/isaac_ros_rtdetr
```

```bash
source ${ISAAC_ROS_WS}/install/setup.bash
```

## Train

```bash
python scripts/rl_games/train.py --task=Galaxea-Planetary-Gear-Assembly-v0 --enable_camera --num_envs=1 --device cpu
```

```bash
python scripts/rule_based_agent.py --task=Galaxea-Planetary-Gear-Assembly-v0 --enable_camera --device cpu
python scripts/rl_games/train.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_camera --num_envs=1
```

# Task

- Task 1

```bash
python scripts/jensen_lovers_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras --num_envs=1 --device cpu
```

- Task 2

```bash
python scripts/jensen_lovers_agent.py --task=Gearbox-Partial-Lackfourth --enable_cameras --num_envs=1 --device cpu
```

- Task 3

```bash
python scripts/jensen_lovers_agent.py --task=Gearbox-Recovery-Misplacedfourth --enable_cameras --num_envs=1 --device cpu
```

This project is tested in the environment of Docker and Window 11.

---