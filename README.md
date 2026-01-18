<div align="center">

  # Gearbox Assembly by Galaxea R1
  
  Robotic Collaborative Assembling Challenge (RoCo Challenge) - HMI Workshop @ AAAI 2026 

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

4. Run the Policy

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