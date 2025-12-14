<div align="center">

  # Gearbox Assembly by Galaxea R1
  
  We, Jensen Lovers, participated in Robotic Collaborative Assembling Challenge (RoCo Challenge) - HMI Workshop @ AAAI 2026 

  [![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
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

git clone https://github.com/wontothree/galaxear1-gearbox-assembly.git
```

2. Create the Docker container

```bash
cd galaxear1-gearbox-assembly
./docker/container.py start
```

3. Run the container

```bash
cd galaxear1-gearbox-assembly
./docker/container.py enter
```

4. Install dependencies

```bash
cd galaxear1-gearbox-assembly/source/gearboxAssembly
python -m pip install -e source/Galaxea_Lab_External
```

5. Run the Policy in the IsaacLab

```bash
cd source/gearboxAssembly
python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras
python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras --device cpu
```

This project is tested in the environment of Docker and Window 11.

--- 