# Gearbox Assembly by Galaxea R1

![RoCo Challenge Poster](docs/images/poster.png)
[RoCo Challenge@AAAI 2026](https://rocochallenge.github.io/RoCo2026/doc.html)

## Getting Started on Docker

0. Install `Docker`, `Docker Compose`, `NVIDIA Container Toolkit`

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
```

This project is tested in the environment of Docker and Window 11.