# Gearbox Assembly by Galaxea R1

![RoCo Challenge Poster](docs/images/poster.png)
[RoCo Challenge@AAAI 2026](https://rocochallenge.github.io/RoCo2026/doc.html)

## Getting Started on Docker

```bash
cd source/gearboxAssembly
python -m pip install -e source/Galaxea_Lab_External
```

```bash
cd galaxear1-gearbox-assembly
./docker/container.py enter

cd source/gearboxAssembly
python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras
```
