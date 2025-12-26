#!/bin/bash
# Mounting workspace
WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/isaac_ros_ws"

# execute container as root
docker run -it --rm \
    --privileged \
    --network host \
    --gpus all \
    -u root \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $WS_PATH:/workspace/isaac_ros_ws \
    -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} \
    --name isaac_ros \
    isaac_ros
