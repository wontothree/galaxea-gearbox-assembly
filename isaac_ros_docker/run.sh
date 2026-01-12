#!/bin/bash

CONTAINER_NAME="isaac_ros"
WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/isaac_ros_ws"

# 1. Check if the container is already running
if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container is already running. Joining session..."

    docker exec -it $CONTAINER_NAME /bin/bash

# 2. Check if the container exists but is stopped
elif [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container exists but is stopped. Starting and joining..."
    docker start $CONTAINER_NAME
    docker exec -it $CONTAINER_NAME /bin/bash

# 3. If it doesn't exist, run the full command
else
    echo "Container not found. Launching new instance..."
    # Note: I removed --rm so the container persists after you close it
    docker run -dt \
        --privileged \
        --network host \
        --gpus all \
        -u root \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/.Xauthority:/root/.Xauthority:rw \
        -e XAUTHORITY=/root/.Xauthority \
        -v $WS_PATH:/workspace/isaac_ros_ws \
        -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} \
        --name $CONTAINER_NAME \
        isaac_ros /bin/bash
        
    docker exec -it $CONTAINER_NAME /bin/bash
fi