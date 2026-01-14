#!/bin/bash

CONTAINER_NAME="isaac_ros"
WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/isaac_ros_ws"

# --- X11 AUTHENTICATION SETUP ---
# Create a temporary Xauth file specifically for the container
XAUTH=/tmp/.docker.xauth
touch $XAUTH
chmod 644 $XAUTH

# Extract the cookie for the current DISPLAY, modify it to work for 'any' hostname, 
# and merge it into our temporary file.
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# 1. Check if the container is already running
if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container is already running. Joining session..."
    docker exec -it $CONTAINER_NAME /bin/bash

# 2. Check if the container exists but is stopped
elif [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container exists but is stopped. Starting..."
    docker start $CONTAINER_NAME
    docker exec -it $CONTAINER_NAME /bin/bash

# 3. If it doesn't exist, run the full command
else
    echo "Container not found. Launching new instance..."
    docker run -dt \
        --privileged \
        --network host \
        --gpus all \
        -u root \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $XAUTH:$XAUTH \
        -e XAUTHORITY=$XAUTH \
        -v $WS_PATH:/workspace/isaac_ros_ws \
        --name $CONTAINER_NAME \
        isaac_ros /bin/bash
        
    docker exec -it $CONTAINER_NAME /bin/bash
fi