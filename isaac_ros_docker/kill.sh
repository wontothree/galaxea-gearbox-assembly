#!/bin/bash

# 1. Define the image name you are looking for
IMAGE_NAME=isaac_ros

if [ -z "$IMAGE_NAME" ]; then
    echo "Error: Please provide an image name."
    echo "Usage: ./kill_by_image.sh <image_name>"
    exit 1
fi

echo "Searching for containers running image: $IMAGE_NAME..."

# 2. Get the IDs of containers matching the image (ancestor)
CONTAINER_IDS=$(docker ps -q -f "ancestor=$IMAGE_NAME")

# 3. Check if any containers were found
if [ -z "$CONTAINER_IDS" ]; then
    echo "No active containers found for image '$IMAGE_NAME'."
else
    echo "Found containers: $CONTAINER_IDS"
    
    # 4. Kill the containers
    docker kill $CONTAINER_IDS
    
    echo "Successfully sent kill signal to containers."
fi