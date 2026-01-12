#!/bin/bash

# 1. Define the container name you are looking for
CONTAINER_NAME="isaac_ros"

# 2. Get the IDs of containers matching the name exactly
# Note the name= prefix and the anchors ^/ and $ for exact matching
CONTAINER_IDS=$(docker ps -q -f "name=^/${CONTAINER_NAME}$")

# 3. Check if any containers were found
if [ -z "$CONTAINER_IDS" ]; then
    echo "No active running containers found with name: '$CONTAINER_NAME'."
else
    echo "Found container(s): $CONTAINER_IDS"
    
    # 4. Kill the containers
    docker kill $CONTAINER_IDS
    
    echo "Successfully sent kill signal to '$CONTAINER_NAME'."
fi