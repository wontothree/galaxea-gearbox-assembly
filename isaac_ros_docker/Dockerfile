FROM nvcr.io/nvidia/isaac/ros:noble-ros2_jazzy_d3e84470d576702a380478a513fb3fc6-amd64

ENV ROS_DOMAIN_ID=0

RUN apt-get update && apt-get install -y sudo git python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc && \
    echo "source /workspace/isaac_ros_ws/install/setup.bash 2>/dev/null || true" >> /etc/bash.bashrc

WORKDIR /workspace/isaac_ros_ws
RUN chmod 777 /workspace/isaac_ros_ws