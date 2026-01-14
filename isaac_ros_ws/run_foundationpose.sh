source ${ISAAC_ROS_WS}/install/setup.bash

ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose.launch.py \
    refine_engine_file_path:=/workspace/isaac_ros_ws/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
    score_engine_file_path:=/workspace/isaac_ros_ws/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
    mesh_file_path:=/workspace/isaac_ros_ws/models/planetary_carrier_3x.obj