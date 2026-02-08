ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
		launch_fragments:=rtdetr \
		interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_rtdetr/quickstart_interface_specs.json \
		engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/sewon6.plan
        # model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/sewon6.onnx
#     /camera_info_rect
# /events/read_split
# /image_rect
# /parameter_events
# /rosout