import os
import cv2
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from cv_bridge import CvBridge

# Global variable to store current depth frame for the mouse callback
current_depth_map = None

def on_mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if current_depth_map is not None:
            # Get the depth value at the mouse coordinates
            # Note: OpenCV uses (y, x) for array indexing
            depth_value = current_depth_map[y, x]
            
            # Create a copy of the display image to draw text on
            display_copy = param.copy()
            text = f"X: {x}, Y: {y} | Depth: {depth_value}"
            
            # Draw text backdrop for readability
            cv2.putText(display_copy, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(display_copy, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow("Depth Inspector", display_copy)

def inspect_depth_data(bag_path, topic_name='/depth_image'):
    global current_depth_map
    
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    bridge = CvBridge()
    
    cv2.namedWindow("Depth Inspector")
    print(f"Inspecting topic: {topic_name}")
    print("CONTROLS: Press ANY KEY for next frame | Press 'ESC' to quit.")

    while reader.has_next():
        (topic, data, t_nanoseconds) = reader.read_next()

        if topic == topic_name:
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            
            # Convert ROS Image to OpenCV (using passthrough to keep 16-bit or 32-bit float values)
            cv_depth = bridge.imgmsg_to_cv2(msg, "passthrough")
            current_depth_map = cv_depth

            # Prepare a visual representation (Colormap)
            # 1. Normalize to 0-255 for visualization
            depth_adj = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_adj = np.uint8(depth_adj)
            # 2. Apply a colormap so it's not just a dark gray image
            color_depth = cv2.applyColorMap(depth_adj, cv2.COLORMAP_JET)

            # Set the mouse callback for this specific frame
            cv2.setMouseCallback("Depth Inspector", on_mouse_move, param=color_depth)
            
            # Show the frame and wait for user input
            cv2.imshow("Depth Inspector", color_depth)
            
            key = cv2.waitKey(0)
            if key == 27: # ESC key to exit
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', help="Path to the ROS 2 bag folder")
    parser.add_argument('--topic', default='/depth_image', help="Depth topic name")
    args = parser.parse_args()
    
    inspect_depth_data(args.bag_path, args.topic)