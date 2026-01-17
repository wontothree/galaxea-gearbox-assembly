import os
import cv2
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from cv_bridge import CvBridge

def extract_synchronized_data(bag_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize ROS 2 bag reader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    # Map topics to types
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    bridge = CvBridge()
    
    # State variables
    last_extraction_time = -1.0
    interval = 1.0 
    frame_index = 0
    info_saved = False
    
    # Buffers to hold messages for the current "slot"
    current_slot_rgb = None
    current_slot_depth = None

    print(f"Reading bag: {bag_path}...")

    while reader.has_next():
        (topic, data, t_nanoseconds) = reader.read_next()
        t_seconds = t_nanoseconds / 1e9

        # 1. Save Camera Info once
        if topic == '/rgb/camera_info' and not info_saved:
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            with open(os.path.join(output_dir, 'camera_info.txt'), 'w') as f:
                f.write(str(msg))
            info_saved = True

        # 2. Check if we are in a new 1-second window
        if last_extraction_time == -1.0 or (t_seconds - last_extraction_time) >= interval:
            
            # Logic: We capture the FIRST rgb and depth we see after the interval
            if topic == '/rgb/image_rect_color' and current_slot_rgb is None:
                msg_type = get_message(topic_types[topic])
                current_slot_rgb = deserialize_message(data, msg_type)

            if topic == '/depth_image' and current_slot_depth is None:
                msg_type = get_message(topic_types[topic])
                current_slot_depth = deserialize_message(data, msg_type)

            # 3. Once we have both for this timestamp, save them
            if current_slot_rgb is not None and current_slot_depth is not None:
                # Process RGB
                cv_rgb = bridge.imgmsg_to_cv2(current_slot_rgb, "bgr8")
                rgb_filename = f"{frame_index}_rgb.jpg"
                cv2.imwrite(os.path.join(output_dir, rgb_filename), cv_rgb)

                # Process Depth
                cv_depth = bridge.imgmsg_to_cv2(current_slot_depth, "passthrough")
                depth_filename = f"{frame_index}_depth.png" # PNG is better for depth to avoid compression artifacts
                cv2.imwrite(os.path.join(output_dir, depth_filename), cv_depth)

                print(f"Saved frame {frame_index} at bag time {t_seconds:.2f}s")
                
                # Reset for next interval
                frame_index += 1
                last_extraction_time = t_seconds
                current_slot_rgb = None
                current_slot_depth = None

    print(f"\nFinished! Extracted {frame_index} image pairs to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', help="Path to the ROS 2 bag folder")
    parser.add_argument('output_dir', help="Directory to save images and text file")
    args = parser.parse_args()
    
    extract_synchronized_data(args.bag_path, args.output_dir)