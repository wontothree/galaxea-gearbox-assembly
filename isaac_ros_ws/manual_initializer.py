import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ManualMaskInitializer(Node):
    def __init__(self):
        super().__init__('manual_mask_initializer')
        self.bridge = CvBridge()
        
        # Subscribe to the camera feed to pick a frame
        self.sub = self.create_subscription(Image, 'rgb/image_rect_color', self.image_callback, 10)
        
        # Publish to the segmentation topic defined in the C++ node
        self.pub = self.create_publisher(Image, 'segmentation', 10)
        
        self.done = False
        self.get_logger().info("Manual Mask Initializer Node Started.")

    def image_callback(self, msg):
        if self.done:
            return

        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape

        self.get_logger().info("Image received. Draw a box around the GEAR and press ENTER.")
        
        # 1. Select ROI
        roi = cv2.selectROI("Manual Mask Selection", cv_image)
        cv2.destroyWindow("Manual Mask Selection")

        x, y, w, h = [int(v) for v in roi]

        if w > 0 and h > 0:
            # 2. Create the Mono8 Mask (Black background)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 3. Fill the ROI with White (255)
            # FoundationPose uses this mask to find the object's centroid and boundaries
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # 4. Convert and Publish
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            mask_msg.header = msg.header # Keep timestamps synced!
            
            self.pub.publish(mask_msg)
            self.get_logger().info(f"Published mask for gear at [{x}, {y}, {w}, {h}]")
            self.done = True
        else:
            self.get_logger().warn("Selection cancelled or invalid.")

def main():
    rclpy.init()
    node = ManualMaskInitializer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()