#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions andd=
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This script listens for images and object detections on the image,
# then renders the output boxes on top of the image and publishes
# the result as an image message

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class RtDetrVisualizer(Node):
    QUEUE_SIZE = 10
    color = (0, 255, 0)
    bbox_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_type = 2

    def __init__(self):
        super().__init__('rtdetr_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._processed_image_pub = self.create_publisher(
            Image, 'rtdetr_processed_image',  self.QUEUE_SIZE)

        self._detections_subscription = message_filters.Subscriber(
            self,
            Detection2DArray,
            'detections_output')
        self._image_subscription = message_filters.Subscriber(
            self,
            Image,
            'image_rgb')


        self.time_synchronizer = message_filters.TimeSynchronizer(
            [self._detections_subscription, self._image_subscription],
            self.QUEUE_SIZE)

        self.time_synchronizer.registerCallback(self.detections_callback)

    def detections_callback(self, detections_msg, img_msg):
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg)
        H, W = cv2_img.shape[:2]

        for detection in detections_msg.detections:
            try:
                # ================================
                # 1. pixel bbox (Isaac ROS output)
                # ================================
                cx = detection.bbox.center.position.x
                cy = detection.bbox.center.position.y
                w  = abs(detection.bbox.size_x)
                h  = abs(detection.bbox.size_y)

                # ================================
                # 2. cxcywh → xyxy (PIXEL)
                # ================================
                x1 = int(cx - w / 2.0)
                y1 = int(cy - h / 2.0)
                x2 = int(cx + w / 2.0)
                y2 = int(cy + h / 2.0)

                # ================================
                # 3. clipping
                # ================================
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(0, min(x2, W - 1))
                y2 = max(0, min(y2, H - 1))

                # ================================
                # 4. label & score
                # ================================
                result = detection.results[0]
                label = result.hypothesis.class_id
                score = result.hypothesis.score

                # ================================
                # 5. draw bbox
                # ================================
                cv2.rectangle(
                    cv2_img,
                    (x1, y1),
                    (x2, y2),
                    self.color,
                    self.bbox_thickness
                )

                # center (검증용)
                cv2.circle(
                    cv2_img,
                    (int(cx * W), int(cy * H)),
                    3,
                    (255, 0, 0),
                    -1
                )

                cv2.putText(
                    cv2_img,
                    f"{label} {score:.3f}",
                    (x1, max(y1 - 5, 15)),
                    self.font,
                    self.font_scale,
                    self.color,
                    self.line_type
                )

            except (ValueError, IndexError):
                pass

        processed_img = self._bridge.cv2_to_imgmsg(
            cv2_img,
            encoding=img_msg.encoding
        )
        self._processed_image_pub.publish(processed_img)

def main():
    rclpy.init()
    rclpy.spin(RtDetrVisualizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
