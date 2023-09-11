#! /usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from rostopic import get_topic_type
from sensor_msgs.msg import Image
from yolov8_ros.msg import BoundingBox, BoundingBoxes
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov8"


class Yolov8Detector:
    def __init__(self):
        self.model = YOLO("/home/fc/catkin_ws/src/gm/yolov8_ros/weights/best.pt")
        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
        self.pred_pub = rospy.Publisher("/traffic_light_color_detect_bbox", BoundingBoxes, queue_size=10)
        self.pred_img_pub = rospy.Publisher('/traffic_light_color_detect_img', Image, queue_size=10)
        self.bridge = CvBridge()
    def callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        results  = self.model(cv_image)
        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = img_msg.header
        bounding_boxes.image_header = img_msg.header
        boxes = results[0].boxes.cpu().numpy()

        for i in range(len(boxes)):
            bbox_msg = BoundingBox()
            box = boxes[i]
            bbox_msg.Class = str(int(box.cls[0]))
            bbox_msg.probability = box.conf[0] 
            bbox_msg.xmin = int(box.xyxy[0][0])
            bbox_msg.ymin = int(box.xyxy[0][1])
            bbox_msg.xmax = int(box.xyxy[0][2])
            bbox_msg.ymax = int(box.xyxy[0][3])

            bounding_boxes.bounding_boxes.append(bbox_msg)
        self.pred_img_pub.publish(self.bridge.cv2_to_imgmsg(results[0].plot(), encoding='bgr8'))
        self.pred_pub.publish(bounding_boxes)
    
if __name__ == "__main__":
    rospy.init_node("yolov8", anonymous=True)

    detector = Yolov8Detector()
    rospy.spin()