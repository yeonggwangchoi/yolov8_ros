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
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from yolov8_ros.msg import BoundingBox, BoundingBoxes
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov8"
bridge = CvBridge()
loop = False


class Yolov8Detector:
    def __init__(self):
        self.model = YOLO("/home/fc/catkin_ws/src/gm/yolov8_ros/weights/yolov8x.pt")
        self.img_sub = rospy.Subscriber('/usb_cam2/image_raw', Image, self.callback)
        self.pred_img_pub = rospy.Publisher('/traffic_light_detect_img', Image, queue_size=1)
        self.traffic_detect_pub = rospy.Publisher('/traffic_light_detect', Bool, queue_size=1)
    
    def callback(self, img_msg):
        cv_image = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        
        results = self.model.predict(source=cv_image, device='cuda')
        r_img = results[0].plot()
        cv2.imshow("r_img", r_img)
        boxes = results[0].boxes.cpu().numpy()

        for i in range(len(boxes)):
            box = boxes[i]
            if int(box.cls[0]) == 9 and box.conf[0] > 0.1 and ((int(box.xyxy[0][2])-int(box.xyxy[0][0])) * (int(box.xyxy[0][3]) - int(box.xyxy[0][1])))>3000:
                img = cv_image[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
                print('box :', int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                cv2.imshow("img", img)
                self.pred_img_pub.publish(bridge.cv2_to_imgmsg(img, encoding='bgr8'))
                self.traffic_detect_pub.publish(True)
            else:
                self.traffic_detect_pub.publish(False)
            cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("traffic_light_detect")

    detector = Yolov8Detector()
    rospy.spin()
