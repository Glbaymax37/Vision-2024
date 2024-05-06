#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.common import DetectMultiBackend
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors, save_one_box
from geometry_msgs.msg import Twist

bridge = CvBridge()

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class CameraPublisher(Node):

    def __init__(self):
        super().__init__('coordinate_Publisher')

        weights = 'los.pt'  # model.pt path(s)
        self.imgsz = (640, 480)  # inference size (pixels)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 3  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.stride = 32
        device_num = 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.dnn = False
        self.data = 'data/coco128.yaml'  # dataset.yaml path
        self.half = False  # use FP16 half-precision inference
        self.augment = False  # augmented inference

        self.radius = 150

        self.objects_within_radius = 0

        self.max_RedBall = 1
        self.max_PurpleBall = 1
        self.max_Blueball = 1
        self.max_Silo = 1

        # Initialize
        self.device = select_device(device_num)

        self.red_balls_count = 0

        # Load model
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Run inference
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup

        self.silo_publisher = self.create_publisher(Twist, 'Silo', 10)
        self.red_publisher = self.create_publisher(Twist, 'Red_ball', 10)
        self.purple_publisher = self.create_publisher(Twist, 'Purple_ball', 10)
        self.blue_publisher = self.create_publisher(Twist, 'Blue_ball', 10)

        # Open webcam
        self.cap = cv2.VideoCapture(2)

        self.timer_period = 0.1  # seconds
        self.create_timer(self.timer_period, self.camera_callback)

    def camera_callback(self):
        ret, frame = self.cap.read()

        # Inference
        img = frame.copy()
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        save_dir = '/home/baymax/yolobot/src/yolobo'
        path = '/yolobot/src/yolobot_recognition/scripts/data/images/zidane.jpg'

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
        pred = self.model(img, augment=self.augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Reset counts for each class
        self.red_balls_count = 0
        self.silo_count = 0
        self.purple_balls_count = 0
        self.blue_balls_count = 0

        self.silo_with_ball = None
        self.largest_bbox_area = 0



        detected_classes = set()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))

            # Reset detected classes set
            detected_classes = set()

            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Limit the maximum number of detected objects for each class
        count_limits = {
            'RedBall': self.max_RedBall,
            'PurpleBall': self.max_PurpleBall,
            'BlueBall': self.max_Blueball,
            'Silo': self.max_Silo
        }
        silo_msg = Twist()
        red_msg = Twist()
        purple_msg = Twist()
        blue_msg = Twist()

   

        objects_within_radius = []

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))

            bbox_width = xyxy[2] - xyxy[0]
            bbox_height = xyxy[3] - xyxy[1]

            bbox_area = bbox_width * bbox_height

            if self.names[c] not in detected_classes:
                # Add the class to the set
                detected_classes.add(self.names[c])

            # Check if the maximum count for the current class has been reached
            if self.names[c] in count_limits and count_limits[self.names[c]] <= 0:
                continue  # 

            # Calculate center point of the bounding box
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)

           
            COLORS = (0, 255, 0)

            # Calculate distance between object and 'silo'
            distance_to_silo = calculate_distance(center_x, center_y, silo_msg.linear.x, silo_msg.linear.y)

            
            if distance_to_silo <= self.radius:
                cv2.circle(img0, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)  # Green circle


                objects_within_radius.append((center_x, center_y))


            self.objects_within_radius = len(objects_within_radius)


            if bbox_area > self.largest_bbox_area:
                # Update information about the silo with largest bbox area
               self.largest_bbox_area = bbox_area
               self.largest_bbox_silo = (center_x, center_y)

        

            if self.silo_with_ball is not None:
                silo_coords = self.silo_with_ball
            else:
                silo_coords = self.largest_bbox_silo



            cv2.putText(img0, f'({center_x}, {center_y})', (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            class_name = self.names[c] if c < len(self.names) else 'unknown'
            text = f'{class_name}, Coordinates: ({center_x}, {center_y})'

            bloxarea = f'{class_name}, Area: ({bbox_area})'

            if silo_coords is not None:
                silo_msg = Twist()
                silo_msg.linear.x = float(silo_coords[0])
                silo_msg.linear.y = float(silo_coords[1])
                silo_msg.linear.z = 0.0  # Set z to 0, assuming it's not used
                silo_msg.angular.x = 0.0  # Set angular components to 0
                silo_msg.angular.y = 0.0
                silo_msg.angular.z = 0.0
                self.silo_publisher.publish(silo_msg)

            elif self.names[c] == 'RedBall':
                red_msg = Twist()
                red_msg.linear.x = float(center_x)
                red_msg.linear.y = float(center_y)

            elif self.names[c] == 'PurpleBall':
                purple_msg = Twist()
                purple_msg.linear.x = float(center_x)
                purple_msg.linear.y = float(center_y)

            elif self.names[c] == 'BlueBall':
                blue_msg = Twist()
                blue_msg.linear.x = float(center_x)
                blue_msg.linear.y = float(center_y)

            self.silo_publisher.publish(silo_msg)
            self.red_publisher.publish(red_msg)
            self.purple_publisher.publish(purple_msg)
            self.blue_publisher.publish(blue_msg)

            cv2.putText(img0, text, (center_x - 10, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)
            print(text)
            cv2.putText(img0, bloxarea, (center_x - 10, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)

            print(f'Bounding box area: {bbox_area}')

        # Show object counts
        print("Red Balls Count:", self.red_balls_count)
        print("Blue Balls Count:", self.blue_balls_count)
        print("Purple Balls Count:", self.purple_balls_count)
        print("Silo Count:", self.silo_count)
        print("Ball_in_Silo",self.objects_within_radius)

        cv2.imshow("IMAGE", img0)
        cv2.waitKey(10)

    def shutdown_callback(self):
        # Release the camera and destroy any OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rclpy.init(args=None)
    camera_publisher = CameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.shutdown_callback()
        camera_publisher.destroy_node()
        rclpy.shutdown()
