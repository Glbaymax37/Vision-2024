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
from geometry_msgs.msg import Point

bridge = CvBridge()

class CameraPublisher(Node):

    def __init__(self):
        super().__init__('coordinate_Publisher')

        weights='best.pt'  # model.pt path(s)
        self.imgsz=(640, 480)  # inference size (pixels)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.stride = 32
        device_num='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.dnn = False
        self.data= 'data/coco128.yaml'  # dataset.yaml path
        self.half=False  # use FP16 half-precision inference
        self.augment=False  # augmented inference

        # Initialize
        self.device = select_device(device_num)

        self.red_ball_positions = []
        self.purple_ball_positions = []


        # Load model
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Run inference
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup

        self.coord_publisher = self.create_publisher(Point, 'circle_coordinates', 10)

        # Open webcam
        self.cap = cv2.VideoCapture(0)

        self.timer_period = 0.1  # seconds
        self.create_timer(self.timer_period, self.camera_callback)

    def camera_callback(self):
        ret, frame = self.cap.read()

        # Run inference
        img = frame.copy()
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Preprocess image
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))  
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  
        img /= 255  

        if len(img.shape) == 3:
            img = img[None]  

        # Inference
        pred = self.model(img, augment=self.augment, visualize=self.visualize)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Process detections
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                for xyxy, conf, cls in det:
                    c = int(cls)
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')

                    # Calculate center of bounding box
                    center_x = int((xyxy[0] + xyxy[2]) / 2)
                    center_y = int((xyxy[1] + xyxy[3]) / 2)

                    # Determine the object class and store its position accordingly
                    if self.names[c] == 'red_ball':
                        self.red_ball_positions.append((center_x, center_y))
                    elif self.names[c] == 'purple_ball':
                        self.purple_ball_positions.append((center_x, center_y))

                    # Publish position
                    coord_msg = Point()
                    coord_msg.x = float(center_x)
                    coord_msg.y = float(center_y)
                    coord_msg.z = 0.0  
                    self.coord_publisher.publish(coord_msg)

                    # Draw circle and text on the image
                    cv2.circle(img0, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)  
                    cv2.putText(img0, f'({center_x}, {center_y})', (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the image
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


