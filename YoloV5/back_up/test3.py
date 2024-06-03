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

bridge = CvBridge()

class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        weights='yolov5s.pt'  # model.pt path(s)
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
        device_num='0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.dnn = False
        self.data= 'data/coco128.yaml'  # dataset.yaml path
        self.half=False  # use FP16 half-precision inference
        self.augment=False  # augmented inference

        # Initialize
        self.device = select_device(device_num)

        # Load model
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Run inference
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup

        self.publisher = self.create_publisher(Image, 'detected_objects', 10)

        # Open webcam
        self.cap = cv2.VideoCapture(0)

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

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
        pred = self.model(img, augment=self.augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        cv2.imshow("IMAGE", img0)
        cv2.waitKey(10)

        # Convert image to ROS message
        ros_img = bridge.cv2_to_imgmsg(img0, encoding="bgr8")
        # Publish ROS message
        self.publisher.publish(ros_img)

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

