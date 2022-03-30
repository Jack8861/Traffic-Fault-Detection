import argparse
import os
import sys
from pathlib import Path
import imutils
import easyocr
import cv2
import torch
from utils.augmentations import letterbox
import numpy as np
from utils.datasets import LoadWebcam
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size, check_requirements, non_max_suppression, print_args, scale_coords)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
import matplotlib.pyplot as plt
import os


ROOT = os.getcwd()

if ROOT not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.no_grad()
class Detector():
    vehicle_weights = 'vehicle.pt'  # model.pt path(s)
    helmet_weights = 'helmet.pt'
    source = ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
    data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz = (640, 640)  # inference size (height, width)
    conf_thres = 0.50  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    half = False  # use FP16 half-precision inference
    dnn = False  # use
    line_thickness = 3
    device = select_device(device)

    vehicle_model = DetectMultiBackend(
        vehicle_weights, device=device, dnn=dnn, data=data)
    stride, vehicle_names = vehicle_model.stride, vehicle_model.names

    helmet_model = DetectMultiBackend(
        helmet_weights, device=device, dnn=dnn, data=data)
    stride, helmet_names = helmet_model.stride, helmet_model.names
    print(helmet_names, ROOT)
    data = ROOT / 'data/coco128.yaml',  # dataset.yaml path

    def __init__(self, actual_image=''):
        pass

    def image_preprocessing(self, image, actual_image, xmin=0, ymin=0):
        """
        Proess the image so it can be used for detection
        """
        self.image = image
        self.actual_image = actual_image
        self.xmin = xmin
        self.ymin = ymin
        img_size = check_img_size(
            self.imgsz, s=self.stride)  # check image size
        stride = 32
        auto = True
        self.img = letterbox(self.image, img_size, stride=stride, auto=auto)[0]
        self.img = self.img.transpose(
            (2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        self.im = np.ascontiguousarray(self.img)
        self.im = torch.from_numpy(self.im).to(self.device)
        self.im = self.im.half() if self.half else self.im.float()  # uint8 to fp16/32
        self.im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(self.im.shape) == 3:
            self.im = self.im[None]  # expand for batch dim

    def vehicle(self):
        """
        Detect vehicles and return their bounding boxes and labels
        """

        pred = self.vehicle_model(self.im, augment=self.augment)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   self.classes, self.agnostic_nms, max_det=self.max_det)

        vehicles = {'car': [], 'auto': [], 'bike': []}
        count = {'car': 0, 'auto': 0, 'bike': 0}
        self.actual_image = np.ascontiguousarray(self.actual_image)
        annotator = Annotator(
            self.actual_image, line_width=self.line_thickness, example=str(self.vehicle_names))

        for i, det in enumerate(pred):

            det[:, :4] = scale_coords(
                self.im.shape[2:], det[:, :4], self.image.shape).round()
            for i, det in enumerate(pred):
                det[:, :4] = scale_coords(
                    self.im.shape[2:], det[:, :4], self.image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    vehicles[self.vehicle_names[int(cls)]].append(
                        (xyxy, self.vehicle_names[int(cls)]))

            for *xyxy, conf, cls in reversed(det):
                xmin, ymin, xmax, ymax = xyxy
                xyxy = (xmin+self.xmin, ymin+self.ymin,
                        xmax+self.xmin, ymax+self.ymin)
                c = int(cls)  # integer class
                label = None if False else (
                    self.vehicle_names[c] if False else f'{self.vehicle_names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if False:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' /
                                 self.vehicle_names[c] / f'{p.stem}.jpg', BGR=True)

        return (annotator.result(), vehicles)

    def helmet(self, bbox):
        """
        detect helmet, if not present then scrape its license plate
        """
        xmin, ymin, xmax, ymax = bbox
        pred = self.helmet_model(self.im, augment=self.augment)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   self.classes, self.agnostic_nms, max_det=self.max_det)

        helmet = {'helmet': [], 'hi': [], 'no_helmet': []}
        count = {'helmet': 0, 'hi': 0, 'no_helmet': 0}
        self.actual_image = np.ascontiguousarray(self.actual_image)
        annotator = Annotator(
            self.actual_image, line_width=self.line_thickness, example=str(self.helmet_names))

        plates = []
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(
                self.im.shape[2:], det[:, :4], self.image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if self.helmet_names[int(cls)] == 'no helmet':
                    license = self.license_plate(self.image)
                    if license != '':
                        plates.append(license)

        return (plates)

    def license_plate(self, img):
        """
        scrape the license plate
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
            edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

            keypoints = cv2.findContours(
                edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            location = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break

            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))

            cropped_image = gray[x1:x2+1, y1:y2+1]
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)

            return result
        except cv2.error as e:
            print(e)
        return ""
