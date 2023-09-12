from ultralytics import YOLO
from ultralytics.yolo.v8.detect.pridict import DetectionPredector
import cv2

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=3)  # train the model