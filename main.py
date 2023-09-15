from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

model = YOLO(r"C:\Users\HP\PycharmProjects\yolo_v8\runs\detect\train\weights\best.pt")
result = model.predict(source=r"0", show=True)
print(result)
