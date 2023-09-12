import cv2
import torch
from torch.hub import download_url_to_file
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# Define the paths to the YOLOv5 model and the video source (0 for webcam)
weights_path = 'best.pt'
video_source = 0  # 0 for webcam

# Initialize YOLOv5 model
device = select_device('')
model = attempt_load(weights_path, map_location=device)
stride = int(model.stride.max())  # Assume all scales have same stride
imgsz = check_img_size(640, s=stride)
if not model.names:
    model.names = ['class'] * model.nc  # Replace with your class names

# Initialize webcam or video source
vid = cv2.VideoCapture(video_source)

while True:
    ret, img0 = vid.read()  # Read a frame from the webcam or video source

    if not ret:
        break

    img = torch.from_numpy(img0).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=True)[0]

    # NMS and post-processing
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    for det in pred[0]:
        p, s, im0 = scale_coords(img.shape[2:], det[:4], img0.shape[:2])
        plot_one_box(p, im0, label=f'{model.names[int(det[-1])]} {det[4]:.2f}', color=(0, 255, 0), line_thickness=3)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Real-Time Object Detection', im0)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
