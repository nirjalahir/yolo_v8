from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml") # If we want to create the new model from scratch then use this

# load a pretrained model (If we are training our existing model than use this)
model = YOLO(r'yolov8m.pt')

# Use the mode(Give the path of config.yaml file in which we have classified the classes)
model.train(data=r"C:\Users\HP\PycharmProjects\yolo_v8\config.yaml", epochs=1)  # train the model
