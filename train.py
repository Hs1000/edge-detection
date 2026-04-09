from ultralytics import YOLO

# Load base model
model = YOLO("yolov8n.pt")

# Train on your dataset
model.train(
    data="./train/data.yaml",
    epochs=30,
    imgsz=640
)
