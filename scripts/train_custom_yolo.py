from ultralytics import YOLO

yolo_11_path = 'model/yolo11s.pt'
dataset_yaml_path = 'data/data.yaml'
model = YOLO(yolo_11_path)
model.train(
    data=dataset_yaml_path,
    epochs=100, # You may increase the number of epochs for better results
    imgsz=640,
    batch=8, # No. of images per batch
    patience=50, # Early stopping patience
    name='yolo11s_custom_dataset', # Name of the training run
    workers=0, # Number of data loading workers
    device=0 # Use device=0 for GPU, device='cpu' for CPU
)