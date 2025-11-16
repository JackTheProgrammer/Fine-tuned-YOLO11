from ultralytics import YOLO

best_model_path = 'runs/detect/yolo11s_custom_dataset/weights/best.pt'
vid_path = 'data/video/Traffic lights sequence theory test UK part 1.mp4'

model = YOLO(best_model_path)
model.predict(
    source=vid_path, # Video path or directory of the image to be predicted
    show=True, # Set to True to display the image with predictions
    save=True, # Set to True to save the media with predictions made on it
    save_dir='runs/detect/video_predict', # Directory to save predictions
    save_crop=False, # Save cropped prediction boxes
    save_txt=False, # Save predictions in a .txt file
    show_labels=True, # Show labels on the predicted boxes
    show_conf=True, # Show confidence scores on the predicted boxes
    # classes=None, # List of class indices to filter predictions, empty for all classes, none for all classes
    conf=0.7, # Confidence threshold for predictions
    line_width=2, # Width of the bounding box lines
    device=0, # Use device=0 for GPU, device='cpu' for CPU
)