from ultralytics import YOLO

best_model_path = 'runs/detect/yolo11s_custom_dataset/weights/best.pt'
test_image_path = 'data/test/images/20221120_165722_jpg.rf.730d6ef7f112828c75e429b714c633ff.jpg'

model = YOLO(best_model_path)
model.predict(
    source=test_image_path, # Image path or directory of the image to be predicted
    show=True, # Set to True to display the media with predictions
    save=True, # Set to True to save the image with predictions made on it
    save_crop=True, # Save cropped prediction boxes
    save_txt=True, # Save predictions in a .txt file
    show_labels=True, # Show labels on the predicted boxes
    show_conf=True, # Show confidence scores on the predicted boxes
    # classes=None, # List of class indices to filter predictions, empty for all classes, none for all classes
    conf=0.6, # Confidence threshold for predictions
    line_width=2, # Width of the bounding box lines
    device=0, # Use device=0 for GPU, device='cpu' for CPU
)