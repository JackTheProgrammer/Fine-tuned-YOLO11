from ultralytics import YOLO

best_model_path = 'runs/detect/yolo11s_custom_dataset/weights/best.pt'
model = YOLO(best_model_path)

# Export the model to ONNX format with an image size of 640x640.
# Exported model is saved in the 'runs/detect/yolo11s_custom_dataset/weights/' directory
model.export(format='onnx', imgsz=640, device=0)

# Export the model to TensorFlow Lite format with 
# an image size of 640x640
# model.export(format='tflite', imgsz=640, device=0)