# YOLO11s.pt fine tuning
In this project, I created an upto 100 epochs custom dataset based fine tuned YOLO model using [`yolo11s.pt`](model\yolo11s.pt) and then tested it on the image and video, after that I exported the cusotm trained model to ONNX.

## Get started
[YOLO11 training on custom data](https://www.youtube.com/watch?v=A1V8yYlGEkI&t=1006s).

## Dataset
[Roboflow traffic signal detection](https://universe.roboflow.com/signal-sense-traffic-signal-detection/traffic-signals-mcmpm)

*IMPORTANT!!* Do download the dataset in the **YOLO** format

## Results
![alt Training results on 100 epochs](runs\detect\yolo11s_custom_dataset\results.png)

![alt Confusion matrix](runs\detect\yolo11s_custom_dataset\confusion_matrix.png)

![alt Box PR curve](runs\detect\yolo11s_custom_dataset\BoxPR_curve.png)

### Image based prediction results
![Test result](runs\detect\predict\20221120_165722_jpg.rf.730d6ef7f112828c75e429b714c633ff.jpg)

### Video based prediction results
![alt Video](runs\detect\video_predict\TrafficlightssequencetheorytestUKpart1-ezgif.com-optimize.gif)

### Saved models
* [Saved best.pt file](runs\detect\yolo11s_custom_dataset\weights\best.pt).
* [Saved best.onnx file](runs\detect\yolo11s_custom_dataset\weights\best.pt).
* For model's visualization, go [here](https://netron.app/) and upload your `*.pt` or `*.onnx` file.