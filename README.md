# Real-Time-Violence-Detection-using-Deep-Learning-and-Raspberry-Pi
Train YOLO Models


Click below to acces a Colab notebook for training YOLO models. It makes training a custom YOLO model as easy as uploading an image dataset and running a few blocks of code.

https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb

Results of my trained yolo v8model![train_batch1](https://github.com/user-attachments/assets/f4356a9c-06fa-4d41-b966-2ec22f022fa7)
![val_batch2_labels](https://github.com/user-attachments/assets/d53addb6-af1e-4e95-bd6e-5914df4fa6db)
![results](https://github.com/user-attachments/assets/bdbdec07-73e4-4413-8b71-e51dd3c6e72b)
![confusion_matrix](https://github.com/user-attachments/assets/f4589158-2ebe-42af-b078-2e598720a366)

Deploy YOLO Models
The yolo_detect.py script provides a basic example that shows how to load a model, run inference on an image source, parse the inference results, and display boxes around each detected class in the image. This script shows how to work with YOLO models in Python, and it can be used as a starting point for more advanced applications.

To run inference with a yolov8s model on a USB camera at 1280x720 resolution, issue:

python yolo_detect.py --model yolov8s.pt --source usb0 --resolution 1280x720

Here are all the arguments for yolo_detect.py:

--model: Path to a model file (e.g. my_model.pt). If the model isn't found, it will default to using yolov8s.pt.
--source: Source to run inference on. The options are:
Image file (example: test.jpg)
Folder of images (example: my_images/test)
Video file (example: testvid.mp4)
Index of a connected USB camera (example: usb0)
Index of a connected Picamera module for Raspberry Pi (example: picamera0)
--thresh (optional): Minimum confidence threshold for displaying detected objects. Default value is 0.5 (example: 0.4)
--resolution (optional): Resolution in WxH to display inference results at. If not specified, the program will match the source resolution. (example: 1280x720)
--record (optional): Record a video of the results and save it as demo1.avi. (If using this option, the --resolution argument must also be specified.)

Setup Picture
![WhatsApp Image 2025-05-19 at 10 37 30 AM](https://github.com/user-attachments/assets/64061e2b-1b3f-4128-9b1b-9a12fd997f95)



Final Results
![20250509_10h22m35s_grim](https://github.com/user-attachments/assets/2d9f6f50-531d-4c84-8ed1-bafff50e7463)

Alert Message Sent

![Screenshot 2025-05-04 002257](https://github.com/user-attachments/assets/3215853b-65c1-40bb-9992-f958bbf3a2e3)


