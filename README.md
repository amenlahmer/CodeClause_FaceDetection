# FaceDetection-YOLOV5

The project I undertook was focused on developing an efficient and accurate face detection system using a pretrained YOLOv5 model on the WIDER Face dataset. The primary goal was to leverage the power of deep learning and computer vision techniques to identify and localize faces in images and videos.

## Table of contents 
- Dataset
- Model
- Requirements
- Usage

## Dataset
The dataset comprises over 32,000 images, making it substantial in size. It contains more than 390,000 labeled faces, making it a valuable resource for training and evaluating face detection models. The dataset includes faces with different poses, occlusions, and lighting conditions, representing the challenges faced in real-life scenarios.

One notable feature of the WIDER Face dataset is its extensive annotation. Each image is carefully annotated with bounding boxes around the faces, providing precise location information for training and evaluation purposes. This level of annotation enables researchers and developers to fine-tune face detection models and assess their performance accurately.

## Model 
YOLOv5 is a state-of-the-art object detection algorithm renowned for its accuracy and speed. It follows a single-shot detection approach, processing the entire image in one go rather than using multiple stages. This design makes it highly efficient for real-time applications.
The architecture of YOLOv5 comprises a backbone network, neck network, and detection head. The backbone extracts image features, the neck network enhances these features, and the detection head predicts bounding boxes, object classes, and confidence scores.
YOLOv5 offers flexibility by supporting various backbone architectures like CSPDarknet53 and EfficientNet. It also provides different model sizes to balance speed and accuracy based on available computational resources.
Overall, YOLOv5 stands as a prominent object detection solution, offering remarkable performance, efficiency, flexibility, and ease of use. Its impact spans diverse fields, including autonomous driving, surveillance systems, robotics, and beyond, advancing the field of computer vision.

## Requirements
-A GP100 session with the following requirements installed:
gitpython>=3.1.30
matplotlib>=3.3
numpy>=1.18.5
opencv-python>=4.1.1
Pillow>=7.1.2
psutil  # system resources
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
thop>=0.1.1  # FLOPs computation
torch>=1.7.0  # see https://pytorch.org/get-started/locally (recommended)
torchvision>=0.8.1
tqdm>=4.64.0
ultralytics>=8.0.111.

## Usage
1.Clone the repository and install the required dependencies.
2.Download the WiderFace dataset and the YOLOv5 model weights from the official repository.
3.run the notebook in the repository in order to train the YOLOv5 on the WiderFace dataset 
4.Specify the video source in the code (video_path) as either a video file or webcam.
5.Run the code to start real-time face detection.

## Acknowledgments
This project utilizes the YOLOv5 model developed by the Ultralytics team. More information about the YOLOv5 model and its training can be found in their repository: https://github.com/ultralytics/yolov5

