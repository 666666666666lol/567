# Small Car License Plate Detection and Recognition System

This system provides an end-to-end solution for detecting and recognizing license plates in images. It consists of three main components:
- YOLOv5-based model for initial vehicle detection
- DetectNet for precise license plate localization
- OcrNet for optical character recognition of license plate text

## Setup

Ensure you have the required dependencies installed:
```
torch
opencv-python (cv2)
numpy
PIL
einops
```

## Pre-trained Weights

Pre-trained model weights are available in the `weights` folder:
- `yolov5s.pt` - YOLOv5 model for vehicle detection
- `detect_net.pt` - DetectNet model for license plate localization
- `ocr_net.pth` - OcrNet model for character recognition

These weights can be used directly for inference without additional training.

## Training

### Training the License Plate Detection Model

To train the license plate detection model (DetectNet):

```
$ python detect_train.py
```

This process uses the configuration settings in `detect_config.py` and trains the model to locate license plates in images. The trained weights will be saved to the path specified in the config file.

### Training the OCR Model

To train the optical character recognition model (OcrNet):

```
$ python ocr_train.py
```

This process uses the configuration in `ocr_config.py` and trains the model to recognize characters from license plate images. The trained weights will be saved to the path specified in the config file.

## Inference

To run inference on example images in the test_image folder:

```
$ python Main_code.py
```