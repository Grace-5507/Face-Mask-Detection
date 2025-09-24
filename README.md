

# Face Mask Detection with PyTorch and OpenCV

This project detects whether a person is wearing a face mask correctly, incorrectly, or not at all.
It uses a ResNet-18 convolutional neural network fine-tuned on a custom dataset.
The model supports training, evaluation, and real-time webcam detection.

## Project Structure

data/
├── train/
├── val/
└── test/

models/
└── resnet18\_mask.pth

notebooks/
├── training.ipynb
└── inference.ipynb

realtime.py
README.md

## Features

* Training a ResNet-18 model for 3 classes: with\_mask, without\_mask, mask\_weared\_incorrect
* Training and validation with PyTorch
* Evaluation on a test dataset with accuracy and confusion matrix
* Real-time face mask detection using webcam with OpenCV

## Setup

1. Clone the repository
2. Install required libraries: torch, torchvision, matplotlib, opencv-python, pillow
3. Prepare dataset in the following structure:

dataoutputs/train/with\_mask
dataoutputs/train/without\_mask
dataoutputs/train/mask\_weared\_incorrect
dataoutputs/val/...
dataoutputs/test/...

## Training

The ResNet-18 model is fine-tuned on the dataset.
Use the training notebook to run training and save the best model weights.

## Evaluation

Use the inference notebook or script to evaluate the model on the test dataset.
Generates accuracy, confusion matrix, and sample predictions.

## Real-time Detection

Run realtime.py to use the webcam for detection.
Press "q" to quit manually or the script will stop automatically after 15 seconds.

## Results

Training accuracy: about 89%
Validation accuracy: about 85%
Good generalization on test images.

## Future Improvements

* Add bounding boxes for detected faces
* Train longer or on a larger dataset
* Use a deeper model such as ResNet-50
* Deploy as a web or mobile app

