# Embedded-AI

## Abstract
Hi, this is the repo for Group 12 -- Dynamic Gesture Recognition System on Raspberry Pi using a Custom Tiny-CNN.

By the following steps, you could successful built a tiny CNN model on raspberry PI 5.

## Training
Upload the files to the Raspberry PI.I hope the environment should not be a problem, if not, here is some tips.
```
python3 venv
python -m venv venv
source /venv/bin/activate
pip install numpy opencv-python matplotlib scikit-learn torch natsort
```

Then, you may need to change the root dir in the python file: ```tiny_cnn.py```. Then the command to train is :
```
python3 tiny_cnn.py
```
The log information would be shown in the terminal. It may take up to 4 hours for training. For the file size, the whole dataset is not provided in the files. If you want to use the dataset, 
go to https://ieee-dataport.org/documents/dataset-dynamic-hand-gesture-recognition-systems or email us.

## Inferencing
We provide you with pre-trained model. You can find it under the folder, ending with ```.pt``` or ```.pth```. Change the configuration in ```inference.py```.  Install the environment by:
```
python3 venv
python -m venv venv
source /venv/bin/activate
pip install numpy opencv-python torch
```
The Inferencing can be done by
```
python3 inference.py <video>
```
