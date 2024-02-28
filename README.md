# Traffic Sign Detection

### Detect traffic signs with Deep Learning models

This project aims to develop two different models: one from scratch and one using fine-tuning and transfer learning to classify traffic signs. 
The dataset used is the famous GTSRB, a german traffic sign benchmark dataset. 

Currently, only a simple CNN is available with an accuracy of ~88% over the test dataset. 


### Tools

- Python
- OpenCV
- Pytorch
- Matplotlib
- Jupyter Notebooks

### Resources

- Dataset: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

### Steps to run

- Install the requirements.txt
- Download the dataset into a dataset folder
- Run one of the Makefile commands
- Done!

### In-progress features

- Improve the performance of the model
- Docker image
- Logging
- Better documentation/comments
- Use transfer learning and fine tuning to compare the performance between a from scratch model and a fine tuned model
- Change params to *args and **kwargs
- Generate actually good plots to demonstrate the performance of the model during training/validation and the results of the testing stage.
- Better code organization, with a major focus in the main.py file