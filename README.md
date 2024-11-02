# Project Title

**Project Title**: (Your project's name)

**Description**: This Project is for MSU's Building Classification Project that identifies 10 different buildings around MSU's campus.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Challenges and Approach](#challenges-and-approach)
- [Results](#results)
- [Testing Instructions](#testing-instructions)
- [Requirements](#requirements)
- [Model Download](#model-download)

## Introduction

This Project is for MSU's Building Classification Project that identifies 10 different buildings around MSU's campus.

## Model Architecture

**Key features**:
- **ResNet34 pre-trained weights** were utilized as a base.
- **Cross-validation approach**: For every 5 epochs, 20% of the dataset was rotated as validation data.
- **Learning Rate Scheduler**: The learning rate was halved whenever the validation accuracy did not improve within two epochs.

## Training Process

- **Batch size**: 32
- **Epochs**: 20
- **Validation approach**: Cross-validation by updating validation data every 5 epochs.
- **Model Saving**: Saved weights after every epoch to handle interruptions (e.g., from Colab session terminations).
- **Learning Rate Adjustment**: Reduced by half if validation accuracy did not increase over two consecutive epochs.
- **Confusion Matrix**: Printed after each epoch to observe class distribution and validation accuracy.


## Challenges and Approach

- **Model Selection**: We had initally tried to run the model with ResNet50 but the training time was taking too long along with a poor training accuracy. We had then decided to use ResNet 34 by removing 16 layers from ResNet50's model and decided to train the pre-trained model of ResNet50
- **Cross-Validation-Like Technique**: Implemented a strategy similar to cross-fold validation by varying the validation data every 5 epochs. This enabled better generalization and evaluation.
- **Handling Colab Interruptions**: Model weights were saved after each epoch to ensure the ability to continue training even if the session terminated.
- **Learning Rate Adjustment**: Implemented a dynamic learning rate scheduler to improve model performance and prevent overfitting.

## Results

The model achieved a **best validation accuracy of 95%** at epoch 17. The confusion matrix, though initially challenging to interpret, confirmed this high accuracy. Testing on the provided dataset without any image augmentation yielded similarly high accuracy levels.

## Testing Instructions



## Requirements

- Python (version X.X)
- TensorFlow (version X.X)
- Keras (version X.X)
- NumPy
- (Other libraries if needed)

To install all dependencies:
```bash
pip install -r requirements.txt
```


## Model Download
- 

