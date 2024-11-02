# Project Title

**Project Title**: (Your project's name)

**Description**: This project is an image classification model built using ResNet34, fine-tuned to achieve high accuracy on a specified dataset through careful adjustments in learning rate, cross-validation techniques, and model saving.

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

This project focuses on building a robust image classification model, initially tested with ResNet50 but ultimately adapted to ResNet34. The aim was to achieve a high validation accuracy by iteratively tuning hyperparameters, implementing learning rate adjustments, and experimenting with cross-validation-like techniques.

## Model Architecture

The model architecture used here is **ResNet34**, chosen for its balance between complexity and computational efficiency. Initially, **ResNet50** was tested; however, due to poor training accuracy, we switched to ResNet34 for improved performance and faster convergence.

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

After switching to ResNet34, we observed improvement in training and validation accuracy, achieving the best performance by epoch 17 with a **validation accuracy of 95%**.

## Challenges and Approach

- **Model Selection**: ResNet50 initially showed poor performance with no learning rate scheduler and a high batch size, which led to experimenting with **ResNet34**.
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

