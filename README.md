# MSU Building Classification Challenge

**Description**: This Model is for MSU's Building Classification Project that identifies 10 different buildings around MSU's campus.

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

The Model that is for the project is our submission for MSU's Building Classification Project, to where it would identify 10 different buildings being:
- Butler Hall
- McCool Hall
- McCain Hall
- Walker Hall
- Carpenter Hall
- Swalm Hall
- Lee Hall
- Old Main
- Student Union
- Simrall Hall

## Model Architecture

This project uses ResNet34 as the main model that we decided to train for building classification.


## Training Process

- **Batch size**: 32
- **Epochs**: 20
- **Validation approach**: Cross-validation by updating validation data every 5 epochs.
- **Model Saving**: Saved weights after every epoch to handle interruptions (e.g., from Colab session terminations).
- **Learning Rate Adjustment**: Reduced by half if validation accuracy did not increase over two consecutive epochs.
- **Confusion Matrix**: Printed after each epoch to observe class distribution and validation accuracy.


## Challenges and Approach

- **Cross-Validation-Like Technique**: Implemented a strategy similar to cross-fold validation by varying the validation data every 5 epochs. This enabled better generalization and evaluation.
- **Handling Colab Interruptions**: To prevent the problem of a session interruption in Google Colab and deleting and removing our trained models after each epoch, we decided to save the model after each epoch the model runs on.
- **Learning Rate Adjustment**: Implemented a dynamic learning rate scheduler to improve model performance and prevent overfitting.

## Results

The model achieved a **best validation accuracy of 95%** at epoch 17. The confusion matrix, though initially challenging to interpret, confirmed this high accuracy. Testing on the provided dataset without any image augmentation yielded similarly high accuracy levels.

## Testing Instructions
The model was entirely trained in Google Colab, so it is easiest to run it there. Add the model and dataset to the memory of the Colab instance and switch the hosted runtime to TPU. Change the paths in the notebook to their corresponding path in the Colab instance. Alternatively you can add the files to your google drive.

To install all dependencies:
```bash
pip install -r requirements.txt
```


## Model Download
- 

