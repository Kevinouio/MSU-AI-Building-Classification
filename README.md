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

Our campus vision AI competition project initially employed a ResNet50 architecture as a feature extractor, intending to fine-tune it on the competition dataset. However, the model's size proved computationally prohibitive, hindering convergence speed during training. Consequently, we transitioned to a ResNet34 backbone, which yielded improved convergence rates and faster training times The ResNet34 model was pretrained on ImageNet, providing a strong initialization for our task. We replaced the final fully connected layer with a new layer appropriate for the number of classes in our competition dataset. Data augmentation techniques, including random cropping, horizontal flipping, and gaussian blur, were used to enhance model generalization and robustness to variations in the input images. The model was trained using the Adam optimizer with a learning rate scheduler to further improve convergence.

## Training Process

- **Batch size**: 32
- **Epochs**: 20
- **Validation approach**: Cross-validation by updating validation data every 5 epochs.
- **Model Saving**: Saved weights after every epoch to handle interruptions (e.g., from Colab session terminations).
- **Learning Rate Adjustment**: Reduced by half if validation accuracy did not increase over two consecutive epochs.
- **Confusion Matrix**: Printed after each epoch to observe class distribution and validation accuracy.

## Challenges and Approach

Training deep learning models for our campus vision AI competition presented several challenges, particularly given the limited timeframe and computational resources available through Google Colab. One early challenge was getting a cleaned, robust dataset to train the model off of. Thus, we took roughly 1,000 images of each building on campus, manually processing each one to ensure that the model was not trained on bad data. We decided to use only our dataset, as we felt the image quality of other participants was lacking and our dataset covered the majority of angles for each building. Another key hurdle was ensuring robust model evaluation and generalization. To address this, we implemented a cross-validation-like strategy, rotating the validation set every 5 epochs. This approach allowed us to assess performance on a larger portion of the dataset and mitigate potential biases associated with a fixed validation split.

Another significant challenge stemmed from the inherent instability of Colab sessions, which are prone to interruptions and resource limitations. To prevent the loss of trained models due to session disconnections, we implemented a model checkpointing strategy, saving the model weights after each epoch. This ensured that progress was preserved, even if a Colab session was terminated prematurely.

Finally, we recognized the importance of fine-tuning the learning rate throughout training. To achieve this, we incorporated a dynamic learning rate scheduler, likely reducing the learning rate based on validation performance or a pre-defined schedule (e.g., step decay or cosine annealing). This dynamic adjustment helped optimize convergence and prevented overfitting, particularly crucial given the potential complexity of our vision task.

## Results

The model achieved a **best validation accuracy of 95%** at epoch 17. The confusion matrix for the full dataset is in the repo as FullDatasetMatrix.png and the confusion matrix for the cleaned dataset is CleanedDatasetMatrix.png. On the full dataset, we achieved 11237/12584, or roughly 89.3% accuracy. On the cleaned dataset, we achieved 10062/10193, or roughly 98.7% accuracy.

## Testing Instructions

The model was entirely trained in Google Colab, so it is easiest to run it there. Add the model and dataset to a google drive. Open the test_model notebook in Colab. Switch the hosted runtime to TPU. Change the paths in the notebook to their corresponding path in the google drive (after it is mounted). Alternatively you can add the files manually and specify the paths in the colab instance. 

To install all dependencies:
```bash
pip install -r requirements.txt
```


## Model Download
- https://drive.google.com/file/d/1-AUcm_axuGYRgV79xWiUsgr61aXTv1jl/view?usp=sharing


