# MSU Building Classification Challenge - Gavin Jiang Fan Club

**Description**: This Model is for MSU's Building Classification Project that identifies 10 different buildings around MSU's campus.

Members:
- Ryan Goodwin - rdg291@msstate.edu
- Kevin Ho - kth258@msstate.edu


## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Challenges and Approach](#challenges-and-approach)
- [Results](#results)
- [Testing Instructions](#testing-instructions)
- [Model Download](#model-download)

## Introduction

This project develops an image classification model using a modified ResNet50 architecture, aiming to classify the below buildings into different categories. The model utilizes transfer learning, leveraging pre-trained ResNet50 weights on ImageNet and fine-tuning them on a our personal dataset. The training process incorporates data augmentation techniques like shearing, zooming, rotation, and flipping to enhance model robustness. Custom callbacks are implemented to save the best model weights at different intervals, reset the validation data periodically, visualize the confusion matrix after each epoch, and dynamically adjust the learning rate based on validation loss. The model is evaluated on a held-out test set, and its performance is visualized using a confusion matrix, providing insights into the classification accuracy for each building category.

The model identifies the following buildings on the Mississippi State University Starkville Campus:
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

The model architecture is based on a modified ResNet50, adapted to resemble a ResNet34 by removing some of the deeper layers. The pre-trained ResNet50 (excluding its classification head) serves as a feature extractor. This feature extractor is followed by a Global Average Pooling layer to reduce the dimensionality of the feature maps. Next, a dense layer with 1024 neurons and ReLU activation, regularized by L2 regularization, is added. A dropout layer with a rate of 0.5 is included to prevent overfitting. Finally, an output dense layer with a softmax activation function provides the classification probabilities for each of the building categories. 

## Training Process
- **Data Loading and Augmentation**: Images are loaded from a directory on Google Drive (/content/drive/MyDrive/AIDataset) and augmented using ImageDataGenerator. Augmentations include rescaling, shearing, zooming, rotation, horizontal flipping, width/height shifts. The dataset is split into training and validation sets (80/20 split).
- **Model Compilation**: The model is compiled using the Adam optimizer with a learning rate of 0.0005, categorical cross-entropy loss (suitable for multi-class classification), and accuracy as the evaluation metric.
- **Callbacks**:
  - SaveBestEveryEpoch: Saves model every epoch and best model per 5-epoch block.

  - ReduceLROnPlateau: Reduces learning rate on plateau (factor=0.5, patience=3).

  - ResetValidationDataCallback: Resets validation data every 10 epochs.

  - ConfusionMatrixCallback: Displays confusion matrix after each epoch.

## Challenges and Approach

Training deep learning models for our campus vision AI competition presented several challenges, particularly given the limited timeframe and computational resources available through Google Colab. One early challenge was getting a cleaned, robust dataset to train the model off of. Thus, we took roughly 1,000 images of each building on campus, manually processing each one to ensure that the model was not trained on bad data. We decided to use only our dataset, as we felt the image quality of other participants was lacking and our dataset covered the majority of angles for each building. Another key hurdle was ensuring robust model evaluation and generalization. To address this, we implemented a cross-validation-like strategy, rotating the validation set every 5 epochs. This approach allowed us to assess performance on a larger portion of the dataset and mitigate potential biases associated with a fixed validation split.

Another significant challenge stemmed from the inherent instability of Colab sessions, which are prone to interruptions and resource limitations. To prevent the loss of trained models due to session disconnections, we implemented a model checkpointing strategy, saving the model weights after each epoch. This ensured that progress was preserved, even if a Colab session was terminated prematurely.

Finally, we recognized the importance of fine-tuning the learning rate throughout training. To achieve this, we incorporated a dynamic learning rate scheduler, likely reducing the learning rate based on validation performance or a pre-defined schedule (e.g., step decay or cosine annealing). This dynamic adjustment helped optimize convergence and prevented overfitting, particularly crucial given the potential complexity of our vision task.

## Results

The model achieved a **best validation accuracy of 95%** at epoch 17. The confusion matrix for the full dataset is in the repo as FullDatasetMatrix.png and the confusion matrix for the cleaned dataset is CleanedDatasetMatrix.png. On the full dataset, we achieved 11237/12584, or roughly 89.3% accuracy. On the cleaned dataset, we achieved 10062/10193, or roughly 98.7% accuracy.

## Testing Instructions

The model was entirely trained in Google Colab, so it is easiest to run it there. Add the model and dataset to a google drive. Open the test_model notebook in Colab. Switch the hosted runtime to TPU. Change the paths in the notebook to their corresponding path in the google drive (after it is mounted). Alternatively you can add the files manually and specify the paths in the colab instance. It is possible to run it locally, but it is not consistent and is not reproducible. The code would be the same as the test_model notebook and everything is stored locally. The requirements.txt is only for attempting to run it locally, but it is *highly suggested* to use Colab.

To install all dependencies:
```bash
pip install -r requirements.txt
```


## Model Download
- https://drive.google.com/file/d/1-AUcm_axuGYRgV79xWiUsgr61aXTv1jl/view?usp=sharing


