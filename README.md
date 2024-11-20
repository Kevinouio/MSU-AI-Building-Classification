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

This project develops an image classification model using a modified ResNet50 architecture, aiming to classify the below buildings into different categories. The model utilizes transfer learning, leveraging pre-trained ResNet50 weights on ImageNet and fine-tuning them on our personal dataset. The training process incorporates data augmentation techniques like shearing, zooming, rotation, and flipping to enhance model robustness. Custom callbacks are implemented to save the best model weights at different intervals, reset the validation data periodically, visualize the confusion matrix after each epoch, and dynamically adjust the learning rate based on validation loss. The model is evaluated on a held-out test set, and its performance is visualized using a confusion matrix, providing insights into the classification accuracy for each building category. The model took roughly 48 hours to train. 

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
- **Data Loading and Augmentation**: These augmentations artificially increased the diversity of our training data, helping the model learn more robust features and reducing overfitting. The ImageDataGenerator applied these transformations randomly during each epoch, ensuring that the model saw a slightly different version of the training data in each iteration. This approach significantly improved the model's ability to generalize to unseen images. The augmentations are as follows:
  - Rescaling: Pixel values were normalized by dividing by 255, scaling them to the range [0, 1].

  - Shearing: Images were randomly sheared horizontally or vertically within a range of ±20%.

  - Zooming: Random zooming was applied with a range of ±20%, simulating variations in image scale.

  - Rotation: Images were randomly rotated by up to 20 degrees in either direction.

  - Horizontal Flipping: A random horizontal flip was applied to some images, introducing further variation.

  - Width/Height Shifts: Images were randomly shifted horizontally or vertically by up to 20% of their width or height, respectively.
  
- **Model Compilation**: The model is compiled using the Adam optimizer with a learning rate of 0.0005, categorical cross-entropy loss (suitable for multi-class classification), and accuracy as the evaluation metric.
- **Callbacks**:
  - SaveBestEveryEpoch: Saves model every epoch and best model per 5-epoch block.

  - ReduceLROnPlateau: Reduces learning rate on plateau (factor=0.5, patience=3).

  - ResetValidationDataCallback: Resets validation data every 10 epochs.

  - ConfusionMatrixCallback: Displays confusion matrix after each epoch.

## Challenges and Approach

Developing our campus vision AI model within the competition's constraints presented several key challenges. Beyond the instability of Colab sessions, effectively managing training parameters and mitigating overfitting proved crucial. To combat overfitting, we employed a combination of techniques. Firstly, we introduced L2 regularization within dense layers, penalizing large weights and encouraging smoother decision boundaries. Secondly, a dropout layer was strategically placed within the architecture to further discourage over-reliance on individual neurons and promote more generalized feature learning.

Additionally, a high quality dataset is critical to any well trained model. We took roughly 1000 images per building, varying lighting conditions, angles, and distances. We manually cleaned each set of pictures to ensure even distribution of angles and high-quality data. After evaluating the publicly available data, we opted to utilize only our curated dataset to maximize training effectiveness.

Given the complexity of distinguishing between similar-looking buildings, we optimized our model architecture for robust feature extraction. Starting with a pre-trained ResNet50 base provided a strong foundation. However, we found that the full ResNet50 architecture was computationally expensive and potentially prone to overfitting on our dataset. Therefore, we adapted it to resemble ResNet34 by removing some of the deeper layers, striking a balance between feature extraction capabilities and computational efficiency. This modification allowed for faster training cycles within the limited Colab environment while still leveraging the benefits of transfer learning.

Finally, visualizing model performance throughout the training process was essential. Beyond tracking standard metrics like loss and accuracy, we implemented a custom callback to generate and display the confusion matrix after each epoch. This visualization provided valuable insights into class-specific performance, highlighting which buildings the model struggled to differentiate and guiding further refinements to our approach. 

## Results
The confusion matrix for the full dataset is in the repo as FullDatasetMatrix.png and the confusion matrix for the cleaned dataset is CleanedDatasetMatrix.png. On the full dataset, we achieved 11237/12584, or roughly 89.3% accuracy. On the cleaned dataset, we achieved 10062/10193, or roughly 98.7% accuracy. 
The loss graph and the validation accuracy through epochs is also attached. The noticable dips in loss and validation accuracy were a result of stopping and restarting the model. The model ended with a final loss of 0.4132 and a validation accuracy of 0.9395.

## Testing Instructions

The model is designed to run optimally within a Google Colab environment. For best results, upload the model and dataset to your Google Drive and utilize the provided test_model notebook within Colab, selecting a TPU runtime. Ensure all file paths within the notebook are updated to reflect your Drive's directory structure. While local execution is possible, it's not recommended due to potential reproducibility issues. For local testing, replicate the test_model notebook's code and adjust paths accordingly; the requirements.txt file lists necessary dependencies for this approach. However, using Colab is strongly encouraged for consistent and reproducible results.

To install all dependencies:
```bash
pip install -r requirements.txt
```
```
tensorflow~=2.15.0
scikit-learn~=1.5.2
seaborn~=0.13.2
matplotlib~=3.8.0
```


## Model Download
- https://drive.google.com/file/d/1-AUcm_axuGYRgV79xWiUsgr61aXTv1jl/view?usp=sharing


