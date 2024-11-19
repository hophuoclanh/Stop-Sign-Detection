# Transfer Learning with Convolutional Neural Networks for Classification

**Version:** 0.2  
**Project Name:** Final_project_stop_signs  
**Training Run:** Transfer Learning for Stop Sign Classification  

---

## Overview

This project demonstrates how to use transfer learning with a Convolutional Neural Network (CNN) for image classification. The goal is to classify images into two categories: "Stop Sign" and "Not Stop Sign." By leveraging the power of pre-trained models, we train the output layer of a ResNet-18 model using a custom dataset. The dataset is annotated using [CV Studio](https://www.skills.network/).

---

## Objectives

- Train a state-of-the-art image classifier using transfer learning.  
- Preprocess and augment the dataset for better model generalization.  
- Experiment with different hyperparameters to improve accuracy.  
- Utilize CV Studio for collaborative data management and reporting.  

---

## Project Workflow

### 1. **Data Preparation**
- The image dataset is downloaded directly from CV Studio.
- 90% of the data is used for training, and 10% for validation.
- Data preprocessing includes resizing images to `(224, 224)`, normalization, and data augmentation (e.g., horizontal flipping, slight rotations).

### 2. **Model Architecture**
- **Base Model:** ResNet-18 pre-trained on ImageNet.  
- The output layer of the model (`fc`) is replaced with a `nn.Linear` layer to classify 2 categories (`Stop Sign`, `Not Stop Sign`).  
- Feature extraction is enabled by freezing the pre-trained layers (`requires_grad=False`).  

### 3. **Training**
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum.  
- **Learning Rate Scheduler:** CyclicLR with a triangular2 policy for adaptive learning rates.  
- Hyperparameters:
  - Batch size: `32`
  - Learning rate: `1e-6` (base), `1e-2` (max)
  - Momentum: `0.9`
  - Epochs: `10`
  
### 4. **Evaluation**
- Validation accuracy and loss are recorded after every epoch.
- Final model achieves a validation accuracy of **100%** on several epochs.

### 5. **Visualization**
- Loss and accuracy trends are plotted for analysis.  
- Sample predictions are displayed with corresponding images.

### 6. **Results**
- Training time: ~4 minutes per epoch (depending on dataset size).  
- Best validation accuracy: **100%**

## Tools and Technologies
- Frameworks: PyTorch, torchvision
- Data Annotation: CV Studio
- Visualization: Matplotlib, PIL
- Cloud Storage: IBM Cloud Object Storage

## Acknowledgments
- CV Studio: For providing an efficient and collaborative annotation tool.
- Skills Network Labs: For the pre-built tools and datasets.
- PyTorch Community: For the ResNet pre-trained model.
