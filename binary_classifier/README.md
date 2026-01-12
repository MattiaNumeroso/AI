# Binary Classifier for Breast Cancer Detection

This project implements a neural network using PyTorch to perform binary classification on the Breast Cancer Wisconsin dataset.

## Description
The notebook includes the following steps:
* Data loading and exploration using scikit-learn and Pandas.
* Data normalization using StandardScaler (mean = 0, std = 1).
* Dataset splitting into training and test sets.
* Definition of a feed-forward neural network architecture.
* Model training using the Adam optimizer and Binary Cross Entropy (BCE) loss.
* Performance evaluation on the test set.

## Model Architecture
The neural network (`classifierNN`) consists of:
* **Input Layer**: 30 features.
* **Hidden Layer**: 15 neurons with Sigmoid activation.
* **Output Layer**: 1 neuron with Sigmoid activation for binary probability.

## Results
Based on the notebook execution, the model achieves the following metrics on the test set:
* **Accuracy**: ~98%
* **Precision**: ~97%
* **Recall**: 100%
* **F1-score**: ~98%
