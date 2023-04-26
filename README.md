# Two Layer Neural Network CIFAR-10 Classification

This repository contains an implementation of a two-layer neural network for classifying images in the CIFAR-10 dataset. The main goal of this project is to gain experience with implementing and training a neural network from scratch. It also provides a simple example of using gradient checking to validate the correctness of the backward pass implementation.

## Background

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The dataset is divided into five training batches and one test batch, each with 10,000 images. The test batch contains exactly 1,000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5,000 images from each class.

## Implementation

The neural network is implemented in the network.py file. It is a simple two-layer neural network with one hidden layer. The activation function used in the hidden layer is the ReLU (Rectified Linear Unit) function. The output layer uses the softmax function to convert the scores to probabilities.

## The code provided in this repository:

* Initializes a small toy neural network and toy data to check the correctness of the forward and backward passes.
* Trains the neural network on the CIFAR-10 dataset.
* Visualizes the learned weights for each class.

## Key Code Considerations

* The code uses the numpy library for efficient numerical computations and the matplotlib library for visualizing the learned weights and loss history.
* The Network class in network.py is used to represent the two-layer neural network, including the forward and backward pass, and the training procedure.
* The eval_numerical_gradient function from the gradient_check.py file is used to numerically check the gradients of the loss function with respect to the network parameters. This is useful for validating the correctness of the backward pass implementation.
* The load_CIFAR10 function from data_utils.py is used to load the CIFAR-10 dataset from disk and preprocess it for training the neural network.
* The visualize_grid function from vis_utils.py is used to visualize the learned weights for each class.

## Analyzing the Model Performance

From the visualizations provided by the code, we can observe that the loss is decreasing more or less linearly. This could indicate that the learning rate might be too low. Moreover, there is no noticeable gap between the training and validation accuracy, suggesting that the model's capacity might be low. In this case, we can consider increasing the model's size. However, we need to be cautious with very large models as they may lead to overfitting, which manifests as a significant gap between the training and validation accuracy.

## Tuning Hyperparameters

One of the crucial aspects of using Neural Networks is tuning hyperparameters and understanding how they impact the final performance. Experimenting with different values for various hyperparameters, including the hidden layer size, learning rate, the number of training epochs, and regularization strength can help in model performance. Additionally, adjusting the learning rate decay may be provide some additional model performance uplift, although using the default value should still yield satisfactory performance.

## Target Results

The goal is to aim to achieve a classification accuracy of greater than 48% on the validation set. With optimal hyperparameter tuning, it is possible to achieve over 52% accuracy on the validation set. 

## Usage

To run the code, simply execute the provided Jupyter or Google Colab notebook. The notebook provides some comments around the main steps in the process, from loading the data and training the network to visualizing the results.
