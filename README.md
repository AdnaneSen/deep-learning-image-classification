# Deep Learning Image Classification



## Description

The CIFAR-10 is a commonly used dataset in computer vision for image classification tasks, it contains 60,000 32x32 color images spread accross 10 different classes which are : 
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck.

The objective of this project is to write a complete image classification program in Python. Two classification models will be successively developed and tested: k-nearest neighbors (KNN) and neural networks (NN).

## Details about the files:

### data :
It contains six files: five data batches and one test batch. There is also an HTML file named 'readme' containing a description of the dataset and some links to download it.

### read_cifar.py :
It contains the code for three functions (with their descriptions):
 - 'read_cifar_batch'
 - 'read_cifar'
 - 'split_dataset' 

### knn.py :
It contains the code for four functions (with their descriptions):
 - 'distance_matrix'
 - 'knn_predict'
 - 'evaluate_knn'
 - 'KNN_plot'

### mlp.py :
It contains the code for multiple functions related to training and testing multilayer perceptron (MLP) neural network for the image classification task:
 - Activation functions: when deriv=True, the function returns its derivative.
      - 'sigmoid(x, deriv=False)': Sigmoid activation function.
      - 'softmax(x, deriv=False)': Softmax activation function.
 - One-Hot Encoding: 
      - 'one_hot': Converts integer labels to one-hot encoded vectors.
 - Training Function:
      - 'learn_once_mse': Updates weights and biases using Mean Squared Error loss.
      - 'learn_once_cross_entropy': Updates weights and biases using Cross-Entropy loss.
 - Training MLP: 
      - 'train_mlp': Trains the MLP for a specified number of epochs and returns the training accuracies.
 - Testing MLP:
      - 'test_mlp': Tests the MLP on the testing dataset and returns the accuracy.
 - Complete Training and Testing:
      - 'run_mlp_training': Runs the complete training and testing process for the MLP.
 - Plotting:
      - 'ANN_plot': Plots the evolution of the learning accuracy across learning epochs and saves the plot figure.

### results :
This file contains two PNG images:
 - 'knn.png': Illustrates the variation of the KNN accuracy as a function of the number of neighbors parameter k.
 - 'mlp.png': Represents the evolution of the learning accuracy across learning epochs.

### tests :
This directory contains all unit tests corresponding to each one of the main function described previously.

# Comment on the results:
- The k-nearest neighbors algorithm reaches for the optimal parameter k an accuracy of about 36%.
- The ANN learning accuracy stabilize at about 11% with a close value as a test accuracy. 

So, the current results highlight the need for model refinement, hyperparameter tuning, and potentially the exploration of more advanced architectures (such as convolutional neural networks which are really effective on image classification tasks) to achieve more satisfying performance.

While the quantitative results may not be satisfactory, the process of developing these algorithms from scratch provides a valuable opportunity for gaining a deep understanding of the inner workings of machine learning and neural network algorithms.

# Requirements : 
- Python version: 3.9
- Python libraries used: Numpy, Pickle, Os, Matplotlib, Collections, Pytest.

# Author:
Adnane Sennoune.



