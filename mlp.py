# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:17:35 2023

@author: asen7
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from read_cifar import read_cifar, split_dataset


"""
N : number of input data
d_in :  input dimension
d_h : number of neurons in the hidden layer
d_out : output dimension (number of neurons of the output layer)

w1 : first layer weights (shape: (d_in,d_h))
b1 : first layer biaises (shape: (1, d_h))
w2 : second layer weights(shape: (d_h,d_out))
b2 : second layer biaises (shape: (1, d_out))"""

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1/ (1 + np.exp(-x))


def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    
    # Forward pass
    a0 = data                     # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1   # input of the hidden layer
    a1 = sigmoid(z1, deriv=False) # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2   # input of the output layer
    a2 = sigmoid(z2, deriv=False) # output of the output layer (sigmoid activation function)
    predictions = a2              # the predicted values are the outputs of the output layer

    # Compute loss (MSE)
    loss = np.mean(np.square(predictions - targets))
    
    # Backward pass
    N = data.shape[0]
    da2 = (2/N)*(predictions - targets)
    dz2 = da2*sigmoid(a2, deriv=True)
    
    dw2 = np.dot(a1.T, dz2) / N
    db2 = np.sum(dz2, axis=0, keepdims=True) / N
    
    da1 = np.dot(dz2, w2.T)
    dz1 = da1*sigmoid(a1, deriv=True)   

    dw1 = np.dot(a0.T, dz1) / N
    db1 = np.sum(dz1, axis=0, keepdims=True) / N
    
    w1 -= learning_rate*dw1
    w2 -= learning_rate*dw2
    b1 -= learning_rate*db1
    b2 -= learning_rate*db2

    return w1, b1, w2, b2, loss

#------------------------------------------------------------------------------------------------------------------------

def one_hot(labels):
    
    labels_encoded= np.zeros((labels.shape[0], np.unique(labels).shape[0]))   
    for i in range(labels.shape[0]):
        labels_encoded[i, labels[i].astype(int)]=1
        
    return labels_encoded

#------------------------------------------------------------------------------------------------------------------------

def softmax(x, deriv=False):
    
    if deriv:
        return x * (1 - x)
    else:
        e_x = np.exp(x - np.max(x)) 
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    


def learn_once_cross_entropy(w1, b1, w2, b2, data, targets, learning_rate):
    
    # Forward pass
    a0 = data                     # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1   # input of the hidden layer
    a1 = sigmoid(z1, deriv=False) # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2   # input of the output layer
    a2 = softmax(z2, deriv=False) # output of the output layer (softmax activation function)
    predictions = a2              # the predicted values are the outputs of the output layer
    
    targets = one_hot(targets)

    # Compute loss (Cross Entropy)
    N = data.shape[0]
    epsilon = 1e-12
    loss = -np.sum(targets * np.log(predictions + epsilon)) / N
    
    # Backward pass
    da2 = predictions - targets
    dz2 = da2*softmax(a2, deriv=True)
    
    dw2 = np.dot(a1.T, dz2) / N
    db2 = np.sum(dz2, axis=0, keepdims=True) / N
    
    da1 = np.dot(dz2, w2.T)
    dz1 = da1*sigmoid(a1, deriv=True)   

    dw1 = np.dot(a0.T, dz1) / N
    db1 = np.sum(dz1, axis=0, keepdims=True) / N
    
    w1 -= learning_rate*dw1
    w2 -= learning_rate*dw2
    b1 -= learning_rate*db1
    b2 -= learning_rate*db2

    return w1, b1, w2, b2, loss

#-------------------------------------------------------------------------------------------------------------------------

def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    
    train_accuracies=[]
    for i in range(num_epoch):
       w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
       
       a0 = data_train  
       z1 = np.matmul(a0, w1) + b1
       a1 = sigmoid(z1, deriv=False)
       z2 = np.matmul(a1, w2) + b2
       a2 = softmax(z2, deriv=False)
       predictions = np.argmax(a2, axis=1)
       
       accuracy = np.mean(predictions == labels_train)
       train_accuracies.append(accuracy)

       print(f"Epoch [{i + 1}/{num_epoch}] - Loss : {loss} - Accuracy : {accuracy}")

    return w1, b1, w2, b2, train_accuracies   
    
#---------------------------------------------------------------------------------------------------------------------------

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    
    a0 = data_test
    z1 = np.matmul(a0, w1) + b1
    a1 = sigmoid(z1, deriv=False)
    z2 = np.matmul(a1, w2) + b2
    a2 = softmax(z2, deriv=False)
    predictions = np.argmax(a2, axis=1)
    
    test_accuracy = np.mean(predictions == labels_test) 
    
    return test_accuracy
    
#-----------------------------------------------------------------------------------------------------------------------------    

def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    
    d_in = data_train.shape[1]  
    d_out = len(np.unique(labels_train)) 
    w1 = 2 * np.random.rand(d_in, d_h) - 1 
    b1 = np.zeros((1, d_h))  
    w2 = 2 * np.random.rand(d_h, d_out) - 1  
    b2 = np.zeros((1, d_out))  
    
    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)
    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)

    return train_accuracies, test_accuracy
    
#-----------------------------------------------------------------------------------------------------------------------------    
    
def ANN_plot(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    
    train_accuracies, test_accuracy = run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch)
   
    epochs=[epoch for epoch in range(num_epoch)]
    plt.plot(epochs, train_accuracies) 
    plt.xlabel('Epochs')
    plt.ylabel('Training accuracy')
    plt.title('The evolution of learning accuracy across learning epochs')
    results_directory = 'C:/Users/asen7/OneDrive/Documents/3A/MOD/DL/TD1/deep-learning-image-classification/results'
    save = os.path.join(results_directory, 'mlp.png')
    plt.savefig(save)
    print(test_accuracy)

#-----------------------------------------------------------------------------------------------------------------------------    
    
if __name__ == "__main__":
    
    split_factor=0.9
    path="C:/Users/asen7/OneDrive/Documents/3A/MOD/DL/TD1/deep-learning-image-classification/data"
    data, labels = read_cifar(path)
    data_train, data_test, labels_train, labels_test = split_dataset(data, labels, split_factor)

    num_epoch=100
    d_h=64
    learning_rate=0.1
    ANN_plot(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch)
    
    
    