# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:26:49 2023

@author: asen7
"""

import numpy as np
import pickle
import os


def read_cifar_batch(path):
    """Reads a CIFAR batch and extracts data and labels.

    Args:
    path (string): Path to the CIFAR batch file to be read.

    Returns:
    data_matrix (numpy array): A 2D array containing image data.
    labels_vector (numpy array): A 1D array containing corresponding labels."""
    
    with open(path,'rb') as file:
        batch=pickle.load(file,encoding='bytes')
    
    data=batch[b'data']
    labels=batch[b'labels']
    
    data_matrix=np.array(data, dtype=np.float32)
    labels_vector=np.array(labels, dtype=np.int64)
    
    return data_matrix, labels_vector

#-----------------------------------------------------------------------------------------------------------------------

def read_cifar(origin_path):
    """Reads all the CIFAR batches from the given directory and concatenate their data and labels.

    Args:
    origin_path (string): Path to the directory containing the 6 CIFAR batch files.

    Returns:
    data (numpy array): A 2D array containing concatenated image data from all batches.
    labels (numpy array): A 1D array containing concatenated image labels from all batches."""
    
    data_batches = []
    label_batches = []

    for batch_file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        batch_path = os.path.join(origin_path, batch_file)
        data, labels = read_cifar_batch(batch_path)
        data_batches.append(data)
        label_batches.append(labels)

    data = np.concatenate(data_batches, axis=0, dtype=np.float32)
    labels = np.concatenate(label_batches, axis=0, dtype=np.int64)

    return data, labels
   
#------------------------------------------------------------------------------------------------------------------------    


def split_dataset(data, labels, split):
    """Splits a dataset into training and testing sets based on a specified split ratio.

    Args:
    data (numpy array): A 2D array containing the dataset.
    labels (numpy array): A 1D array containing the corresponding labels for the data.
    split (float): The ratio of the data to be used for training (between 0 and 1).

    Returns:
    training_data (numpy array): A 2D array containing training data.
    testing_data (numpy array): A 2D array containing testing data.
    training_labels (numpy array): A 1D array containing labels for the training data.
    testing_labels (numpy array): A 1D array containing labels for the testing data."""
    
    first_dim=len(labels)
    training_size=int(first_dim*split)
    
    labels=labels.reshape(data.shape[0],1)
    stack=np.hstack((data, labels))
    np.random.shuffle(stack)
    
    training_data= stack[:training_size,:-1]
    training_labels=np.array(stack[:training_size,-1])
    
    testing_data=stack[training_size:,:-1]
    testing_labels= np.array(stack[training_size:,-1])
    
    return training_data, testing_data, training_labels, testing_labels
  
#------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    path="C:/Users/asen7/OneDrive/Documents/3A/MOD/DL/TD1/deep-learning-image-classification/data"
    data, labels=read_cifar(path)
    
    split = 0.8
    training_data, testing_data, training_labels, testing_labels = split_dataset(data, labels, split)
    print("Training Data Shape:", training_data.shape)
    print("Testing Data Shape:", testing_data.shape)
    print("Training Labels Shape:", training_labels.shape)
    print("Testing Labels Shape:", testing_labels.shape)

    print("Values type in data:", data.dtype)
    print("Values type in labels:", labels.dtype)
