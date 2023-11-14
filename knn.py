# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:14:10 2023

@author: asen7
"""

import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from read_cifar import read_cifar, split_dataset


def distance_matrix(A, B):
    """Computes the L2 Euclidean distance matrix between two sets of data points.

    Args:
    A (numpy array): A 2D array containing the first set of data points.
    B (numpy array): A 2D array containing the second set of data points.

    Returns:
    dists (numpy array): A 2D array representing L2 Euclidean distance between data points in sets A and B."""
    
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True).reshape(1, B.shape[0])
    AB = A.dot(B.T)
    dists = np.sqrt(A2 - 2 * AB + B2)

    return dists

#------------------------------------------------------------------------------------------------------------------------

def knn_predict(dists, labels_train, k):
    """Predicts labels for test data points using k-nearest neighbors.

    Args:
    dists (numpy array): A 2D array of distances between test data points and training data points.
    labels_train (numpy array): A 1D array containing the labels of the training data.
    k (integer): The number of nearest neighbors to consider for the prediction.

    Returns:
    y_pred (numpy array): A 1D array containing predicted labels for the testing data."""
    
    test_dim=dists.shape[1]
    y_pred = np.zeros(test_dim, dtype=labels_train.dtype)
    for i in range(test_dim):    
       d=dists[:,i]
       index=np.argpartition(d,k)
       index=index[:k]
       knn_labels=[labels_train[j] for j in index]
       
       best_label, _= Counter(knn_labels).most_common(1)[0]
       y_pred[i]=best_label
       
    return y_pred
       
#--------------------------------------------------------------------------------------------------------------------------

def evaluate_knn(data_train,labels_train,data_test,labels_test,k):
    """Evaluate the accuracy of k-nearest neighbors classification on testing data.

    Args:
    data_train (numpy array): A 2D array containing the training data.
    labels_train (numpy array): A 1D array containing the  training labels.
    data_test (numpy array): A 2D array containing the testing data.
    labels_test (numpy array): A 1D array containing true labels for the testing data.
    k (integer): The number of nearest neighbors to consider for the prediction.

    Returns:
    accuracy (float): The accuracy of k-nearest neighbors classification on the testing data.""" 
    
    dists=distance_matrix(data_train, data_test)
    labels_knn= knn_predict(dists, labels_train, k)
    count=0
    for i in range(len(labels_knn)):
        if labels_knn[i]==labels_test[i]:
            count+=1
        
    accuracy=count/(len(labels_knn))
    return accuracy
  
#----------------------------------------------------------------------------------------------------------------------------  
   
def KNN_plot(data_train,labels_train,data_test,labels_test,k):
   """Plots the variation of the accuracy as a function of k (from 1 to 20). 
   Then saves the plot as an image named 'knn.png' in the results directory"""
   
   num_k=[x for x in range(1,k+1)]
   accuracy=[evaluate_knn(data_train,labels_train,data_test,labels_test,k) for k in num_k]
   plt.plot(num_k,accuracy, marker='o')
   plt.xticks(num_k)
   plt.xlabel('Number of neighbours')
   plt.ylabel('Accuracy')
   plt.title('The variation of the KNN accuracy as a function of k')
   results_directory = 'C:/Users/asen7/OneDrive/Documents/3A/MOD/DL/TD1/deep-learning-image-classification/results'
   save = os.path.join(results_directory, 'knn.png')
   plt.savefig(save)
   
#---------------------------------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([[7, 8], [9, 10]])
    dists = distance_matrix(A, B)
    print(dists)
    
    split_factor=0.9
    path="C:/Users/asen7/OneDrive/Documents/3A/MOD/DL/TD1/deep-learning-image-classification/data"
    data, labels=read_cifar(path)
    data_train,data_test,labels_train,labels_test=split_dataset(data, labels, split_factor)
    print(data_train.shape)
    print(data_test.shape)

    k=20
    KNN_plot(data_train,labels_train,data_test,labels_test,k)