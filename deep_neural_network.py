# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:55:39 2018

@author: RachnaGupta
"""

#import time
import numpy as np
import pandas as pd
#import h5py
import matplotlib.pyplot as plt
#import scipy
#from PIL import Image
#from scipy import ndimage
#from dnn_app_utils_v3 import *


train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

#print(train_data.head())
#print(test_data.head())

# Read it only once, unless restarting kernel, or variables have been cleared

train_data = np.loadtxt("mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv", 
                       delimiter=",") 

no_of_different_labels=10
fac = 255  *0.99 + 0.01

train_imgs = np.asfarray(train_data[:, 1:]) / fac
test_imgs = np.asfarray(test_data[:, 1:]) / fac
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

#train_imgs=train_imgs.T
#test_imgs=test_imgs.T
#train_labels=train_labels.T
#test_labels=test_labels.T

print("train_imgs.shape", train_imgs.shape)
print("test_imgs.shape", test_imgs.shape)

image_size = 28 
image_pixels = image_size * image_size


lr = np.arange(no_of_different_labels)
# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99



n_x=image_pixels
n_h=100
n_y=10
layers_dims=(n_x, n_h, n_y)
costs=[] 

#for i in range(10):
#    img = train_imgs[i].reshape((28,28))
#    plt.imshow(img, cmap="Greys")
#    plt.show()

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1= np.random.rand(n_h, n_x)*0.01
    b1=np.zeros([n_h, 1])
    W2=np.random.randn(n_y, n_h)*0.01
    b2=np.zeros([n_y, 1])
    
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    
    return parameters
    

def relu(Z):
    print("relu Z.shape", Z.shape)
    if Z.any()>0:
        value=Z
    else:
        value=0
    #print(value)
    cache=(Z, value)
    print("relu value.shape", value.shape)
    return value, cache


def sigmoid(Z):
    print("sigmoid forward Z.shape", Z.shape)
    value= 1/(1+np.power(np.e, -Z))
    cache=(Z, value)
    print("sigmoid forward value.shape", value.shape)
    return value, cache


def linear_forward(A, W, b):
    print((W.shape, A.shape))
    Z=np.dot(W, A)+b
    print("Z.shape in linear forward:",Z.shape)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache=(A, W, b)
    
    return Z, cache
    
    
def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=sigmoid(Z)
        
    if activation == 'relu':
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=relu(Z)
    
    assert(A.shape == (W.shape[0], A.shape[1]))
    cache=(linear_cache, activation_cache)
    
    return A, cache
    

def compute_cost(A, y):
    m=y.shape[1]
    
    cost = -np.sum((y*np.log(A)) +((1-y)*np.log(1-A))) / m
    cost=np.squeeze(cost)
    assert (cost.shape == ())
    
    return cost
    

def relu_backward(dA, cache):
    A, x=cache
    dZ=np.zeros(A.shape)
    for i in range(A.shape[0]):
        if A[i]>0:
            dZ[i]=1
        else:
            pass
        
#    print('A.shape and dZ.shape in relu bckwd', A.shape, dZ.shape)
    return dZ
    

def sigmoid_backward(dA, cache):
    A, x=cache
    #A=np.squeeze(A)
#    print('A.shape in sigmoid bckwd', A.shape)
    return A*(1.0-A)
    
    
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m= A_prev.shape[1]
#    print('A_prev.shape in linear backward: ', A_prev.shape)
#    print('dZ.shape in linear backward: ', dZ.shape)
    dW = np.dot(dZ, A_prev.T)/m
#    print('dW.shape in linear backward: ', dW.shape)
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

    
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache=cache
    
    if activation =='relu':
#        print('dA.shape in linear bckwrd for relu', dA.shape)
        dZ = relu_backward(dA, activation_cache)
#        print('dZ.shape in linear bckwrd for relu: ', dZ.shape)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
#        print('dA.shape in linear bckwrd for sigmoid', dA.shape)
        dZ = sigmoid_backward(dA, activation_cache)
#        print('dZ.shape in linear bckwrd for sigmoid: ', dZ.shape)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA, dW, db


def update_parameters(parameters, grads, learning_rate):
    
    parameters["W1"]=parameters["W1"]-(learning_rate*grads["dW1"])
    parameters["W2"]=parameters["W2"]-(learning_rate*grads["dW2"])
    parameters["b1"]=parameters["b1"]-(learning_rate*grads["db1"])
    parameters["b2"]=parameters["b2"]-(learning_rate*grads["db2"])
        
    return parameters
    

def two_layer_network(X, y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
#    print('Hi')
    grads={}
#    costs=[]
#    m=X.shape[1]
    (n_x, n_h, n_y)=layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    
    for i in range(0, num_iterations):
        A1, cache1=linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2=linear_activation_forward(A1, W2, b2, activation='sigmoid')
        
        cost = compute_cost(A2, y)
        
        dA2= -(np.divide(y, A2) - np.divide(1-y, 1-A2))
        
        dA1, dW2, db2=linear_activation_backward(dA2, cache2, activation='sigmoid')
        dA0, dW1, db1=linear_activation_backward(dA1, cache1, activation='relu')
        
        grads['dW1']=dW1
        grads['dW2']=dW2
        grads['db1']=db1
        grads['db2']=db2
        
        parameters=update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)
        
        W1=parameters["W1"]
        b1=parameters["b1"]
        W2=parameters["W2"]
        b2=parameters["b2"]
        
        
        # Print the cost every 100 training example
        if print_cost and i%50 == 0:
            print("cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    
#    plt.plot(costs)
#    plt.xlabel('Costs')
#    plt.ylabel('100th Iteration')
#    plt.show()
    
    return parameters
    
    
def predict(X, parameters):
    
    A1, cache1=linear_activation_forward(X, parameters["W1"], parameters["b1"], activation='relu')
    A2, cache2=linear_activation_forward(A1, parameters["W2"], parameters["b2"], activation='sigmoid')
    print('A2 from predict: ', A2.shape)
    return A2


#for i in range(train_imgs.shape[0]):
  
for i in range(1000):
    X=train_imgs[i].reshape(train_imgs.shape[1],1)
    Y=train_labels_one_hot[i].reshape(1, train_labels_one_hot.shape[1])
    parameters = two_layer_network(X, Y, layers_dims=(n_x, n_h, n_y),num_iterations=10, print_cost=True)

#print("*"*10, "\n\n\nparameters::", parameters)
for i in range(1000, 1010):
    X=train_imgs[i].reshape(train_imgs.shape[1],1)
    
    predictions_train = predict(X, parameters)
    print(np.argmax(predictions_train), '  ', train_labels[i],'  ',  np.max(predictions_train))
    

plt.plot(costs)
plt.xlabel('Costs')
plt.ylabel('100th Iteration')
plt.show()