# activation function 
# gradiante of the activation function 
# add bias units 
# feed forward 
# encoding labels 
   ## when we load the data, the traning labels comes just an integer or numerical representation of 0-9
   ## since we want to perform multi-class classification, to do that we want column vector  
# perform model learning 
# calculate the cost function
# gradiante with respect to the weights 
# initialize those weights 


import struct
import numpy as np
import matplotlib.pyplot as plt 
import os
from scipy.special import expit

# lets load the data from the mnist dataset
def load_data():
    with open('train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('train-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_imgs = np.fromfile(imgs, dtype=np.uint8).reshape(num,784)
    with open('t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('t10k-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_imgs = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    return train_imgs, train_labels, test_imgs, test_labels

# data visualization
def visualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(1):
        img = img_array[label_array==8][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()

#train_x, train_y, test_x, test_y = load_data()
#visualize_data(train_x, train_y)


# encoding labels 
   ## when we load the data, the traning labels comes just an integer or numerical representation of 0-9
   ## since we want to perform multi-class classification, to do that we want column vector  
# the output of the following function will be a vector, simply we change the labels into vectors 
def enc_one_hot(y,num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    for i, val in enumerate(y):
        one_hot[val,i] = 1.0
    return one_hot

#lets if the above function works 
#y = np.array([4 ,5 , 9 ,0])
#z = enc_one_hot(y)
#print(y)
#print(z)

# activation function
def sigmoid(z):
    return (1/(1+np.exp(-z)))
    #return expit(z)

def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)

def visualize_sigmoid():
    x = np.arange(-10,10,0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

#visualize_sigmoid()

def calc_cost(y_enc, output):
    t1 = -y_enc * np.log(output)
    t2 = (1 - y_enc) * np.log(1 - output)
    cost = np.sum(t1 - t2)
    return cost

def add_bias_unit(x, where):
    # where us just row or column

    if where == 'column':
        x_new = np.ones((x.shape[0], x.shape[1] + 1))
        x_new[:,1:] = x
    elif where == 'row':
        x_new = np.ones((x.shape[0] + 1, x.shape[1]))
        x_new[1:,:] = x
    return x_new

    
