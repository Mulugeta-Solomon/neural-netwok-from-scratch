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
    #return (1/(1+np.exp(-z)))
    return expit(z)

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

#bias unit
def add_bias_unit(x, where):
    # where: is just row or column

    if where == 'column':
        x_new = np.ones((x.shape[0], x.shape[1] + 1))
        x_new[:,1:] = x
    elif where == 'row':
        x_new = np.ones((x.shape[0] + 1, x.shape[1]))
        x_new[1:,:] = x
    return x_new

#weights for every layer in this case three
def init_weights(n_features, n_hidden, n_output):
    w1 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_features + 1))
    w1 = w1.reshape(n_hidden,n_features + 1)
    w2 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_hidden +1))
    w2 = w2.reshape(n_hidden, n_hidden + 1)
    w3 = np.random.uniform(-1.0, 1.0, size=n_output*(n_hidden + 1))
    w3 = w3.reshape(n_hidden, n_output + 1)
    return w1, w2, w3

# forward propagation 
def feed_forward(x, w1, w2, w3):
    #add bias unit to the input
    #column within the row is just a byte of data
    #so we need to add a column vector of ones

    a1 = add_bias_unit(x, where='column')
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)

    #since we have transposed we have to add the bias unit to the row
    a2 = add_bias_unit(a2, where='row')
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)
    a3 = add_bias_unit(z3, where='row')
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4

def predict(x, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4 = feed_forward(x, w1, w2, w3)
    y_predict = np.argmax(a4, axis=0)
    return y_predict

# Backprobagation 
def calc_gradient(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    delta4 = a4 - y_enc
    z3 = add_bias_unit(z3, where='row')
    delta3 = w3.T.dot(delta4*sigmoid_gradient(z3))
    delta3 = delta3[1:, :]
    z2 = add_bias_unit(z2, where='row')
    delta2 = w2.T.dot(delta3*sigmoid_gradient(z2))
    delta2 = delta2[1:, :]

    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)

    return grad1, grad2, grad3
