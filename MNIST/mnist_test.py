"""Image Recognition using an Artificial Neural Network (Forward Prop/Back Prop)

    Note: We use numpy instead of pandas as there is no need for labels
         Since pixels will be unrolled, all we need is a matrix functionality
         Therefore, numpy will allow a ML system to learn faster than in pandas

    Python 3.7
    First version on MNIST dataset to test algorithm
    Multiclass classification

    1 hidden layer neural network

    MNIST has 60,000 training samples, 10,000 test samples
    Each image has 28x28 pixels
"""

import numpy as np
import CostFunction as cf
import struct, gzip, os, sys

#Clearing interpreter
os.system('cls')

#Function definitions
def random_init(input_layer_size, hidden_layer_size, episilon = 0.12):
    """Will initialize random parameters for each unit in each layer to
        ensure high likelyhood of algorithm converging
    """
    rando = np.random.rand(hidden_layer_size, 1 + input_layer_size)
    return (rando * 2 * episilon) - episilon

def read_idx(filename):
    """
    From TylerNeylon:
    https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

#ML Implementation
#Preprocessing Start
input_layer_size = 784
hidden_layer_size = 25

train_set_array = np.zeros([60000, input_layer_size])
test_set_array = np.zeros([10000, input_layer_size])
#Preprocessing End

#Data Processing Start
train_set = read_idx("train-images.idx3-ubyte") #shape = (60000, 28, 28)
train_labels = read_idx("train-labels.idx1-ubyte") #shape = (60000,)
test_set = read_idx("t10k-images.idx3-ubyte") #shape = (10000, 28, 28)
test_labels = read_idx("t10k-labels.idx1-ubyte") #shape = (10000,)

for i in range(len(train_set)):
    unrolled = train_set[i].ravel()         #unrolled pixels for all images into a vector using np.ravel()
    train_set_array[i] = unrolled                 #train_set_array now has shape of (60000, 784)
    if i >= len(test_set):
        continue
    else:
        unrolled2 = test_set[i].ravel()
        test_set_array[i] = unrolled2              #test_set_array has shape of (10000, 784)

#Data Processing End

#Algorithm Start
#Forward Prop
#Parameter Setup
Theta1 = random_init(input_layer_size, hidden_layer_size)
Theta2 = random_init(hidden_layer_size, 10)         #x10 because we have 10 classes to tag

train_set_array = np.insert(train_set_array, 0, np.ones(len(train_set_array)), 1)   #added ones for bias unit
test_set_array = np.insert(test_set_array, 0, np.ones(len(test_set_array)), 1)     #added ones for bias unit
#Parameter Setup End
#ANN
train_matrix = np.asmatrix(train_set_array)
test_matrix = np.asmatrix(test_set_array)

J = cf.cost_func(train_matrix, train_labels, lamb = 0, theta1 = Theta1, theta2 = Theta2)

print J

#Algorithm End
