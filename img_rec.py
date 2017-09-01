"""Image Recognition using an Artificial Neural Network"""

"""Note: We use numpy instead of pandas as there is no need for labels
         Since pixels will be unrolled, all we need is a matrix functionality
         Therefore, numpy will allow a ML system to learn faster than in pandas
"""

import numpy as np
import scipy.misc as sp
import numpy.linalg as alg
import os

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def random_init(input_layer_size, hidden_layer_size, episilon = 0.12):
    """Will randomly initialize parameters for each unit in each layer to
        ensure high likelyhood of algorithm converging
    """
    zeroes = np.zeros(input_layer_size, 1 + hidden_layer_size)
    zeroes =  

def img_process(path, image_name):
    """Will check whether file is an image file
        Convert image into numpy array if file has an image type
    """
    formats = ['jpg', 'gif', 'jpeg', 'png', 'bmp']
    file_type = image_name.split('.')
    return None if file_type[-1].lower() not in formats else sp.imread(path + '\\' + image_name, flatten = True)

def unroll(img_array):
    """Function will unroll a numpy array into a single vector
    """
    return img_array.ravel()



print(sigmoid_grad(0))