"""Image Recognition using an Artificial Neural Network

    Note: We use numpy instead of pandas as there is no need for labels
         Since pixels will be unrolled, all we need is a matrix functionality
         Therefore, numpy will allow a ML system to learn faster than in pandas

    Python 3.7
"""

import numpy as np
import scipy.misc as sp
import numpy.linalg as alg
import os
import sys

#Clearing interpreter
os.system('cls')

#Function definitions
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def random_init(input_layer_size, hidden_layer_size, episilon = 0.12):
    """Will initialize random parameters for each unit in each layer to
        ensure high likelyhood of algorithm converging
    """
    rando = np.random.rand(input_layer_size, 1 + hidden_layer_size)
    return (rando * 2 * episilon) - episilon

def is_img(image):
    """Checks if file is an image"""
    formats = ['jpg', 'gif', 'jpeg', 'png', 'bmp']
    file_type = image.split('.')
    return None if file_type[-1].lower() not in formats else image

#ML implementation starts here
try:
    img_path = input("Folder Path: ")
    files = os.listdir(img_path)
except:
    print("Path not found.")
    print("Exiting.")
    sys.exit()

#ANN dimensions
temp = sp.imread(img_path + '\\' + files[1], flatten = True)
input_layer_size = len(temp.ravel())            #I know, it's a shitty fix
hidden_layer_size = 3
img_list = []

for item in files:
    check = is_img(item)
    if not check is None:
        img_list.append(check)

del temp                                   #Clear memory
final_array = np.zeros([len(img_list), input_layer_size])

#Data/Image processing
i = 0                                      #So we can change each row in the matrix

for img in img_list:
    #Using scipy instead of PIL Image library because we want to flatten easier
    file_ = sp.imread(img_path + '\\' + img, flatten = True)
    file_ = file_.ravel()
    final_array[i] = file_
    i = i + 1
#End Data/Image Processing

#Algorithm Implementation

#print(len(final_array))
