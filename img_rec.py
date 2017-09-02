"""Image Recognition using an Artificial Neural Network

    Note: We use numpy instead of pandas as there is no need for labels
         Since pixels will be unrolled, all we need is a matrix functionality
         Therefore, numpy will allow a ML system to learn faster than in pandas

    Python 3.7
    OpenCV not supported in this version of python
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

def img_process(path, image_name):
    """Will check whether file is an image file
        Convert image into numpy array if file has an image type
    """
    formats = ['jpg', 'gif', 'jpeg', 'png', 'bmp']
    file_type = image_name.split('.')
    return None if file_type[-1].lower() not in formats else sp.imread(path + '\\' + image_name, flatten = True)

#ML implementation starts here
try:
    img_path = input("Folder Path: ")
    files = os.listdir(img_path)
except:
    print("Path not found.")
    print("Exiting.")
    sys.exit()

#ANN dimensions
temp = img_process(img_path, files[1])
input_layer_size = len(temp.ravel())            #I know, it's a shitty fix
hidden_layer_size = 3

del temp                                   #Clear memory
final_matrix = np.empty([input_layer_size, len(files)])

i = 0                                      #So we can change each row in the matrix

for img in files:
    i = i + 1
    file_ = img_process(img_path, img)
    if file_ is None:
        continue

    file_ = file_.ravel()
    np.vstack([final_matrix, file_]) #Need to figure out a way to append to matrix
    """Need to test this loop"""

#need to unroll each image and append one by one to a single numpy array
print(final_matrix)
