"""MNIST Cost Function"""
import numpy as np

def sigmoid(z):
    z = np.clip(z, -300, 300)           #prevents overflow error
    return 1/(1 + np.exp(-1.0 * z))

def cost_func(set_matrix, y, lamb, theta1, theta2):
    """1 layer neural network"""
    m = len(set_matrix)

    Z_2 = np.dot(set_matrix, theta1.T)
    A_2 = sigmoid(Z_2)

    A_2 = np.insert(A_2, 0, np.ones(len(set_matrix)), 1)          #ones
    Z_3 = np.dot(A_2, theta2.T)
    prediction = sigmoid(Z_3)

    err = (-y * np.log(prediction)) - ((1 - y) * np.log(1 - prediction))
    J = (float(1)/ m) * err.sum()

    t1_fix = np.delete(theta1, 0, axis = 1)
    t2_fix = np.delete(theta2, 0, axis = 1)
    return J + ((lamb / (2 * m)) * (t1_fix.sum() + t2_fix.sum()))
