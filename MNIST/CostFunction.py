"""MNIST Cost Function"""

def cost_func(set_matrix, y, lamb, theta1, theta2):

    Z_2 = np.dot(set_matrix, theta1.T)
    A_2 = sigmoid(Z_2)

    A_2 = np.insert(A_2, 0, np.ones(len(set_matrix)), 1)          #ones
    Z_3 = np.dot(A_2, theta2.T)
    prediction = sigmoid(Z_3)

    J = (1/m) * sum(sum((-y * log(prediction)) - ((1 - y) * log(1 - prediction))))
    t1_fix = np.delete(theta1, 0, axis = 1)
    t2_fix = np.delete(theta2, 0, axis = 1)
    return J + (lamb / (2 * m)) * (sum(sum(np.square(t1_fix))) + sum(sum(np.square(t2_fix))))
