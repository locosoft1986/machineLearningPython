import numpy as np
from computeCost import *

def gradientDescent(X, y, theta, alpha, numIters):
    m = y.size
    JHistory = np.zeros((numIters, 1))
    XData = np.array([X[:, 1]]).T

    for iter in range(numIters):
        k = np.sum(X.dot(theta) - y)
        g = np.sum((X.dot(theta) - y) * XData)
        theta[0] -= alpha / m * k
        theta[1] -= alpha / m * g
        JHistory[iter] = computeCost(X, y, theta)

    return theta, JHistory
