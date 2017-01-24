import numpy as np

def computeCost(X, y, theta):
    return 0.5 * np.sum((X.dot(theta) - y)**2) / y.size