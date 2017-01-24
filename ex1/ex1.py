from warmUpExercise import *
from plotData import *
from computeCost import *
from gradientDescent import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

def pause():
    programPause = input("Press the <ENTER> key to continue...")


# ==================== Part 1: Basic Function ====================
print("Running warmUpExercise...")
print("5x5 Identity Matrix: ")

print(warmUpExercise())
# pause()


# ======================= Part 2: Plotting =======================
print("Plotting Data...")

data = np.loadtxt("ex1data1.txt", delimiter=',')
X = np.array([data[:, 0]]).T
y = np.array([data[:, 1]]).T
m = y.size

plotData(X, y)
plt.show()
# pause()

# =================== Part 3: Gradient descent ===================
print("Running Gradient Descent ...")
X = np.hstack((np.ones((m, 1)), X))
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

initialCost = computeCost(X, y, theta)

print("Initial Cost is: ", initialCost)

theta, JHistory = gradientDescent(X, y, theta, alpha, iterations)

print("Theta found by gradient descent", theta[0], theta[1])
scatterPlot = plotData(X[:, 1], y)
pridictLine = plt.plot(X[:, 1], X.dot(theta), '-')
plt.legend(['Training Data', 'Linear regression'])
plt.show()

print('For population = 35,000, we predict a profit of', np.array([1, 3.5]).dot(theta) * 10000)
print('For population = 70,000, we predict a profit of', np.array([1, 7]).dot(theta) * 10000)

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

theta0Vals = np.linspace(-10, 10, 100)
theta1Vals = np.linspace(-1, 4, 100)

JVals = np.zeros((theta0Vals.size, theta1Vals.size))

for i in range(theta0Vals.size):
    for j in range(theta1Vals.size):
        t = np.array([[theta0Vals[i]], [theta1Vals[j]]])
        JVals[i, j] = computeCost(X, y, t)

fig = plt.figure()
ax = fig.gca(projection='3d')
t0, t1 = np.meshgrid(theta0Vals, theta1Vals)
ax.plot_surface(t0, t1, JVals.T, rstride=5, cstride=5, cmap=cm.coolwarm)
plt.show()