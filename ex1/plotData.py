import matplotlib.pyplot as plt


def plotData(x, y):
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    return plt.plot(x, y, 'rx', ms=10)