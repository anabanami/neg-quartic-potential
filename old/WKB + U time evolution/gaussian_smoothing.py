import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


Nx = 1024
x = np.linspace(-10, 10, Nx)

y = np.zeros_like(x)
for i, element in enumerate(y):
    if i % 100 == 0:
        y[i] = 1

plt.plot(x, y)
plt.show()

pts = 25
y = gaussian_smoothing(y, pts)
plt.plot(x, y)
plt.show()