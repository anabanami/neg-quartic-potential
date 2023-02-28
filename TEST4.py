# Ana Fabela Hinojosa, 27/02/2023
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def restricted_V(x):
    V = np.negative(np.ones_like(x))
    V[100:412] = -x[100:412] ** 4
    return V


x = np.linspace(-15, 15, 512)
y = restricted_V(x)
plt.plot(x, y)
plt.show()

pts = 5
y = gaussian_smoothing(y, pts)
plt.plot(x, y)
plt.show()
