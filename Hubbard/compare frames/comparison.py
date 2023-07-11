# Comparing HDF5 files
# Ana Fabela 11/07/2023

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

plt.rcParams['figure.dpi'] = 200


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def smooth_restricted_V(x):
    V = np.ones_like(x) * x[cut] ** 4
    V[cut : Nx - cut] = x[cut : Nx - cut] ** 4
    ## smoooth by pts=3
    V = gaussian_smoothing(V, 3)
    return V


def V(x):
    return - α * smooth_restricted_V(x)
    # return np.zeros_like(x)
    # return - 0.5 * (x ** 2)


def globals():
    # coefficient for quartic potential
    α = 4
    ## space dimension
    dx = 0.1
    x_max = 45
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)
    cut = 225
    
    return (
        α,
        Nx,
        x,
        cut
        )


if __name__ == "__main__":
    """FUNCTION CALLS"""

    α, Nx, x, cut = globals()

    file1 = h5py.File('9.1.hdf5', 'r')
    file2 = h5py.File('10.1.hdf5', 'r')

    # Print out the names of all items in the root of the HDF5 file
    times = []
    for key in file2.keys():
        # print(key)
        times.append(float(key))
    times = np.array(times)
    print(f'{np.shape(times) = }')

    t = times[100]

    # Print out the contents of a single timestep
    state1 = np.array(file1[f'{t}'])
    state2 = np.array(file2[f'{t}'])

    plt.plot(x, V(x), color="black", linewidth=2, label="V(x)")
    plt.plot(x, abs(state1) ** 2, label='U')
    plt.plot(x, abs(state2) ** 2, label='FSS')
    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel("x")
    plt.legend()
    plt.ylim(-1.5, 3)
    plt.xlim(-5, 5)
    plt.title(f"t = {t:05f}")
    plt.show()

    file1.close()
    file2.close()
