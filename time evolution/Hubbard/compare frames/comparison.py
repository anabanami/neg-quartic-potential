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
    # return - α * smooth_restricted_V(x)
    return np.zeros_like(x)
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

    file1 = h5py.File('9.2.hdf5', 'r')
    file2 = h5py.File('11.hdf5', 'r')

    # Print out the names of all items in the root of the HDF5 file
    times = []
    for key in file2.keys():
        print(key)
        times.append(float(key))
    times = np.array(times)
    print(f'{np.shape(times) = }')

    t0 = times[0]
    t1 = times[1000]
    t2 = times[2000]
    t3 = times[3000]
    t4 = times[4000]
    t5 = times[5000]

    # Print out the contents of a single timestep
    state1_0 = np.array(file1[f'{t0}'])
    state2_0 = np.array(file2[f'{t0}'])

    state1_1 = np.array(file1[f'{t1}'])
    state2_1 = np.array(file2[f'{t1}'])
    
    state1_2 = np.array(file1[f'{t2}'])
    state2_2 = np.array(file2[f'{t2}'])
    
    state1_3 = np.array(file1[f'{t3}'])
    state2_3 = np.array(file2[f'{t3}'])
    
    state1_4 = np.array(file1[f'{t4}'])
    state2_4 = np.array(file2[f'{t4}'])
    
    state1_5 = np.array(file1[f'{t5}'])
    state2_5 = np.array(file2[f'{t5}'])

    state1_list = [state1_0, state1_1, state1_2, state1_3, state1_4, state1_5]
    state2_list = [state2_0, state2_1, state2_2, state2_3, state2_4, state2_5]
    
    color1 = "blue"
    color2 = "orange"

    plt.plot(x, V(x), color="black", linewidth=2, label="V(x)")

    for i in range(6):
        plt.plot(x, abs(state1_list[i]) ** 2, color=color1, label=f'U(t)' if i==0 else None)
        plt.plot(x, abs(state2_list[i]) ** 2, color=color2, label=f'Analytic(t)' if i==0 else None)

    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel("x")
    plt.legend()
    # plt.title(f"")
    plt.show()


    file1.close()
    file2.close()
