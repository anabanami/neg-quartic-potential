# Comparing HDF5 files
# Ana Fabela 11/07/2023

# Import necessary libraries and modules
import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

plt.rcParams['figure.dpi'] = 200


def gaussian_smoothing(data, pts):
    """
    Function to smooth input data with a Gaussian kernel.
    The data is convolved with a Gaussian kernel to achieve smoothing.
    Parameters:
    - data: The array to be smoothed.
    - pts: Number of points to be considered in the Gaussian kernel.
    Returns:
    - The smoothed array.
    """
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def smooth_restricted_V(x):
    """
    Function to generate a potential V(x) based on input x, with smoothing applied in a restricted domain.
    Parameters:
    - x: Array defining the spatial coordinates.
    Returns:
    - The smoothed potential V.
    """
    V = np.ones_like(x) * x[cut] ** 4
    V[cut : Nx - cut] = x[cut : Nx - cut] ** 4
    ## smoooth by pts=3
    V = gaussian_smoothing(V, 3)
    return V


def V(x):
    """
    Function defining a potential V as a function of position x.
    Parameters:
    - x: Position.
    Returns:
    - Potential V at position x.
    """
    # Select test potential by uncommenting

    # # Free space (no potential)
    # return np.zeros_like(x)

    # # upside-down harmonic oscillator
    # return - (x ** 2)

    # # # unmodified negative quartic potential
    # return -alpha * x ** 4

    # restricted and smoothed negative quartic potential
    return -alpha * smooth_restricted_V(x)

    # # Higher order perturbation
    # return - (x ** 8)


def globals():
    """
    Function to define and return global variables used throughout the script.
    Includes physical constants, potential coefficients, spatial and temporal discretization, initial wave function, etc.
    Returns:
    - A tuple containing all global parameters.
    """
    # makes folder for simulation frames
    folder = Path(f'Unitary_hubbard')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    hbar = 1

    # Bender units
    m = 1 / 2
    omega = 2
    # # natural units
    # m = 1
    # omega = 1

    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * omega))

    # coefficient for quartic potential
    alpha = 1

    # space dimension
    x_max = 45
    dx = 0.08
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # Lattice parameters
    N_sites = Nx
    cut = 5
    # Hopping strength
    t = 1 / (2 * dx ** 2)

    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 5.7

    # initial conditions: HO ground state
    wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))

    # # initial conditions: shifted HO ground state
    # wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-((x - 1) ** 2) / (2 * l1 ** 2))

    return (
        folder,
        hbar,
        m,
        omega,
        l1,
        alpha,
        N_sites,
        cut,
        t,
        x_max,
        dx,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
        wave,
    )


if __name__ == "__main__":

    # Retrieve global parameters

    (
        folder,
        hbar,
        m,
        omega,
        l1,
        alpha,
        N_sites,
        cut,
        t,
        x_max,
        dx,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
        wave,
    ) = globals()

    file1 = h5py.File('neg_quart.hdf5', 'r')

    # Print out the names of all items in the root of the HDF5 file
    times = []
    for key in file1.keys():
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

    state1_1 = np.array(file1[f'{t1}'])

    state1_2 = np.array(file1[f'{t2}'])

    state1_3 = np.array(file1[f'{t3}'])

    state1_4 = np.array(file1[f'{t4}'])

    state1_5 = np.array(file1[f'{t5}'])

    state1_list = [state1_0, state1_1, state1_2, state1_3, state1_4, state1_5]

    for i in range(6):
        plt.plot(
            x,
            abs(state1_list[i]) ** 2,
            # color=color1,
            label=f'U(t)' if i == 0 else None,
        )

    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel("x")
    plt.legend()
    # plt.title(f"")
    plt.show()

    file1.close()
