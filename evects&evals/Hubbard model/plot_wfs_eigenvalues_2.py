# Diagonalising Hubbard matrices
# Ana Fabela 17/07/2023

# Import necessary libraries and modules
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import convolve
import h5py

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

    # # Free space
    # return np.zeros_like(x)

    #Harmonic oscillator
    return x ** 2

    # # # unmodified negative quartic potential
    # return -alpha * x ** 4

    # # restricted and smoothed negative quartic potential
    # return -alpha * smooth_restricted_V(x)

    # # Higher order perturbation
    # return -(x ** 8)


def explore_hdf5_group(group, indent=0):
    """Recursively explore and print the contents of an HDF5 group."""
    items = list(group.items())
    for name, item in items:
        print(
            "  " * indent
            + f"- {name}: {'Group' if isinstance(item, h5py.Group) else 'Dataset'}"
        )
        if isinstance(item, h5py.Group):
            explore_hdf5_group(item, indent + 1)


def plot_wavefunctions(N, x, evals, evects):
    """To visualize the states in the spatial basis,
    I transform these site-basis eigenvectors back to the spatial basis.
    Since, the site basis in the Hubbard model is typically associated
    with a specific position in real space, I can map each grid site
    as a delta function at that position.

    I plot an eigenvector's distribution across the spatial
    grid (x) by treating the index of the eigenvector's
    components as corresponding to the index in the x array."""

    print(">>>> Plotting wavefunctions")
    for i in range(11):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        # if 3 < i < 9:
        if i < 5:
            plt.plot(
                x,
                (150 * (abs(evects[:, i])**2) + evals[i]),
                "-",
                linewidth=1,
                label=fR"$\psi_{{{i}}}(x)$",
                color=color,
            )

            # plt.plot(
                # x,
            #     (5 * abs(evects[:, i])**2 + evals[i]),
            #     "-",
            #     linewidth=1,
            #     label=fR"$\psi_{{{i}}}(x)$",
            #     color=color,
            # )
            # plt.plot(
            #     x,
            #     (5 * np.imag(evects[:, i])) + np.real(evals[i]),
            #     "--",
            #     linewidth=1,
            #     color=color,
            # )

        else:
            plt.plot(
                x,
                (150 * (abs(evects[:, i])**2) + evals[i]),
                "-",
                linewidth=1,
                color=color,
            )

            # plt.plot(
            #     x,
            #     (5 * np.real(evects[:, i]) + evals[i]),
            #     "-",
            #     linewidth=1,
            #     color=color,
            # )
            # plt.plot(
            #     x,
            #     (5 * np.imag(evects[:, i])) + np.real(evals[i]),
            #     "--",
            #     linewidth=1,
            #     color=color,
            # )

    textstr = '\n'.join(
        (
           fr'$E_0 = {np.real(evals[0]):.06f}$',
            fr'$E_1 = {np.real(evals[1]):.06f}$',
            fr'$E_2 = {np.real(evals[2]):.06f}$',
            fr'$E_3 = {np.real(evals[3]):.06f}$',
            fr'$E_4 = {np.real(evals[4]):.06f}$',
            # fr'$E_5 = {np.real(evals[5]):.06f}$',
            # fr'$E_6 = {np.real(evals[6]):.06f}$',
            # fr'$E_7 = {np.real(evals[7]):.06f}$',
            # fr'$E_8 = {np.real(evals[8]):.06f}$',
            # fr'$E_9 = {np.real(evals[9]):.06f}$',
            # fr'$E_{{10}} = {np.real(evals[10]):.06f}$',
        )
    )
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    plt.plot(
        x,
        V(x),
        linewidth=3,
        label=R"$V(x) = x^2$",
        color="gray",
        alpha=0.3,
    )

    plt.ylim(-2, 8)
    plt.xlim(-16.5, 16.5)

    plt.legend(loc="upper right")
    plt.xlabel(R'$x$')
    plt.ylabel('Probability density with vertical energy shift')
    # plt.ylabel('Amplitude')
    plt.grid(color='gray', linestyle=':')
    plt.title('First few eigenstates')
    # plt.title('"Ground state" for the negative quartic Hamiltonian')
    plt.show()


def read_hdf5_data(file_name):
    """
    Read eigenvalues and eigenvectors data from an HDF5 file.

    Parameters:
        - file_name: The name of the HDF5 file to read.

    Returns:
        - evals: The eigenvalues data read from the file.
        - evects: The eigenvectors data read from the file.
    """
    with h5py.File(file_name, 'r') as file:
        # Ensure the datasets 'eigenvalues' and 'eigenvectors' exist in the file
        if 'eigenvalues' in file.keys() and 'eigenvectors' in file.keys():
            evals = file['eigenvalues'][:]
            evects = file['eigenvectors'][:]
        else:
            raise KeyError(
                "Datasets 'eigenvalues' and/or 'eigenvectors' not found in the file."
            )

    return evals, evects


def global_params():
    """
    Function to define and return global variables used throughout the script.
    Includes physical constants, potential coefficients, spatial and temporal
    discretization, initial wave function, etc.
    Returns:
    - A tuple containing all global parameters.
    """
    # space dimension
    dx = 0.006
    x_max = 20
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)
    alpha = 1
    cut = 5

    return dx, x_max, Nx, x, alpha, cut


if __name__ == "__main__":
    dx, x_max, Nx, x, alpha, cut = global_params()

    print("TESTING PARAMETERS:")
    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"\n{dx = }")

    file_name = 'Evals_hubbard.h5'

    # with h5py.File(file_name, 'r') as file:
    #     print(f"Contents of {file_name}:")
    #     explore_hdf5_group(file)

    evals, evects = read_hdf5_data(file_name)

    # plot eigenfunctions
    plot_wavefunctions(Nx, x, evals, evects)

 