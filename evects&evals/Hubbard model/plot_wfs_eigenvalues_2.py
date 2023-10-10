

# Diagonalising Hubbard matrices
# Ana Fabela 17/07/2023

# Import necessary libraries and modules
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import convolve
import h5py

plt.rcParams['figure.dpi'] = 200


def plot_wavefunctions(N, x, evals, evects):
    """To visualize the states in the spatial basis,
    I transform these site-basis eigenvectors back to the spatial basis.
    Since, the site basis in the Hubbard model is typically associated
    with a specific position in real space, I can map each grid site
    as a delta function at that position.

    I plot an eigenvector's distribution across the spatial
    grid (x) by treating the index of the eigenvector's
    components as corresponding to the index in the x array."""

    print(">>>>Plotting wavefunctions")
    for i in range(11):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        if i < 7:
            plt.plot(
                x,
                np.real(evects[:, i] + evals[i]),
                "-",
                linewidth=1,
                color=color,
            )
            plt.plot(
                x,
                np.imag(evects[:, i]) + np.real(evals[i]),
                "--",
                linewidth=1,
                color=color,
            )

        else:
            plt.plot(
                x,
                np.real(evects[:, i] + evals[i]),
                "-",
                linewidth=1,
                label=fR"$\psi_{{{i}}}(x)$",
                color=color,
            )
            plt.plot(
                x,
                np.imag(evects[:, i]) + np.real(evals[i]),
                "--",
                linewidth=1,
                color=color,
            )


def globals():
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

    return (
        dx,
        x_max,
        Nx,
        x,
    )


if __name__ == "__main__":

    (
        dx,
        x_max,
        Nx,
        x,
    ) = globals()

    print("TESTING PARAMETERS:")
    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"\n{dx = }")

    # FILTER AND SORT
    evals, evects = filter_sorting(evals, evects)



# Create a new HDF5 file
    print("\n>>>> saving Evals_hubbard.h5")
    with h5py.File(f'Evals_hubbard.h5', 'r') as file:
        # Create datasets for eigenvalues and eigenvectors in hdf5 file
        evals_dset = file.create_dataset('eigenvalues', data=evals)
        evects_dset = file.create_dataset('eigenvectors', data=evects)

            # plot eigenfunctions
    plot_wavefunctions(Nx, x, evals_dset, evects_dset)

    # Close the hdf5 file
    file.close()