# Comparing HDF5 files of eigenvalues y eigenvectors
# Ana Fabela 11/07/2023

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

plt.rcParams['figure.dpi'] = 200


def generate_file_names():
    # Initialise an array to store eigenvalues for each α
    eigenvalues = []

    # generate file names
    files = [f'evals{i}_{α:.3f}.hdf5'.format(α) for i, α in enumerate(alphas)]
    # print(files)
    # # open and extract all items in each HDF5 file
    for i, filename in enumerate(files):
        with h5py.File(filename, 'r') as f:
            # extract first 5 items in each HDF5 file if they exist
            eigenvalues.append(f['eigenvalues'][:evals_no])

    # list into array
    eigenvalues = np.array(eigenvalues)
    # print(f"\n{alphas=}\n")
    # print(f"\n {eigenvalues=}\n")
    return eigenvalues


def globals():
    # space dimension
    dx = 0.1
    # # Hopping strength
    # t = 1 / (2 * dx ** 2)
    # scaling coefficient for kinetic energy
    # β = 1.8
    x_max = 40
    Nx = int(2 * x_max / dx)
    cut = 5
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    return (Nx, x, cut)


if __name__ == "__main__":
    """FUNCTION CALLS"""
    Nx, x, cut = globals()

    # number of eigenvalues to check
    evals_no = 5

    # HO energies to compare
    E_HO = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    # # Bender energies to compare
    # E_bender = np.array([1.477150, 6.003386, 11.802434, 18.458819, 25.791792])

    # scaling coefficients for quartic potential
    alphas = np.linspace(0.4, 1, 100)
    # # # rescale alphas to scale energy according to our resonance condition
    # alphas = 0.5 * (2 * alphas) ** (1 / 3)

    eigenvalues = generate_file_names()

    # TEST 1
    for i in range(evals_no):
        eigenvalue_i = [
            np.real(eig[i]) if i < len(eig) else np.nan for eig in eigenvalues
        ]
        plt.scatter(alphas, eigenvalue_i, marker='.', color='k', linewidth=1,)# label=fR"$E_{i}$")

    for i, ref_val in enumerate(E_HO):
        if i == 0:  # Only assign a label to the first line
            plt.hlines(
                ref_val,
                alphas[0],
                alphas[-1],
                linewidth=0.5,
                linestyles='dashed',
                label=R'$E_{n,\mathrm{ref}}$',
            )
        else:  # For the rest of the lines, don't assign any label
            plt.hlines(
                ref_val, alphas[0], alphas[-1], linewidth=0.5, linestyles='dashed'
            )

    plt.title(R'Comparing eigenvalues (HO vs eigenvalues.py)')
    # plt.title(R'Comparing eigenvalues (Bender vs eigenvalues.py)')
    plt.xlabel(R'$\frac{1}{2}(2\alpha)^{\frac{1}{3}}$')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()



