# Comparing HDF5 files of eigenvalues y eigenvectors
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
    return -α * smooth_restricted_V(x)


def first_5_evals_comparison():
    # Plot Bender's energies
    plt.plot(range(1, 6), E_bender, marker='.', linewidth=1, label='Bender')
    # Plot eigenvalues.py energies
    for filename, alpha in zip(files, alphas):
        with h5py.File(filename, 'r') as f:
            # extract first 5 items in each HDF5 file if they exist
            eigenvalues = np.array(f['eigenvalues'][:5])
            plt.plot(
                range(1, 6),
                np.real(eigenvalues),
                marker='.',
                linewidth=1,
                label=fR'$\alpha$={alpha:.3f}',
            )

    plt.xticks(np.arange(1, 6, 1))
    plt.title(R'Comparing eigenvalues (Bender vs eigenvalues.py)')
    plt.xlabel('First few eigenvalues')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()


def globals():
    # space dimension
    dx = 0.1
    # # Hopping strength
    # t = 1 / (2 * dx ** 2)
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

    # Bender energies to compare
    E_bender = np.array([1.477150, 6.003386, 11.802434, 18.458819, 25.791792])

    # scaling coefficients for quartic potential
    alphas = np.linspace(1e-6, 1.5, 15)

    # Initialise an array to store eigenvalues for each α
    eigenvalues = []

    # generate file names
    files = ['evals{:.3f}.hdf5'.format(α) for α in alphas]
    # print(files)
    # # open and extract all items in each HDF5 file
    for i, filename in enumerate(files):
        with h5py.File(filename, 'r') as f:
            # extract first 5 items in each HDF5 file if they exist
            eigenvalues.append(f['eigenvalues'][:evals_no])

    #list into array
    eigenvalues = np.array(eigenvalues)

    # rescale alphas to linearise the plot according to our resonance condition
    alphas = 0.5 * (2 * alphas) ** (1 / 3)

    print(f"\n{alphas=}\n")
    print(f"\n {eigenvalues=}\n")


    # # TEST 1
    # first_5_evals_comparison()

    # TEST 2
    for i in range(evals_no):
        plt.plot(alphas, np.real(eigenvalues[:, i]), marker='.', linewidth=1, label=fR"$E_{i}$" )

    for i, ref_val in enumerate(E_bender):
        if i == 0:  # Only assign a label to the first line
            plt.hlines(ref_val, alphas[0], alphas[-1], linewidth=0.5, linestyles='dashed', label=R'$E_{n,\mathrm{Bender}}$')
        else:  # For the rest of the lines, don't assign any label
            plt.hlines(ref_val, alphas[0], alphas[-1], linewidth=0.5, linestyles='dashed')

    plt.xlabel(R'$\frac{1}{2}(2\alpha)^{\frac{1}{3}}$')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()

