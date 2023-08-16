# Comparing HDF5 files of eigenvalues y eigenvectors
# Ana Fabela 14/08/2023

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

plt.rcParams['figure.dpi'] = 200


def generate_file_names():
    # Initialise an array to store eigenvalues for each ω
    eigenvalues = []

    # generate file names
    files = [f'evals_{i}_{ω:.3f}.hdf5'.format(ω) for i, ω in enumerate(omegas)]
    files[0] = 'evals_V(x)=-0.5x**4.hdf5'
    # print(files)
    # # open and extract all items in each HDF5 file
    for i, filename in enumerate(files):
        with h5py.File(filename, 'r') as f:
            # extract first 5 items in each HDF5 file if they exist
            eigenvalues.append(f['eigenvalues'][:evals_no])

    # list into array
    eigenvalues = np.array(eigenvalues)
    return eigenvalues


def globals():
    # # natural units
    m = 1
    hbar = 1

    dx = 0.01
    # Hopping strength
    t = 1 / (2 * dx ** 2)

    # space dimension
    x_max = 15
    Nx = int(2 * x_max / dx)
    cut = 5
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    return (
        hbar,
        m,
        Nx,
        cut,
        t,
        x_max,
        dx,
        Nx,
        x,
    )


if __name__ == "__main__":

    (
        hbar,
        m,
        Nx,
        cut,
        t,
        x_max,
        dx,
        Nx,
        x,
    ) = globals()

    # number of eigenvalues to check
    evals_no = 5

    # HO energies to compare
    E_HO = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    # # Bender energies to compare
    # E_bender = np.array([1.477150, 6.003386, 11.802434, 18.458819, 25.791792])
    # # Bender energies to compare
    # E_wkb = np.array([1.3765, 5.9558, 11.7690, 18.4321, 257692])

    # scaling frequencies
    omegas = np.linspace(1e-6, 1, 10)

    eigenvalues = generate_file_names()


    print(np.shape(eigenvalues))
    print(eigenvalues[0])
    print(eigenvalues[1])




    ass

    # TEST 1
    for i in range(evals_no):
        eigenvalue_i = [
            np.real(eig[i]) if i < len(eig) else np.nan for eig in eigenvalues
        ]
        plt.scatter(
            omegas,
            eigenvalue_i,
            marker='.',
            color='k',
            linewidth=1,
        )  # label=fR"$E_{i}$")

    for i, ref_val in enumerate(E_HO):
        if i == 0:  # Only assign a label to the first line
            plt.hlines(
                ref_val,
                omegas[0],
                omegas[-1],
                linewidth=0.5,
                linestyles='dashed',
                label=R'$E_{n,\mathrm{ref}}$',
            )
        else:  # For the rest of the lines, don't assign any label
            plt.hlines(
                ref_val, omegas[0], omegas[-1], linewidth=0.5, linestyles='dashed'
            )

    plt.title(R'Comparing energy (eigenvalues_3.py vs. $E_{HO}$)')
    plt.xlabel(R'$\omega$')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()
