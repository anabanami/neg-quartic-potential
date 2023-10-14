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

    # #Harmonic oscillator
    # return x ** 2

    # # # unmodified negative quartic potential
    # return -alpha * x ** 4

    # # restricted and smoothed negative quartic potential
    # return -alpha * smooth_restricted_V(x)

    # Higher order perturbation
    return -(x ** 8)


def Bose_Hubbard_Hamiltonian():
    # Initialize the Hamiltonian as a zero matrix
    H = np.zeros((Nx, Nx))
    # On-site interaction potential
    V_values = V(x)

    # Define the hopping and interaction terms
    # PERIODIC BCS
    print(">>>> Generating Hubbard matrix")
    for i in range(Nx):
        # Hopping terms
        H[i, (i + 1) % Nx] = -t
        H[(i + 1) % Nx, i] = -t

        # On-site interaction
        H[i, i] = V_values[i]

    return H


def plot_matrix(H):
    # Plot the Hamiltonian as a heat map
    plt.imshow(H, cmap='magma', interpolation='nearest')
    plt.colorbar(label='Matrix element value')
    plt.title('Hubbard Hamiltonian')
    plt.xlabel('Site index')
    plt.ylabel('Site index')
    plt.show()

    # Calculate absolute values and add a small constant to avoid log(0)
    H_abs = np.abs(H) + 1e-9

    # Plot the absolute value of the Hamiltonian as a heat map on a logarithmic scale
    plt.imshow(np.log(H_abs), cmap='magma', interpolation='nearest')
    plt.colorbar(label='Log of absolute matrix element value')
    plt.title('Absolute value of Hubbard Hamiltonian\n(log scale)')
    plt.xlabel('Site index')
    plt.ylabel('Site index')
    plt.show()


def filter_sorting(evals, evects):
    print(">>>> Filter and sort eigenvalues")
    # filtering
    mask = (-10 < evals.real) & (evals.real < 50)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(np.round(evals.real, 3) + np.round(evals.imag, 3) / 1e6)
    evals = evals[order]
    evects = evects[:, order]
    # (evals.shape= (Nx,))
    # example: becomes (29,), (evects.shape= (Nx, 29) where the column v[:, i])
    return evals, evects


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

        if i < 1:
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

    textstr = '\n'.join(
        (
            fr'$E_0 = {np.real(evals[0]):.06f}$',
            fr'$E_1 = {np.real(evals[1]):.06f}$',
            fr'$E_2 = {np.real(evals[2]):.06f}$',
            fr'$E_3 = {np.real(evals[3]):.06f}$',
            fr'$E_4 = {np.real(evals[4]):.06f}$',
            fr'$E_5 = {np.real(evals[5]):.06f}$',
            fr'$E_6 = {np.real(evals[6]):.06f}$',
            fr'$E_7 = {np.real(evals[7]):.06f}$',
            fr'$E_8 = {np.real(evals[8]):.06f}$',
            fr'$E_9 = {np.real(evals[9]):.06f}$',
            fr'$E_{{10}} = {np.real(evals[10]):.06f}$',
        )
    )
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    # plt.plot(x, V(x), linewidth=2, alpha=0.4, color='k')
    plt.axvline(0, linestyle=":", alpha=0.4, color="black")

    # plt.ylim()
    # plt.xlim(-10, 10)

    plt.legend(loc="upper right")
    plt.xlabel(R'$x$')
    plt.ylabel('Amplitude')
    plt.title('First few eigenstates')
    # plt.savefig(f"evals.png")
    plt.show()


def globals():
    """
    Function to define and return global variables used throughout the script.
    Includes physical constants, potential coefficients, spatial and temporal
    discretization, initial wave function, etc.
    Returns:
    - A tuple containing all global parameters.
    """
    os.system(f'rm *.h5')

    hbar = 1

    # Bender units
    m = 1 / 2
    omega = 2

    # coefficient for quartic potential
    alpha = 1

    # space dimension
    dx = 0.006

    x_max = 20
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # Hopping strength
    t = 1 / (2 * dx ** 2)

    cut = 5

    return (
        hbar,
        m,
        omega,
        alpha,
        dx,
        x_max,
        Nx,
        x,
        t,
        cut,
    )


if __name__ == "__main__":

    (
        hbar,
        m,
        omega,
        alpha,
        dx,
        x_max,
        Nx,
        x,
        t,
        cut,
    ) = globals()

    print("TESTING PARAMETERS:")
    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"\n{dx = }")

    print("only relevant to smooth restricted potential:")
    print(f"x_cut_left = {x[cut]= }")
    print(f"x_cut_right = {x[Nx-cut]= }")

    # generate Hubbard matrix
    M = Bose_Hubbard_Hamiltonian()
    plot_matrix(M)

    evals, evects = linalg.eig(M)  # remember that evects are columns! v[:, j]

    # fix Energy shift of - 2t
    print("\n>>>> correcting energy shift due to hopping term approx")
    for i, value in enumerate(evals):
        evals[i] = np.real(value) + 2 * t

    # FILTER AND SORT
    evals, evects = filter_sorting(evals, evects)

    # plot eigenfunctions
    plot_wavefunctions(Nx, x, evals, evects)

    # Create a new HDF5 file
    print("\n>>>> saving Evals_hubbard.h5")
    with h5py.File(f'Evals_hubbard.h5', 'w') as file:
        # Create datasets for eigenvalues and eigenvectors in hdf5 file
        evals_dset = file.create_dataset('eigenvalues', data=evals)
        evects_dset = file.create_dataset('eigenvectors', data=evects)

    # Close the hdf5 file
    file.close()

    print("\n>>>> Comparing with Bender's spectrum")
    _2evals = 2 * evals
    # Bender energies to compare
    E_bender = np.array([1.477150, 6.003386, 11.802434, 18.458819, 25.791792])
    print(
        f"E0 = {np.real(evals[6]):.06f} vs {E_bender[0]}\n, E1 = {np.real(evals[7]):.06f} vs {E_bender[1]}\n, E2 = {np.real(evals[8]):.06f} vs {E_bender[2]}\n, E3 = {np.real(evals[9]):.06f} vs {E_bender[3]}\n, E4 = {np.real(evals[10]):.06f} vs {E_bender[4]}"
    )
