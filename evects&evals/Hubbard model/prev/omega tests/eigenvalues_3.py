# Diagonalising Hubbard matrices
# HDF5 protocol
# Ana Fabela 13/08/2023

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import convolve
import h5py

plt.rcParams['figure.dpi'] = 200


#######################################################################################################


def V(x, ω):
    return 0.5 * m * (ω * x) ** 2
    # return -0.5 * x ** 4


def Bose_Hubbard_Hamiltonian(ω):
    # Initialize the Hamiltonian as a zero matrix
    H = np.zeros((Nx, Nx))
    # On-site interaction potential
    V_values = V(x, ω)

    # Define the hopping and interaction terms
    # PERIODIC BCS
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
    # filtering
    mask = (-5 < evals.real) & (evals.real < 50)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(np.round(evals.real, 3) + np.round(evals.imag, 3) / 1e6)
    evals = evals[order]
    evects = evects[:, order]
    # (evals.shape= (Nx,))
    # example: becomes (29,), (evects.shape= (Nx, 29) where the column v[:, i])
    return evals, evects


def RESONANCE_omega_hdf5(omegas):
    for i, ω in enumerate(omegas):
        print(f"{i = }, {ω = }")

        # Create a new HDF5 file
        file = h5py.File(f'evals_{i}_{ω:.3f}.hdf5', 'w')

        # Generate Hubbard matrix
        M_ω = Bose_Hubbard_Hamiltonian(ω)

        # Diagonalize the matrix
        evals, evects = linalg.eig(M_ω)

        # Create datasets for eigenvalues and eigenvectors in the hdf5 file
        evals_dset = file.create_dataset('eigenvalues', data=evals)
        evects_dset = file.create_dataset('eigenvectors', data=evects)

        # Close the hdf5 file
        file.close()


def neg_quart_hdf5():
    # Create a new HDF5 file
    file = h5py.File(f'evals_V(x)=-0.5x**4.hdf5', 'w')

    # Generate Hubbard matrix
    M = Bose_Hubbard_Hamiltonian(ω)

    # Diagonalize the matrix
    evals, evects = linalg.eig(M)

    # Adjust the eigenvalues by the shift
    evals = evals.real + 2 * t

    # Filter and sort
    evals, evects = filter_sorting(evals, evects)

    # Create datasets for eigenvalues and eigenvectors in the hdf5 file
    evals_dset = file.create_dataset('eigenvalues', data=evals)
    evects_dset = file.create_dataset('eigenvectors', data=evects)

    # Close the hdf5 file
    file.close()


def plot_wavefunctions(N, x, evals, evects):
    """To visualize the states in the spatial basis,
    I transform these site-basis eigenvectors back to the spatial basis.
    Since, the site basis in the Hubbard model is typically associated
    with a specific position in real space, I can map each grid site
    as a delta function at that position.

    I plot an eigenvector's distribution across the spatial
    grid (x) by treating the index of the eigenvector's
    components as corresponding to the index in the x array."""

    for i in range(5):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(
            x,
            np.real(evects[:, i] + evals[i]),
            "-",
            linewidth=1,
            label=fR"$\psi_{i}(x)$",
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
        )
    )
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    plt.plot(x, V(x), linewidth=2, alpha=0.4, color='k')
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

    # os.system(f'rm *.hdf5')

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

    print("TESTING PARAMETERS:")
    print("\n")
    print(f"{dx = }")
    print(f"{t = }")
    print(f"\nEigenvalues No. = {Nx = }")
    print(f"{x_max = }")
    print("\nONLY for restricted V(x):")
    print(f"{x[cut] = }")
    print(f"index {cut = }\n")

    ## FOR RESONANCE TESTS
    # frequency adjusting
    omegas = np.linspace(1e-6, 2, 10)
    RESONANCE_omega_hdf5(omegas)

    # ω = 1
    # neg_quart_hdf5()
