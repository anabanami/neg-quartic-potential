# Diagonalising Hubbard matrices
# Ana Fabela 17/07/2023

# Import necessary libraries and modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d

plt.rcParams['figure.dpi'] = 200


def V(x):
    """
    Function defining a potential V as a function of position x.
    Parameters:
    - x: Position.
    Returns:
    - Potential V at position x.
    """
    # # unmodified negative quartic potential
    return -alpha * x ** 4


def get_files_RK4(filename):
    eigenvalues = []
    wavefunctions = []

    with h5py.File(f"{filename}.h5", "r") as file:
        # Get the eigenvalue data and convert it to a float
        evalue = file["eigenfunction"].attrs["eigenvalue"]
        eigenvalues.append(evalue)

        numpy_array = file["eigenfunction"][:]  # Get the wavefunction
        wavefunctions.append(numpy_array)

    return eigenvalues, wavefunctions



def get_files_Hubbard(filename):
    eigenvalues = []
    wavefunctions = []

    with h5py.File(f"{filename}.h5", "r") as file:
        # Get the eigenvalue data and convert it to a float
        evalue = file["eigenvalues"][:]
        eigenvalues.append(evalue)

        numpy_array = file["eigenvectors"][:]  # Get the wavefunction
        wavefunctions.append(numpy_array)

    return eigenvalues, wavefunctions



def initialisation_parameters():
    """
    Function to define and return global variables used throughout the script.
    Includes physical constants, potential coefficients, spatial and temporal
    discretization, initial wave function, etc.
    Returns:
    - A tuple containing all global parameters.
    """

    alpha = 1

    dx_RK4 = 1e-5
    x_max_RK4 = 30
    Nx_RK4 = int(2 * x_max_RK4 / dx_RK4)
    x_RK4 = np.linspace(-x_max_RK4, x_max_RK4, Nx_RK4, endpoint=False)

    dx_Hubb= 0.08
    x_max_Hubb = 45
    Nx_Hubb = int(2 * x_max_Hubb / dx_Hubb)
    x_Hubb = np.linspace(-x_max_Hubb, x_max_Hubb, Nx_Hubb, endpoint=False)

    return (
        alpha,
        dx_RK4,
        x_max_RK4,
        Nx_RK4,
        x_RK4,
        dx_Hubb,
        x_max_Hubb,
        Nx_Hubb,
        x_Hubb,
    )


if __name__ == "__main__":

    alpha, dx_RK4, x_max_RK4, Nx_RK4, x_RK4, dx_Hubb, x_max_Hubb, Nx_Hubb, x_Hubb = initialisation_parameters()

    # RK4 numerics
    evalue_RK4, wf_RK4 = get_files_RK4("0")
    # eigenvalue for gs
    evalue_RK4 = evalue_RK4[0]
    #wave function
    wf_RK4 = np.array(np.squeeze(wf_RK4, axis=0))
    # # obtain both sides of wave function (recall that gs is even)
    wf_RK4 = np.concatenate([wf_RK4[::-1], wf_RK4[0:1], wf_RK4])
    print(f"{evalue_RK4 = }")
    print(f"{np.shape(wf_RK4) = }")
    # print(f"{wf_RK4 = }") #<<< a list full OF nan :( when I use the dx=0.08 and x_max =45 


    # Hubbard model
    # Fuck my life
    eigenvalues_Hubb, wavefunctions_Hubb = get_files_Hubbard("Evals_hubbard")
    # make the list 'eigenvalues_Hubb' a (1, 22) array
    eigenvalues_Hubb_array = np.array(eigenvalues_Hubb)
    # Squeeze array into 1D, i.e. (22,)
    eigenvalues_Hubb_array = np.squeeze(eigenvalues_Hubb_array, axis=0)
    # get the ground state
    evalue_Hubb = eigenvalues_Hubb_array[4]
    print(f"\n{evalue_Hubb = }")

    # make the list 'wavefunctions_Hubb' a (1, 1125, 22) array
    wavefunctions_Hubb = np.array(wavefunctions_Hubb)
    # Squeeze array into 2D, i.e. (1125, 22)
    wavefunctions_Hubb = np.squeeze(wavefunctions_Hubb, axis=0)
    # Accessing one of the 22 arrays (for example, the ground state one)
    # making it a row vector
    wf_Hubb = wavefunctions_Hubb[:, 4].reshape((1125,))
    # 'wf_Hubb' is now one of the 22 arrays and has a shape of (1125,)
    print(f"{np.shape(wf_Hubb) = }")  # should print (1125,)

    # Plotting
    ax = plt.gca()

    print(f"{np.shape(x_RK4) = }")
    print(f"{np.shape(x_Hubb) = }")

    print("\n>>>> Plotting wavefunctions")
    ax = plt.gca()
    # color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(
            x_RK4,
            ((1 / 2.107) * abs(wf_RK4)**2),
            "-",
            linewidth=1,
            label=R"$\psi_{0,\mathrm{RK4}}(x)$",
            )

    plt.plot(
            x_Hubb,
            (20.82 * abs(wf_Hubb)**2),
            "-",
            linewidth=1,
            label=R"$\psi_{0,\mathrm{Hubb}}(x)$",
            color="mediumpurple",
            )

    # textstr = '\n'.join(
    #     (
    #     r'$E_{{0,\mathrm{{RK4}}$ =',f'{np.real(evalue_RK4):.06f}$',
    #     r'$E_{{0,\mathrm{{Hubb}}$ =',f'{np.real(evalue_Hubb):.06f}$',
    #     )
    # )
    # # place a text box in upper left in axes coords
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    plt.plot(
        x_RK4,
        V(x_RK4),
        linewidth=3,
        label=R"$V(x) = -x^4$",
        color="gray",
        alpha=0.3,
    )

    plt.ylim(-0.1, 1.2)
    # plt.xlim(-16.5, 16.5)

    plt.legend(loc="upper right")
    plt.xlabel(R'$x$')
    plt.ylabel('')
    plt.ylabel('Probability density')# with vertical energy shift')
    plt.grid(color='gray', linestyle=':')
    plt.title('First few eigenstates')
    plt.title('Ground state of negative quartic Hamiltonian\n(with RK4 and Hubbard methods)')
    plt.show()

