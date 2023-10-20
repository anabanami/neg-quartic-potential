# Diagonalising Hubbard matrices
# Ana Fabela 17/07/2023

# Import necessary libraries and modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams['figure.dpi'] = 300


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

    dx = 0.08
    x_max = 45
    
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    return (
        alpha,
        dx,
        x_max,
        Nx,
        x,
    )


if __name__ == "__main__":

    alpha, dx, x_max, Nx, x = initialisation_parameters()

    # RK4 numerics
    evalue_RK4, wf_RK4 = get_files_RK4("Eval_gs_RK4")
    evalue_RK4 = evalue_RK4[0]
    wf_RK4 = np.array(wf_RK4).reshape((562,))
    # plot both sides of wave function (gs is even)
    wf_RK4 = np.concatenate([wf_RK4[::-1], wf_RK4[0:1], wf_RK4])
    print(f"{np.shape(wf_RK4) = }")

    print(f"{wf_RK4 = }") <<< THIS IS A BUNCH OF NAN IN :(

    ass

    # Hubbard model
    # Fuck my life
    eigenvalues_Hubb, wavefunctions_Hubb = get_files_Hubbard("Evals_hubbard")
    # make the list 'eigenvalues_Hubb' a (1, 22) array
    eigenvalues_Hubb_array = np.array(eigenvalues_Hubb)
    # Squeeze array into 1D, i.e. (22,)
    eigenvalues_Hubb_array = np.squeeze(eigenvalues_Hubb_array, axis=0)
    print(f"{np.shape(eigenvalues_Hubb_array)}")
    # get the ground state
    evalue_Hubb = eigenvalues_Hubb_array[4]
    print(f"{evalue_Hubb = }")

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

    print("\n>>>> Plotting wavefunctions")
    ax = plt.gca()
    # color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(
            x,
            (10 * abs(wf_RK4)**2) + evalue_RK4,
            "-",
            linewidth=1,
            label=R"$\psi_{0,\mathrm{RK4}}(x)$",
            color="purple",
            )

    # plt.plot(
    #         x,
    #         (10 * abs(wf_Hubb)**2) + evalue_Hubb,
    #         "-",
    #         linewidth=1,
    #         label=R"$\psi_{0,\mathrm{Hubb}}(x)$",
    #         )

    textstr = '\n'.join(
        (
        r'$E_{{0,\mathrm{{RK4}}$ =',f'{np.real(evalue_RK4):.06f}$',
        r'$E_{{0,\mathrm{{Hubb}}$ =',f'{np.real(evalue_Hubb):.06f}$',
        )
    )
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    # plt.plot(
    #     x,
    #     V(x),
    #     linewidth=3,
    #     label=R"$V(x) = -x^4$",
    #     color="gray",
    #     alpha=0.3,
    # )

    plt.ylim(0, 1.5)
    # plt.xlim(-16.5, 16.5)

    plt.legend(loc="upper right")
    plt.xlabel(R'$x$')
    plt.ylabel('')
    plt.ylabel('Probability density')
    plt.grid(color='gray', linestyle=':')
    plt.title('First few eigenstates')
    plt.title('Ground state probability density with vertical energy shift\n(negative quartic Hamiltonian with Hubbard and RK4 methods)')
    plt.show()

