import os
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


def V(x):
    return - (x ** 4)


def initialisation_parameters():
    dx = 1e-3

    # space dimension
    x_max = 10
    Nx = int(x_max / dx)
    x = np.linspace(-x_max, 0, Nx, endpoint=False)

    return (
        dx,
        x_max,
        Nx,
        x,
    )


if __name__ == "__main__":

    dx, x_max, Nx, x = initialisation_parameters()

    eigenvalues = []
    wavefunctions = []


    for i in range(2):

        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        with h5py.File(f"{i}.h5", "r") as file:
            # Get the eigenvalue data and convert it to a float
            evalue = file["eigenfunction"].attrs["eigenvalue"]
            eigenvalues.append(evalue)

            numpy_array = file["eigenfunction"][:]  # Get the wavefunction
            wavefunctions.append(numpy_array[::-1])


    # Plotting
    for i, (wf, evalue) in enumerate(zip(wavefunctions, eigenvalues)):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(x, np.real(wf) + evalue, linewidth=1, label=Rf"$\psi_{i}$", color=color)
        plt.plot(x, np.imag(wf) + evalue, "--", linewidth=1, color=color)

        # plt.plot(x, abs(wf)**2 + evalue, linewidth=1, label=Rf"$\psi_{i}$", color=color)

    textstr = '\n'.join(
        (
            # fr'$E_2 = {eigenvalues[2]:.06f}$',
            fr'$E_1 = {eigenvalues[1]:.06f}$',
            fr'$E_0 = {eigenvalues[0]:.06f}$',    
        )
    )
    
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    # plt.plot(x, V(x), linewidth=2, alpha=0.4, color='k')
    plt.legend()
    plt.xlabel(R'$x$')
    plt.ylabel('Amplitude')
    plt.title("First few eigenstates")
    plt.show()
