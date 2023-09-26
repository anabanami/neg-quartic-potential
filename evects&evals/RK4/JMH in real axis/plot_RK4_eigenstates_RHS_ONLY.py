import os
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


def V(x):
    return - (x ** 4)


def initialisation_parameters():
    x_max = 30
    dx = 1e-5
    Nx = int(x_max / dx)
    x = np.linspace(0, x_max, Nx, endpoint=False)

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

    for i in range(1):

        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        with h5py.File(f"{i}.h5", "r") as file:
            # Get the eigenvalue data and convert it to a float
            evalue = file["eigenfunction"].attrs["eigenvalue"]
            eigenvalues.append(evalue)

            numpy_array = file["eigenfunction"][:]  # Get the wavefunction
            wavefunctions.append(numpy_array)


    # Plotting
    for i, (wf, evalue) in enumerate(zip(wavefunctions, eigenvalues)):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        # Logarithmic Plots:
        # Plot the logarithm of the absolute value of the wavefunction against x
        # to infer exponential, power law, or other decays
        plt.plot(
            np.log(x),
            np.log(abs(wf) ** 2),#+ evalue,
            linewidth=1,
            # label=Rf"$\psi_{i}$",
            color=color,   
        )

    textstr = '\n'.join(
        (
            # fr'$E_4 = {eigenvalues[4]:.06f}$',
            # fr'$E_3 = {eigenvalues[3]:.06f}$',
            # fr'$E_2 = {eigenvalues[2]:.06f}$',
            # fr'$E_1 = {eigenvalues[1]:.06f}$',
            fr'$E_0 = {eigenvalues[0]:.06f}$',    
        )
    )
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    # plt.plot(x, V(x), linewidth=2, alpha=0.4, color='k')

    plt.grid(color='gray', linestyle=':', linewidth=0.5)

    # # INVESTIGATING THE DECAY
    # Define the specific x values.
    logx1 = 1.0675  # First x value
    logx2 = 2.0048  # Second x value

    # Find the indices of the closest values to logx1 and logx2 in x_values array.
    index1 = np.argmin(np.abs(np.log(x) - logx1))
    index2 = np.argmin(np.abs(np.log(x) - logx2))

    # Extract the corresponding wavefunction values.
    y1 = np.log(abs(wf) ** 2)[index1]
    y2 = np.log(abs(wf) ** 2)[index2]

    # Plot the points on the wavefunction.
    plt.scatter([logx1, logx2], [y1, y2], marker='.', color='red')
    # Draw a line between the points to visualize the gradient.
    plt.plot([logx1, logx2], [y1, y2], color='red', linestyle='--', label=f'gradient = {(y2 - y1) / (logx2 - logx1)}')


    plt.legend()
    plt.xlabel(R'Log($x$)')
    plt.ylabel(R'Log($|\psi(x)^{2}|$)')
    # plt.title("First few eigenstates")
    plt.title("Ground state of negative quartic Hamiltonian")
    plt.show()

