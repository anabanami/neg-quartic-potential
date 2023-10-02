import os
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


def V(x):
    return -(x ** 4)


def initialisation_parameters():
    x_max = 30
    dx = 1e-5
    Nx = int(x_max / dx)
    x = np.linspace(0.9, x_max, Nx, endpoint=False)

    return (
        dx,
        x_max,
        Nx,
        x,
    )


def get_files():
    eigenvalues = []
    wavefunctions = []

    for i in range(5):

        with h5py.File(f"{i}.h5", "r") as file:
            # Get the eigenvalue data and convert it to a float
            evalue = file["eigenfunction"].attrs["eigenvalue"]
            eigenvalues.append(evalue)

            numpy_array = file["eigenfunction"][:]  # Get the wavefunction
            wavefunctions.append(numpy_array)

    return eigenvalues, wavefunctions


if __name__ == "__main__":

    dx, x_max, Nx, x = initialisation_parameters()

    eigenvalues, wavefunctions = get_files()

    # Plotting
    ax = plt.gca()

    wf, evalue = wavefunctions[0], eigenvalues[0]
    # wf, evalue = wavefunctions[1], eigenvalues[1]
    # wf, evalue = wavefunctions[2], eigenvalues[2]
    # wf, evalue = wavefunctions[3], eigenvalues[3]
    # wf, evalue = wavefunctions[4], eigenvalues[4]

    # Logarithmic Plots:
    # Plot the logarithm of the absolute value of the wavefunction against x
    # to infer exponential, power law, or other decays
    plt.plot(
        np.log(x),
        np.log(abs(wf) ** 2),
        linewidth=1,
        label=R"$\psi_0$",
        # label=R"$\psi_1$",
        # label=R"$\psi_2$",
        # label=R"$\psi_3$",
        # label=R"$\psi_4$",
    )

    textstr = '\n'.join((fr'$E = {eigenvalues[0]:.06f}$',))
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    plt.grid(color='gray', linestyle=':', linewidth=0.5)

    # # INVESTIGATING THE DECAY
    # Select the specific points for gradient sampling for each of the first 5 states

    logx1 = 0.9535  # First x value
    logx2 = 2.0067  # Second x value

    # logx1 = 0.9535  # First x value
    # logx2 = 2.00335  # Second x value

    # logx1 = 0.9535  # First x value
    # logx2 = 2.00011  # Second x value

    # logx1 = 0.9535  # First x value
    # logx2 = 1.99710  # Second x value

    # logx1 = 0.9535  # First x value
    # logx2 = 1.99388  # Second x value

    # Find the indices of the closest values to logx1 and logx2 in x_values array.
    index1 = np.argmin(np.abs(np.log(x) - logx1))
    index2 = np.argmin(np.abs(np.log(x) - logx2))

    # Extract the corresponding wavefunction values.
    y1 = np.log(abs(wf) ** 2)[index1]
    y2 = np.log(abs(wf) ** 2)[index2]

    # Plot the points on the wavefunction.
    plt.scatter([logx1, logx2], [y1, y2], marker='.', color='red')
    # Draw a line between the points to visualize the gradient.
    plt.plot(
        [logx1, logx2],
        [y1, y2],
        color='red',
        linestyle='--',
        label=f'gradient = {(y2 - y1) / (logx2 - logx1)}',
    )

    plt.legend()
    plt.xlabel(R'Log($x$)')
    plt.ylabel(R'Log($|\psi(x)^{2}|$)')
    # plt.title("First few eigenstates")
    plt.title("Ground state of negative quartic Hamiltonian")
    plt.show()
