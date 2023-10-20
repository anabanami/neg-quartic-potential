# Time evolution using Hubbard Hamiltonian with unitary operator
# Ana Fabela 22/06/2023

# Import necessary libraries and modules
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import convolve
import h5py

# Configure matplotlib to display high DPI figures
plt.rcParams['figure.dpi'] = 500

#######################################################################################################


def plot_matrices():
    """
    Function to visualize the Hubbard Hamiltonian and its absolute value (on a log scale) as heat maps.
    The Hamiltonian is generated via the Bose_Hubbard_Hamiltonian function.
    """
    # Generate the Hamiltonian matrix
    H = Bose_Hubbard_Hamiltonian()

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

    # # Free space (no potential)
    # return np.zeros_like(x)

    # # harmonic oscillator
    # return (x ** 2)

    # # upside-down harmonic oscillator
    # return - (x ** 2)

    # unmodified negative quartic potential
    return -alpha * x ** 4

    # # a similar higher order deformation of HO
    # return -(x ** 8)


def plot_evolution_frame(y, state, time, i):
    """
    Function to plot and save the evolution of the wave function in position space at a specific time.
    Parameters:
    - y: Position coordinates.
    - state: Wave function at time `time`.
    - time: The particular instant of time.
    - i: Index used for saving the plot.
    """
    ax = plt.gca()
    # potential plot
    plt.plot(
        y,
        V(y),
        # label=R"$V(x) = 0$",
        # label=R"$V(x) = x^2$",
        # label=R"$V(x) = -x^2$",
        label=R"$V(x) = -x^4$",
        # label=R"$V(x) = -x^8$",
        color="gray",
        alpha=0.4,
    )
    # plot of prob. density of state
    plt.plot(
        y,
        3 * abs(state) ** 2,
        label=R"$|\psi(x, t)|^2$",
    )

    plt.legend()
    plt.ylim(-0.2, 2.5)
    # plt.xlim(-5, 5)
    # plt.ylabel(R"$|\psi(x, t)|^2$")
    # plt.xlabel(R"$x$")

    textstr = f"t = {time:05f}"
    # place a text box in upper left in axes coords
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        verticalalignment='top',
    )
    plt.tight_layout()
    plt.savefig(f"{folder}/{i}.pdf")
    # plt.show()
    plt.clf()


def plot_vs_k(state, time, i):
    ax = plt.gca()
    # for Fourier space
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Nx, dx))
    state = np.fft.fftshift(np.fft.fft(state))

    # prob. density plot
    plt.plot(
        kx,
        abs(state) ** 2,
        label=R"$|\psi(k_x, t)|^2$",
    )

    plt.legend()
    # plt.ylabel(R"$|\psi(k_x, t)|^2$")
    # plt.xlabel(R"$k_x$")
    # plt.ylim(-1.5, 3)
    # plt.xlim(-5, 5)
    textstr = f"t = {time:05f}"
    # place a text box in upper left in axes coords
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        verticalalignment='top',
    )
    plt.tight_layout()
    plt.savefig(f"{folder2}/{i}.pdf")
    # plt.show()
    plt.clf()



def x_variance(x, dx, Ψ):
    # Calculate Spatial variance of wavefunction (Ψ) per unit time
    # means squared distance from the mean
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]
    return dx / 2 * np.sum(g_right + g_left)


def Bose_Hubbard_Hamiltonian():
    """
    Function to generate the Hubbard Hamiltonian for a Bose system.
    Returns:
    - The Hubbard Hamiltonian matrix H.
    """
    # Initialize the Hamiltonian as a zero matrix
    H = np.zeros((N_sites, N_sites))
    # On-site interaction potential
    V_values = V(x)

    # Define the hopping and interaction terms
    # PERIODIC BCS
    for i in range(N_sites):
        # Hopping terms
        H[i, (i + 1) % N_sites] = -t
        H[(i + 1) % N_sites, i] = -t

        # On-site interaction
        H[i, i] = V_values[i]

    return H


def Unitary(M):
    """
    Function to generate the unitary matrix for time evolution.
    Returns:
    - The unitary time evolution operator U
    """
    A = -1j * M * dt / hbar
    return linalg.expm(A)


def TEV(x, wave):
    """
    Function to perform Time Evolution via the Hubbard Hamiltonian
    and store the results in an HDF5 file.
    Parameters:
    - x: Spatial coordinates array.
    - wave: Initial wave function.
    """
    # spatial variance
    SIGMAS_x_SQUARED = []

    states = []

    # Create a new HDF5 file
    file = h5py.File('Unitary_hubbard.hdf5', 'w')

    # time evolution
    H = Bose_Hubbard_Hamiltonian()
    U = Unitary(H)

    state = wave
    states.append(state)

    # VARIANCE
    sigma_x_squared = x_variance(x, dx, state)
    # VARIANCE
    # # ONLY IF USING a shifted IC
    # sigma_x_squared = x_variance(x-1, dx, state)

    SIGMAS_x_SQUARED.append(sigma_x_squared)
    dset = file.create_dataset("0.0", data=state)

    # generate timesteps
    times = np.arange(t_initial, t_final, dt)

    # ALL OTHER ts
    for time in times[1:]:
        print(f"t = {time}")
        state = U @ state
        states.append(state)
        # create a new dataset for each frame
        dset = file.create_dataset(f"{time}", data=state)
        # store variance
        sigma_x_squared = x_variance(x, dx, state)
        SIGMAS_x_SQUARED.append(sigma_x_squared)

    # Close the hdf5 file
    file.close()
    SIGMAS_x_SQUARED = np.array(SIGMAS_x_SQUARED)
    np.save(f"Unitary_hubbard_variance.npy", SIGMAS_x_SQUARED)

    PLOT_INTERVAL = 100
    for j, state in enumerate(states):
        if j % PLOT_INTERVAL == 0:
            print(f"t = {times[j]}")
            plot_evolution_frame(x, state, times[j], j)
            plot_vs_k(state, times[j], j)


def globals():
    """
    Function to define and return global variables used throughout the script.
    Includes physical constants, potential coefficients, spatial and temporal
    discretization, initial wave function, etc.
    Returns:
    - A tuple containing all global parameters.
    """
    # makes folders for simulation frames
    folder = Path(f'Unitary_hubbard')
    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.pdf')

    folder2 = Path(f'Unitary_hubbard-momentum')
    os.makedirs(folder2, exist_ok=True)
    os.system(f'rm {folder2}/*.pdf')

    hbar = 1

    # Bender units
    m = 1 / 2
    omega = 2

    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * omega))

    # coefficient for quartic potential
    alpha = 1

    # space dimension
    x_max = 45
    dx = 0.08
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # Lattice parameters
    N_sites = Nx
    cut = 5
    # Hopping strength
    t = 1 / (2 * dx ** 2)

    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 15

    # # # initial conditions: neg_quartic GS
    # # Open the HDF5 file for reading
    # with h5py.File('neg_quart_eigenvectors.h5', 'r') as file:
    #     # Access the 'eigenvectors' dataset
    #     eigenvectors = file['eigenvectors']
        
    #     # Get the Hubbard negative quartic GS
    #     first_set_of_eigenvectors = eigenvectors[:, 4]
        
    #     # Convert it to a numpy array
    #     wave = np.array(first_set_of_eigenvectors)


    # # initial conditions: HO ground state
    wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))

    # # # initial conditions: shifted HO ground state
    # wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-((x - 1) ** 2) / (2 * l1 ** 2))

    return (
        folder,
        folder2,
        hbar,
        m,
        omega,
        l1,
        alpha,
        N_sites,
        cut,
        t,
        x_max,
        dx,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
        wave,
    )


if __name__ == "__main__":

    # Retrieve global parameters

    (
        folder,
        folder2,
        hbar,
        m,
        omega,
        l1,
        alpha,
        N_sites,
        cut,
        t,
        x_max,
        dx,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
        wave,
    ) = globals()

    # # Generate and plot the Hamiltonian matrices
    # plot_matrices()


    # Perform time evolution and visualize
    TEV(x, wave)

    # [Print or log statements that help understanding the flow and outcomes...]
    # CHECK that IC is normalised
    print(f"\n{np.sum(abs(wave)**2)*dx = }")

    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"\n{dx = }")
    print(f"{dt = }")
