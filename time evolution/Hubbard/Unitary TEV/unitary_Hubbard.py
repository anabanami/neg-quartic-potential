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

def x_variance(x, dx, Ψ):
    """
    Calculate Spatial variance of wave function (Ψ) per unit time
    """
    # means squared distance from the mean
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]
    return dx / 2 * np.sum(g_right + g_left)


def average_position(x, dx, Ψ):
    """
    Calculate the average position, <x>, of wave function (Ψ) per unit time
    """
    integrand = np.conj(Ψ) * x * Ψ
    integrand_right = integrand[1:]
    integrand_left = integrand[:-1]
    return dx / 2 * np.sum(integrand_right + integrand_left)


def average_momentum(x, dx, Ψ):
    """
    Calculate the average momentum, <p>, of wave function (Ψ) per unit time.
    """
    # Calculate the derivative of Ψ using the central difference method
    dΨ_dx = (Ψ[2:] - Ψ[:-2]) / (2 * dx)

    # We'll truncate Ψ and x by 1 on each side since dΨ_dx is shorter.
    integrand = -1j * hbar * np.conj(Ψ[1:-1]) * dΨ_dx
    integrand_right = integrand[1:]
    integrand_left = integrand[:-1]
    return dx / 2 * np.sum(integrand_right + integrand_left)


def Shannon_entropy(x, dx, Ψ):
    """
    Calculate the Entropy, S of wave function (Ψ) per unit time
    """
    prob_dens = np.conj(Ψ) * Ψ
    integrand = prob_dens * np.log(prob_dens)
    integrand_right = integrand[1:]
    integrand_left = integrand[:-1]
    return dx / 2 * np.sum(integrand_right + integrand_left)


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
    # positions
    average_positions = []
    # momenta
    average_momenta = []
    # Shannon entropies
    shannon_entropies = []

    # states
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
    # # ONLY IF USING a shifted IC
    # sigma_x_squared = x_variance(x-1, dx, state)

    average_positions.append(sigma_x_squared)
    dset = file.create_dataset("0.0", data=state)

    # <x>
    average_x = average_position(x, dx, state)
    average_positions.append(average_x)
    # <p>
    average_p = average_momentum(x, dx, state)
    average_momenta.append(average_p)

    entropy = Shannon_entropy(x, dx, state)
    shannon_entropies.append(entropy)


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
        # <x>
        average_x = average_position(x, dx, state)
        average_positions.append(average_x)
        # <p>
        average_p = average_momentum(x, dx, state)
        average_momenta.append(average_p)
        # Entropy
        entropy = Shannon_entropy(x, dx, state)
        shannon_entropies.append(entropy)


    # Close the hdf5 file
    file.close()

    SIGMAS_x_SQUARED = np.array(SIGMAS_x_SQUARED)
    np.save(f"Unitary_hubbard_variance.npy", SIGMAS_x_SQUARED)

    average_positions = np.array(average_positions)
    np.save(f"Unitary_hubbard_avg_positions.npy", average_positions)

    average_momenta = np.array(average_momenta)
    np.save(f"Unitary_hubbard_avg_momenta.npy", average_momenta)

    shannon_entropies = np.array(shannon_entropies)
    np.save(f"Unitary_hubbard_shannon_entropies.npy", shannon_entropies)

    PLOT_INTERVAL = 100
    for j, state in enumerate(states):
        if j % PLOT_INTERVAL == 0:
            print(f"t = {times[j]}")
            plot_evolution_frame(x, state, times[j], j)
            # plot_vs_k(state, times[j], j)


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
