# Time evolution using Hubbard Hamiltonian with unitary operator
# Ana Fabela 22/06/2023

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import convolve


plt.rcParams['figure.dpi'] = 200

#######################################################################################################


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def smooth_restricted_V(x):
    V = np.ones_like(x) * x[cut] ** 4
    V[cut : Nx - cut] = x[cut : Nx - cut] ** 4
    ## smoooth by pts=3
    V = gaussian_smoothing(V, 3)
    return V


def V(x):
    # return np.zeros_like(x)
    # return - 0.5 * (x ** 2)
    return -α * smooth_restricted_V(x)


def plot_evolution_frame(y, state, time, i):
    # potential
    plt.plot(y, V(y), color="black", linewidth=2, label="V(x)")
    # prob. density plot
    plt.plot(y, abs(state) ** 2, label=R"$|\psi(x, t)|^2$")
    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel("x")
    plt.legend()
    plt.ylim(-1.5, 3)
    plt.xlim(-5, 5)
    plt.title(f"t = {time:05f}")
    plt.savefig(f"{folder}/{i}.png")
    # plt.show()
    plt.clf()

###############################################################################
def x_variance(x, dx, Ψ):
    # Calculate Spatial variance of wavefunction (Ψ) per unit time
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]
    return dx / 2 * np.sum(g_right + g_left)


###############################################################################

#######################################################################################################


def Bose_Hubbard_Hamiltonian():
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
    A = -1j * M * dt / hbar
    return linalg.expm(A)


def TEV(x, wave):
    # time evolution
    H = Bose_Hubbard_Hamiltonian()
    U = Unitary(H)

    state = wave
    time_steps = np.arange(t_initial, t_final, dt)

    # spatial variance
    SIGMAS_x_SQUARED = []

    states = []
    times = []
    for time in time_steps:
        times.append(time)
        state = U @ state
        states.append(state)

    i = 0
    PLOT_INTERVAL = 12
    for j, state in enumerate(states):
        if i % PLOT_INTERVAL == 0:
            print(f"t = {times[j]}")
            plot_evolution_frame(x, state, times[j], i)
        sigma_x_squared = x_variance(x, dx, state)
        SIGMAS_x_SQUARED.append(sigma_x_squared)
        i += 1

    SIGMAS_x_SQUARED = np.array(SIGMAS_x_SQUARED)
    np.save(f"8._variance.npy", SIGMAS_x_SQUARED)



def plot_matrices():
    # Generate the Hamiltonian
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


def globals():
    # makes folder for simulation frames
    folder = Path(f'Hubbard_Unitary_TEV')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # natural units
    hbar = 1
    m = 1
    ω = 1
    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))

    # coefficient for quartic potential
    α = 4

    N_sites = 900
    cut = 225

    dx = 0.1
    # Hopping strength
    t = 1 / (2 * dx ** 2)

    # space dimension
    x_max = 45
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 1

    # initial conditions: HO ground state
    wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))

    return (
        folder,
        hbar,
        m,
        ω,
        l1,
        α,
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
    """FUNCTION CALLS"""

    (
        folder,
        hbar,
        m,
        ω,
        l1,
        α,
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

    plot_matrices()

    TEV(x, wave)

    print(f"\n{np.sum(abs(wave)**2)*dx = }")  # is IC normalised???

    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"x_cut_left = {x[cut]= }")
    print(f"x_cut_right = {x[Nx-cut]= }")

    print(f"\n{dx = }")
    print(f"{dt = }")
