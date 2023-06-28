# Time evolution using Hubbard Hamiltonian with unitary operator
# Ana Fabela 22/06/2023

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

plt.rcParams['figure.dpi'] = 200

#######################################################################################################

def V(x):
    # return np.zeros_like(x)
    return - α * (x ** 4)

def plot_evolution_frame(y, states):
    i = 0
    for state in states:
        # potential
        # plt.plot(y, V(y), color="black", linewidth=2)

        # prob. density plot
        plt.plot(y, abs(state) ** 2)
        plt.ylabel(R"$|\psi(x, t)|^2$")
        plt.xlabel("x")
        # plt.legend()
        plt.ylim(-1.5, 3)
        plt.xlim(-4, 4)
        plt.savefig(f"{folder}/{i}.png")
        # plt.show()
        plt.clf()
        i += 1

#######################################################################################################

def Bose_Hubbard_Hamiltonian():
    # Initialize the Hamiltonian as a zero matrix
    H = np.zeros((N_sites, N_sites))

    # Define the hopping and interaction terms
    # PERIODIC BCS
    for i in range(N_sites):
        # Hopping terms
        H[i, (i + 1) % N_sites] = -t
        H[(i + 1) % N_sites, i] = -t

        # On-site no potential.
        # H[i, i] = 0

        # On-site interaction term with negative quartic potential
        H[i, i] =  - α * x[i] ** 4

    return H


def Unitary(M):
    A = -1j * M * dt / hbar
    return linalg.expm(A)


def TEV(wave):

    H = Bose_Hubbard_Hamiltonian()
    U = Unitary(H)
    # U = np.load(f"{n_sites}x{n_sites}_BHH.npy")

    WAVES = []
    for step in range(1000):
        wave = U @ wave  ### wave NEEDS TO MATCH DIMENSIONALITY OF unitary
        WAVES.append(wave)
    return WAVES



def globals():
    # makes folder for simulation frames
    folder = Path(f'Hubbard_Unitary')

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

    N_sites = 1024
    t = 1 # Need to find dis
    U = 0

    dx = 0.01
    x_max = 5.12
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # time dimension
    dt = 0.5
    t_initial = 0
    t_final = 2

    # initial conditions: HO ground state
    wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(
        -(x ** 2) / (2 * l1 ** 2)
)
    print(f"\n{np.sum(abs(wave)**2)*dx = }")  # is IC normalised???

    return (
        folder,
        hbar,
        m,
        ω,
        l1,
        α,
        N_sites,
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

    states = TEV(wave)
    plot_evolution_frame(x, states)