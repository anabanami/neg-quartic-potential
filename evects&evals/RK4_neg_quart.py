# RK4 to solve negative quartic Eigenvalue problem with secant shooting method
# Ana Fabela 15/08/2023

import os
from pathlib import Path
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import convolve
import scipy.special as sc
from matplotlib.ticker import FormatStrFormatter


plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)


def V(x, ω):
    # m = 1/2 # (n + 0.5) form for HO
    # return 0.5 * m * (ω * x) ** 2
    return -0.5 * x ** 4


# Schrödinger equation
def Schrödinger_eqn(x, Ψ, Φ, E, ω):
    dΨ = Φ
    dΦ = (V(x, ω) - E) * Ψ
    return dΨ, dΦ


# Algorithm Runge-Kutta 4 for solving TISE eigenvalue problem
def Schrödinger_RK4(x, Ψ, Φ, E, dx):
    k1_Ψ, k1_Φ = Schrödinger_eqn(x, Ψ, Φ, E, ω)
    k2_Ψ, k2_Φ = Schrödinger_eqn(
        x + 0.5 * dx, Ψ + 0.5 * dx * k1_Ψ, Φ + 0.5 * dx * k1_Φ, E, ω
    )
    k3_Ψ, k3_Φ = Schrödinger_eqn(
        x + 0.5 * dx, Ψ + 0.5 * dx * k2_Ψ, Φ + 0.5 * dx * k2_Φ, E, ω
    )
    k4_Ψ, k4_Φ = Schrödinger_eqn(x + dx, Ψ + dx * k3_Ψ, Φ + dx * k3_Φ, E, ω)

    Ψ_new = Ψ + (dx / 6) * (k1_Ψ + 2 * k2_Ψ + 2 * k3_Ψ + k4_Ψ)
    Φ_new = Φ + (dx / 6) * (k1_Φ + 2 * k2_Φ + 2 * k3_Φ + k4_Φ)

    return Ψ_new, Φ_new


"""Running the RK4 Integrator through the Grid one xn in x at the time:
    For each point xn in this grid, I update the wavefunction Ψ using Schrödinger_RK4().


Ψ is the state, Φ is the fist spatial derivative of the state..."""


def Solve(E1, E2, ω, conv_crit):
    # Solve TISE using RK4 with shooting method (Secant method energy update)
    max_iteration = 1000
    iteration = 0

    Ψ1 = 0  # initial conditions for wavefunction
    Φ1 = 1  # initial conditions for dΨ/dx
    
    Ψ2 = ?  # we don't impose this BC we find it 
    Φ2 = 1  # we don't impose this BC we find it 

    if iteration >= max_iteration:
        warnings.warn("Maximum number of iterations reached without convergence.")

    while iteration < max_iteration:
        wavefunction = []  # list that stores the value of Ψ at each xn

        for i, xn in enumerate(x):
            Ψ2, Φ2 = Schrödinger_RK4(xn, Ψ2, Φ2, E2, dx)
            wavefunction.append(Ψ2)

        if abs(wavefunction[0]) < conv_crit and abs(wavefunction[-1]) < conv_crit:
            break

        # SECANT METHOD UPDATE STEP
        # Check the boundary condition and update E accordingly
        E_new = E2 - Ψ2 * (E2 - E1) / (Ψ2 - Ψ1)

        # check convergence
        if abs(E_new - E2) < conv_crit:
            break

        # reset for next iteration
        E1 = E2
        E2 = E_new
        Ψ1 = Ψ2
        Ψ2 = 0
        Φ2 = 1

        iteration += 1

    # Normalization of the Wavefunction
    integral = np.sum(np.abs(wavefunction) ** 2) * dx
    Ψ_normalised = wavefunction / np.sqrt(integral)

    return E2, Ψ_normalised


def find_eigenvalues(E_min, E_max, num_intervals, ω, conv_crit):
    # initialise list
    eigenvalues = np.array([])
    # define energy range to search
    E_range = np.linspace(E_min, E_max, num_intervals + 1)

    for i in range(num_intervals):
        E1 = E_range[i]
        E2 = E_range[i] + update_E

        E, Psi = Solve(E1, E2, ω, conv_crit)

        # Check if this eigenvalue is a new one (not a duplicate)
        for i in range(num_intervals):
            if not any(abs(E - E_existing) < conv_crit for E_existing in eigenvalues):
                eigenvalues = np.append(eigenvalues, E)
    print(f"\nFound an array of eigenvalues with shape: {np.shape(eigenvalues)} ")

    return eigenvalues


def globals():

    conv_crit = 1e-6

    m = 1  #
    hbar = 1

    dx = 0.01

    # space dimension
    x_max = 15
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    return (
        conv_crit,
        m,
        hbar,
        dx,
        x_max,
        Nx,
        x,
    )


if __name__ == "__main__":

    conv_crit, m, hbar, dx, x_max, Nx, x = globals()

    # HO energies to compare
    # E_HO = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    # Bender energies to compare
    E_bender_RK4 = np.array([1.477150, 6.003386, 11.802434, 18.458819, 25.791792])
    # more Bender energies to compare
    E_bender_wkb = np.array([1.3765, 5.9558, 11.7690, 18.4321, 25.7692])

    # HO frequency
    ω = 1  ### NEED TO CHANGE THIS UP (make variable)

    E_min = 1
    E_max = 26
    update_E = 0.25
    num_intervals = 500
    eigenvalues = find_eigenvalues(E_min, E_max, num_intervals, ω, conv_crit)
    eigenvalues = np.sort(eigenvalues)  # Sort the eigenvalues in ascending order

    some_threshold = 0.5
    filtered_eigenvalues = np.array([E for E in eigenvalues if E > some_threshold])

    print(f"The first 5 eigenvalues in the list are:\n{filtered_eigenvalues[:5]}")
    print(f"\nMy scaled Energies to match Bender's\n{2 * filtered_eigenvalues[:5]}")

    # print(f"\nThese are the energies that we expect:\n{E_HO =}")
    print(f"\nThese are the energies that we expect:\n{E_bender_RK4 =}")
    # print(f"\nThese are the energies that we expect:\n{E_bender_wkb =}")


    print("\nTESTING PARAMETERS:")
    print(f"{conv_crit = }")
    print(" *~ spatial space ~*")
    print(f"{x_max = }")
    print(f"{dx = }")
    print(f"{Nx = }")
    print(" *~ Energy ~*")
    print(f"{ω = }")
    print(f"{E_min = }, {E_max = }")
    print(f"{update_E = }")
    print(f"{num_intervals = }")
