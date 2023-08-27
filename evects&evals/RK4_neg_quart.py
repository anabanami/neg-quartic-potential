# RK4 to solve negative quartic Eigenvalue problem with shooting method
# Ana Fabela 15/08/2023

"""
This code provides a way to numerically solve the Time-Independent Schrödinger Equation (TISE) for a specific potential, using the shooting method and the Runge-Kutta 4 (RK4) algorithm. The goal is to find the energy eigenvalues of a quantum system for a given potential. Let me break down the main aspects of the code:

    Imports and Settings:
        Essential Python libraries and modules like numpy, matplotlib, scipy.fft, and scipy.signal are imported.
        Default settings for plotting (with matplotlib) and printing (with numpy) are set.

    Potential Function (V):
        V(x)=0.5x4V(x)=0.5x4: Defines the quartic potential function in terms of a position variable xx.

    Schrödinger's Equation:
        This function returns the spatial derivatives of the wavefunction (ΨΨ) and its first spatial derivative (ΦΦ) using the Schrödinger equation. The potential VV and the energy EE are parameters of this equation.

    Runge-Kutta 4 (RK4) Method:
        Schrödinger_RK4 is a numerical integration method to solve ordinary differential equations (ODEs). Here, it's used to solve Schrödinger's equation for the given potential.

    Shooting Method:
        The Solve function uses the shooting method. Given two initial energy guesses (E1 and E2), the function integrates Schrödinger's equation from some boundary towards another boundary and checks if the solution matches the desired boundary condition at the other end. The energies E1 and E2 are then updated using the secant method until the solution converges or a maximum number of iterations is reached.

    Finding Eigenvalues:
        find_eigenvalues divides the energy range into intervals. For each interval, the Solve function is called to find an energy eigenvalue. Duplicate eigenvalues (from neighboring intervals) are filtered out.

    Global Parameters:
        globals function returns the global parameters like conv_crit (convergence criteria), m (mass), hbar (Planck's reduced constant), and space discretization parameters.

    Main Execution (__main__):
        This section initializes all parameters and calls the find_eigenvalues function.
        The found eigenvalues are printed out and compared with a set of known values (E_bender_RK4 and E_bender_wkb).
        Various parameters and results are printed to the console for inspection.

Key Points:
    The code is built to solve the TISE for a quartic potential. This potential is non-analytic, meaning it doesn't have a known exact solution, so numerical methods like the shooting method are appropriate.
    The Runge-Kutta 4 (RK4) method is chosen as the numerical integrator because of its accuracy.
    The shooting method, combined with the secant method, is applied to adjust the energy guesses iteratively until a solution meeting the desired boundary condition is found.
    The eigenvalues found represent the allowed energy levels of the quantum system under the defined quartic potential.

In summary, this code serves as a tool for finding the allowed energy levels of a quantum system governed by a quartic potential, providing an essential part of understanding the behavior of quantum systems in such potentials.
"""

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
    return 0.5 * x ** 4


# Schrödinger equation
def Schrödinger_eqn(x, Ψ, Φ, E, ω):
    """Ψ is the state, Φ is the fist spatial derivative of the state."""
    dΨ = Φ
    # my specific negative quartic PROBLEM
    dΦ = - 2 * (V(x, ω) + E) * Ψ
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

    Ψ_new = Ψ + (dx / 6) * (
        k1_Ψ + 2 * k2_Ψ + 2 * k3_Ψ + k4_Ψ
    )  # updated solution wavefunction
    Φ_new = Φ + (dx / 6) * (
        k1_Φ + 2 * k2_Φ + 2 * k3_Φ + k4_Φ
    )  # updated first derivative of solution

    return Ψ_new, Φ_new


"""Running the RK4 Integrator through the Grid one xn in x at the time:
    For each point xn in this grid, I update the wavefunction Ψ using Schrödinger_RK4().
"""


def Solve(E1, E2, ω, conv_crit):
    # Solve TISE using RK4 with shooting method
    max_iteration = 1000
    iteration = 0
    wavefunction = []  # wavefunction value for each xn

    while iteration < max_iteration:
        # SETUP REVERSED RK4 integrals

        # for E1
        # print(f"{E1 = }")
        Ψ1, Φ1 = 1, 0  # initial conditions for Ψ and dΨ/dx

        # RK4 integration
        for i, xn in reversed(list(enumerate(x))):
            Ψ1, Φ1 = Schrödinger_RK4(xn, Ψ1, Φ1, E1, -dx)
        # A(E1) test condition
        A1 = Ψ1

        # for E2
        # print(f"{E2 = }")
        Ψ2, Φ2 = 1, 0  # initial conditions for Ψ and dΨ/dx

        for i, xn in reversed(list(enumerate(x))):
            Ψ2, Φ2 = Schrödinger_RK4(xn, Ψ2, Φ2, E2, -dx)
            wavefunction.append(Ψ2)
        # A(E2) test condition
        A2 = Ψ2

        # check convergence & normalise
        if abs(A2) < conv_crit:
            wavefunction = np.array(wavefunction)
            integral = np.sum(np.abs(wavefunction) ** 2) * dx
            Ψ_normalised = wavefunction / np.sqrt(integral)
            return E2, Ψ_normalised

        # SECANT METHOD (Energy) UPDATE STEP
        E_new = (E2 * A1 - E1 * A2) / (A1 - A2)
        print(f"{E_new = }")

        # reset for next iteration
        E1 = E2
        E2 = E_new

        iteration += 1
        if iteration >= max_iteration:
            print("Maximum number of iterations reached without convergence.")

    # if the while loop ends without returning, raise an error or return None
    raise ValueError("Did not converge")


def find_eigenvalues(E_min, E_max, num_intervals, ω, conv_crit):
    # initialise list
    eigenvalues = np.array([])
    # define energy range to search
    E_range = np.linspace(E_min, E_max, num_intervals + 1)

    for i in range(num_intervals):
        E1 = E_range[i]
        E2 = E_range[i] + update_E
        
        print(f"shooting between {E1 =} and {E2 = }")

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
    Nx = int(x_max / dx)
    x = np.linspace(0, x_max, Nx, endpoint=False)

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

    # Bender energies to compare to
    E_bender_RK4 = np.array([1.477150, 6.003386, 11.802434, 18.458819, 25.791792])
    # more Bender energies to compare to
    E_bender_wkb = np.array([1.3765, 5.9558, 11.7690, 18.4321, 25.7692])

    # HO frequency
    ω = 1  ### MAYBE CHANGE THIS UP (make variable)

    E_min = 1
    E_max = 26
    update_E = 0.01
    num_intervals = 500
    eigenvalues = find_eigenvalues(E_min, E_max, num_intervals, ω, conv_crit)
    eigenvalues = np.sort(eigenvalues)  # Sort the eigenvalues in ascending order

    # only care about eigenvalues above some value
    # some_threshold = 0.5
    # filtered_eigenvalues = np.array([E for E in eigenvalues if E > some_threshold])
    # print(f"The first 5 eigenvalues in the list are:\n{filtered_eigenvalues[:5]}")

    print(f"The first 5 eigenvalues in the list are:\n{eigenvalues[:5]}")

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
