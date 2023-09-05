# RK4 to solve negative quartic Eigenvalue problem with shooting method
# Ana Fabela 15/08/2023

"""
This code provides a way to numerically solve the Time-Independent Schrödinger Equation (TISE) for a specific potential, using the shooting method and the Runge-Kutta 4 (RK4) algorithm.
 The goal is to find the energy eigenvalues of a quantum system for a given potential. Let me break down the main aspects of the code:

    Imports and Settings:
        Essential Python libraries and modules like numpy, matplotlib, scipy.fft, and scipy.signal are imported.
        Default settings for plotting (with matplotlib) and printing (with numpy) are set.

    Potential Function (V):
        V(x)=0.5x**4: Defines the quartic potential function in terms of a position variable x.

    Schrödinger's Equation:
        This function returns the spatial derivatives of the wavefunction (Ψ) and its first spatial derivative (Φ) using the Schrödinger equation.
         The potential V and the energy E are parameters of this equation.

    Runge-Kutta 4 (RK4) Method:
        Schrödinger_RK4 is a numerical integration method to solve ordinary differential equations (ODEs).
         Here, it's used to solve Schrödinger's equation for the given potential.

    Shooting Method:
        The Solve function uses the shooting method. 
        Given two initial energy guesses (E1 and E2), the function integrates Schrödinger's equation from some boundary towards another boundary 
        and checks if the solution matches the desired boundary condition at the final point of integration.
        The energies E1 and E2 are then updated using an interval bisection method until the solution converges or a maximum number of iterations is reached.

    Finding Eigenvalues:
        find_multiple_eigenvalues divides the energy range into intervals. For each interval, the Solve function is called to find an energy eigenvalue. 
        Duplicate eigenvalues (from neighboring intervals) are filtered out.

    Global Parameters:
        initialisation_parameters function returns the global parameters like tolerance and space discretization parameters.

    Main Execution (__main__):
        This section initializes all parameters and declare an initial value to solve the Schrodinger equation via the integrate and the find_multiple_eigenvalues functions.
        Various parameters and results are printed to the console for inspection.
        The found eigenvalues are printed out and compared with a set of known values (E_bender_RK4 and E_bender_wkb).


Key Points:
    The code is built to solve the TISE for a negative quartic potential. This potential is non-analytic, meaning it doesn't have a known exact solution, so numerical methods
    like the shooting method are appropriate.
    The Runge-Kutta 4 (RK4) method is chosen as the numerical integrator because of its accuracy.
    The shooting method, combined with the bisection method, is applied to adjust the energy guesses iteratively until a solution meeting the desired boundary condition is found.
    The eigenvalues found represent the allowed energy levels of the quantum system under the defined negative quartic potential.

In summary, this code serves as a tool for finding the allowed energy levels of a quantum system governed by a negative quartic potential, 
providing an essential part of understanding the behavior of quantum systems in such potentials.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import h5py


def save_to_hdf5(filename, eigenvalues):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset("eigenvalues", data=eigenvalues)


def V(x):
    return -0.5 * x ** 2 # HO
    # return 0.5 * x ** 4# negative quartic potential


# Schrödinger equation
def Schrödinger_eqn(x, Ψ, Φ, E):
    """Ψ is the state, Φ is the fist spatial derivative of the state."""
    dΨ = Φ
    dΦ = -2 * (V(x) + E) * Ψ  # This applies 2E_A = E_B scaling to match Bender's energy
    return dΨ, dΦ


# Algorithm Runge-Kutta 4 for integrating TISE eigenvalue problem
def Schrödinger_RK4(x, Ψ, Φ, E, dx):
    k1_Ψ, k1_Φ = Schrödinger_eqn(x, Ψ, Φ, E)
    k2_Ψ, k2_Φ = Schrödinger_eqn(x + 0.5 * dx, Ψ + 0.5 * dx * k1_Ψ, Φ + 0.5 * dx * k1_Φ, E)
    k3_Ψ, k3_Φ = Schrödinger_eqn(x + 0.5 * dx, Ψ + 0.5 * dx * k2_Ψ, Φ + 0.5 * dx * k2_Φ, E)
    k4_Ψ, k4_Φ = Schrödinger_eqn(x + dx, Ψ + dx * k3_Ψ, Φ + dx * k3_Φ, E)

    Ψ_new = Ψ + (dx / 6) * (k1_Ψ + 2 * k2_Ψ + 2 * k3_Ψ + k4_Ψ)  # updated solution wavefunction
    Φ_new = Φ + (dx / 6) * (k1_Φ + 2 * k2_Φ + 2 * k3_Φ + k4_Φ)  # updated first derivative of solution

    return Ψ_new, Φ_new


def integrate(E, Ψ, Φ, dx):
    """Reversed running of the RK4 integrator through the grid, one xn in x at a time:
    For each point xn in this grid, I update the wavefunction Ψ using Schrödinger_RK4(...)."""
    for i, xn in reversed(list(enumerate(x))):
        Ψ, Φ = Schrödinger_RK4(xn, Ψ, Φ, E, -dx)
    return Ψ, Φ


def bisection(E1, E2, A1, AΦ1, tolerance, Ψ1, Φ1, dx):
    while abs(E1 - E2) > tolerance:
        E_new = (E1 + E2) / 2 # bisect interval
        Ψ_new, Φ_new = integrate(E_new, Ψ1, Φ1, dx) # integrate from boundary (x_max) to zero

        # find signs of the solution and its first derivative
        A_new = np.sign(Ψ_new)
        AΦ_new = np.sign(Φ_new)
        
        if A_new != A1 or AΦ_new == AΦ1:
            """if sign(Ψ_new) is not equal to sign(Ψ1) or sign(Φ_new) is equal to sign(Φ1). make E_new the right side interval boundary."""
            E2 = E_new
        else:
            E1 = E_new
            
        A = A_new
        AΦ = AΦ_new
    return E_new


def find_multiple_eigenvalues(E_min, E_max, dE, tolerance, Ψ_init, Φ_init, dx):
    eigenvalues = []
    
    E1 = E_min
    while E1 < E_max:
        E2 = E1 + dE
        Ψ1, Φ1 = integrate(E1, Ψ_init, Φ_init, dx)
        Ψ2, Φ2 = integrate(E2, Ψ_init, Φ_init, dx)
        
        A1 = np.sign(Ψ1)
        A2 = np.sign(Ψ2)
        AΦ1 = np.sign(Φ1)
        AΦ2 = np.sign(Φ2)
        
        if A1 != A2 or AΦ1 == AΦ2:
            eigenvalue = bisection(E1, E2, A1, AΦ1, tolerance, Ψ_init, Φ_init, dx)
            eigenvalues.append(eigenvalue)
            E1 = E2 + dE  # skip to next interval, avoiding the eigenvalue just found
        else:
            E1 = E2  # no eigenvalue in this range, move to next interval
    
    return eigenvalues

def initialisation_parameters():
    tolerance = 1e-3

    dx = 0.01

    # space dimension
    x_max = 15
    Nx = int(x_max / dx)
    x = np.linspace(0, x_max, Nx, endpoint=False)

    return (
        tolerance,
        dx,
        x_max,
        Nx,
        x,
    )


if __name__ == "__main__":

    tolerance, dx, x_max, Nx, x = initialisation_parameters()

    E_HO = [0.5, 1.5, 2.5, 3.5, 4.5]
    # Bender energies to compare to
    E_bender_RK4 = [1.477150, 6.003386, 11.802434, 18.458819, 25.791792]
    # more Bender energies to compare to
    E_bender_wkb = [1.3765, 5.9558, 11.7690, 18.4321, 25.7692]

    # * ~ENERGY~ *
    E_min = 0
    E_max = 28
    dE = 0.1
    # dE = 0.002 # <<<<<< THIS ONE IS ANNOYING

    # # HO POTENTIAL I.V.:
    y = x_max * np.sqrt(x_max ** 2) / (2 * np.sqrt(2))
    Ψ_init, Φ_init = (np.exp(y), np.exp(y) * (np.sqrt(x_max ** 2) / np.sqrt(2)))

    # # Neg Quart Potential I.V.
    # # ONLY Checking solution in the form: 2 Real(Ψ) =  2 B cos(y) 
    # y = x_max ** 3 / (3 * np.sqrt(2)) # NEGATIVE QUARTIC POTENTIAL 
    # Ψ_init, Φ_init = (np.cos(y), - x_max**2 * np.sin(y) * np.sqrt(2))

    Ψ1, Φ1 = Ψ_init, Φ_init
    Ψ2, Φ2 = Ψ_init, Φ_init

    # Integrate for given E values
    Ψ1, Φ1 = integrate(E_min, Ψ1, Φ1, dx)
    Ψ2, Φ2 = integrate(E_max, Ψ2, Φ2, dx)

    evals = find_multiple_eigenvalues(E_min, E_max, dE, tolerance, Ψ_init, Φ_init, dx)
    sliced_list = evals[:5]
    formatted_list = [f"{evalue:.5f}" for evalue in sliced_list]

    print(f"\n{dE = }")
    # Printing the formatted list
    print(f"\nevals = {formatted_list}")
    print(f"\n{E_HO = }")
    # print(f"\n{E_bender_RK4 = }")
    # print(f"\n{E_bender_wkb = }")
