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

# RK4 to solve neg quartic Eigenvalue problem with shooting method
# Ana Fabela 27/08/2023

import numpy as np
import h5py


def normalize_wavefunction(wavefunction, dx):
    integral = np.sum(np.abs(wavefunction) ** 2) * dx
    normalization_constant = np.sqrt(2 / integral)
    return wavefunction * normalization_constant


def save_to_hdf5(filename, eigenvalue, eigenfunction):
    with h5py.File(filename, 'w') as hf:
        dataset = hf.create_dataset("eigenfunction", data=eigenfunction)
        dataset.attrs["eigenvalue"] = eigenvalue


def V(x):
    return 0.5 * x ** 4


# Schrödinger equation
def Schrödinger_eqn(x, Ψ, Φ, E):
    """Ψ is the state, Φ is the fist spatial derivative of the state."""
    dΨ = Φ
    # my specific negative quartic problem. This applies 2E_A = E_B scaling to match Bender's energy
    dΦ = -2 * (V(x) + E) * Ψ
    return dΨ, dΦ


# Algorithm Runge-Kutta 4 for integrating TISE eigenvalue problem
def Schrödinger_RK4(x, Ψ, Φ, E, dx):
    k1_Ψ, k1_Φ = Schrödinger_eqn(x, Ψ, Φ, E)
    k2_Ψ, k2_Φ = Schrödinger_eqn(
        x + 0.5 * dx, Ψ + 0.5 * dx * k1_Ψ, Φ + 0.5 * dx * k1_Φ, E
    )
    k3_Ψ, k3_Φ = Schrödinger_eqn(
        x + 0.5 * dx, Ψ + 0.5 * dx * k2_Ψ, Φ + 0.5 * dx * k2_Φ, E
    )
    k4_Ψ, k4_Φ = Schrödinger_eqn(x + dx, Ψ + dx * k3_Ψ, Φ + dx * k3_Φ, E)

    Ψ_new = Ψ + (dx / 6) * (
        k1_Ψ + 2 * k2_Ψ + 2 * k3_Ψ + k4_Ψ
    )  # updated solution wavefunction
    Φ_new = Φ + (dx / 6) * (
        k1_Φ + 2 * k2_Φ + 2 * k3_Φ + k4_Φ
    )  # updated first derivative of solution

    return Ψ_new, Φ_new


def integrate(E, Ψ, Φ, dx, save_wavefunction=False):
    """***Reversed*** running of the RK4 Integrator through the grid one xn in x at a time:
    For each point xn in this grid, I update the wavefunction Ψ using Schrödinger_RK4()."""
    wavefunction = []
    for i, xn in reversed(list(enumerate(x))):
        # Reduction of order of ODE
        Ψ, Φ = Schrödinger_RK4(xn, Ψ, Φ, E, -dx)
        wavefunction.append(Ψ)

    if save_wavefunction:
        normalized_wavefunction = normalize_wavefunction(np.array(wavefunction), dx)
        save_to_hdf5(f"wavefunction_{E}.h5", E, normalized_wavefunction)
    return Ψ, Φ


def bisection(E1, E2, A1, AΦ1, tolerance, Ψ1, Φ1, dx):
    k = 0
    while tolerance <= abs(E1 - E2):
        # print(f"{k = }")
        print(f"************* Entering bisection")
        E_new = (E1 + E2) / 2
        Ψ_new, Φ_new = integrate(E_new, Ψ1, Φ1, dx)

        A_new = np.sign(Ψ_new)
        AΦ_new = np.sign(Φ_new)

        if A_new != A1:
            E2 = E_new
        elif AΦ_new != AΦ1:
            E2 = E_new
        else:
            E1 = E_new

        A = A_new
        AΦ = AΦ_new

        k += 1

    # Save the wavefunction corresponding to E_new
    print(f"*** ~~saving wavefunction for eigenvalue {E_new}~~ ***")
    _, _ = integrate(E_new, Ψ1, Φ1, dx, save_wavefunction=True)
    return E_new


def find_multiple_odd_eigenvalues(E_min, E_max, dE, tolerance, Ψ_init, Φ_init, dx):
    eigenvalues = []
    i = 0
    E1 = E_min
    while E1 < E_max:
        print(f"{i = }")
        E2 = E1 + dE
        Ψ1, Φ1 = integrate(E1, Ψ_init, Φ_init, dx)
        Ψ2, Φ2 = integrate(E2, Ψ_init, Φ_init, dx)

        A1 = np.sign(Ψ1)
        A2 = np.sign(Ψ2)
        AΦ1 = np.sign(Φ1)
        AΦ2 = np.sign(Φ2)

        if A1 != A2:
            # testing sign of solution
            print("\nlet's-a go")
            print(f"{E1 = }")
            print(f"{E2 = }")
            eigenvalue = bisection(E1, E2, A1, AΦ1, tolerance, Ψ1, Φ1, dx)
            eigenvalues.append(eigenvalue)
            E1 = E2 + dE  # skip to next interval, avoiding the eigenvalue just found
        else:
            E1 = E2  # no eigenvalue in this range, move to next interval
        i += 1

    return eigenvalues


def find_multiple_even_eigenvalues(E_min, E_max, dE, tolerance, Ψ_init, Φ_init, dx):
    eigenvalues = []
    j = 0
    E1 = E_min
    while E1 < E_max:
        print(f"{j = }")
        E2 = E1 + dE
        Ψ1, Φ1 = integrate(E1, Ψ_init, Φ_init, dx)
        Ψ2, Φ2 = integrate(E2, Ψ_init, Φ_init, dx)

        A1 = np.sign(Ψ1)
        A2 = np.sign(Ψ2)
        AΦ1 = np.sign(Φ1)
        AΦ2 = np.sign(Φ2)

        if AΦ1 != AΦ2:
            # testing sign of first derivative of solution
            print("\nMama mia!")
            print(f"{E1 = }")
            print(f"{E2 = }")
            eigenvalue = bisection(E1, E2, A1, AΦ1, tolerance, Ψ_init, Φ_init, dx)
            eigenvalues.append(eigenvalue)
            E1 = E2 + dE  # skip to next interval, avoiding the eigenvalue just found
        else:
            E1 = E2  # no eigenvalue in this range, move to next interval
        j += 1

    return eigenvalues


def initialisation_parameters():
    tolerance = 1e-15

    dx = 2.25e-4

    # space dimension
    x_max = 8
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

    # * ~ENERGY~ *
    E_min = 1
    E_max = 27
    dE = 0.04

    E_even = [1.477150, 11.802434, 25.791792]
    E_odd = [6.003386, 18.458819]

    # NEG QUART POTENTIAL I.V.:
    y = x_max ** 3 / (3 * np.sqrt(2))
    Ψ_init, Φ_init = (2 * np.cos(y), - np.sqrt(2) * x_max ** 2 * np.sin(y))

    Ψ1, Φ1 = Ψ_init, Φ_init
    Ψ2, Φ2 = Ψ_init, Φ_init

    # Integrate for given E values
    Ψ1, Φ1 = integrate(E_min, Ψ1, Φ1, dx)
    Ψ2, Φ2 = integrate(E_max, Ψ2, Φ2, dx)

    #################################################

    print("finding odd eigenvalues")
    odd_evals = find_multiple_odd_eigenvalues(
        E_min, E_max, dE, tolerance, Ψ_init, Φ_init, dx
    )

    sliced_odd_list = odd_evals[:3]
    formatted_odd_list = np.array([evalue for evalue in sliced_odd_list])

    #################################################

    print("finding even eigenvalues")
    even_evals = find_multiple_even_eigenvalues(
        E_min, E_max, dE, tolerance, Ψ_init, Φ_init, dx
    )

    sliced_even_list = even_evals[:3]
    formatted_even_list = np.array([evalue for evalue in sliced_even_list])

    #################################################

    print(f"\n{dx = }")
    print(f"{dE = }")

    # Printing the formatted list
    print(f"\nfirst 3 even eigenvalues = {formatted_even_list}")
    print(f"expected:{E_even = }")

    print(f"\nfirst 3 odd eigenvalues = {formatted_odd_list}")
    print(f"expected:{E_odd = }")
