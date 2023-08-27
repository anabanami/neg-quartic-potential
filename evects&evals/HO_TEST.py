# RK4 to solve HO Eigenvalue problem with shooting method
# Ana Fabela 27/08/2023

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import h5py


plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)


def V(x):
    return -0.5 * x ** 2


# Schrödinger equation
def Schrödinger_eqn(x, Ψ, Φ, E):
    """Ψ is the state, Φ is the fist spatial derivative of the state."""
    dΨ = Φ
    # my specific negative quartic PROBLEM
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


def integrate(E, Ψ, Φ, dx):
    """Reversed running of the RK4 Integrator through the grid one xn in x at a time:
    For each point xn in this grid, I update the wavefunction Ψ using Schrödinger_RK4()."""
    for i, xn in reversed(list(enumerate(x))):
        Ψ, Φ = Schrödinger_RK4(xn, Ψ, Φ, E, -dx)
    return Ψ, Φ


def solve_tise(E1, E2, tolerance=1e-6):
    max_iterations = 1000
    iteration = 0

    intervals = [(E1, E2)]
    eigenvalues_found = []
    wavefunctions_found = []

    # Initial values: (Ψ, dΨ/dx) ~ BASED ON WKB ASYMPTOTIC SOLUTION
    y = x_max ** 3 / (3 * np.sqrt(2))
    # ONLY 2 Real(Ψ) =  2 B cos(y) solutions
    Ψ1_init, Φ1_init = (np.cos(y), - x_max**2 * np.sin(y) * np.sqrt(2))

    while intervals and iteration < max_iterations:
        E1, E2 = intervals.pop(0)

        E_new = (E1 + E2) / 2

        Ψ1, Φ1 = Ψ1_init, Φ1_init
        Ψ2, Φ2 = Ψ1_init, Φ1_init
        wavefunction = []  # store wavefunction solution

        # Integrate for given E values
        Ψ1, Φ1 = integrate(E1, Ψ1, Φ1, dx)
        Ψ2, Φ2 = integrate(E2, Ψ2, Φ2, dx)
        wavefunction.append(Ψ2)

        # Test condition at the origin
        A1 = Ψ1
        A2 = Ψ2

        # POTENTIAL ROOT CASE
        if np.sign(A1) != np.sign(A2):
            if abs(E_new - E2) < tolerance:
                # normalise wavefunction
                wavefunction = np.array(wavefunction)
                integral = np.sum(np.abs(wavefunction) ** 2) * dx
                Ψ_normalised = wavefunction / np.sqrt(integral)
                print(
                    f"wavefunction is normalised? {np.sum(np.abs(Ψ_normalised) ** 2) * dx}"
                )

                print(f"\nEIGENVALUE FOUND! {E_new = }\n")
                eigenvalues_found.append(E_new)  # store the eigenvalue
                wavefunctions_found.append(Ψ_normalised)

                # split the interval to search for other potential roots
                intervals.append((E1, E_new - tolerance))
                intervals.append((E_new + tolerance, E2))
            else:
                # REFINING TEH SEARCH
                intervals.append((E1, E_new))
                intervals.append((E_new, E2))

        iteration += 1

    if not intervals:
        print("No more energy intervals to check")

    if iteration >= max_iterations:
        print(
            "\nNO EIGENVALUE FOUND YET...\n<<<<Maximum number of iterations reached without convergence>>>>\n"
        )
        return None

    return np.array(eigenvalues_found), np.array(wavefunctions_found)


def find_eigenvalues(E_min, E_max, num_intervals, tolerance):
    # initialise lists for hdf5 files
    eigenvalues = []
    solutions = []

    # define energy range to search
    E_range = np.linspace(E_min, E_max, num_intervals + 1)

    for i in range(num_intervals):
        E1 = E_range[i]
        E2 = E_range[i + 1]

        print(f"shooting between {E1 = } and {E2 = }")

        eigenvalues_found, wavefunctions_found = solve_tise(E1, E2, tolerance)
        if eigenvalues_found.size > 0:
            for idx, eigenvalue in enumerate(eigenvalues_found):
                if not any(
                    abs(eigenvalue - E_existing) < tolerance
                    for E_existing in eigenvalues
                ):
                    eigenvalues.append(eigenvalue)
                    solutions.append(wavefunctions_found[idx])

    eigenvalues = np.array(eigenvalues)
    eigenvalues = np.sort(eigenvalues)
    save_to_hdf5("HO_eigenvalues_and_solutions.h5", eigenvalues, solutions)
    return eigenvalues


def save_to_hdf5(filename, eigenvalues, solutions):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset("eigenvalues", data=eigenvalues)
        for i, solution in enumerate(solutions):
            hf.create_dataset(f"solution_{i}", data=solution)


def initialisation_parameters():

    tolerance = 1e-6

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

    # Analytical HO energies to compare to
    E_HO = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    # HO frequency
    ω = 1  ### MAYBE need TO CHANGE THIS UP (make variable)

    E_min = 0
    E_max = 26
    num_intervals = 2000
    eigenvalues = find_eigenvalues(E_min, E_max, num_intervals, tolerance)

    print(f"\n{eigenvalues = }")
    print(f"\nThese are the energies that we expect:\n{E_HO =}")

    print("\nTESTING PARAMETERS:")
    print(f"{tolerance = }")
    print("\n*~ spatial space ~*")
    print(f"{x_max = }")
    print(f"{dx = }")
    print(f"{Nx = }")
    print("\n*~ Energy ~*")
    print(f"{ω = }")
    print(f"{E_min = }, {E_max = }")
    print(f"{num_intervals = }")
