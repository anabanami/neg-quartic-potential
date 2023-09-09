# RK4 to solve HO Eigenvalue problem with shooting method
# Ana Fabela 27/08/2023

import numpy as np
import h5py


def normalize_wavefunction(wavefunction, dx):
    integral = np.sum(np.abs(wavefunction) ** 2) * dx
    normalization_constant = np.sqrt(2 / integral)
    return wavefunction * normalization_constant


def save_to_hdf5(filename, eigenvalues, eigenfunctions):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(f"{eigenvalues}", data=eigenvalues)
        hf.create_dataset(f"{eigenfunctions}", data=eigenfunctions)


def V(x):
    return -0.5 * x ** 2


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
        # print(f"{np.shape(wavefunction) = }")
    return Ψ, Φ


def bisection(E1, E2, A1, AΦ1, tolerance, Ψ1, Φ1, dx):
    k = 0
    while tolerance <= abs(E1 - E2):
        # print(f"{k = }")
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

    dx = 9e-3

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
    E_min = 0
    E_max = 6
    dE = 9e-3

    E_HO_even = [0.5, 2.5, 4.5]
    E_HO_odd = [1.5, 3.5, 5.5]

    # HO POTENTIAL I.V.:
    y = x_max * np.sqrt(x_max ** 2) / (2 * np.sqrt(2))
    Ψ_init, Φ_init = (np.exp(y), np.exp(y) * (np.sqrt(x_max ** 2) / np.sqrt(2)))

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

    print(f"\n{dx = }")
    print(f"{dE = }")

    # Printing the formatted list
    print(f"\nfirst 3 even eigenvalues = {formatted_even_list}")
    print(f"expected:{E_HO_even = }")

    print(f"\nfirst 3 odd eigenvalues = {formatted_odd_list}")
    print(f"expected:{E_HO_odd = }")
    #################################################
