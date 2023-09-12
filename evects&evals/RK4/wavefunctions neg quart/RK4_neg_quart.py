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
    return -(x ** 4)  # epsilon=2


# Schrödinger equation
def Schrödinger_eqn(x, Ψ, Φ, E):
    """Ψ is the state, Φ is the fist spatial derivative of the state."""
    dΨ = Φ
    dΦ = -(-V(x) + E) * Ψ
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
        # Save the wavefunction corresponding to E_new
        print(f"*** ~~saving wavefunction for eigenvalue {E}~~ ***")
        normalized_wavefunction = normalize_wavefunction(np.array(wavefunction), dx)
        save_to_hdf5(f"wavefunction_{E}.h5", E, normalized_wavefunction)
    return Ψ, Φ


def bisection(E1, E2, A1, AΦ1, A2, AΦ2, tolerance, Ψ_init, Φ_init, dx, even):
    k = 0
    while tolerance <= abs(E1 - E2):
        # print(f"{k = }")
        print(f"************* Entering bisection")
        print(f"{E1 = }")
        print(f"{E2 = }")

        E_new = (E1 + E2) / 2
        Ψ_new, Φ_new = integrate(E_new, Ψ_init, Φ_init, dx)

        A_new = np.sign(Ψ_new)
        AΦ_new = np.sign(Φ_new)

        if even:
            if AΦ_new != AΦ1:
                E2 = E_new
                AΦ2 = AΦ_new
            else:
                E1 = E_new
                AΦ1 = AΦ_new
        else:
            if A_new != A1:
                E2 = E_new
                A2 = A_new
            else:
                E1 = E_new
                A1 = A_new

        k += 1

    _, _ = integrate(E_new, Ψ_init, Φ_init, dx, save_wavefunction=True)
    return E_new


def find_multiple_odd_eigenvalues(Es, tolerance, Ψ_init, Φ_init, dx):
    eigenvalues = []
    i = 0
    for E1, E2 in zip(Es[:-1], Es[1:]):
        print(f"{i = }")
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
            eigenvalue = bisection(
                E1, E2, A1, AΦ1, A2, AΦ2, tolerance, Ψ_init, Φ_init, dx, False
            )
            eigenvalues.append(eigenvalue)

        i += 1

    return eigenvalues


def find_multiple_even_eigenvalues(Es, tolerance, Ψ_init, Φ_init, dx):
    eigenvalues = []
    j = 0
    for E1, E2 in zip(Es[:-1], Es[1:]):
        print(f"{j = }")
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
            eigenvalue = bisection(
                E1, E2, A1, AΦ1, A2, AΦ2, tolerance, Ψ_init, Φ_init, dx, True
            )
            eigenvalues.append(eigenvalue)

        j += 1

    return eigenvalues


def initialisation_parameters():
    tolerance = 1e-6

    dx = 5e-4

    # space dimension
    x_max = 30
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
    E_max = 5
    nE = 512
    dE = (E_max - E_min) / nE

    Es = np.linspace(E_min, E_max, nE)

    E_even = [1.477150, 11.802434, 25.791792]
    E_odd = [6.003386, 18.458819]

    # NEG QUART POTENTIAL I.V.:
    y = x_max ** 3 / (3 * np.sqrt(2))
    Ψ_init, Φ_init = (
        2 * np.cos(y),
        -np.sqrt(2) * x_max ** 2 * np.sin(y),
    )

    #################################################

    print("\nfinding odd eigenvalues")
    odd_evals = find_multiple_odd_eigenvalues(Es, tolerance, Ψ_init, Φ_init, dx)

    sliced_odd_list = odd_evals[:3]
    formatted_odd_list = np.array([evalue for evalue in sliced_odd_list])

    #################################################

    print("\nfinding even eigenvalues")
    even_evals = find_multiple_even_eigenvalues(Es, tolerance, Ψ_init, Φ_init, dx)

    sliced_even_list = even_evals[:3]
    formatted_even_list = np.array([evalue for evalue in sliced_even_list])

    #################################################

    print(f"\n{dx = }")
    print(f"{dE = }")

    print(f"\nfirst few odd eigenvalues = {formatted_odd_list}")
    print(f"expected:{E_odd = }")

    # Printing the formatted list
    print(f"\nfirst few even eigenvalues = {formatted_even_list}")
    print(f"expected:{E_even = }")
