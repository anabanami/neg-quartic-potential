# RK4 to solve HO Eigenvalue problem with shooting method
# Ana Fabela 14/09/2023

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
    return x ** 2


# Schrödinger equation
def Schrödinger_eqn(x, Ψ, Φ, E):
    """Ψ is the state, Φ is the fist spatial derivative of the state."""
    dΨ = Φ
    dΦ = -(E - V(x)) * Ψ
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


def integrate(E, Ψ, Φ, dx, save_wavefunction=False):
    """***Reversed*** running of the RK4 Integrator through the grid one xn in x at a time:
    For each point xn in this grid, I update the wavefunction Ψ using Schrödinger_RK4()."""
    wavefunction = []
    for i, xn in reversed(list(enumerate(x))):
        # Reduction of order of ODE
        Ψ, Φ = Schrödinger_RK4(xn, Ψ, Φ, E, -dx)
        if save_wavefunction:
            wavefunction.append(Ψ)

    if save_wavefunction:
        # Save the wavefunction corresponding to E_new
        print(f"*** ~~saving wavefunction for eigenvalue {E}~~ ***")
        normalized_wavefunction = normalize_wavefunction(np.array(wavefunction), dx)
        save_to_hdf5(f"wavefunction_{E}.h5", E, normalized_wavefunction)
    return Ψ, Φ


def bisection_odd(A1, A2, E1, E2, tolerance):
    print("\n~*~*~ BISECTION ~*~*~")
    while abs(E2 - E1) >= tolerance:

        E_new = (E1 + E2) / 2
        Ψ_new, Φ_new = integrate(E_new, Ψ_init, Φ_init, dx, save_wavefunction=False)
        A_new, AΦ_new = (np.sign(Ψ_new), np.sign(Φ_new))

        print("---Boundaries and the solution signs---")
        print(f"{E1 = } has sign {A1 = }")
        print(f"{E_new = } has sign A_new = {A_new}")
        print(f"{E2 = } has sign {A2 = }")


        if  A1 != A_new:
            print(f"%%% Root in left interval %%% ")
            E2 = E_new

        elif A2 != A_new:
            print(f"%%% Root in right interval %%% ")
            E1 = E_new

    _, _ = integrate(E_new, Ψ_init, Φ_init, dx, save_wavefunction=True)
    return E_new


def bisection_even(A1, A2, E1, E2, tolerance):
    print("\n~*~*~ BISECTION ~*~*~")
    while abs(E2 - E1) >= tolerance:
        
        E_new = (E1 + E2) / 2
        Ψ_new, Φ_new = integrate(E_new, Ψ_init, Φ_init, dx, save_wavefunction=False)
        A_new, AΦ_new = (np.sign(Ψ_new), np.sign(Φ_new))

        print("\n---Boundaries and the derivative signs---")
        print(f"{E1 = } has sign {A1 = }")
        print(f"{E_new = } has sign A_new = {AΦ_new}")
        print(f"{E2 = } has sign {A2 = }")


        if  A1 != AΦ_new:
            print(f"%%% Root in left interval %%% ")
            E2 = E_new

        elif A2 != AΦ_new:
            print(f"%%% Root in right interval %%% ")
            E1 = E_new

    _, _ = integrate(E_new, Ψ_init, Φ_init, dx, save_wavefunction=True)
    return E_new


def find_odd_eigenvalue(Es, tolerance, Ψ_init, Φ_init, dx):
    i = 0
    for E1, E2 in zip(Es[:-1], Es[1:]):
        print(f"{i = }")
        Ψ1, Φ1 = integrate(E1, Ψ_init, Φ_init, dx)
        Ψ2, Φ2 = integrate(E2, Ψ_init, Φ_init, dx)

        A1 = np.sign(Ψ1)
        A2 = np.sign(Ψ2)

        if A1 != A2:
            # testing sign of solution
            print("\nlet's-a go")
            eigenvalue = bisection_odd(A1, A2, E1, E2, tolerance)

        i += 1

    return eigenvalue


def find_even_eigenvalue(Es, tolerance, Ψ_init, Φ_init, dx):
    j = 0
    for E1, E2 in zip(Es[:-1], Es[1:]):
        print(f"{j = }")
        Ψ1, Φ1 = integrate(E1, Ψ_init, Φ_init, dx)
        Ψ2, Φ2 = integrate(E2, Ψ_init, Φ_init, dx)

        AΦ1 = np.sign(Φ1)
        AΦ2 = np.sign(Φ2)

        if AΦ1 != AΦ2:
            # testing sign of first derivative of solution
            print("\nMama mia!")
            eigenvalue = bisection_even(AΦ1, AΦ2, E1, E2, tolerance)

        j += 1

    return eigenvalue


def initialisation_parameters():
    tolerance = 1e-6

    dx = 1e-5

    # space dimension
    x_max = 5
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

    # HO POTENTIAL I.V.:
    y = -(x_max ** 2) / 2
    Ψ_init, Φ_init = (np.exp(y), -x_max * np.exp(y))


    nE = 50
    #################################################

    print("\nfinding even wavefunction")
    E_min = 0
    E_max = 2
    
    dE = (E_max - E_min) / nE
    Es = np.linspace(E_min, E_max, nE)

    eval_0 = find_even_eigenvalue(Es, tolerance, Ψ_init, Φ_init, dx)

    #################################################
    print("\nfinding odd wavefunction")

    E_min = 2
    E_max = 4
    dE = (E_max - E_min) / nE
    Es = np.linspace(E_min, E_max, nE)

    eval_1 = find_odd_eigenvalue(Es, tolerance, Ψ_init, Φ_init, dx)

    #################################################
    print("\nfinding even wavefunction")

    E_min = 4
    E_max = 6
    dE = (E_max - E_min) / nE
    Es = np.linspace(E_min, E_max, nE)

    eval_2 = find_even_eigenvalue(Es, tolerance, Ψ_init, Φ_init, dx)

    print(f"{tolerance = }")
    print(f"\n{dx = }")
    print(f"{dE = }")

    print(f"\n{eval_0 = }")
    print(f"{eval_1 = }")
    print(f"{eval_2 = }")

    print("\nERROR:")
    print(f"{1 - eval_0 = }")
    print(f"{3 - eval_1 = }")
    print(f"{5 - eval_2 = }")







