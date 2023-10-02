import numpy as np
import h5py


def normalize_wavefunction(wavefunction, dx):
    integral = np.sum(np.abs(wavefunction) ** 2) * dx
    normalization_constant = np.sqrt(2 / integral)
    return wavefunction * normalization_constant


def save_to_hdf5(filename, eigenvalue, eigenfunction):
    """ Save """
    with h5py.File(filename, 'w') as hf:
        dataset = hf.create_dataset("eigenfunction", data=eigenfunction)
        dataset.attrs["eigenvalue"] = eigenvalue


def Schrödinger_eqn(x, Ψ, Φ, E):
    dΨ = Φ
    dΦ = - (x ** 4 + E) * Ψ 
    return dΨ, dΦ


def Schrödinger_RK4(x, Ψ, Φ, E, dx):
    # Algorithm Runge-Kutta 4 for integrating TISE eigenvalue problem
    # coeffs for RK4 of solution, derivative = evaluation of Schrödinger_eqn
    k1_Ψ, k1_Φ = Schrödinger_eqn(x, Ψ, Φ, E)
    k2_Ψ, k2_Φ = Schrödinger_eqn(x + 0.5 * dx, Ψ + 0.5 * dx * k1_Ψ, Φ + 0.5 * dx * k1_Φ, E)
    k3_Ψ, k3_Φ = Schrödinger_eqn(x + 0.5 * dx, Ψ + 0.5 * dx * k2_Ψ, Φ + 0.5 * dx * k2_Φ, E)
    k4_Ψ, k4_Φ = Schrödinger_eqn(x + dx, Ψ + dx * k3_Ψ, Φ + dx * k3_Φ, E)

    Ψ_new = Ψ + (dx / 6) * (k1_Ψ + 2 * k2_Ψ + 2 * k3_Ψ + k4_Ψ)  # updated solution wavefunction
    Φ_new = Φ + (dx / 6) * (k1_Φ + 2 * k2_Φ + 2 * k3_Φ + k4_Φ)  # updated first derivative of solution

    return Ψ_new, Φ_new


def integrate(E, Ψ, Φ, dx, save_wavefunction=False):
    """***Reversed*** running of the RK4 Integrator through the grid one rn in r at a time:
    For each point rn in this grid, I update the wavefunction Ψ using Schrödinger_RK4()."""
    wavefunction = []

    for i, xn in list(enumerate(x)):
        # Reduction of order of ODE
        Ψ, Φ = Schrödinger_RK4(xn, Ψ, Φ, E, dx)
        if save_wavefunction:
            wavefunction.append(Ψ)

    if save_wavefunction:
        # Save the wavefunction corresponding to E_new
        print(f"*** ~~saving wavefunction for eigenvalue {E}~~ ***")
        normalized_wavefunction = normalize_wavefunction(np.array(wavefunction), dx)
        save_to_hdf5(f"{E}.h5", E, normalized_wavefunction)
    return Ψ, Φ

##################################################################################################

def initialisation_parameters():
    # These are the (first 11) eigenvalues we found for epsilon = 2
    E_Bender = np.array([
    1.47714975357798686273,
    6.00338608330813547180,
    11.80243359513386549372,
    18.45881870407350603368,
    25.79179237850933880880,
    33.69427987657709945568,
    42.09380771103636153727,
    50.93740433237080572626,
    60.18436924385155633796,
    69.80209265698014350909,
    79.76551330248462000350
    ])

    # spatial dimension
    x_max = 30
    dx = 1e-5
    Nx = int(x_max / dx)
    x = np.linspace(0, x_max, Nx, endpoint=False)

    return (
        E_Bender,
        x_max,
        dx,
        Nx,
        x,
    )
if __name__ == "__main__":

    E_Bender, x_max, dx, Nx, x = initialisation_parameters()

    # IV for even solutions (at the origin)
    Ψ_init, Φ_init = 1, 0
    Ψ0, Φ0 = integrate(E_Bender[0], Ψ_init, Φ_init, dx, True)
    Ψ2, Φ2 = integrate(E_Bender[2], Ψ_init, Φ_init, dx, True)
    Ψ4, Φ4 = integrate(E_Bender[4], Ψ_init, Φ_init, dx, True)
    Ψ6, Φ6 = integrate(E_Bender[6], Ψ_init, Φ_init, dx, True)
    Ψ8, Φ8 = integrate(E_Bender[8], Ψ_init, Φ_init, dx, True)
    Ψ10, Φ10 = integrate(E_Bender[10], Ψ_init, Φ_init, dx, True)


    # IV for odd solutions (at the origin)
    Ψ_init, Φ_init = 0, 1
    Ψ1, Φ1 = integrate(E_Bender[1], Ψ_init, Φ_init, dx, True)
    Ψ3, Φ3 = integrate(E_Bender[3], Ψ_init, Φ_init, dx, True)
    Ψ5, Φ5 = integrate(E_Bender[5], Ψ_init, Φ_init, dx, True)
    Ψ7, Φ7 = integrate(E_Bender[7], Ψ_init, Φ_init, dx, True)
    Ψ9, Φ9 = integrate(E_Bender[9], Ψ_init, Φ_init, dx, True)




