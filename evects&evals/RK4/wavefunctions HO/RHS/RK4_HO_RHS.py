import numpy as np
import h5py
from scipy.optimize import minimize


def normalize_wavefunction(wavefunction, dr):
    integral = np.sum(np.abs(wavefunction) ** 2) * dr
    normalization_constant = np.sqrt(2 / integral)
    return wavefunction * normalization_constant


def save_to_hdf5(filename, eigenvalue, eigenfunction):
    """ Save """
    with h5py.File(filename, 'w') as hf:
        dataset = hf.create_dataset("eigenfunction", data=eigenfunction)
        dataset.attrs["eigenvalue"] = eigenvalue


def Schrödinger_eqn(r, Ψ, Φ, E):
    dΨ = Φ
    dΦ = - phase ** 2 * (-((r * phase) ** 2) + E) * Ψ # V(x = r * phase) = (r * phase) ** 2
    return dΨ, dΦ


def Schrödinger_RK4(r, Ψ, Φ, E, dr):
    # Algorithm Runge-Kutta 4 for integrating TISE eigenvalue problem
    # coeffs for RK4 of solution, derivative = evaluation of Schrödinger_eqn
    k1_Ψ, k1_Φ = Schrödinger_eqn(r, Ψ, Φ, E)
    k2_Ψ, k2_Φ = Schrödinger_eqn(r + 0.5 * dr, Ψ + 0.5 * dr * k1_Ψ, Φ + 0.5 * dr * k1_Φ, E)
    k3_Ψ, k3_Φ = Schrödinger_eqn(r + 0.5 * dr, Ψ + 0.5 * dr * k2_Ψ, Φ + 0.5 * dr * k2_Φ, E)
    k4_Ψ, k4_Φ = Schrödinger_eqn(r + dr, Ψ + dr * k3_Ψ, Φ + dr * k3_Φ, E)

    Ψ_new = Ψ + (dr / 6) * (k1_Ψ + 2 * k2_Ψ + 2 * k3_Ψ + k4_Ψ)  # updated solution wavefunction
    Φ_new = Φ + (dr / 6) * (k1_Φ + 2 * k2_Φ + 2 * k3_Φ + k4_Φ)  # updated first derivative of solution

    return Ψ_new, Φ_new


def integrate(E, Ψ, Φ, dr, save_wavefunction=False):
    """***Reversed*** running of the RK4 Integrator through the grid one rn in r at a time:
    For each point rn in this grid, I update the wavefunction Ψ using Schrödinger_RK4()."""
    wavefunction = []

    for i, rn in reversed(list(enumerate(r))):
        # Reduction of order of ODE
        Ψ, Φ = Schrödinger_RK4(rn, Ψ, Φ, E, -dr)
        if save_wavefunction:
            wavefunction.append(Ψ)

    if save_wavefunction:
        # Save the wavefunction corresponding to E_new
        print(f"*** ~~saving wavefunction for eigenvalue {E}~~ ***")
        normalized_wavefunction = normalize_wavefunction(np.array(wavefunction), dr)
        save_to_hdf5(f"{E}.h5", E, normalized_wavefunction)
    return Ψ, Φ


def ICS():
    # y = 1j * (r_max**3) / 3
    # Ψ_init = np.exp(y)
    # Φ_init = 1j * (r_max ** 3) * Ψ_init
    # return Ψ_init, Φ_init
    y = - (r_max * phase)**2 / 2
    Ψ_init = np.exp(y)
    Φ_init = -(r_max * phase) *  Ψ_init
    return Ψ_init, Φ_init



def objective_even(E, Ψ_init, Φ_init, dr):
    """Objective function for even states"""
    _, Φ = integrate(E, Ψ_init, Φ_init, dr)
    return np.abs(Φ ** 2)


def objective_odd(E, Ψ_init, Φ_init, dr):
    """Objective function for odd states"""
    Ψ, _ = integrate(E, Ψ_init, Φ_init, dr)
    return np.abs(Ψ ** 2)


def find_even_eigenvalue(Es, Ψ_init, Φ_init, dr):
    j = 0
    # Initialise eigenvalue
    eigenvalue = None
    
    # Initialise list with first two Ψ values
    _, Φ1 = integrate(Es[0], Ψ_init, Φ_init, dr)
    _, Φ2 = integrate(Es[1], Ψ_init, Φ_init, dr)

    # append abs(Φn ** 2) to Φ_values (n = 1, 2)
    Φ_values = [abs(Φ1 ** 2), abs(Φ2 ** 2)]

    for E1, E2, E3 in zip(Es[:-2], Es[1:-1], Es[2:]):
        # Compute the integral for E3 (Ψ3)
        _, Φ3 = integrate(E3, Ψ_init, Φ_init, dr)
        # append abs(Φ3 ** 2) to Φ_values
        Φ_values.append(abs(Φ3 ** 2))
        print(f"{j = }")


        # FIND MINIMUM BETWEEN: Φ1 and Φ3
        if Φ_values[0] > Φ_values[1] < Φ_values[2]:
            print("#### Mama mia!")
            # iteratively search for minimum
            minimum = minimize(objective_even, (E1 + E3) / 2, args=(Ψ_init, Φ_init, dr), bounds=[(E1, E3)])
            eigenvalue = minimum.x[0]
            _ = integrate(eigenvalue, Ψ_init, Φ_init, dr, save_wavefunction=True)

        Φ_values.pop(0)

        j+=1

    return eigenvalue


def find_odd_eigenvalue(Es, Ψ_init, Φ_init, dr):
    i = 0
    # Initialise eigenvalue
    eigenvalue = None
    
    # Initialise list with first two Ψ values
    Ψ1, _ = integrate(Es[0], Ψ_init, Φ_init, dr)
    Ψ2, _ = integrate(Es[1], Ψ_init, Φ_init, dr)

    # append abs(Ψn ** 2) to Ψ_values (n = 1, 2)
    Ψ_values = [abs(Ψ1 ** 2), abs(Ψ2 ** 2)]

    for E1, E2, E3 in zip(Es[:-2], Es[1:-1], Es[2:]):
        # Compute the integral for E3 (Ψ3)
        Ψ3, _ = integrate(E3, Ψ_init, Φ_init, dr)
        # append abs(Ψ3 ** 2) to Ψ_values
        Ψ_values.append(abs(Ψ3 ** 2))
        print(f"{i = }")

        # FIND MINIMUM BETWEEN: Ψ1 and Ψ3
        if Ψ_values[0] > Ψ_values[1] < Ψ_values[2]:
            print("#### Let's-a-go!")
            # iteratively search for minimum
            minimum = minimize(objective_odd, (E1 + E3) / 2, args=(Ψ_init, Φ_init, dr), bounds=[(E1, E3)])
            eigenvalue = minimum.x[0]
            _ = integrate(eigenvalue, Ψ_init, Φ_init, dr, save_wavefunction=True)

        Ψ_values.pop(0)

        i+=1

    return eigenvalue



##################################################################################################

def initialisation_parameters():
    # (centre) opening angle for Stokes wedges corresponding to the negative quartic potential
    theta_right = 0 #- np.pi / 6

    phase = np.exp(1j * theta_right)

    # radial dimension
    r_max = 10
    dr = 1e-3
    Nr = int(r_max / dr)
    r = np.linspace(0, r_max, Nr, endpoint=False)

    return (
        theta_right,
        phase,
        r_max,
        dr,
        r_max,
        Nr,
        r,
    )

if __name__ == "__main__":

    theta_right, phase, r_max, dr, r_max, Nr, r = initialisation_parameters()

    Ψ_init, Φ_init = ICS()

    #################################################

    nE = 15

    E_min = 0
    E_max = 4
    
    dE = (E_max - E_min) / nE
    Es = np.linspace(E_min, E_max, nE)

    print(f"\n<><<><<>< finding even wave function in range E = [{E_min}, {E_max}] ")

    eval_even = find_even_eigenvalue(Es, Ψ_init, Φ_init, dr)

    print(f"<><<><<><")

    #################################################

    E_min = 0
    E_max = 4
    dE = (E_max - E_min) / nE
    Es = np.linspace(E_min, E_max, nE)

    print(f"\n<><<><<>< finding odd wave function in range E = [{E_min}, {E_max}] ")

    eval_odd = find_odd_eigenvalue(Es, Ψ_init, Φ_init, dr)

    print(f"<><<><<><")
    
    print(f"\n{dr = }")
    print(f"{dE = }")

    print(f"\nEven eigenvalue = {eval_even}")
    print(f" Odd eigenvalue = {eval_odd}")

    E_HO = [
    1,
    3,
    5,
    7,
    ]

    print(f"\nError")
    print(f"{E_HO[0] - eval_even = }")
    print(f"{E_HO[1] - eval_odd = }")