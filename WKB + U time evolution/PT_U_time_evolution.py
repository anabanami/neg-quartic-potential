# Ana Fabela Hinojosa, 01/11/2022 
# Based on: Scattering off PT -symmetric upside-down potentials -Carl M. Bender and Mariagiovanna Gianfreda
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy import linalg
from scipy.linalg import expm
from tqdm import tqdm


plt.rcParams['figure.dpi'] = 150


def subdominant_WKB_states(x, ϵ, Energies):
    states = []
    if ϵ == 0:
        for E in Energies:
            x0 = -np.sqrt(E)
            print(f"{x0 = }")
            assert x[0] < x0 < x[-1]

            i = np.searchsorted(x, x0)
            sqrt_P = np.sqrt((E - x ** 2).astype(complex))
            integral = np.cumsum(sqrt_P) * delta_x
            integral-= integral[i]
            wkb = np.exp(1j * integral) / np.sqrt(sqrt_P)
            # print(f"{np.shape(wkb) = }\n")
            states.append(wkb)
            # print(f"\n{wkb = }\n")

    if ϵ == 2:
        for E in Energies:
            # analytic solution for WKB approximation: Mathematica WKB_solution-1.nb
            wkb = np.exp(1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, -x ** 4 / E)) / ((E + x ** 4) ** (1 / 4))
            states.append(wkb)
            print(f"\n{wkb = }\n")
    return states


def PT_normalised_states(x, ϵ, states_ϵ, P_states_ϵ):
    PT_normed_states = []
    for i, P_state in enumerate(P_states_ϵ):
        # print(f"{i = }")
        PT_norm = np.dot(np.conjugate(P_state), states_ϵ[i])
        normed_state = states_ϵ[i] / np.sqrt(PT_norm)
        PT_normed_states.append(normed_state)
    return PT_normed_states


def C_operator(normalised_states):
    wavefunction_products = []
    for state in normalised_states:
        wavefunction_products.append(np.dot(state, state))
    C_op = np.sum(wavefunction_products)
    return C_op


def V(x, ϵ):
    if ϵ == 0:
        return x ** 2
    elif ϵ == 2:
        return - x ** 4


def HΨ(x, ϵ, normaliseded_states):
    for state in normalised_states:
        # Fourier derivative theorem
        KΨ = -hbar ** 2 / (2 * m) * ifft(-(k ** 2) * fft(state))
        VΨ = V(x, ϵ) * state
        return (-1j / hbar) * (KΨ + VΨ)


def element_integrand(x, ϵ, C_op, normalised_state, P_P_normaliseed_state):
    return C_op * np.conj(P_P_normalised_state) * HΨ(x, ϵ, normalised_state)


def Matrix(N):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            element = element_integrand #<<<< WHAT KIND OF INTEGRAL DO I WANT HERE?
            print(element)
            M[m][n] = element
    #print(f"{M = }")
    return M

# def U_operator(N, t):
#     # print(f"{HMatrix(N) = }")
#     return expm(-1j * HMatrix(N) * t / hbar)

# def U_time_evolution(N, t):
#     HO_GS = np.zeros(N, complex)
#     HO_GS[0] = 1
#     # print(HO_GS)

#     ## create time evolution operator
#     U = U_operator(N, t)
#     # print(f"\ntime evolution operator:\n")
#     # for line in U:
#     #     print ('  '.join(map(str, line)))
#     ## state vector
#     return np.einsum('ij,j->i', U, HO_GS)

def plot_states(states, ϵ):
    ax = plt.gca()
    for i, state in enumerate(states[:5]):

        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(x, np.real(state), color=color, label=fR"$\psi_{i}(x)$")
        plt.plot(x, np.imag(state), linestyle='--', color = color)
        plt.ylabel(R"$\psi_{n}(x, E_{WKB})$", labelpad=6)

        # Energy shifted states
        # plt.plot(x, 5 * np.real(state) + Energies[i], color=color, label=fR"$\psi_{i}(x)$")
        # plt.plot(x, 5 * np.imag(state) + Energies[i], linestyle='--', color = color)
        # plt.axhline(Energies[i], linestyle=":", linewidth=0.6, color="grey")
        # plt.ylabel(R"$\psi_{n}(x, E)$", labelpad=6)

        # # # Probability density plot
        # # plt.plot(x, abs(state)**2 + Energies[i], label=fR"$|\psi_{i}(x)|^{{2}}$")
        # plt.plot(x, abs(state)**2, label=fR"$|\psi_{i}(x)|^{{2}}$")
        # plt.ylabel(R"$|\psi_{n}(x, E)|^2$", labelpad=6)

    plt.legend()
    plt.xlim(xmin=-5, xmax=5)
    plt.xlabel(r'$x$', labelpad=6)
    # plt.twinx()
    # plt.ylabel(r'$Energy$', labelpad=6)

    if ϵ == 0:
        plt.title(fR"First subdominant WKB states for $H = p^{{2}} + x^{{2}}$")
    elif ϵ == 2:
        plt.title(fR"First subdominant WKB states for $H = p^{{2}} - x^{{4}}$")

    plt.show()



def globals():
    # #makes folder for simulation frames
    # folder = Path('U_time_evolution')

    # os.makedirs(folder, exist_ok=True)
    # os.system(f'rm {folder}/*.png')

    # units based on "Bender's PT-symmetry book"
    hbar = 1
    m = 1/2
    ω = 2
    g = 1

    # spatial dimension
    Nx = 1024
    x = np.linspace(-100, 100, Nx)
    x[x==0] = 1e-200
    delta_x = x[1] - x[0]
    # FFT variable
    k = 2 * np.pi * np.fft.fftfreq(Nx, delta_x) 

    # # time interval
    # t_d = m * delta_x ** 2 / (np.pi * hbar)
    # t = 0
    # t_final = 1
    # delta_t = t_d

    ϵ0 = 0
    ϵ2 = 2

    Energies_ϵ0 = np.load("Energies_HO_WKB_N=10.npy")
    Energies_ϵ0 = Energies_ϵ0.reshape(len(Energies_ϵ0))


    Energies_ϵ2 = np.load("Energies_WKB_N=10.npy")
    Energies_ϵ2 = Energies_ϵ2.reshape(len(Energies_ϵ2))

    N = len(Energies_ϵ2)

    return hbar, m, ω, g, Nx, x, delta_x, k, ϵ0, ϵ2, Energies_ϵ0, Energies_ϵ2, N


if __name__ == "__main__":

    hbar, m, ω, g, Nx, x, delta_x, k, ϵ0, ϵ2, Energies_ϵ0, Energies_ϵ2, N = globals()


# HARMONIC OSCILLATOR STUFF
    # print(f"\n{np.shape(Energies_ϵ0) = }\n")
    states_ϵ0 = subdominant_WKB_states(x, ϵ0, Energies_ϵ0)
    # parity flipped
    P_states_ϵ0 = states_ϵ0[::-1]
    normalised_states_ϵ0 = PT_normalised_states(x, ϵ0, states_ϵ0, P_states_ϵ0)

    plot_states(states_ϵ0, ϵ0)
    
    # C_ϵ0 = C_operator(normalised_states_ϵ0) #NEEED TO CHECK THIS FURTHER
    # print(f"\nCheck if C^2 = 1\n{C_ϵ0**2 = }\n")
    # # print("\nCheck if CΨ_j = (-1)^n Ψ_j\n")
    # # for j in normalised_states_ϵ0:
    # #     print(f"\n{C * normalised_states_ϵ0[j] = }\n")

    
