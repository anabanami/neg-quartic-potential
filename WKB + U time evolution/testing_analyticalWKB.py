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


plt.rcParams['figure.dpi'] = 180

def V(x, ϵ):
    if ϵ == 0:
        return x ** 2
    elif ϵ == 2:
        return - x ** 4


def subdominant_WKB_states(x, ϵ, Energies):
    states = []
    for E in Energies:
        x0 = -np.sqrt(E)
        # ANALYTICAL VERSION
        Q = E - x **2
        y = x / x0 # <<<----THIS IS WRONG 
        # According to Mathematica:
        A = np.sqrt(E) * (1/2) * (y * np.sqrt(1 - y **2) + np.arcsin(y))
        wkb = np.exp(1j * A) / np.sqrt(np.sqrt(Q))
        states.append(wkb)

    return states, y


def PT_normalised_states(x, ϵ, states_ϵ, P_states_ϵ):
    PT_normed_states = []
    PT_normed_P_states = []
    for i, P_state in enumerate(P_states_ϵ):
        # print(f"{P_state = }")
        # print(f"{np.conj(P_state) = }")
        PT_norm = np.dot(np.conj(P_state), states_ϵ[i])
        # print(f"with PT norm {PT_norm}")

        normed_state = states_ϵ[i] / np.sqrt(PT_norm)
        normed_P_state = P_state / np.sqrt(PT_norm)
        # print(f"normalised state: {normed_state}")
        # print(f"state shape:{np.shape(normed_state)}\n")
        PT_normed_states.append(normed_state)
        PT_normed_P_states.append(normed_P_state)

    return PT_normed_states, PT_normed_P_states



def plot_states(y, states, ϵ, Energies):
    ax = plt.gca()
    for i, state in enumerate(states[:5]):

        color = next(ax._get_lines.prop_cycler)['color']
        # Raw states
        plt.plot(y, np.real(state), color=color, label=fR"$\psi_{i}$")
        plt.plot(y, np.imag(state), linestyle='--', color = color)
        plt.ylabel(R"$\psi_{n}(x, E)$", labelpad=6)

        # # # Probability density plot
        # plt.plot(y, abs(state)**2 , label=fR"$|\psi_{i}|^{{2}}$")
        # plt.ylabel(R"$|\psi_{n}(x, E_{wkb})|^2$", labelpad=6)

        # # # Probability density plot
        # for i, E in enumerate(Energies):
        #     y = np.linspace(-np.sqrt(E), np.sqrt(E), Nx) ####<<<<< THIS PLOTTING IS KIND OF IMPORTANT
        #     plt.plot(y, abs(state)**2 , label=fR"$|\psi_{i}|^{{2}}$")
        # plt.ylabel(R"$|\psi_{n}(x, E_{wkb})|^2$", labelpad=6)

    plt.legend()
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
    Nx = 4096
    x = np.linspace(-10, 10, Nx)
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

    print("\n#################### Harmonic oscillator ####################")
    # print(f"\n{Energies_ϵ0 = }\n")
    states_ϵ0, y = subdominant_WKB_states(x, ϵ0, Energies_ϵ0) 
    ## parity flipped states
    # P_states_ϵ0 = [state[::-1] for state in states_ϵ0]

    # normalised_states_ϵ0, normalised_P_states_ϵ0 = PT_normalised_states(x, ϵ0, states_ϵ0, P_states_ϵ0)
    plot_states(y, states_ϵ0, ϵ0, Energies_ϵ0)
    # plot_states(normalised_states_ϵ0, ϵ0, Energies_ϵ0)
