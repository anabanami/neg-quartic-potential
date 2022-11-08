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

def V(x, ϵ):
    if ϵ == 0:
        return x ** 2
    elif ϵ == 2:
        return - x ** 4

def subdominant_WKB_states(x, Energies):
    states = []
    for E in Energies:
        # WKB approximation from Mathematica WKB_solution-1.nb
        states.append(np.exp(-1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, -x ** 4 / E))/ ((E + x ** 4) ** (1 / 4)))
    return states

def PT_normalised_states(x, states, Energies):
    P_states = subdominant_WKB_states(-x, Energies)
    PT_states = []
    for i, state in enumerate(states):
        print(f"{i = }")
        PT_norm = np.dot(np.conjugate(P_states[i]), state)
        state/=np.sqrt(PT_norm)
        PT_states.append(state)

        # print(f"{np.shape(PT_states) = }")
    return PT_states

def Matrix(PT_states):


# def C-operator(PT_states):


def plot_WKBstates(states):
    ax = plt.gca()
    for i, state in enumerate(states[:5]):

        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(x,  np.real(state), color=color, label=fR"$\psi_{i}(x)$")
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
    # plt.twinx()
    plt.xlim(xmin=-5, xmax=5)
    plt.xlabel(r'$x$', labelpad=6)
    # plt.ylabel(r'$Energy$', labelpad=6)
    
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
    x = np.linspace(-5, 5, Nx)
    x[x==0] = 1e-200
    delta_x = x[1] - x[0]

    # time interval
    t_d = m * delta_x ** 2 / (np.pi * hbar)
    t = 0
    t_final = 1
    delta_t = t_d
    
    # Harmonic oscillator ground state
    HO_GS = np.zeros_like(x)
    HO_GS[0] = 1

    f_x = np.zeros_like(x)
    # print(f"Initialised {f_x = }\n")


    Energies_HO = np.load("Energies_HO_WKB_N=10.npy")
    Energies_HO = Energies_HO.reshape(len(Energies_HO))


    Energies_ϵ2 = np.load("Energies_WKB_N=10.npy")
    Energies_ϵ2 = Energies_ϵ2.reshape(len(Energies_ϵ2))

    return hbar, m, ω, g, Nx, x, delta_x, f_x, Energies_HO, Energies_ϵ2


if __name__ == "__main__":

    hbar, m, ω, g, Nx, x, delta_x, f_x, Energies_HO, Energies_ϵ2 = globals()

    states_ϵ0 = subdominant_WKB_states(x, Energies_HO)
    # plot_WKBstates(states_ϵ0)

    PT_states_ϵ0 = PT_normalised_states(x, states_ϵ0, Energies_HO)
    # plot_WKBstates(PT_states_ϵ0)

    # ass

    states_ϵ2 = subdominant_WKB_states(x, Energies_ϵ2)
    # plot_WKBstates(states_ϵ2)

    PT_states_ϵ2 = PT_normalised_states(x, states_ϵ2, Energies_ϵ2)
    # plot_WKBstates(PT_states_ϵ2)

