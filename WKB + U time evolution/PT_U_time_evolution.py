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

def subdominant_WKB_states(x, ϵ):
    states = []
    # if ϵ == 0:
    #     return HO_GS
    if ϵ == 2:
        for E in Energies:
            # WKB approximation from Mathematica WKB_solution-1.nb
            states.append(np.exp(-1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, -x ** 4 / E))/ ((E + x ** 4) ** (1 / 4)))
        return states

# # def PT_normalise(psi_n, psi_m):


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
    x = np.linspace(-10, 10, Nx)
    x[x==0] = 1e-200
    delta_x = x[1] - x[0]

    # time interval
    t_d = m * delta_x ** 2 / (np.pi * hbar)
    t = 0
    t_final = 1
    delta_t = t_d
    
    # ϵ = 0
    ϵ = 2

    # Harmonic oscillator ground state
    HO_GS = np.zeros_like(x)
    HO_GS[0] = 1

    f_x = np.zeros_like(x)
    # print(f"Initialised {f_x = }\n")


    Energies = np.load("Energies_WKB_N=10.npy")
    Energies = Energies.reshape(len(Energies))

    return hbar, m, ω, g, Nx, x, delta_x, ϵ, HO_GS, f_x,  Energies


if __name__ == "__main__":

    hbar, m, ω, g, Nx, x, delta_x, ϵ, HO_GS, f_x, Energies = globals()


    states_ϵ2 = subdominant_WKB_states(x, ϵ)

    # for i, state in enumerate(states_ϵ2):
    #     plt.plot(x, np.real(state), label=fR"Re($\psi_{i}(x)$)")
    #     plt.plot(x, np.imag(), label=fR"Im($\psi_{i}(x)$)")
    # plt.legend()
    # plt.show()
    