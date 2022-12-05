# Ana Fabela Hinojosa, 01/11/2022 
# Based on: Scattering off PT -symmetric upside-down potentials -Carl M. Bender and Mariagiovanna Gianfreda
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy import special
from scipy import linalg
from scipy.linalg import expm
from tqdm import tqdm


plt.rcParams['figure.dpi'] = 180

def V(x):
    return x ** 2

def WKB_states(x, Energies):
    states = []
    for n, E in enumerate(Energies):
        wkb = np.zeros(Nx, dtype=complex)

        x0 = np.sqrt(E) # MUST THINK ABOUT THIS TO FIT THE NEG QUARTIC SITUATION
        a = -x0
        b = x0

        F0 = -(2 * a)
        F1 = 2 * b


        u0 = F0**(1/3) * (a - x[(a - δminus < x) & (x < a + δplus)])
        u1 = F1**(1/3) * (x[(b - δLeft < x) & (x < b + δRight)] - b)

        Q = np.sqrt((V(x) - E).astype(complex))
        P = np.sqrt((E - V(x)).astype(complex))

        # LHS of potential barrier
        integral_left = np.cumsum(Q[x < a - δminus]) * delta_x
        integral_left = -(integral_left - integral_left[-1])
        wkb[x < a - δminus] = np.exp(-integral_left) / (3 * np.sqrt(Q[x < a - δminus]))

        # around left turning point "a"
        Ai_a, Aip_a, Bi_a, Bip_a = special.airy(u0)
        wkb[(a - δminus < x) & (x < a + δplus)] = Ai_a * np.sqrt(np.pi) / F0 ** (1/6)

        # inside potential barrier 
        excessively_long_array = np.cumsum(P[x > a]) * delta_x
        integral_a_x = excessively_long_array[x[x > a] > a + δplus]
        wkb[x > a + δplus] = np.cos(integral_a_x - np.pi/4) / np.sqrt(P[x > a + δplus])

        # around right turning point "b"
        Ai_b, Aip_b, Bi_b, Bip_b = special.airy(u1)
        if n % 2 == 0:
            wkb[(b - δLeft < x) & (x < b + δRight)] = Ai_b * np.sqrt(np.pi) / F1 ** (1/6)
        else:
            wkb[(b - δLeft < x) & (x < b + δRight)] = -Ai_b * np.sqrt(np.pi) / F1 ** (1/6)

        # RHS of potential barrier
        integral_right = np.cumsum(Q[x > b + δRight]) * delta_x
        if n % 2 == 0:
            wkb[x > b + δRight] = np.exp(-integral_right) / (3 * np.sqrt(Q[x > b + δRight])) # I scaled these down to 1/3
        else:
            wkb[x > b + δRight] = -np.exp(-integral_right) / (3 * np.sqrt(Q[x > b + δRight])) #####

        states.append(wkb)

    return states


def PT_normalised_states(x, states, P_states):
    PT_normed_states = []
    PT_normed_P_states = []
    for i, P_state in enumerate(P_states):
        # print(f"{P_state = }")
        # print(f"{np.conj(P_state) = }")
        PT_norm = np.dot(np.conj(P_state), states[i])
        # print(f"with PT norm {PT_norm}")

        normed_state = states[i] / np.sqrt(PT_norm)
        normed_P_state = P_state / np.sqrt(PT_norm)
        # print(f"normalised state: {normed_state}")
        # print(f"state shape:{np.shape(normed_state)}\n")
        PT_normed_states.append(normed_state)
        PT_normed_P_states.append(normed_P_state)

    return PT_normed_states, PT_normed_P_states


def plot_states(states, ϵ, Energies):
    ax = plt.gca()
    for i, state in enumerate(states):

        x0 = np.sqrt(Energies[i]) # MUST THINK THIS THROUGH TO FIT THE NEG QUARTIC SITUATION
        a = -x0
        b = x0

        y = np.linspace(-10, 80, Nx).T
        plt.axvline(a, linestyle="--", linewidth=0.5, color="red")
        plt.axvline(b, linestyle="--", linewidth=0.5, color="red")
        plt.fill_betweenx(y, a - δminus, a + δplus , alpha=0.1, color="pink")
        plt.fill_betweenx(y, b - δLeft, b + δRight , alpha=0.1, color="pink")

        color = next(ax._get_lines.prop_cycler)['color']
        # # Energy shifted states
        # plt.plot(x, np.real(state) + Energies[i], color=color, label=fR"$\psi_{i}$")
        # plt.plot(x, np.imag(state) + Energies[i], linestyle='--', color=color)
        # plt.axhline(Energies[i], linestyle=":", linewidth=0.6, color="grey")

        # # Probability density plot
        plt.plot(x,  abs(state)**2 + Energies[i], color=color, label=fR"$|\psi_{i}|^{{2}}$")
        plt.axhline(Energies[i], linewidth=0.5, linestyle=":", color="gray")
        plt.axvline(a, linewidth=0.3, linestyle=":", color="red")
        plt.axvline(b, linewidth=0.3, linestyle=":", color="red")

    if ϵ == 0:
        plt.plot(x, V(x), linewidth=2, color="grey")
        plt.title(fR"WKB states for $H = p^{{2}} + x^{{2}}$")
        plt.axis(xmin=-5,xmax=5, ymin=-1, ymax=20)
    # elif ϵ == 2:
    #     plt.plot(x, V(x, ϵ), linewidth=2, color="grey")
    #     plt.title(fR"WKB states for $H = p^{{2}} - x^{{4}}$")
    #     plt.axis(xmin=-10,xmax=10, ymin=-10, ymax=80)

    plt.legend()
    plt.ylabel(r'$Energy$', labelpad=6)
    plt.xlabel(r'$x$', labelpad=6)
    plt.show()


def globals():
    # #makes folder for simulation frames
    # folder = Path('U_time_evolution')

    # os.makedirs(folder, exist_ok=True)
    # os.system(f'rm {folder}/*.png')

    # # units based on "Bender's PT-symmetry book"
    hbar = 1
    m = 1/2
    ω = 2
    g = 1

    # spatial dimension
    Nx = 1024 * 10
    x = np.linspace(-10, 10, Nx)
    x[x==0] = 1e-200
    delta_x = x[1] - x[0]

    ϵ0 = 0
    ϵ2 = 2

    Energies_ϵ0 = np.load("Energies_HO_WKB_N=10.npy")
    Energies_ϵ0 = Energies_ϵ0.reshape(len(Energies_ϵ0))


    Energies_ϵ2 = np.load("Energies_WKB_N=10.npy")
    Energies_ϵ2 = Energies_ϵ2.reshape(len(Energies_ϵ2))

    N = len(Energies_ϵ2)

    δminus = 0.4
    δplus = 1
    δRight = δminus
    δLeft = δplus

    return hbar, m, ω, Nx, x, delta_x, ϵ0, ϵ2, Energies_ϵ0, Energies_ϵ2, N, δminus, δplus, δRight, δLeft


if __name__ == "__main__":

    hbar, m, ω, Nx, x, delta_x, ϵ0, ϵ2, Energies_ϵ0, Energies_ϵ2, N, δminus, δplus, δRight, δLeft = globals()

    print("\n#################### Harmonic oscillator ####################")
    # print(f"\n{Energies_ϵ0 = }\n")
    states_ϵ0 = WKB_states(x, Energies_ϵ0) 
    plot_states(states_ϵ0, ϵ0, Energies_ϵ0)

    # parity flipped states
    P_states_ϵ0 = [state[::-1] for state in states_ϵ0]
    # plot_states(P_states_ϵ0, ϵ0, Energies_ϵ0)

    normalised_states_ϵ0, normalised_P_states_ϵ0 = PT_normalised_states(x, states_ϵ0, P_states_ϵ0)
    # plot_states(normalised_states_ϵ0, ϵ0, Energies_ϵ0)


