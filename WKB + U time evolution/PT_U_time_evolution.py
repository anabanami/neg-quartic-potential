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
from scipy.signal import convolve


plt.rcParams['figure.dpi'] = 180

def V(x, ϵ):
    if ϵ == 0:
        return x ** 2
    elif ϵ == 2:
        return - x ** 4

def WKB_states(x, ϵ, Energies):
    states = []
    if ϵ == 0:
        for n, E in enumerate(Energies):
            wkb = np.zeros(Nx, dtype=complex)
            x0 = np.sqrt(E)
            a = -x0
            b = x0
            F0 = -(2 * a)
            F1 = 2 * b
            u0 = F0**(1/3) * (a - x[(a - δminus < x) & (x < a + δplus)])
            u1 = F1**(1/3) * (x[(b - δLeft < x) & (x < b + δRight)] - b)
            Q = np.sqrt((V(x, ϵ) - E).astype(complex))
            P = np.sqrt((E - V(x, ϵ)).astype(complex))

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

            # Gaussian blur
            pts = 25
            wkb = gaussian_blur(wkb, pts)
            states.append(wkb)

    elif ϵ == 2:
        for E in Energies:
            wkb = np.zeros(Nx, dtype=complex)
            x0 = - np.sqrt(np.sqrt(E))
            a = -x0
            b = x0
            # analytic solution for WKB approximation: Mathematica WKB_solution-1.nb
            wkb = np.exp(1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, -x ** 4 / E)) / ((E + x ** 4) ** (1 / 4))
            states.append(wkb)

    return states

def gaussian_blur(data, pts):
    """gaussian blur an array by given number of points"""
    x = np.arange(-2 * pts, 2 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def plot_states(states, ϵ, Energies):
    ax = plt.gca()
    for i, state in enumerate(states):

        y = np.linspace(-10, 80, Nx).T

        if ϵ == 0:
            x0 = np.sqrt(Energies[i])
            a = -x0
            b = x0
            # plt.axvline(a, linestyle="--", linewidth=0.3, color="red")
            # plt.axvline(b, linestyle="--", linewidth=0.3, color="red")
            # plt.fill_betweenx(y, a - δminus, a + δplus , alpha=0.1, color="pink")
            # plt.fill_betweenx(y, b - δLeft, b + δRight , alpha=0.1, color="pink")

        elif ϵ == 2:
            x0 = - np.sqrt(np.sqrt(Energies[i]))  # MUST THINK THIS THROUGH TO FIT THE NEG QUARTIC SITUATION
            a = -x0
            b = x0
            plt.axvline(a, linestyle="--", linewidth=0.3, color="red")
            plt.axvline(b, linestyle="--", linewidth=0.3, color="red")
            # plt.fill_betweenx(y, a - δminus, a + δplus , alpha=0.1, color="pink")
            # plt.fill_betweenx(y, b - δLeft, b + δRight , alpha=0.1, color="pink")

        color = next(ax._get_lines.prop_cycler)['color']
        # Energy shifted states
        plt.plot(x, np.real(state) + Energies[i], color=color, label=fR"$\psi_{i}$")
        plt.plot(x, np.imag(state) + Energies[i], linestyle='--', color=color)

        # # Probability density plot
        # plt.plot(x,  abs(state)**2 + Energies[i], color=color, label=fR"$|\psi_{i}|^{{2}}$")

    if ϵ == 0:
        plt.plot(x, V(x, ϵ), linewidth=2, color="grey")
        plt.title(fR"WKB states for $H = p^{{2}} + x^{{2}}$")
        plt.axis(xmin=-5,xmax=5, ymin=-1, ymax=20)
    elif ϵ == 2:
        plt.plot(x, V(x, ϵ), linewidth=2, color="grey")
        plt.title(fR"WKB states for $H = p^{{2}} - x^{{4}}$")
        # plt.axis(xmin=-10,xmax=10, ymin=-10, ymax=80)
        plt.axis(xmin=-10,xmax=10, ymin=-10, ymax=20)




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
    Nx = 2048
    x = np.linspace(-10, 10, Nx)
    x[x==0] = 1e-200
    delta_x = x[1] - x[0]
    # FFT variable
    k = 2 * np.pi * np.fft.fftfreq(Nx, delta_x) 

    ϵ0 = 0
    ϵ2 = 2

    Energies_ϵ0 = np.load("Energies_HO_WKB_N=10.npy")
    Energies_ϵ0 = Energies_ϵ0.reshape(len(Energies_ϵ0))


    Energies_ϵ2 = np.load("Energies_WKB_N=10.npy")
    Energies_ϵ2 = Energies_ϵ2.reshape(len(Energies_ϵ2))

    N = len(Energies_ϵ2)

    δminus = 0.4
    δplus = 0.4
    δRight = δminus
    δLeft = δplus

    # # time interval
    # t_d = m * delta_x ** 2 / (np.pi * hbar)
    # t = 0
    # t_final = 1
    # delta_t = t_d

    return hbar, m, ω, g, Nx, x, delta_x, k, ϵ0, ϵ2, Energies_ϵ0, Energies_ϵ2, N, δminus, δplus, δRight, δLeft


if __name__ == "__main__":

    hbar, m, ω, g, Nx, x, delta_x, k, ϵ0, ϵ2, Energies_ϵ0, Energies_ϵ2, N, δminus, δplus, δRight, δLeft = globals()

    print("\n#################### Harmonic oscillator ####################")
    # print(f"\n{Energies_ϵ0 = }\n")
    # states_ϵ0 = WKB_states(x, ϵ0, Energies_ϵ0) 
    # plot_states(states_ϵ0, ϵ0, Energies_ϵ0)

    # parity flipped states
    # P_states_ϵ0 = [state[::-1] for state in states_ϵ0]
    # plot_states(P_states_ϵ0, ϵ0, Energies_ϵ0)

    # normalised_states_ϵ0, normalised_P_states_ϵ0 = PT_normalised_states(x, states_ϵ0, P_states_ϵ0)
    # # plot_states(normalised_states_ϵ0, ϵ0, Energies_ϵ0)
    
    print("#################### inverted quartic  ####################")
    # print(f"\n{Energies_ϵ2 = }\n")
    states_ϵ2 = WKB_states(x, ϵ2, Energies_ϵ2) 
    plot_states(states_ϵ2[:4], ϵ2, Energies_ϵ2[:4])