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

def V(x, ϵ):
    if ϵ == 0:
        return x ** 2
    elif ϵ == 2:
        return - x ** 4

def subdominant_WKB_states(x, ϵ, Energies):
    δ = 0.0001
    states = []
    for E in Energies:
        wkb = np.zeros(Nx, dtype=complex)
        x0 = np.sqrt(E)
        a = -x0
        b = x0

        if ϵ == 0:
            F0 = -(2 * a)
            F1 = 2 * b
        elif ϵ == 2:
            F0 = 4 * a ** 3
            F1 = -4 * b ** 3


        Q = np.sqrt((V(x, ϵ) - E).astype(complex))
        P = np.sqrt((E - V(x, ϵ)).astype(complex))

        # LHS of potential barrier
        integral_left = np.cumsum(Q[x < a]) * delta_x
        integral_left = -(integral_left - integral_left[-1])
        wkb[x < a] = np.exp(-integral_left) / (2 * np.sqrt(Q[x < a]))

        # left turning point
        Ai_a, Aip_a, Bi_a, Bip_a = special.airy(F0**(1/3) * (a - x[(x < a - δ) & (a + δ < x)]))
        wkb[(x < a - δ) & (a + δ < x)] = Ai_a * np.sqrt(np.pi) / F0 ** (1/6)

        # inside potential barrier
        integral_centre = np.cumsum(P[(a < x) & (x < b)]) * delta_x
        wkb[(a < x) & (x < b)] = np.cos(integral_centre - np.pi / 4) / np.sqrt(P[(a < x) & (x < b)])# + np.cos(-integral_centre + np.pi / 4) / np.sqrt(P[(a < x) & (x < b)])

        # right turning point
        Ai_b, Aip_b, Bi_b, Bip_b = special.airy(F1**(1/3) * (x[x==b] - b))
        wkb[x==b] = Ai_b * np.sqrt(np.pi) / F1 ** (1/6)

        # RHS of potential barrier
        integral_right = np.cumsum(Q[b < x]) * delta_x
        wkb[b < x] = np.exp(-integral_right) / (2 * np.sqrt(Q[b < x]))

        states.append(wkb)
        # print(f"{np.shape(states) = }")

    # for E in Energies:
    #   # analytic solution for WKB approximation: Mathematica WKB_solution-1.nb
    #   wkb = np.exp(1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, -x ** 4 / E)) / ((E + x ** 4) ** (1 / 4))
    #   states.append(wkb)

    return states


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


def C_operator(normalised_states, normalised_P_states):
    wavefunction_PT_products = []
    for i, P_state in enumerate(normalised_P_states):
        state_j = np.dot(np.conj(P_state), normalised_states[i])
        wavefunction_PT_products.append(state_j)
        # print(f"{state_j = }")

    c_ns = [] 
    for j, prod in enumerate(wavefunction_PT_products):
        c_n = prod * (-1) ** j
        c_ns.append(c_n)
    C_op = np.sum(c_ns)
    return C_op


# def HΨ(x, ϵ, normalised_states):
#     for state in normalised_states:
#         # Fourier derivative theorem
#         KΨ = -hbar ** 2 / (2 * m) * ifft(-(k ** 2) * fft(state))
#         VΨ = V(x, ϵ) * state
#         return (-1j / hbar) * (KΨ + VΨ)


# def element_integrand(x, ϵ, C_op, normalised_state, P_normalised_state):
#     return C_op * np.conj(P_normalised_state) * HΨ(x, ϵ, normalised_state)


# def Matrix(N):
#     M = np.zeros((N, N), dtype="complex")
#     for m in tqdm(range(N)):
#         for n in tqdm(range(N)):
#             element = element_integrand #<<<< WHAT KIND OF INTEGRAL DO I WANT HERE??????????????
#             print(element)
#             M[m][n] = element    
#     print(f"{M = }")
#     return M


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


def plot_states(states, ϵ, Energies):
    ax = plt.gca()
    for i, state in enumerate(states):

        color = next(ax._get_lines.prop_cycler)['color']
        # # Energy shifted states
        # plt.plot(x, np.real(state) + Energies[i], color=color, label=fR"$\psi_{i}$")
        # plt.plot(x, np.imag(state) + Energies[i], linestyle='--', color=color)
        # plt.axhline(Energies[i], linestyle=":", linewidth=0.6, color="grey")
        # plt.ylabel(r'$Energy$', labelpad=6)

        # Probability density plot
        plt.plot(x,  abs(state)**2 + Energies[i], label=fR"$|\psi_{i}|^{{2}}$")
        plt.axhline(Energies[i], linewidth=0.5, linestyle=":", color="gray")
        plt.ylabel(r'$Energy$', labelpad=6)

    plt.legend()
    plt.axis(xmin=-5,xmax=5, ymin=-1, ymax=15)
    plt.xlabel(r'$x$', labelpad=6)
    # plt.twinx()
    # plt.ylabel(r'$Energy$', labelpad=6)

    if ϵ == 0:
        plt.plot(x, V(x, ϵ), linewidth=2, color="grey")
        plt.title(fR"Subdominant WKB states for $H = p^{{2}} + x^{{2}}$")
    elif ϵ == 2:
        plt.plot(x, V(x, ϵ), linewidth=2, color="grey")
        plt.title(fR"Subdominant WKB states for $H = p^{{2}} - x^{{4}}$")

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
    states_ϵ0 = subdominant_WKB_states(x, ϵ0, Energies_ϵ0) 
    # plot_states(states_ϵ0[:4], ϵ0, Energies_ϵ0[:4])

    ## parity flipped states
    P_states_ϵ0 = [state[::-1] for state in states_ϵ0]
    # plot_states(P_states_ϵ0[:4], ϵ0, Energies_ϵ0[:4])

    normalised_states_ϵ0, normalised_P_states_ϵ0 = PT_normalised_states(x, ϵ0, states_ϵ0, P_states_ϵ0)
    # plot_states(normalised_states_ϵ0[:4], ϵ0, Energies_ϵ0[:4])

    ## TEST P squared:
    states_ϵ0_1 = states_ϵ0[3] # is this the same as PP_states_ϵ0_1 ???
    P_states_ϵ0_1 = P_states_ϵ0[3]
    P_operator = [pp / p for pp, p in zip(states_ϵ0_1, P_states_ϵ0_1)] 
    P_operator_squared = [i ** 2 for i in P_operator]
    print(f"\nIs P complex? {np.iscomplex(P_operator)}")
    plt.plot(x, np.real(P_operator_squared))
    plt.plot(x, np.imag(P_operator_squared))
    plt.title(fR"$P^2$ for state $\psi_{3}(x)$")
    plt.show()

    # ## TEST C squared:
    # C_ϵ0 = C_operator(normalised_states_ϵ0, normalised_P_states_ϵ0)
    # print(f"\nIs C complex? {np.iscomplex(C_ϵ0)}")
    # # print(f"C operator = {C_ϵ0}")
    # print(f"Test that C^2 = 1\n{C_ϵ0 ** 2}\n")




    # print("#################### inverted quartic  ####################")
    # # print(f"\n{Energies_ϵ2 = }\n")
    # states_ϵ2 = subdominant_WKB_states(x, ϵ2, Energies_ϵ2) 
    # ## parity flipped states

    # P_states_ϵ2 = [state[::-1] for state in states_ϵ2]
    # normalised_states_ϵ2, normalised_P_states_ϵ2 = PT_normalised_states(x, ϵ2, states_ϵ2, P_states_ϵ2)
    # # plot_states(states_ϵ2, ϵ2, Energies_ϵ2)
    # plot_states(normalised_states_ϵ2, ϵ2, Energies_ϵ2)

    # ## TEST P squared:
    # states_ϵ2_1 = states_ϵ2[1] # is this the same as PP_states_ϵ2_1 ???
    # P_states_ϵ2_1 = P_states_ϵ2[1]
    # P_operator2 = [pp / p for pp, p in zip(states_ϵ2_1, P_states_ϵ2_1)]
    # P_operator2_squared = [i ** 2 for i in P_operator2]
    # # print(f"\nIs P complex? {np.iscomplex(P_operator2)}")
    # plt.plot(x, np.real(P_operator2_squared))
    # plt.plot(x, np.imag(P_operator2_squared))
    # plt.title(fR"$P^2$ for state $\psi_{1}(x)$")
    # plt.show()

    # ## TEST C squared:
    # C_ϵ2 = C_operator(normalised_states_ϵ2, normalised_P_states_ϵ2)
    # print(f"\nIs C complex? {np.iscomplex(C_ϵ2)}")
    # # print(f"C operator = {C_ϵ2}")
    # print(f"Test that C^2 = 1\n{C_ϵ2 ** 2}\n")

