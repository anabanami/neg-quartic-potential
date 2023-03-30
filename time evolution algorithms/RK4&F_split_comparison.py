# comparing RK4&F_split step
# Ana Fabela 16/03/2023

import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import linalg
from scipy.linalg import expm
import scipy.special as sc
from scipy.integrate import quad
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)


"""
THE FOLLOWING BLOCK OF FUNCTIONS CORRESPONDS TO _F_split_step QUENCH
"""

# kinetic energy for F_split_step
def K():
    return (hbar * kx) ** 2 / (2 * m)

# Potentials for F_split_step QUENCH
def V(x, t):
    T = 0.001
    if t < T:
        return (1 / 2) * m * ((hbar / (m * l1 ** 2)) * x) ** 2
    else:
        return -(x ** 4)

# Second order F_split_step
def F_split_step2(Ψ, t, dt):
    """Evolve Ψ in time from t-> t+dt using a single step
    of the second order Fourier Split step method with time step dt"""
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    Ψ = ifft(np.exp(-1j * K() * dt / hbar) * fft(Ψ))
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    return Ψ

def FSS():# # split step time evolution of GS
    state = wave
    time_steps = np.arange(t_initial, t_final, dt)
    # Ψs = []  # LIST of state vectors (list of row vectors shape like x = (512,))
    SIGMAS_SQUARED = []  # spatial variance
    i = 0
    for time in time_steps:
        print(f"t = {time}")
        if time < T:
            state = Quench(time, t_final, i, x, state, x_max, dx, folder, "FSS", Nx)
            sigma_x_squared = variance(x, dx, state)
            # Ψs.append(state)
        else:
            state = Quench(time, t_final, i, x, state, x_max, dx, folder, "FSS", Nx)
            sigma_x_squared = variance(x, dx, state)
            # Ψs.append(state)

        SIGMAS_SQUARED.append(sigma_x_squared)
        i += 1

    # Ψs = np.array(Ψs)
    # np.save("F_states.npy", Ψs)

    SIGMAS_SQUARED = np.array(SIGMAS_SQUARED)
    np.save(f"FSS_SIGMAS_SQUARED.npy", SIGMAS_SQUARED)
    # Ψs = np.load("F_states.npy")


""" END """

######################################################################

"""
THE FOLLOWING BLOCK OF FUNCTIONS CORRESPONDS TO RK4 QUENCH
"""

# Schrodinger equation
def Schrodinger_eqn(t, Ψ):
    # Fourier derivative theorem to calculate derivative operator
    KΨ = -(hbar ** 2) / (2 * m) * ifft(-(kx ** 2) * fft(Ψ))
    VΨ = V(x, t) * Ψ
    return (-1j / hbar) * (KΨ + VΨ)

# TEv: Runga - Kutta 4 on Schrodinger equation
def Schrodinger_RK4(t, dt, Ψ):
    k1 = Schrodinger_eqn(t, Ψ)
    k2 = Schrodinger_eqn(t + dt / 2, Ψ + k1 * dt / 2)
    k3 = Schrodinger_eqn(t + dt / 2, Ψ + k2 * dt / 2)
    k4 = Schrodinger_eqn(t + dt, Ψ + k3 * dt)
    return Ψ + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def RK4():# RK4 time evolution of HO GS
    state = wave
    time_steps = np.arange(t_initial, t_final, dt)
    # Ψs = []  # LIST of state vectors (list of row vectors shape like x = (1024,))
    SIGMAS_SQUARED = []  # spatial variance
    i = 0
    for time in time_steps:
        print(f"t = {time}")
        if time < T:
            state = Quench(time, t_final, i, x, state, x_max, dx, folder, "RK4", Nx)
            sigma_x_squared = variance(x, dx, state)
            # Ψs.append(state)
        else:
            state = Quench(time, t_final, i, x, state, x_max, dx, folder, "RK4", Nx)
            sigma_x_squared = variance(x, dx, state)
            # Ψs.append(state)

        SIGMAS_SQUARED.append(sigma_x_squared)
        i += 1

    # Ψs = np.array(Ψs)
    # np.save("RK4_states.npy", Ψs)

    SIGMAS_SQUARED = np.array(SIGMAS_SQUARED)
    np.save(f"RK4_SIGMAS_SQUARED.npy", SIGMAS_SQUARED)
    # Ψs = np.load("RK4_states.npy")
""" END """


######################################################################

# Calculate Spatial variance of wavefunction (Ψ) per unit time
def variance(x, dx, Ψ):
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]

    return dx / 2 * np.sum(g_right + g_left)

# plot spatial variance of wavefunction (Ψ) vs time
def variance_plot(t, sigmas_list):
    plt.plot(t, sigmas_list, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{x}^2$")
    plt.title(f"Spatial variance")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("Variance_Quench.png")
    plt.show()
    plt.clf()


def Quench(t, t_final, i, y, state, y_max, dy, folder, method, N):
    """generates simulation frame corresponding to time t (Quench occurs at t = T). 
    The function only plots the state every 500 frames of sim."""

    PLOT_INTERVAL = 500

    states = []
    SIGMAS_SQUARED = []

    if method == "FSS":
        ## state vector
        state = F_split_step2(state, t, dt)  # np.array shape like x = (1024,)

        if not i % PLOT_INTERVAL:
            if t < T:
                # initial HO
                plt.plot(y, V(y, t), color="black", linewidth=2)
            else:
                # final HO
                plt.plot(y, V(y, t), color="black", linewidth=2)

            plt.plot(y, abs(state) ** 2, label=fR"$\psi({t:.04f})$")
            plt.ylabel(R"$|\psi(x, t)|^2$")
            plt.legend()
            plt.xlabel("x")
            plt.ylim(-1, 1)
            plt.xlim(-11, 11)
            plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
            plt.clf()
            # plt.show()
        return state


    if method == "RK4":
        # state vector
        state = Schrodinger_RK4(t, dt, state) # np.array shape like x = (1024,)

        if not i % PLOT_INTERVAL:
            if t < T:
                states.append(state)
                plt.plot(y, V(y, t), color="black", linewidth=2)
            else:
                plt.plot(y, V(y, t), color="black", linewidth=2)

            # prob. density plot
            plt.plot(y, abs(state) ** 2, label=fR"$\psi({t:.04f})$")
            plt.ylabel(R"$|\psi(x, t)|^2$")
            plt.title(f"state at t = {t:04f}")
            plt.xlabel("x")
            plt.legend()
            plt.ylim(-1, 1)
            plt.xlim(-15, 15)
            plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
            # plt.show()
            plt.clf()
        return state


def globals(method):
    # makes folder for simulation frames
    if method=="RK4":
        folder = Path('RK4_quench_HO-neg-quartic')
    elif method=="FSS":
        folder = Path('FSS_quench_HO-neg-quartic')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    x_max = 11
    dx = RK4 
    Nx = int(2 * x_max / dx)

    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # for Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    # time dimension
    t_initial = 0
    t_final = 10
    ## Nyquist dt
    dt = 0.4 * m * dx ** 2 / (np.pi * hbar)
    # quench time
    T = 0.001

    # initial conditions
    # HO ground state For FSS
    HO_GS = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))
    # HO ground state for RK4
    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    i = 0
    return folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, kx, t_initial, t_final, dt, T, HO_GS, wave, i


if __name__ == "__main__":
    """FUNCTION CALLS"""

    # folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, kx, t_initial, t_final, dt, T, HO_GS, wave, i = globals(method="FSS")

    # FSS()
    # print(f"\n{Nx = }")


    # F_sigmas_squared_list = np.load("FSS_SIGMAS_SQUARED.npy")
    # F_sigmas_list = np.sqrt(F_sigmas_squared_list)
    # time = np.linspace(t_initial, t_final, len(F_sigmas_list))

    # variance_plot(time, F_sigmas_list)

    # #######################################################################

    folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, kx, t_initial, t_final, dt, T, HO_GS, wave, i = globals(method="RK4")

    RK4()
    print(f"\n{Nx = }")


    # RK4_sigmas_squared_list = np.load("RK4_SIGMAS_SQUARED.npy")
    # RK4_sigmas_list = np.sqrt(RK4_sigmas_squared_list)
    # time = np.linspace(t_initial, t_final, len(RK4_sigmas_list))

    # variance_plot(time, RK4_sigmas_list)

    ##########################################################################

    # F_sigmas_squared_list = np.load("FSS_SIGMAS_SQUARED.npy")
    # RK4_sigmas_squared_list = np.load("RK4_SIGMAS_SQUARED.npy")
    # F_sigmas_list = np.sqrt(F_sigmas_squared_list)
    # RK4_sigmas_list = np.sqrt(RK4_sigmas_squared_list)

    # time = np.linspace(0, 10, len(F_sigmas_list))

    # plt.plot(time, F_sigmas_list, label=R"FSS: $\sigma_{x}$")
    # plt.title(f"Spatial variance Fourier Split Step ")

    # plt.plot(time, RK4_sigmas_list, label=R"RK4: $\sigma_{x}$")
    # plt.title(f"Spatial variance Runge-Kutta 4 ")

    # plt.ylabel(R"$\sigma_{x}$")
    # plt.xlabel("t")
    # plt.legend()
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    # plt.savefig("VARIANCE_COMPARISON.png")
    # plt.show()

    # variance_diff = F_sigmas_list - RK4_sigmas_list

    # plt.plot(time, variance_diff, label=R"Difference between $\sigma_{x}$")
    # plt.ylabel(R"$\sigma_{x}$")
    # plt.title(f"Spatial variance difference between methods")
    # plt.xlabel("t")
    # plt.legend()
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    # plt.savefig("VARIANCE_Difference.png")
    # plt.show()
