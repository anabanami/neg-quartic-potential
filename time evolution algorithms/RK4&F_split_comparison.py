# comparying RK4&F_split step
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

# Calculate Spatial variance of wavefunction (Ψ) per unit time
def variance(x, dx, Ψ):
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)
    print(f"{expectation_value_x =}")

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]

    return dx / 2 * np.sum(g_right + g_left)


# plot spatial variance of wavefunction (Ψ) vs time
def variance_plot(time, sigmas_list):
    plt.plot(time, sigmas_list, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{x}^2$")
    plt.title(f"Spatial variance")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("Variance_Quench.png")
    plt.show()
    plt.clf()


"""
THE FOLLOWING BLOCK OF FUNCTIONS CORRESPONDS TO _F_split_step QUENCH
"""

# BASIS STATES (FOURIER -- EXPONENTIAL FORM)
def F_basis_vector(x, n):
    # Fourier state (exponential form)
    return (1 / np.sqrt(P)) * np.exp(1j * 2 * np.pi * n * x / P)  # (.shape= (512,))


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


def F_split_step_Quench(N, y, t, state, i):
    ## state vector
    psi = F_split_step2(state, t, dt)  # np.array shape like x = (512,)

    PLOT_INTERVAL = 50

    if not i % PLOT_INTERVAL:
        if t < T:
            # initial HO
            plt.plot(y, V(y, t), color="black", linewidth=2)
        else:
            # final HO
            plt.plot(y, V(y, t), color="black", linewidth=2)

        plt.plot(y, abs(psi) ** 2, label=fR"$\psi({t:.04f})$")
        plt.ylabel(R"$|\psi(x, t)|^2$")
        # plt.plot(y, np.real(psi), label=fR"Re($\psi$)")
        # plt.plot(y, np.imag(psi), label=fR"Im($\psi$)")
        # plt.ylabel(R"$\psi(t)$")

        plt.legend()
        plt.xlabel("x")
        plt.ylim(-1, 1)
        # plt.xlim(-15, 15)
        plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
        plt.clf()
        # plt.show()
    return psi

def FSS():# # split step time evolution of GS
    state = HO_GS
    time_steps = np.arange(t_initial, t_final, dt)
    Ψs = []  # LIST of state vectors (list of row vectors shape like x = (512,))
    SIGMAS_SQUARED = []  # spatial variance
    i = 0
    for t in time_steps:
        print(t)
        if t < T:
            state = F_split_step_Quench(Nx, x, t, state, i)
            sigma_x_squared = variance(x, dx, state)
            Ψs.append(state)
        else:
            state = F_split_step_Quench(Nx, x, t, state, i)
            sigma_x_squared = variance(x, dx, state)
            Ψs.append(state)

        SIGMAS_SQUARED.append(sigma_x_squared)
        i += 1

    # Ψs = np.array(Ψs)
    # np.save("State_vectors.npy", Ψs)

    SIGMAS_SQUARED = np.array(SIGMAS_SQUARED)
    np.save(f"SIGMAS_SQUARED.npy", SIGMAS_SQUARED)
    Ψs = np.load("State_vectors.npy")
    # print(Ψs[0])

""" END """


"""
THE FOLLOWING BLOCK OF FUNCTIONS CORRESPONDS TO RK4 QUENCH
"""

# Potential energy. Changes at t = T.
def V(x, t):
    T = 0.001
    if t < T:
        return (
            (1 / 2) * m * ω ** 2 * x ** 2
        )  # <<<< CHECK THIS LINE VS WHAT IS IN F_split_step
    else:
        return -(x ** 4)


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


def RK4_Quench(t, t_final, i, x, wave, x_max, dx, folder):
    """generates simulation frame corresponding to time t (Quench occurs at t = T). 
    The function only plots the state every 50 frames of sim."""
    PLOT_INTERVAL = 50

    waves = []
    SIGMAS_SQUARED = []
    while t < t_final:
        # print(i)
        if not i % PLOT_INTERVAL:

            waves.append(wave)

            # # raw plot
            # plt.plot(x, np.real(wave), label="real part")
            # plt.plot(x, np.imag(wave), label="imaginary part")
            # plt.ylabel(R"$\psi(x,t)$")
            # plt.title(f"state at t = {t:04f}")
            # plt.legend()

            # prob. density plot
            plt.plot(x, abs(wave ** 2))
            plt.ylabel(R"$|\psi(x,t)|^2$")
            plt.title(f"state at t = {t:04f}")

            # # phase plot
            # plt.plot(x, np.angle(wave))
            # plt.ylabel(R"$\theta(x)$")
            # plt.title(f"state's phase at t = {t:04f}")

            plt.xlabel("x")
            plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
            plt.clf()

            # spatial variance
            sigma_x_squared = variance(x, dx, wave)
            SIGMAS_SQUARED.append(sigma_x_squared)
            print(f"variance = {sigma_x_squared}\n")

            h = abs(wave ** 2)
            h_right = h[1:]
            h_left = h[:-1]
            print(f"wave normalisation: {dx / 2 * np.sum(h_right + h_left)}")

        wave = Schrodinger_RK4(t, dt, wave)
        i += 1
        t += dt

    np.save("SIGMAS_SQUARED.npy", SIGMAS_SQUARED)
    np.save("waves_list.npy", waves)

""" END """






def globals(method):
    # makes folder for simulation frames
    if method=="RK4"
        folder = Path('OG RK4')
    elif method=="FSS"       ###########################################REFORMULATE THIS
        folder = Path('F_SS_quench_HO-neg-quartic')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    Nx = 512 # RK4 is 1024
    P = 30

    x_max = 15 # RK4 is 15

    x = np.linspace(-x_max, x_max, Nx, endpoint=False)
    n = x.size
    dx = x[1] - x[0]

    # for Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    # time dimension
    t_initial = 0
    t_final = 40
    ## Nyquist dt
    dt = 0.5 * m * dx ** 2 / (np.pi * hbar) # RK4 is m * dx ** 2 / (np.pi * hbar)
    # quench time
    T = 0.001

    # initial conditions
    # HO ground state For FSS
    HO_GS = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))
    # HO ground state for RK4
    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    i = 0
    return folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, n, kx, P, t_initial, t_final, t, dt, T, HO_GS, wave, i





if __name__ == "__main__":
    """FUNCTION CALLS"""

    folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, n, kx, P, t_initial, t_final, t, dt, T, HO_GS, wave, i = globals(method="FSS")
    # FSS()

    # folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, n, kx, P, t_initial, t_final, t, dt, T, HO_GS, wave, i = globals(method="RK4")
    # RK4_Quench(t, t_final, i, x, wave, x_max, delta_x, folder)


    sigmas_squared_list = np.load("SIGMAS_SQUARED.npy")
    sigmas_list = np.sqrt(sigmas_squared_list)

    time = np.linspace(t, t_final, len(sigmas_list))

    plt.plot(time, sigmas_list, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{x}^2$")
    plt.title(f"Spatial variance")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("OG_RK4_Variance_Quench.png")
    plt.show()
    plt.clf()