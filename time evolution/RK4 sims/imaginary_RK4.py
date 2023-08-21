# imaginary RK4 TEv v2
# Ana Fabela 19/03/2023

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



def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def restricted_V(x):
    # gassuan smooth vertices
    pts = 5
    V = np.zeros_like(x)
    V[50:462] = -x[50:462] ** 4  # <<<< can modify this
    return gaussian_smoothing(V, pts)


# Potentials
def V(x, t):
    return - x ** 4
    # T = 0.001
    # if t < T:
    #     return (1 / 2) * m * ((hbar / (m * l1 ** 2)) * x) ** 2
    # else:
    #     return (1 / 2) * m * ((hbar / (m * l2 ** 2)) * x) ** 2
    #     #return - x ** 4
    #     #return restricted_V(x)
    #     #return 0


# Schrodinger equation
def Schrodinger_eqn(t, Ψ):
    # Fourier derivative theorem
    KΨ = -(hbar ** 2) / (2 * m) * ifft(-(kx ** 2) * fft(Ψ))
    VΨ = V(x, t) * Ψ
    HΨ  = KΨ + VΨ 
    return (-1 / hbar) * HΨ, HΨ   # imaginary time SE, Hamiltonian acting on state(Ψ) from left


# Schrodinger equation TEv: Runga - Kutta 4 
def Schrodinger_RK4(t, dt, Ψ):
    k1, HΨ  = Schrodinger_eqn(t, Ψ)
    k2, HΨ  = Schrodinger_eqn(t + dt / 2, Ψ + k1 * dt / 2)
    k3, HΨ  = Schrodinger_eqn(t + dt / 2, Ψ + k2 * dt / 2)
    k4, HΨ  = Schrodinger_eqn(t + dt, Ψ + k3 * dt)
    return Ψ + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4), HΨ #HERE goes HΨ, but is it ok to use the k4 version???


def RK4(time_steps):# RK4 time evolution of HO GS
    state = wave

    Ψs = []  # LIST of state vectors (list of row vectors shape like x = (1024,))
    SIGMAS_x_SQUARED = []  # spatial variance
    expectation_Ens = [] # expectation values of En

    i = 0
    for time in time_steps:
        print(f"t = {time}")
        if time < T:
            state, HΨ = Quench(time, t_final, i, x, state, x_max, dx, folder, Nx)
            expected_En = find_En(state, HΨ) # recover expectation value of energy at time t
            sigma_x_squared = x_variance(x, dx, state)
            
        else:
            state, HΨ = Quench(time, t_final, i, x, state, x_max, dx, folder, Nx)
            expected_En = find_En(state, HΨ) # recover expectation value of energy at time t
            sigma_x_squared = x_variance(x, dx, state)

        Ψs.append(state)
        expectation_Ens.append(expected_En)
        SIGMAS_x_SQUARED.append(sigma_x_squared)

        i += 1

    Ψs = np.array(Ψs)
    np.save("RK4_states.npy", Ψs)

    expectation_Ens = np.array(expectation_Ens)
    np.save(f"RK4_expectation_Ens.npy", expectation_Ens)

    SIGMAS_x_SQUARED = np.array(SIGMAS_x_SQUARED)
    np.save(f"RK4_SIGMAS_x_SQUARED.npy", SIGMAS_x_SQUARED)


def find_En(Ψ, HΨ):
    return np.sum(np.conj(Ψ) *  HΨ * dx)

######################################################################

# Calculate Spatial variance of wavefunction (Ψ) per unit time
def x_variance(x, dx, Ψ):
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]

    return dx / 2 * np.sum(g_right + g_left)


#def En_variance():


# plot variance of wavefunction (Ψ) vs time
def variance_plot(t, sigmas_list):
    plt.plot(t, sigmas_list)
    plt.ylabel(R"$\sigma^2$")
    plt.title(f"Variance")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("Variance_Quench.png")
    plt.show()
    plt.clf()


def norm(dx, Ψ):
    h = abs(Ψ) ** 2
    h_right = h[1:]
    h_left = h[:-1]
    return dx / 2 * np.sum(h_right + h_left)

######################################################################

def Quench(t, t_final, i, y, state, y_max, dy, folder, N):
    """generates simulation frame corresponding to time t (Quench occurs at t = T). 
    The function only plots the state every 50 frames of sim."""

    PLOT_INTERVAL = 500

    # state vector
    state, HΨ = Schrodinger_RK4(t, dt, state) # np.array shape like x = (1024,)

    state = state / norm(dx, state) # normalise state at every time step

    if not i % PLOT_INTERVAL:
        if t < T:
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
        # plt.xlim(-15, 15)
        plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
        plt.clf()
        # plt.show()
    return state, HΨ


def globals():
    # makes folder for simulation frames
    folder = Path('Imag_RK4_neg-quartic')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    Nx = 1024
    P = 30

    x_max = 15
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)
    dx = x[1] - x[0]

    # for Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    # time dimension
    t_initial = 0
    t_final = 10
    ## Nyquist dt
    dt = 0.2 * m * dx ** 2 / (np.pi * hbar)
    # quench time
    T = 0

    # initial conditions
    # HO ground state for RK4
    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    i = 0
    return folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, kx, P, t_initial, t_final, dt, T, wave, i





if __name__ == "__main__":
    """FUNCTION CALLS"""

    folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, kx, P, t_initial, t_final, dt, T, wave, i = globals()

    time_steps = np.arange(t_initial, t_final, dt)

    RK4(time_steps)

    Ens = np.load("RK4_expectation_Ens.npy")

    print(f"E_0 = {Ens[-1]}")
    plt.plot(time_steps, np.real(Ens))
    plt.show()

