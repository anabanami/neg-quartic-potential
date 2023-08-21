# RK4
# Ana Fabela 13/04/2023

import os
from pathlib import Path
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import convolve 
import scipy.special as sc
from matplotlib.ticker import FormatStrFormatter


plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)



###############################################################################
def x_variance(x, dx, Ψ):
# Calculate Spatial variance of wavefunction (Ψ) per unit time
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)

    g = ((x - expectation_value_x) ** 2) * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]
    return dx / 2 * np.sum(g_right + g_left)

###############################################################################

def gaussian_smoothing(data, pts): 
    """gaussian smooth an array by given number of points""" 
    x = np.arange(-4 * pts, 4 * pts + 1, 1) 
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2)) 
    smoothed = convolve(data, kernel, mode='same') 
    normalisation = convolve(np.ones_like(data), kernel, mode='same') 
    return smoothed / normalisation 


def smooth_restricted_V(x): 
    V = np.ones_like(x) * x[250]** 4
    V[250:Nx-250] = x[250:Nx-250] ** 4 
    ## smoooth by pts=3
    V = gaussian_smoothing(V, 3) 
    return V 


def V(x, t):
    if t < T:
        return (1 / 2) * m * ((hbar / (m * l1 ** 2)) * x) ** 2
    else:
        return np.zeros_like(x)
        # return -(x ** 2) # testing an inverted HO for refocusing (we expect full dissipation)
        # return - α * (x ** 4)
        # return - β * (x ** 8)
        # return - α * smooth_restricted_V(x)


def plot_evolution_frame(y, t, i, state):
    PLOT_INTERVAL = 1000
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
        plt.xlim(-20, 20)
        plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
        # plt.show()
        plt.clf()


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


def Quench(t, t_final, i, y, state, y_max, dy, folder, method, N):
    """generates simulation frame corresponding to time t (Quench occurs at t = T). 
    The function only plots the state every PLOT_INTERVAL."""

    if method == "RK4":
        # state vector
        state = Schrodinger_RK4(t, dt, state) # np.array shape like x = (Nx,)
    return state


def evolve(method="RK4", label=""):#  time evolution
    state = wave
    time_steps = np.arange(t_initial, t_final, dt)

    SIGMAS_x_SQUARED = []  # spatial variance

    i = 0
    
    for time in time_steps:
        print(f"t = {time}")
        state = Quench(time, t_final, i, x, state, x_max, dx, folder, method, Nx)
        plot_evolution_frame(x, time, i, state)

        sigma_x_squared = x_variance(x, dx, state)
        SIGMAS_x_SQUARED.append(sigma_x_squared)

        # if i == i_rand:
        #     np.save(f"state_{method}_{time}_{x_max=}", state)
        
        i += 1
    SIGMAS_x_SQUARED = np.array(SIGMAS_x_SQUARED)
    np.save(f"RK4_SIGMAS_x_SQUARED_no_potential_{dx=}.npy", SIGMAS_x_SQUARED)
    # np.save(f"RK4_SIGMAS_SQUARED_{α=}_{dx=}.npy", SIGMAS_x_SQUARED)



def globals(method):
    α = 4
    β = 1

    # makes folder for simulation frames
    if method=="RK4":
        folder = Path(f'{method}_no_potential')
        # folder = Path(f'{method}_quench_HO-neg_HO')
        # folder = Path(f'{α=}')
        # folder = Path(f'{β=}')
        # folder = Path(f'restricted_V_{α=}')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    x_max = 45
    dx = 0.001
    Nx = int(2 * x_max / dx)

    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # for Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    # time dimension
    t_initial = 0
    t_final = 2
    dt =  5 * m * (0.005) ** 2 / (np.pi * hbar)


    # quench time
    T = 0.001

    # initial conditions
    # HO ground state for RK4
    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    i = 0
    return folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, kx, t_initial, t_final, dt, T, wave, i, α, β


if __name__ == "__main__":
    """FUNCTION CALLS"""

    folder, hbar, m, ω, l1, l2, Nx, x_max, x, dx, kx, t_initial, t_final, dt, T, wave, i, α, β = globals(method="RK4")

    time_range = np.arange(t_initial, t_final, dt)

    # i_rand = int(np.floor(random.uniform(1,  len(time_range)))) - 1
    # i_rand = 16209 # t_final = 2
    # i_rand = 114000 # t_final = 5

    evolve(method="RK4", label="")

    print(f"\n{x_max = }")
    # # x_cut1
    # print(f"x_cut_left = {x[2900]= }")
    # print(f"x_cut_right = {x[Nx-2900]= }")

    # # x_cut2
    # print(f"x_cut_left = {x[1500]= }")
    # print(f"x_cut_right = {x[Nx-1500]= }")

    # x_cut3
    # print(f"x_cut_left = {x[250]= }")
    # print(f"x_cut_right = {x[Nx-250]= }")

    ####

    print(f"x_cut_left = {x[250]= }")
    print(f"x_cut_right = {x[Nx-250]= }")
    print(f"{dx=}")
  


