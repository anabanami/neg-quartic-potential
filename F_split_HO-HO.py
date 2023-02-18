IMPLEMENT FOURIER BASIS CORRECTLY AS WITH TEST_2.py


import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import linalg
from scipy.linalg import expm
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

def F_basis_vector(x, n):
    return (1 / np.sqrt(P)) * np.exp((1j * 2 * np.pi * n * x) / P)

# kinetic energy
def K():
    return - (hbar * kx)**2 / (2 * m)

# Piece-wise potential
def V(x, t):
    if t < T:
        return (hbar / 2) * (x / l1**2) ** 2
    else:
        return (hbar / 2) * (x / l2**2) ** 2

def F_split_step2(Ψ, t, dt):
    """Evolve Ψ in time from t-> t+dt using a single step of the second order Fourier Split step method with time step dt"""
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    Ψ = ifft(np.exp(-1j * K() * dt / hbar) * fft(Ψ))
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    return Ψ

def plot_spatial_wavefunction(N, y, t, state, i):
    ## state vector
    psi = F_split_step2(state, t, dt)

    PLOT_INTERVAL = 50
    
    if not i % PLOT_INTERVAL:
        if t < T:
            # initial HO 
            print("Initial HO")
            plt.plot(y, V(y, t), color="black", linewidth=2)
        else:
            # final HO
            print("QUENCH!")
            plt.plot(y, V(y, t), color="black", linewidth=2)

        plt.plot(y, abs(psi) ** 2, label=fR"$\psi({t:.04f})$")
        plt.ylabel(R"$|\psi(x, t)|^2$")
        # plt.plot(y, np.real(psi), label=fR"Re($\psi$)")
        # plt.plot(y, np.imag(psi), label=fR"Im($\psi$)")
        # plt.ylabel(R"$\psi(t)$")

        plt.legend()
        plt.xlabel("x")
        plt.xlim(-L/2, L/2)
        plt.ylim(-0.2, 1)

        # plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
        # plt.clf()
        # plt.show()

    return psi

def globals():
    #makes folder for simulation frames
    folder = Path('QUENCH_HO-HO_F_second_order')

    # os.makedirs(folder, exist_ok=True)
    # os.system(f'rm {folder}/*.png')
    ## natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    #lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    Nx = 2048
    L = 10
    P = L

    x = np.linspace(-L, L, Nx)
    dx = x[1] - x[0]

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    t_initial = 0
    t_final = 6
    ## Nyquist dt
    dt = m * dx ** 2 / (np.pi * hbar)

    T = 0.001


    return folder, hbar, m, ω, l1, l2, L, Nx, x, dx, kx, P, t_initial, t_final, dt, T


if __name__ == "__main__":

    folder, hbar, m, ω, l1, l2, L, Nx, x, dx, kx, P, t_initial, t_final, dt, T = globals()

    HO_GS = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-x ** 2 /(2 * l1 ** 2))
    # plt.plot(x, HO_GS)
    # plt.plot(x, V(x, 0), color="black", linewidth=2)
    # plt.show()

    # # split step time evolution of GS
    state = HO_GS
    time_steps = np.arange(t_initial, t_final, dt)
    i = 0
    for t in time_steps:
        print(t)
        if t < T:
            state = plot_spatial_wavefunction(Nx, x, t, state, i)
        else:
            state = plot_spatial_wavefunction(Nx, x, t, state, i)
        i += 1