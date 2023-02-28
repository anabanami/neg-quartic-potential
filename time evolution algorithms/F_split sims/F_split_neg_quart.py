import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import linalg
from scipy.linalg import expm
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)


def variance(x, dx, Ψ):
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)
    # print(f"{expectation_value_x =}")

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]
    return dx / 2 * np.sum(g_right + g_left)


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


def F_basis_vector(x, n):
    # Fourier state (exponential form)
    return (1 / np.sqrt(P)) * np.exp(1j * 2 * np.pi * n * x / P)  # (.shape= (512,))


# kinetic energy
def K():
    return (hbar * kx) ** 2 / (2 * m)


# Test potentials
def V(x, t):
    T = 0.001
    if t < T:
        return (1 / 2) * m * ((hbar / (m * l1 ** 2)) * x) ** 2
    else:
        return -(x ** 4)


def F_split_step2(Ψ, t, dt):
    """Evolve Ψ in time from t-> t+dt using a single step of the second order Fourier Split step method with time step dt"""
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    Ψ = ifft(np.exp(-1j * K() * dt / hbar) * fft(Ψ))
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    return Ψ


def evolve_and_plot_spatial_wavefunction(N, y, t, state, i):
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


def globals():
    # makes folder for simulation frames
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

    Nx = 512
    P = 30

    x_max = 15

    x = np.linspace(-x_max, x_max, Nx, endpoint=False)  # HO
    n = x.size
    dx = x[1] - x[0]

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    t_initial = 0
    t_final = 40
    ## Nyquist dt
    dt = 0.5 * m * dx ** 2 / (np.pi * hbar)
    T = 0.001

    return folder, hbar, m, ω, l1, l2, Nx, x, dx, kx, P, t_initial, t_final, dt, T


if __name__ == "__main__":

    folder, hbar, m, ω, l1, l2, Nx, x, dx, kx, P, t_initial, t_final, dt, T = globals()

    HO_GS = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))

    # # split step time evolution of GS
    state = HO_GS
    time_steps = np.arange(t_initial, t_final, dt)
    Ψs = []  # LIST of state vectors (list of row vectors shape like x = (512,))
    SIGMAS_SQUARED = []  # spatial variance
    i = 0
    for t in time_steps:
        print(t)
        if t < T:
            state = evolve_and_plot_spatial_wavefunction(Nx, x, t, state, i)
            sigma_x_squared = variance(x, dx, state)
            Ψs.append(state)
        else:
            state = evolve_and_plot_spatial_wavefunction(Nx, x, t, state, i)
            sigma_x_squared = variance(x, dx, state)
            Ψs.append(state)

        SIGMAS_SQUARED.append(sigma_x_squared)
        i += 1

    # Ψs = np.array(Ψs)
    # np.save("State_vectors.npy", Ψs)

    SIGMAS_SQUARED = np.array(SIGMAS_SQUARED)
    np.save(f"SIGMAS_SQUARED.npy", SIGMAS_SQUARED)

    # Ψs = np.load("State_vectors.npy")
    # print(Ψs[0])

    sigmas_list = np.load("SIGMAS_SQUARED.npy")
    time = np.linspace(t, t_final, len(sigmas_list))

    variance_plot(time, sigmas_list)

