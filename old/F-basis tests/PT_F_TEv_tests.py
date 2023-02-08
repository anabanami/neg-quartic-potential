# Ana Fabela Hinojosa, 30/01/2023
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy import linalg
from scipy.linalg import expm
from scipy.fft import fft, ifft
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['figure.dpi'] = 200

def F_basis_vector(x, n):
    return (1 / np.sqrt(P)) * np.exp((1j * 2 * np.pi * n * x) / P)

# kinetic energy
def K():
    return - (hbar * kx)**2 / (2 * m)

# Piece-wise potential
def V(x, t):
    if t < T:
        return (hbar /2) * (x / l1**2) ** 2
    else:
        return (hbar /2) * (x / l2**2) ** 2

def F_split_step2(Ψ, t, dt):
    """Evolve Ψ in time from t-> t+dt using a single step of the second order Fourier Split step method with time step dt"""
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    Ψ = ifft(np.exp(-1j * K() * dt / hbar) * fft(Ψ))
    Ψ = np.exp(-1j * V(x, t) * dt * 0.5 / hbar) * Ψ
    return Ψ

def expected_x_squared(x, Ψ):
    f = abs(Ψ ** 2) * x ** 2
    f_right = f[1:] # right endpoints
    f_left = f[:-1] # left endpoints
    return (dx / 2) * np.sum(f_right + f_left)


def variance(x, Ψ):
    f = x * abs(Ψ ** 2)
    f_right = f[1:] # right endpoints
    f_left = f[:-1] # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)
    # print(f"{expectation_value_x = }")
    
    g = abs(Ψ ** 2) * (x - expectation_value_x) ** 2
    g_right = g[1:] 
    g_left = g[:-1]

    ## checking normalisation
    # h = abs(Ψ ** 2)
    # h_right = h[1:]
    # h_left = h[:-1]

    return dx / 2 * np.sum(g_right + g_left)


def variance_evolution(t, t_final, dt, i, y, state):
    ## state vector
    state = F_split_step2(state, t, dt)

    EXPECTED_Xs_SQUARED = []
    SIGMAS_SQUARED = []
    while t < t_final:
        expectation_value_x_squared = expected_x_squared(y, state)
        # spatial variance plot
        sigma_squared = variance(y, state)
        # print(f"{sigma_squared = }")
        EXPECTED_Xs_SQUARED.append(expectation_value_x_squared)
        SIGMAS_SQUARED.append(sigma_squared)

        state = F_split_step2(state, t, dt)
        i += 1
        t += dt
        print(t)

    EXPECTED_Xs_SQUARED = np.array(EXPECTED_Xs_SQUARED)
    SIGMAS_SQUARED = np.array(SIGMAS_SQUARED)
    return EXPECTED_Xs_SQUARED, SIGMAS_SQUARED


####################################################################################################

def globals():
    # #makes folder for simulation frames
    # folder = Path('QUENCH_HO-HO_second_order')

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

    x = np.linspace(-L/2, L/2, Nx)
    dx = x[1] - x[0]

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    t_initial = 0
    t_final = 6
    # dt = 0.001
    dt = m * dx ** 2 / (np.pi * hbar)

    T = 0.5

    return hbar, m, ω, l1, l2, L, Nx, x, dx, kx, P, t_initial, t_final, dt, T #, folder


if __name__ == "__main__":

    hbar, m, ω, l1, l2, L, Nx, x, dx, kx, P, t_initial, t_final, dt, T = globals()

    HO_GS = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-x ** 2 /(2 * l1 ** 2))

    state = HO_GS
    time_steps = np.arange(t_initial, t_final, dt)

    t = 0
    i = 0

    ### TEST 1 ### expectation value of x squared and variance for split step time evolution of GS ###

    expected_x_squared_list, sigmas_list = variance_evolution(t, t_final, dt, i, x, state)
    # print(f"{time_steps.shape = }")
    # print(f"{sigmas_list.shape = }")

    # plt.plot(time_steps, expected_x_squared_list, label=R"$\left< x^2 \right>$")
    # plt.ylabel(R"$\left< x^2 \right>$")
    # plt.title(fR"Expectation value: $x^2$")
    # plt.xlabel("t")
    # plt.legend()
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    # # plt.show()


    plt.plot(time_steps, sigmas_list, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{x}^2$")
    plt.title(f"Spatial dispersion")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("Variance_Quench.png")
    

    ### TEST 2 ### P operator ###