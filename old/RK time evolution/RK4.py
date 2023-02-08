# Ana Fabela Hinojosa, 15/04/2022
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.special as sc
from scipy.integrate import quad
from matplotlib.ticker import FormatStrFormatter


plt.rcParams['figure.dpi'] = 200

# Piece-wise potential
def V(x, t):
    print(t)
    T = 0.001
    if t < T:
        return (hbar / 2) * (x / l1**2) ** 2
    else:
        print("QUENCH")
        return (hbar / 2) * (x / l2**2) ** 2

def Schrodinger_eqn(t, Ψ):
    # Fourier derivative theorem
    KΨ = -hbar ** 2 / (2 * m) * ifft(-(k ** 2) * fft(Ψ))
    VΨ = V(x, t) * Ψ
    return (-1j / hbar) * (KΨ + VΨ)


def Schrodinger_RK4(t, dt, Ψ):
    k1 = Schrodinger_eqn(t, Ψ)
    k2 = Schrodinger_eqn(t + dt / 2, Ψ + k1 * dt / 2)
    k3 = Schrodinger_eqn(t + dt / 2, Ψ + k2 * dt / 2)
    k4 = Schrodinger_eqn(t + dt, Ψ + k3 * dt)
    return Ψ + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def variance(x, dx, Ψ):
    f = x * abs(Ψ ** 2)
    f_right = f[1:] # right endpoints
    f_left = f[:-1] # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)
    # print(f"{expectation_value_x =}")
    
    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:] 
    g_left = g[:-1]

    return dx / 2 * np.sum(g_right + g_left)


def simulate_quench(t, t_final, i, x, wave, x_max, dx, folder):
    # generates simulation frame corresponding to time t (Quench occurs at t = 0.001)
    PLOT_INTERVAL = 50

    waves = []
    SIGMAS_SQUARED = []
    while t < t_final:
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

            plt.plot(x, V(x, t), color="black", linewidth=2)
            plt.ylim(-0.2, 1)
            plt.xlabel("x")
            plt.savefig(f"{folder}/{i // PLOT_INTERVAL:06d}.png")
            plt.clf()

            # spatial variance
            sigma_x_squared = variance(x, dx, wave)
            SIGMAS_SQUARED.append(sigma_x_squared)
            print(f"variance = {sigma_x_squared}\n")

            # h = abs(wave) ** 2
            # h_right = h[1:]
            # h_left = h[:-1]
            # print(f"wave normalisation: {dx / 2 * np.sum(h_right + h_left)}")

        wave = Schrodinger_RK4(t, dt, wave)
        i += 1
        t += dt

    np.save(f"SIGMAS_SQUARED.npy", SIGMAS_SQUARED)
    # np.save(f"waves_list_{t_final=}.npy", waves)


def globals():
    #makes folder for simulation frames
    folder = Path('RK_quench_time_evolution')
    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    ## natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    #lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1


    x_max = 10
    x = np.linspace(-x_max/2, x_max/2, 2048, endpoint=False) # HO
    n = x.size
    dx = x[1] - x[0]

    # For Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, dx) 

    # #initial condition
    # HO ground state
    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    t_initial = 0
    t_final = 6
    ## Nyquist dt
    dt = m * dx ** 2 / (np.pi * hbar)

    i = 0

    return hbar, m, ω, l1, l2, x_max, x, dx,  n, k, wave, t_initial, t_final, dt, i, folder



if __name__ == "__main__":

    hbar, m, ω, l1, l2, x_max, x, dx,  n, k, wave, t_initial, t_final, dt, i, folder = globals()

    # simulate_quench(t_initial, t_final, i, x, wave, x_max, dx, folder)

    sigmas_list = np.load("SIGMAS_SQUARED.npy")
    time = np.linspace(t_initial, t_final, len(sigmas_list))
    
    plt.plot(time, sigmas_list, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{x}^2$")
    plt.title(f"Spatial variance")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("Variance_Quench.png")
    plt.show()
    plt.clf()
