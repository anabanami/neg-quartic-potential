# Ana Fabela Hinojosa, 28/02/2023
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['figure.dpi'] = 200


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
    V[50:462] = -x[50:462] ** 4  # <<<< THIS IS WRONG ?
    return gaussian_smoothing(V, pts)


# Potentials
def V(x, t):
    T = 0
    return - x ** 4
    # T = 0.001
    # if t < T:
    #     return (1 / 2) * m * ((hbar / (m * l1 ** 2)) * x) ** 2
    # else:
    #     return (1 / 2) * m * ((hbar / (m * l2 ** 2)) * x) ** 2
        # #return - x ** 4
        # #return restricted_V(x)
        # #return 0


def Schrodinger_eqn(t, Ψ):

    # IMAGINATION SPACE :DDD
    t = 1j * t

    # Fourier derivative theorem
    KΨ = -(hbar ** 2) / (2 * m) * ifft(-(k ** 2) * fft(Ψ))
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
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)
    # print(f"{expectation_value_x =}")

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]

    return dx / 2 * np.sum(g_right + g_left)


def simulate_quench(t, t_final, i, x, wave, x_max, dx, folder):
    # generates simulation frame corresponding to time t (Quench occurs at t = 0)
    PLOT_INTERVAL = 50

    waves = []
    SIGMAS_SQUARED = []
    while t < t_final:
        # print(i)
        if not i % PLOT_INTERVAL:
            waves.append(wave)

            # # raw plot
            # plt.plot(x, V(x, t), color='k', linewidth=2)
            # plt.plot(x, np.real(wave), label="real part")
            # plt.plot(x, np.imag(wave), label="imaginary part")
            # plt.ylabel(R"$\psi(x,t)$")
            # plt.title(f"state at t = {t:04f}")
            # plt.legend()

            # prob. density plot
            plt.plot(x, V(x, t), color='k', linewidth=2)
            plt.plot(x, abs(wave ** 2))
            plt.ylabel(R"$|\psi(x,t)|^2$")
            plt.title(f"state at t = {t:04f}")

            # # phase plot
            # plt.plot(x, np.angle(wave))
            # plt.ylabel(R"$\theta(x)$")
            # plt.title(f"state's phase at t = {t:04f}")

            plt.ylim(-1, 1)
            plt.xlabel("x")
            plt.savefig(f"{folder}/{i // PLOT_INTERVAL:04d}.png")
            # plt.show()
            plt.clf()

            # spatial variance
            sigma_x_squared = variance(x, dx, wave)
            SIGMAS_SQUARED.append(sigma_x_squared)
            # print(f"variance = {sigma_x_squared}\n")

            h = abs(wave) ** 2
            h_right = h[1:]
            h_left = h[:-1]
            print(f"wave normalisation: {dx / 2 * np.sum(h_right + h_left)}")

        wave = Schrodinger_RK4(t, dt, wave)
        i += 1
        t += dt

    np.save(f"SIGMAS_SQUARED.npy", SIGMAS_SQUARED)
    # np.save(f"waves_list.npy", waves)


def variance_plot(time, sigmas_list):
    plt.plot(time, sigmas_list, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{x}^2$")
    plt.title(f"Spatial variance")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("RENAME ME.png")
    plt.show()
    plt.clf()


def globals():
    # makes folder for simulation frames
    folder = Path('imaginary TEv')
    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    hbar = 1
    m = 1
    ω = 1

    # for test potential
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    x_max = 10  # if this is 15 I need much smaller dt!
    x = np.linspace(-x_max, x_max, 512, endpoint=False)  # HO
    n = x.size
    dx = x[1] - x[0]

    # For Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, dx)

    # #initial condition
    # HO ground state
    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    # time interval
    t = 0
    t_final = 40
    # dt = even smaller???
    dt = 0.5 * m * dx ** 2 / (np.pi * hbar)
    i = 0
    return folder, hbar, m, ω, l1, l2, x_max, x, dx, n, k, wave, t, t_final, dt, i


if __name__ == "__main__":

    folder, hbar, m, ω, l1, l2, x_max, x, dx, n, k, wave, t, t_final, dt, i = globals()

    simulate_quench(t, t_final, i, x, wave, x_max, dx, folder)

    sigmas_list = np.load("SIGMAS_SQUARED.npy")
    time = np.linspace(t, t_final, len(sigmas_list))

    variance_plot(time, sigmas_list)
