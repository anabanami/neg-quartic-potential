# Ana Fabela Hinojosa, 15/04/2022
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from matplotlib.ticker import FormatStrFormatter

# Setting the resolution of the figures
plt.rcParams['figure.dpi'] = 200


def V(x, t):
    """Define the piece-wise potential function"""
    T = 0
    if t < T:
        return (1 / 2) * m * ω ** 2 * x ** 2
    else:
        return -(x ** 4)


def Schrodinger_eqn(t, Ψ):
    """Define the Schrodinger equation"""
    # Fourier derivative theorem
    KΨ = -(hbar ** 2) / (2 * m) * ifft(-(k ** 2) * fft(Ψ))
    VΨ = V(x, t) * Ψ
    return (-1j / hbar) * (KΨ + VΨ)


def Schrodinger_RK4(t, delta_t, Ψ):
    """Implement the 4th order Runge-Kutta method on TDSE"""
    k1 = Schrodinger_eqn(t, Ψ)
    k2 = Schrodinger_eqn(t + delta_t / 2, Ψ + k1 * delta_t / 2)
    k3 = Schrodinger_eqn(t + delta_t / 2, Ψ + k2 * delta_t / 2)
    k4 = Schrodinger_eqn(t + delta_t, Ψ + k3 * delta_t)
    return Ψ + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def variance(x, dx, Ψ):
    """Calculate the variance of the wave function using trapezoid rule"""
    f = x * abs(Ψ ** 2)
    f_right = f[1:]
    f_left = f[:-1]
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)
    print(f"{expectation_value_x =}")

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]

    return dx / 2 * np.sum(g_right + g_left)


def plot_evolution_frame(y, state, time, i):
    """
    Function to plot and save the evolution of the wave function in position space at a specific time.
    Parameters:
    - y: Position coordinates.
    - state: Wave function at time `time`.
    - time: The particular instant of time.
    - i: Index used for saving the plot.
    """
    ax = plt.gca()
    # plot of prob. density of state
    plt.plot(
        y,
        3 * abs(state) ** 2,
        label=R"$|\psi(x, t)|^2$",
    )

    plt.legend()
    plt.ylim(-0.2, 2.5)
    # plt.xlim(-5, 5)
    # plt.ylabel(R"$|\psi(x, t)|^2$")
    # plt.xlabel(R"$x$")

    textstr = f"t = {time:05f}"
    # place a text box in upper left in axes coords
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        verticalalignment='top',
    )
    plt.tight_layout()
    plt.savefig(f"{folder}/{i}.pdf")
    # plt.show()
    plt.clf()


def simulate_quench(t, t_final, i, x, wave, x_max, delta_x, folder):
    """Simulate the quantum quench from HO to negative quartic potential.
    Only plot every 50th frame of the simulation and save those files as pdfs in a folder.
    Calculate the variance and normalisation for each wave function"""
    PLOT_INTERVAL = 50
    waves = []
    SIGMAS_SQUARED = []
    while t < t_final:
        if not i % PLOT_INTERVAL:
            print(f"time = {t}\n")

            waves.append(wave)
            plot_evolution_frame(x, wave, t, i)

            sigma_x_squared = variance(x, delta_x, wave)
            SIGMAS_SQUARED.append(sigma_x_squared)
            print(f"variance = {sigma_x_squared}\n")

            h = abs(wave ** 2)
            h_right = h[1:]
            h_left = h[:-1]
            print(f"wave normalisation: {delta_x / 2 * np.sum(h_right + h_left)}")

        wave = Schrodinger_RK4(t, delta_t, wave)
        i += 1
        t += delta_t

    np.save("SIGMAS_SQUARED.npy", SIGMAS_SQUARED)


def globals():
    """Define global variables and initial conditions"""
    folder = Path('OG RK4_bleh')
    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    hbar = 1
    # Bender units
    m = 1 / 2
    ω = 2

    x_max = 10
    x = np.linspace(-x_max, x_max, 1024, endpoint=False)
    n = x.size
    delta_x = x[1] - x[0]

    k = 2 * np.pi * np.fft.fftfreq(n, delta_x)

    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    t = 0
    t_final = 6
    delta_t = m * delta_x ** 2 / (np.pi * hbar)

    i = 0

    return folder, hbar, m, ω, x_max, x, delta_x, n, k, wave, t, t_final, delta_t, i


if __name__ == "__main__":
    """Main function to run the simulation and plot the results"""

    (
        folder,
        hbar,
        m,
        ω,
        x_max,
        x,
        delta_x,
        n,
        k,
        wave,
        t,
        t_final,
        delta_t,
        i,
    ) = globals()

    simulate_quench(t, t_final, i, x, wave, x_max, delta_x, folder)

    sigmas_list = np.load("SIGMAS_SQUARED.npy")
    time = np.linspace(t, t_final, len(sigmas_list))

    plt.plot(time, sigmas_list, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{x}^2$")
    plt.title(f"Spatial variance")
    plt.xlabel("t")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.savefig("OG_RK4_Variance_Quench.pdf")
    plt.grid(color='gray', linestyle=':')
    plt.show()
    plt.clf()
