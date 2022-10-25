# Ana Fabela Hinojosa, 28/06/2022
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.fft import fft, ifft
from scipy.integrate import quad
from scipy import linalg
from scipy.linalg import expm
from tqdm import tqdm
from odhobs import psi as cpsi


plt.rcParams['figure.dpi'] = 200

####################################################################################################

def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integratequad() for a complex function
    def real_func(*args):
        return np.real(func(*args))

    def imag_func(*args):
        return np.imag(func(*args))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]


def V(x):
    return -hbar * np.sqrt(2 * g / m) * x + 4 * g * x ** 4


def Schrodinger_eqn(t, Ψ):
    # Fourier derivative theorem
    KΨ = -hbar ** 2 / (2 * m) * ifft(-(k ** 2) * fft(Ψ))
    VΨ = V(x) * Ψ
    return (-1j / hbar) * (KΨ + VΨ)


def Schrodinger_RK4(t, delta_t, Ψ):
    k1 = Schrodinger_eqn(t, Ψ)
    k2 = Schrodinger_eqn(t + delta_t / 2, Ψ + k1 * delta_t / 2)
    k3 = Schrodinger_eqn(t + delta_t / 2, Ψ + k2 * delta_t / 2)
    k4 = Schrodinger_eqn(t + delta_t, Ψ + k3 * delta_t)
    return Ψ + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_RK4(t, t_final, i, x, wave, x_max, delta_x, folder):
    PLOT_INTERVAL = 5000

    waves = []
    # SIGMAS_SQUARED = []
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
            plt.plot(y, (-hbar * np.sqrt(2 * g / m) * y) + 4 * g * y ** 4, color="black", linewidth=2)
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


            h = abs(wave) ** 2
            h_right = h[1:]
            h_left = h[:-1]
            print(f"wave normalisation: {delta_x / 2 * np.sum(h_right + h_left)}")

        wave = Schrodinger_RK4(t, delta_t, wave)
        i += 1
        t += delta_t

    np.save(f"waves_list_{t_final=}.npy", waves)

####################################################################################################

def HHamiltonian(x, n):
    h = 1e-2
    z = np.array(x)
    z[z == 0] = 1e-200
    psi_nz = cpsi(n, z)
    d2Ψdz2 = (cpsi(n, (z + h)) - 2 * psi_nz + cpsi(n, (z - h))) / h ** 2
    return -(hbar ** 2 / (2 * m)) * d2Ψdz2 + (-hbar * np.sqrt(2 * g / m) * z + 4 * g * z ** 4) * psi_nz


def element_integrand(x, m, n):
    psi_m = cpsi(m, x)
    return np.conj(psi_m) * HHamiltonian(x, n)


# NxN MATRIX
def HMatrix(N):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            b = 2 * np.abs(np.sqrt(4 * min(m, n) + 2) + 2)
            element = complex_quad(
                element_integrand, -b, b, args=(m, n), epsabs=1.49e-03, limit=1000
            )
            # print(element)
            M[m][n] = element
            # # TESTING THE INTEGRAND AND INTEGRATION LIMITS
            # xs = np.linspace(-b, b, 1000)
            # plt.plot(xs, element_integrand(xs, m, n))
            # plt.show()

    # print(f"{M = }")
    return M

def U_operator(N, t):
    # print(f"{HMatrix(N) = }")
    return expm(-1j * HMatrix(N) * t / hbar)

def U_time_evolution(N, t):
    HO_GS = np.zeros(N, complex)
    HO_GS[0] = 1
    # print(HO_GS)

    ## create time evolution operator
    U = U_operator(N, t)
    # print(f"\ntime evolution operator:\n")
    # for line in U:
    #     print ('  '.join(map(str, line)))
    ## state vector
    return np.einsum('ij,j->i', U, HO_GS)


def plot_spatial_wavefunction(N, y, t):
    # calculating basis functions
    y[y == 0] = 1e-200
    PHI_ns = []
    for n in range(N):
        phi_n = cpsi(n, y)
        PHI_ns.append(phi_n)
    PHI_ns = np.array(PHI_ns)

    ## state vector
    c = U_time_evolution(N, t)

    psi_jy = np.zeros(y.shape, complex)

    # making spatial wavefunctions
    for n in range(N):
        psi_jy += c[n] * PHI_ns[n]

    plt.plot(y, (-hbar * np.sqrt(2 * g / m) * y) + 4 * g * y ** 4, color="black", linewidth=2)
    plt.plot(y, 10 * abs(psi_jy) ** 2, label=fR"$\psi({t:.02f})$")
    plt.grid(linestyle=':')
    plt.title(fR"time evolution using $U(t)$")
    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel("x")
    plt.xlim(-1.5, 1.5)
    plt.legend()
    
    plt.savefig(f"{folder2}/{t:.06f}.png")
    plt.clf()


####################################################################################################

def globals():
    #makes folder for simulation frames
    folder = Path('HH_RK4_time_evolution')
    folder2 = Path('HH_U_time_evolution')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    os.makedirs(folder2, exist_ok=True)
    os.system(f'rm {folder2}/*.png')

    # units based on "Bender's PT-symmetry book"
    hbar = 1
    m = 1/2
    ω = 2
    g = 1

    ####################################################################################################
    ## RK4 time evolution

    ## spatial domain
    x_max = 10
    x = np.linspace(-x_max, x_max, 1024 * 20, endpoint=False)
    n = x.size
    delta_x = x[1] - x[0]

    # Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, delta_x) 

    # # initial condition
    # HO ground state
    wave = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    wave = np.array(wave, dtype=complex)

    # time interval
    t = 0
    t_final = 10
    delta_t = 1.5 * m * delta_x ** 2 / (np.pi * hbar)

    # counting index
    i = 0

    ####################################################################################################
    ## U-operator time evolution
    N = 5
    Ny = 2048
    y = np.linspace(-2, 2, Ny)
    delta_y = y[1] - y[0]

    return folder, folder2, hbar, m, ω, g, x_max, x, delta_x,  n, k, wave, t, t_final, delta_t, i, N, Ny, y, delta_y
    # return folder2, hbar, m, ω, g, x_max, x, delta_x,  n, k, wave, t, t_final, delta_t, i, N, Ny, y, delta_y



if __name__ == "__main__":

    folder, folder2, hbar, m, ω, g, x_max, x, delta_x,  n, k, wave, t, t_final, delta_t, i, N, Ny, y, delta_y = globals()
    # folder2, hbar, m, ω, g, x_max, x, delta_x,  n, k, wave, t, t_final, delta_t, i, N, Ny, y, delta_y = globals()


    ## Runga-Kutta time evolution
    evolve_RK4(t, t_final, i, x, wave, x_max, delta_x, folder)

    ####################################################################################################
    ## U-operator time evolution
    time_steps = np.arange(0.0, 10, 0.001)
    for step in time_steps:
        plot_spatial_wavefunction(N, y, step)

    ####################################################################################################