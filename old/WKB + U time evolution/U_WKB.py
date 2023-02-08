# Ana Fabela Hinojosa, 29/10/2022
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

def complex_trapezoid(integrand, y, dy):
    real = np.real(trapezoid(integrand, y, dy))
    imaginary = np.imag(trapezoid(integrand, y, dy))
    return real + 1j * imaginary

# Piece-wise potential for quench
def V(x, t):
    if t < T:
        return (hbar /2) * (x / l1**2) ** 2
    else:
        return (hbar /2) * (x / l2**2) ** 2

def Hamiltonian(x, t, Φ):
    KΦ = -hbar ** 2 / (2 * m) * ifft(-(k ** 2) * fft(Φ))
    VΦ = V(x, t) * Φ
    return KΦ + VΦ

def element_integrand(x, m, n, t):
    Φm = normalised_WKB_HO[m]
    # print("\n",np.shape(Φm))
    Φn = normalised_WKB_HO[n]
    # print("\n",np.shape(Φn))
    # print("\n",np.shape(Hamiltonian(x, t, Φn)))
    return np.conj(Φm) * Hamiltonian(x, t, Φn)

# NxN MATRIX
def HMatrix(N, t):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            element_int = element_integrand(x, m, n, t)
            M[m][n] = complex_trapezoid(element_int, x, dx)
    return M

def U_operator(M):
    return expm(-1j * M * dt / hbar)


def U_time_evolution(M, state):
    ## create time evolution operator
    U = U_operator(M)
    ## state vector
    # print(np.shape(U))
    return np.einsum('ij,j->i', U, state)


def plot_spatial_wavefunction(N, y, t, state, i):
    ## state vector
    c = U_time_evolution(M, state)
    # print(c.shape)

    PLOT_INTERVAL = 50
    
    if not i % PLOT_INTERVAL:
        # IC wavefunction
        psi_jy = np.zeros(y.shape, complex)
        # print(y.shape)

        # making spatial wavefunctions
        for n in range(N):
            psi_jy += c[n] * normalised_WKB_HO[n]

        if t < T:
            # initial HO 
            print("Initial HO!")
            plt.plot(y, V(y, t), color="black", linewidth=2)
        else:
            # final HO
            print("Quench!")
            plt.plot(y, V(y, t), color="black", linewidth=2)

        plt.plot(y, abs(psi_jy) ** 2, label=fR"$\psi({t:.02f})$")
        plt.ylabel(R"$|\psi(x, t)|^2$")
        # plt.plot(y, np.real(psi_jy), label=fR"Re($\psi$)")
        # plt.plot(y, np.imag(psi_jy), label=fR"Im($\psi$)")
        # plt.ylabel(R"$\psi(t)$")

        plt.legend()
        plt.xlabel("x")
        plt.xlim(-L/2, L/2)
        plt.ylim(-0.2, 1)

        plt.savefig(f"{folder}/{i // PLOT_INTERVAL:04d}.png")
        plt.clf()
        # plt.show()

    return c



####################################################################################################

def globals():
    # # units based on "Bender's PT-symmetry book"
    # hbar = 1
    # m = 1/2
    # ω = 2

    ## natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    l1 = np.sqrt(hbar /(m * ω))
    l2 = 2 * np.sqrt(hbar /(m * ω))

    Nx = 2048
    x = np.linspace(-5, 5, Nx)
    x[x==0] = 1e-200
    dx = x[1] - x[0]
    n = x.size
    L =10

    # Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, dx) 

    t_initial = 0
    t_final = 6
    ## Nyquist dt
    dt = m * dx ** 2 / (np.pi * hbar)

    T = 0.001

    # wkb basis
    normalised_WKB_HO = np.load(f"normalised_wkb_states_HO.npy")
    N = len(normalised_WKB_HO)


    return hbar, m, ω, l1, l2, Nx, x, dx, L, k, normalised_WKB_HO, N, t_initial, t_final, dt, T 


if __name__ == "__main__":

    hbar, m, ω, l1, l2, Nx, x, dx, L, k, normalised_WKB_HO, N, t_initial, t_final, dt, T  = globals()

    # makes folder for simulation frames
    folder = Path('QUENCH_U_time_evolution')
    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')
    ##################################################################################################

    time_steps = np.arange(t_initial, t_final, dt)
    # # Making HO matrix
    M = HMatrix(N, t_initial)
    # plt.matshow(np.real(M))
    # plt.colorbar()

    # plt.matshow(np.imag(M))
    # plt.colorbar()
    # plt.show()

    state = np.zeros(N, dtype="complex")
    state[0] = 1
    i = 0
    for t in time_steps:
        print(t)
        if t < T:
            state = plot_spatial_wavefunction(N, x, t, state, i)
        else:
            state = plot_spatial_wavefunction(N, x, t, state, i)
        i += 1

