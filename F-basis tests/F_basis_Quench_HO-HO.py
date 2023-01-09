import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import linalg
from scipy.linalg import expm
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integratequad() for a complex function
    def real_func(*args):
        return np.real(func(*args))
    def imag_func(*args):
        return np.imag(func(*args))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]

def F_basis_vector(x, n):
    return (1 / np.sqrt(P)) * np.exp((1j * 2 * np.pi * n * x) / P)

#IC
def IC_integrand(x, n):
    HO1_GS = (np.sqrt(np.pi) / l1) * np.exp(-x **2 /(2 * l1))
    return np.conj(F_basis_vector(x, n)) *  HO1_GS

# Piece-wise potential
def V(x, t):
    if t < T:
        return (1/2) * (x / l1**2) ** 2
    else:
        return (1/2) * (x / l2**2) ** 2

def U_operator(M):
    return expm(-1j * M * delta_t / hbar)


def U_time_evolution(M, state):
    ## create time evolution operator
    U = U_operator(M)
    ## state vector
    return np.einsum('ij,j->i', U, state)


def plot_spatial_wavefunction(N, M, y, t, state, i):

    ## state vector
    c = U_time_evolution(M, state)
    print(c[0])

    PLOT_INTERVAL = 1
    
    if not i % PLOT_INTERVAL:
        # calculating basis functions
        s_ns = []
        for n in range(N):
            s_n = F_basis_vector(n, y)
            s_ns.append(s_n)
        PHI_ns = np.array(s_ns)

        # IC wavefunction
        psi_jy = np.zeros(y.shape, complex)

        # making spatial wavefunctions
        for n in range(N):
            psi_jy += c[n] * PHI_ns[n]

        if t < T:
            # initial HO 
            print("Initial HO!")
            plt.plot(y, V(y, t), color="black", linewidth=2)
        else:
            # final HO
            print("after quench!")
            plt.plot(y, V(y, t), color="black", linewidth=2)


        plt.plot(y, abs(psi_jy) ** 2, label=fR"$\psi({t:.02f})$")
        plt.ylabel(R"$|\psi(x, t)|^2$")
        # plt.plot(y, np.real(psi_jy), label=fR"Re($\psi$)")
        # plt.plot(y, np.imag(psi_jy), label=fR"Im($\psi$)")
        # plt.ylabel(R"$\psi(t)$")


        plt.legend()
        plt.xlabel("x")
        plt.xlim(-L/2, L/2)
        plt.ylim(-2, 10)

        plt.savefig(f"{folder}/{i // PLOT_INTERVAL:04d}.png")
        plt.clf()

    return c



####################################################################################################

def globals():
    #makes folder for simulation frames
    folder = Path('QUENCH_HO-HO')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    ## natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    #lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    N = 100

    L = 10
    Nx = 2048
    x = np.linspace(-L / 2, L / 2, Nx)
    delta_x = x[1] - x[0]
    P = L

    t = 0
    t_final = 2
    delta_t = 0.01

    T = 0.5

    M1 = np.load(f"matrix_100_HO_1.npy")
    M2 = np.load(f"matrix_100_HO_2.npy")


    return folder, hbar, m, ω, l1, l2, N, L, Nx, x, delta_x, P, t, t_final, delta_t, T, M1, M2


if __name__ == "__main__":

    folder, hbar, m, ω, l1, l2, N, L, Nx, x, delta_x, P, t, t_final, delta_t, T, M1, M2 = globals()


    HO_GS = np.zeros(N, complex)
    for n in tqdm(range(N)):
        HO_GS[n] = complex_quad(IC_integrand, -L / 2, L / 2, args=(n), epsabs=1.49e-08, epsrel=1.49e-08, limit=500)

    # # U-operator time evolution
    state = HO_GS
    time_steps = np.arange(t, t_final, delta_t)
    i = 0
    for step in time_steps:
        if step < T:
            state = plot_spatial_wavefunction(N, M1, x, step, state, i)
        else:
            state = plot_spatial_wavefunction(N, M2, x, step, state, i)
        i += 1

