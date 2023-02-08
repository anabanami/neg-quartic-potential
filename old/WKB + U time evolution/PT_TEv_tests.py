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


def complex_trapezoid(integrand, y, dy):
    real = np.real(trapezoid(integrand, y, dy))
    imaginary = np.imag(trapezoid(integrand, y, dy))
    return real + 1j * imaginary


# Piece-wise potential for quench
def V(x, t):
    if t < T:
        return (hbar / 2) * (x / l1**2) ** 2
    else:
        return (hbar / 2) * (x / l2**2) ** 2


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

    return (dx / 2) * np.sum(g_right + g_left)


def variance_evolution(t, t_final, i, y, state):
    ## state vector
    c = U_time_evolution(M, state) #<---need to use this to create the state in spatial basis and to obtain its variance 
    # print(f"{c.shape = }")

    EXPECTED_Xs_SQUARED = []
    SIGMAS_SQUARED = []

    # IC wavefunction
    psi_jy = np.zeros(y.shape, complex)
    # making spatial wavefunctions
    for n in range(N):
        # print(f"{c[n] = }")
        # print(f"{normalised_WKB_HO[n].shape = }")

        psi_jy += c[n] * normalised_WKB_HO[n]

        expectation_value_x_squared = expected_x_squared(y, psi_jy)
        # spatial variance plot
        sigma_squared = variance(y, psi_jy)
        # print(f"{sigma_squared = }")
        EXPECTED_Xs_SQUARED.append(expectation_value_x_squared)
        SIGMAS_SQUARED.append(sigma_squared)

        c = U_time_evolution(M, c)
        i += 1

    EXPECTED_Xs_SQUARED = np.array(EXPECTED_Xs_SQUARED)
    SIGMAS_SQUARED = np.array(SIGMAS_SQUARED)
    return EXPECTED_Xs_SQUARED, SIGMAS_SQUARED


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
    l2 = 4 * np.sqrt(hbar /(m * ω))

    Nx = 2048
    x = np.linspace(-5, 5, Nx)
    x[x==0] = 1e-200
    dx = x[1] - x[0]
    n = x.size
    L = 10

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
    # os.makedirs(folder, exist_ok=True)
    # os.system(f'rm {folder}/*.png')
    ##################################################################################################

    # # Making HO matrix
    M = HMatrix(N, t_initial)

    time_steps = np.arange(t_initial, t_final, dt)

    state = np.zeros(N, dtype="complex")
    state[0] = 1

    i = 0

    ### TEST 1 ### expectation value of x squared and variance for split step time evolution of GS ###


        expected_x_squared_list, sigmas_list = variance_evolution(t, t_final, i, x, state) <<<< CORRECT THIS CALL CAUSE ITS STUPID
     
    
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

    ### TEST 2 ###  ###