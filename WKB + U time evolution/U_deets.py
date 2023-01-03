# Ana Fabela Hinojosa, 29/10/2022
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import linalg
from scipy.linalg import expm
from scipy.fft import fft, ifft
from tqdm import tqdm


plt.rcParams['figure.dpi'] = 200

def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integratequad() for a complex function
    def real_func(*args):
        return np.real(func(*args))
    def imag_func(*args):
        return np.imag(func(*args))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]

# Piece-wise potential for quench
def V(x, t):
    return (1/2) * (m * ((length_1**2) * m / hbar)**2 * x ** 2) #<<<<---X is acting weird here because I want to use it as the integration variable below.
    # if t < T:
        # return (1/2) * (m * ((length_1**2) * m / hbar)**2 * x ** 2)
    # else:
    #     return (1/2) * (m * ((length_2**2) * m / hbar)**2 * x ** 2)

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
             # HO turning points plus a bit
            b = 2 * np.abs(np.sqrt(4 * min(m, n) + 2) + 2)
            element = complex_quad(
                element_integrand, -b, b, args=(m, n, t), epsabs=1.49e-03, limit=1000 #<<<<---X is acting weird because I use it as the integration variable.
            )
            M[m][n] = element
    return M

# def U_operator(M):
#     return expm(-1j * M * delta_t / hbar)


# def U_time_evolution(M, state):
#     ## create time evolution operator
#     U = U_operator(M)
#     ## state vector
#     return np.einsum('ij,j->i', U, state)

####################################################################################################

def globals():
    #makes folder for simulation frames
    folder = Path('QUENCH_U_time_evolution')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # units based on "Bender's PT-symmetry book"
    hbar = 1
    m = 1/2
    ω = 2
    g = 1

    length_1 = np.sqrt(hbar /(m * ω))
    length_2 = 4 * np.sqrt(hbar /(m * ω))

    Nx = 2048
    x = np.linspace(-10, 10, Nx)
    x[x==0] = 1e-200
    delta_x = x[1] - x[0]
    n = x.size

    # Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, delta_x) 

    t = 0
    t_final = 6
    delta_t = 0.001

    T = 0.1

    # wkb basis
    normalised_WKB_HO = np.load(f"normalised_wkb_states_HO.npy")
    N = len(normalised_WKB_HO)

    #ICS
    HO_GS = normalised_WKB_HO[0]

    return folder, hbar, m, ω, g, length_1, length_2, Nx, x, delta_x, k, normalised_WKB_HO, N, HO_GS, t, t_final, delta_t, T 


if __name__ == "__main__":

    folder, hbar, m, ω, g, length_1, length_2, Nx, x, delta_x, k, normalised_WKB_HO, N, HO_GS, t, t_final, delta_t, T  = globals()

    ##################################################################################################
    # Making HO matrix
    M = HMatrix(N, t)
    plt.pyplot.matshow(M)
    plt.show()
