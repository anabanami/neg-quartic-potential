# Ana Fabela Hinojosa, 29/10/2022
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import linalg
from scipy.linalg import expm
from tqdm import tqdm
from odhobs import psi as cpsi


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

# Piece-wise potential
def V(x, t):
    if t < T:
        return (1/2) * (m * ω**2 * x ** 2)
    else:
        return -x ** 4

def Hamiltonian(x, n, t):
    h = 1e-2
    z = np.array(x)
    z[z == 0] = 1e-200
    psi_nz = cpsi(n, z)
    d2Ψdz2 = (cpsi(n, (z + h)) - 2 * psi_nz + cpsi(n, (z - h))) / h ** 2
    return -(hbar ** 2 / (2 * m)) * d2Ψdz2 + V(z, t) * psi_nz

def element_integrand(x, m, n, t):
    psi_m = cpsi(m, x)
    return np.conj(psi_m) * Hamiltonian(x, n, t)

# NxN MATRIX
def HMatrix(N, t):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
             # HO turning points plus a bit
            b = 2 * np.abs(np.sqrt(4 * min(m, n) + 2) + 2)
            element = complex_quad(
                element_integrand, -b, b, args=(m, n, t), epsabs=1.49e-03, limit=1000
            )
            # print(element)
            M[m][n] = element
            # TESTING THE INTEGRAND AND INTEGRATION LIMITS
            # xs = np.linspace(-b, b, 1000)
            # plt.plot(xs, element_integrand(xs, m, n))
            # plt.show()
    # print(f"{M = }")
    return M

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

    PLOT_INTERVAL = 10
    
    if not i % PLOT_INTERVAL:
        # calculating basis functions
        PHI_ns = []
        for n in range(N):
            phi_n = cpsi(n, y)
            PHI_ns.append(phi_n)
        PHI_ns = np.array(PHI_ns)

        # IC wavefunction
        psi_jy = np.zeros(y.shape, complex)

        # making spatial wavefunctions
        for n in range(N):
            psi_jy += c[n] * PHI_ns[n]

        if t < T:
            # HO plot
            print("Harmonic Oscillator!")
            plt.plot(y, (1/2) * V(y, t), color="black", linewidth=2)
        else:
            # negative quartic potential
            print("QUENCH!")
            plt.plot(y, V(y, t), color="black", linewidth=2)
        plt.axis(xmin=-6, xmax=6, ymin=-25, ymax=25)
        plt.ylabel("E")

        plt.twinx()

        plt.plot(y, abs(psi_jy) ** 2, label=fR"$\psi({t:.02f})$")
        # plt.plot(y, np.real(psi_jy), label=fR"Re($\psi$)")
        # plt.plot(y, np.imag(psi_jy), label=fR"Im($\psi$)")


        plt.ylim(ymin=-1, ymax=1)
        plt.legend()
        # plt.ylabel(R"$|\psi(x, t)|^2$")

        plt.xlabel("x")
        # plt.title(fR"time evolution using $U(t)$")

        plt.title(f"state at t = {t:04f}")

        plt.savefig(f"{folder}/{i // PLOT_INTERVAL:04d}.png")
        plt.clf()


    return c


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

    N = 300
    Nx = 1024
    x = np.linspace(-10, 10, Nx)
    x[x == 0] = 1e-200
    delta_x = x[1] - x[0]

    t = 0
    t_final = 6
    delta_t = 0.001

    T = 0.1

    M1 = np.load(f"HMATRIX_HO.npy")
    M2 = np.load(f"HMATRIX_neg_quartic.npy")

    #ICS
    HO_GS = np.zeros(N, complex)
    HO_GS[0] = 1

    # # Superposition TEST
    # HO_1ex = np.zeros(N, complex)
    # HO_1ex[1] = 1
    # TEST_ICS = (HO_GS + HO_1ex) / np.sqrt(2)

    return folder, hbar, m, ω, g, t, t_final, delta_t, T,  N, Nx, x, delta_x, M1, M2, HO_GS


if __name__ == "__main__":

    folder, hbar, m, ω, g, t, t_final, delta_t, T, N, Nx, x, delta_x, M1, M2, HO_GS = globals()

    # for n in range(N):
    #     plt.plot(x, cpsi(n, x), label=f"{n=}")
    # plt.legend()
    # plt.show()

    ###################################################################################################
    # Making HO and neg quartic matrices
    # t = 1
    # print("HO!")
    # np.save(f"HMATRIX_HO.npy", HMatrix(N, t))
    # t = 2
    # print("-x^4!")
    # np.save(f"HMATRIX_neg_quartic.npy", HMatrix(N, t))

    # # U-operator time evolution
    state = HO_GS
    time_steps = np.arange(t, t_final, delta_t)
    i = 0
    for step in time_steps:
        if step < T:
            # print(state[0])
            state = plot_spatial_wavefunction(N, M1, x, step, state, i)
        else:
            # print(state[0])
            state = plot_spatial_wavefunction(N, M2, x, step, state, i)
        i += 1

    # ###################################################################################################

        
    