# Ana Fabela Hinojosa, 14/06/2022
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.special import hyp2f1
from scipy.special import gamma
from scipy.integrate import quad
from scipy import linalg
from tqdm import tqdm
from odhobs import psi as cpsi

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


def HHamiltonian(y, n):
    h = 1e-6
    
    z = np.array(y)
    z[z == 0] = 1e-200
    psi_nz = cpsi(n, z)
    d2Ψdz2 = ((cpsi(n, z + h) - 2 * psi_nz + cpsi(n, z - h)) / h) ** 2

    return -(hbar ** 2 / (2 * m)) * d2Ψdz2 + (-hbar * np.sqrt(2 * g / m) * z + 4 * g * z ** 4) * psi_nz


def element_integrand(x, m, n):
    psi_m = cpsi(m, x)
    return np.conj(psi_m) * HHamiltonian(x, n)


# NxN MATRIX
def Matrix(N):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            b = 2 * np.abs(np.sqrt(4 * min(m, n) + 2) + 2)
            element = complex_quad(
                element_integrand, -b, b, args=(m, n), epsabs=1.49e-08, limit=1000
            )
            # print(element)
            M[m][n] = element
            # # TESTING THE INTEGRAND AND INTEGRATION LIMITS
            # xs = np.linspace(-b, b, 1000)
            # plt.plot(xs, element_integrand(xs, m, n))
            # plt.show()
    return M


def filtering_sorting_eigenstuff(evals, evects):
    # filtering
    mask = (0 < evals.real) & (evals.real < 100)
    # print(mask)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    # print(f"{evals[0] = }\n")
    order = np.argsort(np.round(evals.real, 3) + np.round(evals.imag, 3) / 1e6)
    # print(f"{order = }\n")
    evals = evals[order]
    evects = evects[:, order]
    # print(f"{evals[0] = }\n")
    return evals, evects


def spatial_wavefunctions(N, y, evals, evects):
    # calculating basis functions
    y[y == 0] = 1e-200
    PHI_ns = []
    for n in range(N):
        phi_n = cpsi(n, y)
        PHI_ns.append(phi_n)
    PHI_ns = np.array(PHI_ns)

    np.save(f"PHI_ns.npy", PHI_ns)
    PHI_ns = np.load('PHI_ns.npy')

    eigenfunctions = []
    # print(f"{np.shape(evects) = }\n")

    for j in range(4):
        c = evects[:, j]
        print(f"{c= }\n")

        # print(f"{np.shape(PHI_ns[0]) = }\n")

        psi_jy = np.zeros(y.shape, complex)
        # print(f"{np.shape(psi_jy) = }\n")

        # for each H.O. basis vector
        for n in range(N):
            print(f"{n = }\n")

            psi_jy += c[n] * PHI_ns[n]
            # print(f"{psi_jy = }\n")

        eigenfunctions.append(psi_jy)

    fig, ax = plt.subplots()

    # probability density plot
    for i in range(4):

        # plt.plot(
        #     y, abs(eigenfunctions[i]) ** 2, "-", linewidth=1, label=fr"$|\psi_{i}|^2$"
        # )
        plt.plot(y, eigenfunctions[i] + evals[i], "-", linewidth=1, label=fr"$\psi_{i}$")
        plt.legend(loc="upper right")
        plt.xlabel(r'$y$')
        # plt.ylabel(r'$ |\psi_{n}|^2$')
        plt.ylabel(r'$\psi_{n}$')

    textstr = '\n'.join(
        (
            fr'$E_0 = {evals[0]:.01f}$',
            fr'$E_1 = {evals[1]:.01f}$',
            fr'$E_2 = {evals[2]:.01f}$',
            fr'$E_3 = {evals[3]:.01f}$',
            fr'$E_4 = {evals[4]:.01f}$',
        )
    )
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')
    plt.show()
    return PHI_ns


################################################################
# GLOBALS
def globals():

    N = 100

    hbar = 1
    m = 2
    # ω = 1
    g = 1
    λ = (hbar ** 2 / (m * g)) ** -(1 / 6)
    L = λ * (hbar ** 2 / (m * g)) ** (1 / 6)

    Ny = 2048
    y = np.linspace(-15, 15, Ny)
    delta_y = y[1] - y[0]

    # Energies = np.load("Energies_WKB_N=10.npy")
    # Energies = Energies.reshape(len(Energies))

    # return hbar, m, ω, L, Energies
    return hbar, m, g, L, N, Ny, y, delta_y


if __name__ == "__main__":

    hbar, m, g, L, N, Ny, y, delta_y = globals()

    matrix = Matrix(N)
    # np.save(f"matrix_512.npy", matrix)
    np.save(f"matrix_100.npy", matrix)

    # matrix = np.load(f"matrix_512.npy")
    # matrix = np.load(f"matrix_100.npy")
    print(f"\n\nMatrix\n{matrix}")

    # remember that evects are columns!
    evals, evects = linalg.eigh(matrix)
    # evals, evects = filtering_sorting_eigenstuff(evals, evects)

    print(f"\neigenvalues\n{evals}")
    print(f"\neigenvectors\n{evects}")

    PHI_ns = spatial_wavefunctions(N, y, evals, evects)
