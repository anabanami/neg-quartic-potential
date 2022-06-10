# Ana Fabela Hinojosa, 18/04/2022
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
import scipy.special as sc
from scipy.special import hyp2f1
from scipy.special import gamma
from scipy.integrate import quad
from tqdm import tqdm


def globals():

    hbar = 1
    m = 2
    ω = 1

    x_max = 100
    x = np.linspace(-x_max, x_max, 1024, endpoint=False)
    x_step = x[1] - x[0]

    # time interval
    t_d = m * x_step ** 2 / (np.pi * hbar)
    t = 0
    t_final = 1
    delta_t = t_d

    # negative quartic potential problem
    Energies = np.load("Energies_WKB.npy")
    Energies = Energies.reshape(len(Energies))

    return hbar, m, ω, x_max, x, t, t_final, delta_t, Energies


def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integrate.quad() for a complex function
    def real_func(*args):
        return np.real(func(*args))

    def imag_func(*args):
        return np.imag(func(*args))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]


def initial_condition(x):
    # HO ground state
    HO_gs = (m * ω / (np.pi * hbar)) ** (1 / 4) * np.exp(-m * ω * x ** 2 / (2 * hbar))
    return np.array(HO_gs, dtype=complex)


def norm_phi(E):
    return 2 * 4 * sc.gamma(5 / 4) ** 2 / (E ** (1 / 4) * np.sqrt(np.pi))

# negative quartic potential problem
def phi(x, E):
    return (
        (1 / np.sqrt(norm_phi(E)))
        * np.exp(-1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, -x ** 4 / E))
        / ((E + x ** 4) ** (1 / 4))
    )

# #positive quartic potential problem
# def phi(x, E):
#     return (
#         (1 / np.sqrt(norm_phi(E)))
#         * np.exp(-1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, x ** 4 / E))
#         / ((E - x ** 4) ** (1 / 4))
#     )

def orthogonality_test(Energies):
    
    M = np.zeros((len(Energies), len(Energies)), dtype="complex")

    for m, E_m in enumerate(Energies):
        for n, E_n in enumerate(Energies):
            print(f"{m = }")
            print(f"{n = }")
            print(f"{E_m = }\n{E_n = }")

            def integrand(x):
                return np.conj(phi(x, E_m)) * phi(x, E_n)

            M[m][n] = complex_quad(
                integrand, 1e-200, x_max, epsabs=1.49e-08, limit=3000
            )
            print(f"{M[m][n] = }")
    return M


def normalisation_test(Energies):
    total = 0
    for E in Energies:

        def integrand(x):
            return np.conj(phi(x, E)) * initial_condition(x)

        total += (
            abs(complex_quad(integrand, -x_max, x_max, epsabs=1.49e-08, limit=3000))
            ** 2
        )

    print(f"Normalisation test result: {total}")


if __name__ == "__main__":
    hbar, m, ω, x_max, x, t, t_final, delta_t, Energies = globals()

    # TEST: Are the WKB states properly normalised?
    def integrand(x):
        return abs(phi(x, Energies[7])) ** 2
    print(quad(integrand, -10 * x_max, 10 * x_max))

    M = orthogonality_test(Energies)
    plt.imshow(abs(M))
    plt.colorbar()
    plt.show()

    print(f"\nmax_value in M: {abs(M).max()}")
    print("After zeroing the main diagonal:")
    np.fill_diagonal(M, 0)
    print(f"max_value in M: {abs(M).max()}\n")

    plt.imshow(abs(M))
    plt.colorbar()
    plt.show()

    normalisation_test(Energies)

    # TRY WKB energies and Analytic solutions
