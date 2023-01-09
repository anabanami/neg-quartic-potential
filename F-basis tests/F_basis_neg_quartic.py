import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import linalg
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


def Hamiltonian(x, n):
    return - ((1/2) * (1j * 2 * np.pi * n / P) ** 2 + x ** 4)


def element_integrand(x, m, n):
    s_m = F_basis_vector(x, m)
    s_n = F_basis_vector(x, n)
    return np.conj(s_m) * Hamiltonian(x, n) * s_n


# NxN MATRIX
def Matrix(N):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            element = complex_quad(
                element_integrand,
                -L / 2,
                L / 2,
                args=(m, n),
                epsabs=1.49e-08,
                epsrel=1.49e-08,
                limit=500
            )
            M[m][n] = element
            # # TESTING THE INTEGRAND AND INTEGRATION LIMITS
            # xs = np.linspace(-L, L, 1000)
            # plt.plot(xs, element_integrand(xs, m, n))
            # plt.show()
    return M

def filtering_sorting_eigenstuff(evals, evects):
    # filtering
    # print(f"{evals = }\n")
    mask = (0 < evals.real) & (evals.real < 100)
    # print(f"\n{mask}\n")
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(np.round(evals.real, 3) + np.round(evals.imag, 3) / 1e6)
    # print(f"{order = }\n")
    evals = evals[order]
    evects = evects[:, order]
    # print(f"{evals = }\n")
    return evals, evects

def spatial_wavefunctions(N, x, evals, evects):
    #calculating basis functions
    x[x == 0] = 1e-200
    s_ns = []
    for n in range(N):
        s_n = F_basis_vector(x, n)
        s_ns.append(s_n)

    s_ns = np.array(s_ns)

    eigenfunctions = []
    # print(f"{np.shape(evects) = }\n")

    for j in range(5):
        c = evects[:, j]
        # print(f"{c= }\n")

        S_jx = np.zeros(x.shape, complex)
        # print(f"{np.shape(psi_jx) = }\n")

        # for each Fourier basis vector
        for n in range(N):
            S_jx += c[n] * s_ns[n]
            # print(f"{S_jx = }\n")

        eigenfunctions.append(S_jx / np.max(np.abs(S_jx)))

    for i in range(5):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(
            x,
            np.real(eigenfunctions[i]) + evals[i],
            "-",
            linewidth=1,
            label=fR"$\psi_{i}$",
            color=color,
        )
        plt.plot(
            x, np.imag(eigenfunctions[i]) + evals[i], "--", linewidth=1, color=color
        )

        # # probability density
        # plt.plot(
        #     x,
        #     abs(eigenfunctions[i] **2 ) + evals[i],
        #     linewidth=1,
        #     label=fR"$|\psi_{i}^2|$",
        #     color=color,
        # )
        # plt.ylabel(r'$ |\psi_{n}|^2$')
        

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

    plt.plot(x, - x ** 4, alpha=0.4, color="black")
    plt.axvline(0, linestyle=":", alpha=0.4, color="black")
    plt.legend(loc="upper right")
    plt.xlabel(r'$x$')
    plt.xlim(-L/2, L/2)
    plt.ylim(-2, 10)
    plt.show()
    return s_ns


################################################################
## GLOBALS
## natural units according to wikipedia
hbar = 1
m = 1
ω = 1

N = 100

L = 10
Nx = 2048 * 8
xs = np.linspace(-L / 2, L / 2, Nx)
delta_x = xs[1] - xs[0]
P = L

matrix = Matrix(N)
np.save(f"matrix_100_neg_quartic.npy", matrix)

# remember that evects are columns!
evals, evects = linalg.eigh(matrix)

# evals, evects = filtering_sorting_eigenstuff(evals, evects)

# s_ns = spatial_wavefunctions(N, xs, evals, evects)
