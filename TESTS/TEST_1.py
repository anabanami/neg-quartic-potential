import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy import linalg
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

def complex_trapezoid(integrand, y, dy):
    real = np.real(trapezoid(integrand, y, dy))
    imaginary = np.imag(trapezoid(integrand, y, dy))
    return real + 1j * imaginary

def F_basis_vector(x, n):
    return (1 / np.sqrt(P)) * np.exp(1j * 2 * np.pi * n * x / P)

def V(x):
    return (hbar / (2 * m)) * (x / l1 ** 2) ** 2
    # return - x ** 4

def Hamiltonian(x, n):
    return (-hbar ** 2 /(2 * m)) * ((1j * 2 * np.pi * n / P) ** 2) + V(x)

def element_integrand(x, m, n):
    s_m = F_basis_vector(x, m)
    s_n = F_basis_vector(x, n)
    return np.conj(s_m) * Hamiltonian(x, n) * s_n

# NDxND MATRIX
def Matrix(N):
    ND = N * 2
    M = np.zeros((ND, ND), dtype="complex")
    for i, m in tqdm(enumerate(range(-N, N))):
        for j, n in tqdm(enumerate(range(-N, N))):
            y = element_integrand(xs, m, n)
            element = complex_trapezoid(y, xs, delta_x)
            M[i][j] = element
    return M


def spatial_wavefunctions(N, x, evals, evects): # <<<  SUSPECT MY BUG IS HERE
    #calculating basis functions
    s_ns = []
    for n in range(-N, N):
        s_n = F_basis_vector(x, n)
        s_ns.append(s_n)
    s_ns = np.array(s_ns)

    eigenfunctions = []
    for j in range(5):
        cj = evects[:, j]
        S_jx = np.zeros(x.shape, complex)
        # for each Fourier basis vector
        for n in range(2 * N):
            S_jx += cj[n] * s_ns[n]
        S_jx /= np.exp(1j * np.angle(S_jx[Nx // 2]))
        eigenfunctions.append(S_jx / np.max(np.abs(S_jx))) # this makes the function's height = 1. Maybe want integral mod2 = 1

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
            x, np.imag(eigenfunctions[i]) + evals[i], 
            "--", linewidth=1, 
            color=color
        )

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

    plt.ylim(-1, 6)

    plt.axvline(0, linestyle=":", alpha=0.4, color="black")
    plt.legend(loc="upper right")
    plt.xlabel(r'$x$')
    return s_ns

def globals():
    ## natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    # Harmonic oscillator length
    l1 = np.sqrt(hbar / (m * ω))
    P = 30 # <The n harmonic makes n cycles in the function's period P 

    # Basis states / 2
    N = 150

    # X-space
    L = 30
    Nx = 2048
    xs = np.linspace(-L/2, L/2, Nx)

    delta_x = xs[1] - xs[0]

    return hbar, m, ω, l1, P, N, L, Nx, xs, delta_x
################################################################

hbar, m, ω, l1, P, N, L, Nx, xs, delta_x = globals()

# Make  matrix 
# M = Matrix(N)
# np.save("matrix_HO.npy", M)

M = np.load("matrix_HO.npy")

# remember that evects are columns!
evals, evects = linalg.eigh(M)
# print(f"\n{evects.shape}")
# print(f"\n{evects[0]}")


s_ns = spatial_wavefunctions(N, xs, evals, evects)
plt.plot(xs, V(xs), alpha=0.4, color="black")
plt.show()

