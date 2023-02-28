import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy import linalg
from tqdm import tqdm
from scipy.signal import convolve


plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation

def complex_trapezoid(integrand, y, dy):
    # complex integrals
    real = np.real(trapezoid(integrand, y, dy))
    imaginary = np.imag(trapezoid(integrand, y, dy))
    return real + 1j * imaginary

def filter_sorting(evals, evects):
    # filtering
    mask = (0 < evals.real) & (evals.real < 50)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(np.round(evals.real,3) + np.round(evals.imag, 3) / 1e6)
    evals = evals[order]
    evects = evects[:, order]
    return evals, evects # (evals.shape= (300,)) becomes (29,), (evects.shape= (300, 29) where the column v[:, i])

def F_basis_vector(x, n):
    # Fourier state (exponential form)
    return (1 / np.sqrt(P)) * np.exp(1j * 2 * np.pi * n * x / P) # (.shape= (512,))

def basis_functions(x, N):
    # calculating basis functions
    S_ns = []
    for n in range(-N, N):
        S_n = F_basis_vector(x, n)
        S_ns.append(S_n)
    return np.array(S_ns) # (S_ns.shape= (300,)

def restricted_V(x):
    # gassuan smooth vertices
    pts = 5
    V = np.zeros_like(x)
    V[128:384] = -x[128:384] ** 4
    return gaussian_smoothing(V, pts)

def V(x):
    # return (1 / 2) * m * ((hbar / (m * l1 ** 2)) * x) ** 2
    # return (1 / 2) * m * ((hbar / (m * l2 ** 2)) * x) ** 2
    return restricted_V(x)
    # return -x ** 4

def Hamiltonian(x, n):
    return (-hbar ** 2 /(2 * m)) * ((1j * 2 * np.pi * n / P) ** 2) + V(x)

def element_integrand(x, m, n):
    S_m = F_basis_vector(x, m)
    S_n = F_basis_vector(x, n)
    return np.conj(S_m) * Hamiltonian(x, n) * S_n # (element.shape= (300,))

# NDxND Hamiltonian MATRIX (in Fourier space)
def Matrix(N):
    M = np.zeros((ND, ND), dtype="complex")
    for i, m in tqdm(enumerate(range(-N, N))):
        for j, n in tqdm(enumerate(range(-N, N))):
            y = element_integrand(xs, m, n)
            element = complex_trapezoid(y, xs, delta_x)
            M[i][j] = element
    return M # (M.shape= (300, 300))


def wavefunctions(x, N, S_ns, evects):
    wavefunctions = []
    for j in range(len(evals)):
        cj = evects[:, j]
        S_jx = np.zeros(x.shape, complex)
        # for each Fourier basis vector
        for n in range(len(evals)):
            S_jx += cj[n] * S_ns[n]
        ## fix phase factor
        # S_jx /= np.exp(1j * np.angle(S_jx[Nx // 2]))
        # this makes the function's height = 1. Maybe want integral mod2 = 1
        wavefunctions.append(S_jx / np.max(np.abs(S_jx))) 
    return np.array(wavefunctions) # (wavefunctions.shape= (300, 512))


def plot_wavefunctions(N, x, evals, wavefunctions):
    for i in range(5):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(
            x,
            np.real(wavefunctions[i]) + 5*evals[i], # scaled offset for visibility 
            "-",
            linewidth=1,
            label=fR"$\psi_{i}$",
            color=color,
        )
        plt.plot(
            x, np.imag(wavefunctions[i]) + 5*evals[i], 
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

    # plt.ylim(-1, 30)
    # plt.ylim(-0.5, 25)
    # plt.xlim(-10,10)

    plt.axvline(0, linestyle=":", alpha=0.4, color="black")
    plt.legend(loc="upper right")
    plt.xlabel(r'$x$')


################################################################

def globals():
    ## natural units according to wikipedia
    hbar = 1
    m = 1
    ω = 1
    # Harmonic oscillator length #lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))
    l2 = 2 * l1

    P = 30 # <The n harmonic makes n cycles in the function's period P 

    # Basis states / 2
    N = 150
    ND = 2 * N

    # X-space
    x_max = 30
    Nx = 512
    xs = np.linspace(-x_max/2, x_max/2, Nx)
    delta_x = xs[1] - xs[0]

    ks = 2 * np.pi* np.fft.fftfreq(Nx, delta_x)

    return hbar, m, ω, l1, l2, P, N, ND, x_max, Nx, xs, delta_x, ks

################################################################

hbar, m, ω, l1, l2, P, N, ND, x_max, Nx, xs, delta_x, ks = globals()

S_ns = basis_functions(xs, N)
# Make  matrix 
# M = Matrix(N)
# np.save("matrix_HO.npy", M)
# np.save("matrix_2ndHO.npy", M)
# np.save("matrix_neg_quartic.npy", M)

# M = np.load("matrix_HO.npy")
# M = np.load("matrix_2ndHO.npy")
M = np.load("matrix_neg_quartic.npy")

# plt.matshow(np.real(M))
# plt.colorbar()
# plt.show()

# plt.matshow(np.imag(M))
# plt.colorbar()
# plt.show()

# remember that evects are columns! v[:, j]
evals, evects = linalg.eigh(M)
print(evals.shape)
print(evects.shape)

evals, evects = filter_sorting(evals, evects)
print(evals.shape)
print(evects.shape)

# Plotting wavefunctions
eigenfunctions = wavefunctions(xs, N, S_ns, evects)
plot_wavefunctions(N, xs, evals, eigenfunctions)
plt.plot(xs, V(xs), alpha=0.4, color="black")
plt.show()