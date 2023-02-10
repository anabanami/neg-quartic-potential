import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
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
    # return (1/2) * m * (((m * l1**2) / hbar) * x) ** 2
    return - x ** 4

def Hamiltonian(x, n):
    return (-hbar**2 /(2 * m)) * ((1j * 2 * np.pi * n / P) ** 2) + V(x)

def element_integrand(x, m, n):
    s_m = F_basis_vector(x, m)
    s_n = F_basis_vector(x, n)
    return np.conj(s_m) * Hamiltonian(x, n) * s_n

# NxN MATRIX
def Matrix(N):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            y = element_integrand(xs, m, n)
            element = complex_trapezoid(y, xs, delta_x)
            M[m][n] = element
    return M

def filter_sorting(evals, evects):
    # filtering
    mask = (0 < evals.real) & (evals.real < 50)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(np.round(evals.real,3) + np.round(evals.imag, 3) / 1e6)
    evals = evals[order]
    evects = evects[:, order]
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
            x, np.imag(eigenfunctions[i]) + evals[i], 
            "--", linewidth=1, 
            color=color
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

    # plt.ylim(-0.2, 10)

    plt.ylim(-0.5, 25)
    plt.xlim(-10,10)

    plt.axvline(0, linestyle=":", alpha=0.4, color="black")
    plt.legend(loc="upper right")
    plt.xlabel(r'$x$')
    
    return s_ns


################################################################
## GLOBALS
## natural units according to wikipedia
hbar = 1
m = 1
ω = 1
#lengths for HO quench
l1 = np.sqrt(hbar / (m * ω))

N = 300

L = 30
P = L

Nx = 2048
xs = np.linspace(-L/2, L/2, Nx)
delta_x = xs[1] - xs[0]

# matrix = Matrix(N)
# np.save(f"matrix_300_HO.npy", matrix)
# np.save(f"matrix_300_neg_quartic.npy", matrix)

matrix = np.load(f"matrix_300_neg_quartic.npy")

# remember that evects are columns!
evals, evects = linalg.eigh(matrix)
evals, evects = filter_sorting(evals, evects)

s_ns = spatial_wavefunctions(N, xs, evals, evects)
plt.plot(xs, V(xs), alpha=0.4, color="black")
plt.show()

## Normalising F states in te conventional way
# normalised_F_HO = []
# for state in evects:
#     N = np.vdot(state, state) * delta_x
#     state /= np.sqrt(N)
#     normalised_F_HO.append(state)

# print(np.shape(normalised_F_HO))
# # checking conventional orthogonality for normalised F states
# M = np.zeros_like(matrix)
# for n, istate in enumerate(normalised_F_HO):
#     for m, jstate in enumerate(normalised_F_HO):
#         Orthogonality_check = (np.vdot(istate, jstate) * delta_x)
#         # print(f"{n, m = }: {Orthogonality_check}")
#         M[n][m] = Orthogonality_check

# plt.matshow(np.real(M))
# plt.colorbar()
# plt.show()

# plt.matshow(np.imag(M))
# plt.colorbar()
# plt.show()


#### PT NORMALISE ###
# Normalising F states
normalised_F_HO = []
for state in evects:
    N = np.vdot(state, state) * delta_x <<<<<<<<
    state /= np.sqrt(N)
    normalised_F_HO.append(state)

print(np.shape(normalised_F_HO))
# checking conventional orthogonality for normalised F states
M = np.zeros_like(matrix)
for n, istate in enumerate(normalised_F_HO):
    for m, jstate in enumerate(normalised_F_HO):
        Orthogonality_check = (np.vdot(istate, jstate) * delta_x)
        # print(f"{n, m = }: {Orthogonality_check}")
        M[n][m] = Orthogonality_check

plt.matshow(np.real(M))
plt.colorbar()
plt.show()

plt.matshow(np.imag(M))
plt.colorbar()
plt.show()
