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

def Hamiltonian(x, n):
    # return (1/2) * (-((1j * 2 * np.pi * n / P) ** 2) + (x / l1**2) ** 2)
    return (1/2) * (-((1j * 2 * np.pi * n / P) ** 2) + (x / l2**2) ** 2)

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
l2 = 2 * l1

N = 300

L = 10
P = L

Nx = 2048
xs = np.linspace(-L / 2, L / 2, Nx)
delta_x = xs[1] - xs[0]

# matrix = Matrix(N)
# print(np.shape(matrix))
# np.save(f"matrix_300_HO_1.npy", matrix)
# np.save(f"matrix_300_HO_2.npy", matrix)

matrix1 = np.load(f"matrix_300_HO_1.npy")
matrix2 = np.load(f"matrix_300_HO_2.npy")

# remember that evects are columns!
evals1, evects1 = linalg.eigh(matrix1)
evals2, evects2 = linalg.eigh(matrix2)

s_ns1 = spatial_wavefunctions(N, xs, evals1, evects1)
plt.plot(xs, (1/2) * (xs / l1**2) ** 2, alpha=0.4, color="black")
plt.show()

s_ns2 = spatial_wavefunctions(N, xs, evals2, evects2)
plt.plot(xs, (1/2) * (xs/ l2**2) ** 2, alpha=0.4, color="black")
plt.show()


# Normalising F states
normalised_F_HO = []
for state in evects1:
    N = np.vdot(state, state) * delta_x
    state /= np.sqrt(N)
    normalised_F_HO.append(state)

print(np.shape(normalised_F_HO))
# checking conventional orthogonality for normalised F states
M = np.zeros_like(matrix1)
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
