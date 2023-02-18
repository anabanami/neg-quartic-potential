import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy import linalg
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

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
    return evals, evects # (evals.shape= (300,) ), (evects.shape= (300, 300) where the column v[:, i])

def F_basis_vector(x, n):
    # Fourier state (exponential form)
    return (1 / np.sqrt(P)) * np.exp(1j * 2 * np.pi * n * x / P) # (.shape= (2048,))

def basis_functions(x, N):
    # calculating basis functions
    S_ns = []
    for n in range(-N, N):
        S_n = F_basis_vector(x, n)
        S_ns.append(S_n)
    return np.array(S_ns) # (S_ns.shape= (300,)

def V(x):
    return (hbar / (2 * m)) * (x / l1 ** 2) ** 2
    # return - x ** 4

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
    for j in range(2 * N):
        cj = evects[:, j]
        S_jx = np.zeros(x.shape, complex)
        # for each Fourier basis vector
        for n in range(2 * N):
            S_jx += cj[n] * S_ns[n]
        ## fix phase factor
        # S_jx /= np.exp(1j * np.angle(S_jx[Nx // 2]))
        # this makes the function's height = 1. Maybe want integral mod2 = 1
        wavefunctions.append(S_jx / np.max(np.abs(S_jx))) 
    return np.array(wavefunctions) # (wavefunctions.shape= (300, 2048))

def plot_wavefunctions(N, x, evals, wavefunctions):
    for i in range(5):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(
            x,
            np.real(wavefunctions[i]) + evals[i],
            "-",
            linewidth=1,
            label=fR"$\psi_{i}$",
            color=color,
        )
        plt.plot(
            x, np.imag(wavefunctions[i]) + evals[i], 
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
    # plt.ylim(-0.5, 25)
    # plt.xlim(-10,10)

    plt.axvline(0, linestyle=":", alpha=0.4, color="black")
    plt.legend(loc="upper right")
    plt.xlabel(r'$x$')


# ################################################################

def P_states(states):
    # Operator is basis dependent*
    P_states = []
    for state in states:
        # apply parity inversion (reverse array)
        P_state = state[::-1]
        # append to new list
        P_states.append(P_state)
    return np.array(P_states) # (P_states.shape= (300, 300) where the column v[:, i])

def PT_innerproducts(states, P_states):
    inner_prods = np.zeros(ND, dtype="complex")
    # print(inner_prods.shape)
    for i, istate in enumerate(P_states): 
        for j, jstate in enumerate(states):
            """PT inner product: PT(Ψ) * Ψ * dx = (PΨ)^* * Ψ * dx
            np.vdot() conjugates first input and performs a dot product of arrays"""
            inner_prods[i] = np.vdot(istate, jstate) * delta_x
    return inner_prods # (inner_prods.shape= (300,))

def PT_normalise(evects, inner_prods):
    PT_normed_evects = np.zeros_like(evects, dtype="complex")
    for j in range(ND):
        PT_normed_evects[:, j] = evects[:, j] / np.sqrt(inner_prods[j])
    return PT_normed_evects # (PT_normed_evects.shape= (300, 300) where the column v[:, i])


################################################################

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
    ND = 2 * N

    # X-space
    L = 30
    Nx = 2048
    xs = np.linspace(-L/2, L/2, Nx)
    delta_x = xs[1] - xs[0]

    ks = 2 * np.pi* np.fft.fftfreq(Nx, delta_x)


    return hbar, m, ω, l1, P, N, ND, L, Nx, xs, delta_x, ks

################################################################

hbar, m, ω, l1, P, N, ND, L, Nx, xs, delta_x, ks = globals()

S_ns = basis_functions(xs, N)
# Make  matrix 
# M = Matrix(N)
# np.save("matrix_HO.npy", M)
# np.save("matrix_neg_quartic.npy", M)

M = np.load("matrix_HO.npy")
# M = np.load("matrix_neg_quartic.npy")

# remember that evects are columns!
evals, evects = linalg.eigh(M)
# print(f"\n{evects.shape=}\n")

# # Plotting wavefunctions
# eigenfunctions = wavefunctions(xs, N, S_ns, evects)
# plot_wavefunctions(N, xs, evals, eigenfunctions)
# plt.plot(xs, V(xs), alpha=0.4, color="black")
# plt.show()

#### PT NORMALISE in Fourier space###
P_evects = P_states(evects)
# print(f"{P_evects.shape=}\n")

# # Plot P flipped eigenstates
# P_S_ns = P_states(S_ns)
# P_eigenfunctions = wavefunctions(xs, N, P_S_ns, evects)
# plot_wavefunctions(N, xs, evals, P_eigenfunctions)
# plt.plot(xs, V(xs), alpha=0.4, color="black")
# plt.show()


PT_inner_prods = PT_innerproducts(evects, P_evects)
# print(f"{PT_inner_prods.shape=}\n")

PT_evects = PT_normalise(evects, PT_inner_prods) ###################### (PT_normed_evects.shape= (300, 300) where the column v[:, i])
print(f"{PT_evects.shape=}\n")

# # Plotting PT_normed_wavefunctions
# PT_eigenfunctions = wavefunctions(xs, N, S_ns, PT_evects)
# plot_wavefunctions(N, xs, evals, PT_eigenfunctions)
# plt.plot(xs, V(xs), alpha=0.4, color="black")
# plt.show()

## checking PT orthogonality for PT normalised eigenvectors
M2 = np.zeros_like(M)
for i in range(ND):
    u = PT_evects[:, i]
    # Spatial reflection
    Pu = u[::-1]
    for j in range(ND):
        v = PT_evects[:, j]
        M2[i][j] += np.vdot(Pu, v)

plt.matshow(np.real(M2))
plt.colorbar()
plt.show()

plt.matshow(np.imag(M2))
plt.colorbar()
plt.show()


## checking PT orthogonality for PT normalised eigenvectors
M3 = np.zeros_like(M)
for i in range(ND):
    u = PT_evects[:, i]
    for j in range(ND):
        v = PT_evects[:, j]
        M3[i][j] += np.vdot(u, v)

plt.matshow(np.real(M3))
plt.colorbar()
plt.show()

plt.matshow(np.imag(M3))
plt.colorbar()
plt.show()


