import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import convolve


plt.rcParams['figure.dpi'] = 180

def V(x):
    return x ** 2

def gaussian_blur(data, pts):
    """gaussian blur an array by given number of points"""
    x = np.arange(-2 * pts, 2 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation

Nx = 2048
x = np.linspace(-10, 10, Nx)
x[x==0] = 1e-200
delta_x = x[1] - x[0]
y = np.linspace(-3, 22, Nx).T

ϵ = 0

Energies = np.load("Energies_HO_WKB_N=10.npy")
Energies = Energies.reshape(len(Energies))


for n, E in enumerate(Energies):
    print(f"{E = }")
    print(f"{n = }")

    # does my method work for all E's?

    δminus = 0.4
    δplus = 0.4

    δRight = δminus
    δLeft = δplus

    wkb = np.zeros(Nx, dtype=complex)
    x0 = np.sqrt(E)
    a = -x0
    b = x0
    F0 = -(2 * a)
    F1 = 2 * b

    u0 = F0**(1/3) * (a - x[(a - δminus < x) & (x < a + δplus)])
    u1 = F1**(1/3) * (x[(b - δLeft < x) & (x < b + δRight)] - b)

    Q = np.sqrt((V(x) - E).astype(complex))
    P = np.sqrt((E - V(x)).astype(complex))

    # LHS of potential barrier
    integral_left = np.cumsum(Q[x < a - δminus]) * delta_x
    integral_left = -(integral_left - integral_left[-1])
    wkb[x < a - δminus] = np.exp(-integral_left) / (2 * np.sqrt(Q[x < a - δminus]))

    # around left turning point "a"
    Ai_a, Aip_a, Bi_a, Bip_a = special.airy(u0)
    wkb[(a - δminus < x) & (x < a + δplus)] = Ai_a * np.sqrt(np.pi) / F0 ** (1/6)

    # inside potential barrier 
    excessively_long_array = np.cumsum(P[x > a]) * delta_x
    integral_a_x = excessively_long_array[x[x > a] > a + δplus]
    wkb[x > a + δplus] = np.cos(integral_a_x - np.pi/4) / np.sqrt(P[x > a + δplus])
    # wkb[x > a + δplus] = (np.cos(integral_a_x) + np.sin(integral_a_x)) / 2 < Do I still need to figure out a more correct inside state given psi_1 = c * psi_2???

    # around right turning point "b"
    Ai_b, Aip_b, Bi_b, Bip_b = special.airy(u1)
    if n % 2 == 0:
        wkb[(b - δLeft < x) & (x < b + δRight)] = Ai_b * np.sqrt(np.pi) / F1 ** (1/6)
    else:
        wkb[(b - δLeft < x) & (x < b + δRight)] = -Ai_b * np.sqrt(np.pi) / F1 ** (1/6)

    # RHS of potential barrier
    integral_right = np.cumsum(Q[x > b + δRight]) * delta_x
    if n % 2 == 0:
        wkb[x > b + δRight] = np.exp(-integral_right) / (2 * np.sqrt(Q[x > b + δRight]))
    else:
        wkb[x > b + δRight] = -np.exp(-integral_right) / (2 * np.sqrt(Q[x > b + δRight]))

    ############## PLOTTING with and without gaussian smoothing ###################################
    pts = 25
    wkb = gaussian_blur(wkb, pts)

    ax = plt.gca()
    color = next(ax._get_lines.prop_cycler)['color']

    plt.plot(x, np.real(wkb) + E, label=fR"$\psi_{n}$", color=color)
    plt.plot(x, np.imag(wkb) + E, linestyle='--', color=color)
    plt.axhline(E, linestyle=":", linewidth=0.6, color="grey")
    plt.ylabel(r'$Energy$', labelpad=6)
    plt.xlabel(r'$x$', labelpad=6)
    
    plt.plot(x, V(x), linewidth=2, color="grey")

    # plt.axvline(a, linestyle="--", linewidth=0.5, color="red")
    # plt.axvline(b, linestyle="--", linewidth=0.5, color="red")
    # plt.fill_betweenx(y, a - δminus, a + δplus , alpha=0.1, color="pink")
    # plt.fill_betweenx(y, b - δLeft, b + δRight , alpha=0.1, color="pink")

    # plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-0.2, 21)

plt.show()







## IDEAS ########################################################## TO DO

    # ## TEST P squared:
    # for n, state in enumerate(states_ϵ0):
    #     # is this the same as PP_states_ϵ0_1 ???
    #     P_state = P_states_ϵ0[n]
    #     P_operator = [pp / p for pp, p in zip(state, P_state)] 
    #     P_operator_squared = [i ** 2 for i in P_operator]
    #     print(f"\nIs P complex? {np.iscomplex(P_operator)}")
    #     plt.plot(x, np.real(P_operator_squared))
    #     plt.plot(x, np.imag(P_operator_squared))
    #     plt.title(fR"$P^2$ for state $\psi_{n}(x)$")
    #     plt.show()





# def C_operator(normalised_states, normalised_P_states):
#     wavefunction_PT_products = []
#     for i, P_state in enumerate(normalised_P_states):
#         state_j = np.dot(np.conj(P_state), normalised_states[i])
#         wavefunction_PT_products.append(state_j)
#         # print(f"{state_j = }")

#     c_ns = [] 
#     for j, prod in enumerate(wavefunction_PT_products):
#         c_n = prod * (-1) ** j
#         c_ns.append(c_n)
#     C_op = np.sum(c_ns)
#     return C_op



# ## TEST C squared:
# C_ϵ0 = C_operator(normalised_states_ϵ0, normalised_P_states_ϵ0)
# print(f"\nIs C complex? {np.iscomplex(C_ϵ0)}")
# # print(f"C operator = {C_ϵ0}")
# print(f"Test that C^2 = 1\n{C_ϵ0 ** 2}\n")




# def HΨ(x, ϵ, normalised_states):
#     for state in normalised_states:
#         # Fourier derivative theorem
#         KΨ = -hbar ** 2 / (2 * m) * ifft(-(k ** 2) * fft(state))
#         VΨ = V(x, ϵ) * state
#         return (-1j / hbar) * (KΨ + VΨ)


# def element_integrand(x, ϵ, C_op, normalised_state, P_normalised_state):
#     return C_op * np.conj(P_normalised_state) * HΨ(x, ϵ, normalised_state)


# def Matrix(N):
#     M = np.zeros((N, N), dtype="complex")
#     for m in tqdm(range(N)):
#         for n in tqdm(range(N)):
#             element = element_integrand #<<<< WHAT KIND OF INTEGRAL DO I WANT HERE??????????????
#             print(element)
#             M[m][n] = element    
#     print(f"{M = }")
#     return M


# def U_operator(N, t):
#     # print(f"{HMatrix(N) = }")
#     return expm(-1j * HMatrix(N) * t / hbar)


# def U_time_evolution(N, t):
#     HO_GS = np.zeros(N, complex)
#     HO_GS[0] = 1
#     # print(HO_GS)

#     ## create time evolution operator
#     U = U_operator(N, t)
#     # print(f"\ntime evolution operator:\n")
#     # for line in U:
#     #     print ('  '.join(map(str, line)))
#     ## state vector
#     return np.einsum('ij,j->i', U, HO_GS)






# TIME DEPENDENT SIMULATION

    # FFT variable
    # k = 2 * np.pi * np.fft.fftfreq(Nx, delta_x) 
    # # time interval
    # t_d = m * delta_x ** 2 / (np.pi * hbar)
    # t = 0
    # t_final = 1
    # delta_t = t_d