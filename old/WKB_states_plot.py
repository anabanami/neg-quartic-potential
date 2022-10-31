# Ana Fabela Hinojosa, 10/06/2022
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.special import hyp2f1
from scipy.special import gamma

plt.rcParams['figure.dpi'] = 200

def globals():
    hbar = 1
    m = 1
    ω = 1
    g = 1
    λ = (hbar ** 2 / (m * g)) ** -(1 / 6)
    L = λ * (hbar ** 2 / (m * g)) ** (1 / 6)

    x_max = 100
    x = np.linspace(-x_max, x_max, 1024*100, endpoint=False)
    x_step = x[1] - x[0]

    # time interval
    t_d = m * x_step ** 2 / (np.pi * hbar)
    t = 0
    t_final = 1
    delta_t = t_d

    Energies = np.load("Energies_WKB_N=10.npy")
    Energies = Energies.reshape(len(Energies))

    return hbar, m, ω, L, x, x_max, t, t_final, delta_t, Energies


def norm_phi(E):
    return 2 * 4 * sc.gamma(5 / 4) ** 2 / (E ** (1 / 4) * np.sqrt(np.pi))


def phi(x, E):
    return (
        (1 / np.sqrt(norm_phi(E)))
        * np.exp(-1j * np.sqrt(E) * sc.hyp2f1(-1 / 2, 1 / 4, 5 / 4, -x ** 4 / E))
        / ((E + x ** 4) ** (1 / 4))
    )


if __name__ == "__main__":
    hbar, m, ω, L, x, x_max, t, t_final, delta_t, Energies = globals()

    print(f"Energies matrix shape: {Energies.shape}")


    for i, E in enumerate(Energies):
        WKB_state = phi(x, E)
        plt.plot(x, (np.real(WKB_state)) * 7 + E, label=fR'$\phi_{i}$')
        plt.plot(x, (np.imag(WKB_state)) * 7  + E, linestyle='--', color='grey')

    plt.ylabel("E")
    plt.xlabel("x")
    plt.xlim(-25, 25)
    plt.legend()
    plt.show()



