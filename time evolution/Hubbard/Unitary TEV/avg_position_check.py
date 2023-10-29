# Import necessary libraries and modules
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

def globals():
    hbar = 1

    # Bender units
    m = 1 / 2
    omega = 2

    # space dimension
    dx = 0.08

    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 15

    return (
        dt,
        t_initial,
        t_final,
    )


if __name__ == "__main__":

    # Retrieve global parameters
    (
        dt,
        t_initial,
        t_final,
    ) = globals()

    data = np.load('Unitary_hubbard_avg_positions.npy')

    time = np.linspace(t_initial, t_final, len(data))
    
    ### TITLES
    # plt.title(f"Shannon entropy for wavepacketin free space")
    # plt.title(f"Shannon entropy for wavepacket in an upside down HO")
    plt.title(f"Position expectation value for wavepacket\nin the negative quartic potential")

    plt.plot(time, data)
    plt.ylabel(R"$< x >$")

    plt.xlabel("t")
    plt.grid()
    plt.show()