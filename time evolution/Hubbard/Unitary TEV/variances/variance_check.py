# Comparing HDF5 files
# Ana Fabela 11/07/2023

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
    t_final = 6

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

    with open('Unitary_hubbard_variance.npy', 'rb') as f:
        data = np.load(f)

    time = np.linspace(t_initial, t_final, len(data))
    
    ### TITLES
    # plt.title(f"Spatial dispersion for wavepacket in free space")
    plt.title(f"Spatial dispersion for wavepacket in an upside down HO")
    # plt.title(f"Spatial dispersion for wavepacket in the negative quartic potential")
    # plt.title(f"Spatial dispersion for a shifted gaussian in a negative quartic potential")
    # plt.title(f"Spatial dispersion for wavepacket in a negative octic potential")

    plt.plot(time, data, label=R"$\left< x^2 \right> - \left< x \right>^2$")
    plt.ylabel(R"$\sigma_{(x)}^2$")

    # plt.ylabel(R"$\sigma_{(x-1)}^2$")
    # plt.plot(time, data, label=R"$\left< (x - 1)^2 \right> - \left< (x - 1) \right>^2$")

    # plt.ylabel(R"$\sigma_{(x-2)}^2$")
    # plt.plot(time, data, label=R"$\left< (x - 2)^2 \right> - \left< (x - 2) \right>^2$")

    plt.xlabel("t")
    plt.grid()
    plt.legend()
    plt.show()
