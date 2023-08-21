# Time Evolution of Analytical free space solution
# HDF5 protocol
# Ana Fabela 13/07/2023

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import h5py

plt.rcParams['figure.dpi'] = 200


def plot_evolution_frame(y, state, time, i):
    # potential
    V = np.zeros_like(y)
    plt.plot(y, V, color="black", linewidth=2, label="V(x)")
    # probability density
    plt.plot(y, abs(state) ** 2, label=R"$|\psi(x, t)|^2$")
    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel(R"$x$")
    plt.legend()
    plt.title(f"t = {time:05f}")
    plt.savefig(f"{folder}/{i}.png")
    plt.clf()


def TEV(x, times):
    states = []
    file = h5py.File('11.hdf5', 'w')

    # time evolution
    for t in times:
        print(f"t = {t}")
        state = np.exp(-(x ** 2) / (2 * l1 ** 2 * (1 + 1j * ω * t))) / np.sqrt(
            (np.pi ** (1 / 2) * l1 * (1 + 1j * ω * t))
        )
        states.append(state)
        # create a new dataset for each frame
        dset = file.create_dataset(f"{t}", data=state)

    file.close()

    PLOT_INTERVAL = 20
    for j, state in enumerate(states):
        if j % PLOT_INTERVAL == 0:
            print(f"t = {times[j]}")
            plot_evolution_frame(x, state, times[j], j)
            print(f"\n{np.sum(abs(state)**2)*dx = }")  # is normalisation preserved???


def globals():
    folder = Path(f'11')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm{folder}/*.png')

    # natural units
    hbar = 1
    m = 1
    ω = 1
    l1 = np.sqrt(hbar / m * ω)

    N_sites = 900
    dx = 0.1

    # space dimension
    x_max = 45
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)
    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 2

    # generate timesteps
    times = np.arange(t_initial, t_final, dt)

    return folder, hbar, m, ω, l1, N_sites, dx, x_max, Nx, x, dt, times


if __name__ == "__main__":
    """FUNCTION CALLS"""
    folder, hbar, m, ω, l1, N_sites, dx, x_max, Nx, x, dt, times = globals()

    TEV(x, times)

    print("\n,>>>>>>>>>>>>>>>>>>")
    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"{dx = }")
    print(f"{dt = }")
    print("\n,>>>>>>>>>>>>>>>>>>")

