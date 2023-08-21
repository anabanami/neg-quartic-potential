# Time evolution using Hubbard Hamiltonian with FSS2
# HDF5 protocol
# Ana Fabela 03/07/2023

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from numpy.fft import fft, ifft, fftfreq
import h5py

plt.rcParams['figure.dpi'] = 200

#######################################################################################################


def K(x):
    # return fft(-t)
    return -2 * t * np.cos(kx)


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def smooth_restricted_V(x):
    V = np.ones_like(x) * x[cut] ** 4
    V[cut : Nx - cut] = x[cut : Nx - cut] ** 4
    ## smoooth by pts=3
    V = gaussian_smoothing(V, 3)
    return V


def V(x):
    # return - α * smooth_restricted_V(x)
    return np.zeros_like(x)
    # return - 0.5 * (x ** 2)


def plot_evolution_frame(y, state, t, i):
    # potential
    plt.plot(y, V(y), color="black", linewidth=2, label="V(x)")
    # prob. density plot
    plt.plot(y, abs(state) ** 2, label=R"$|\psi(x, t)|^2$")
    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel("x")
    plt.legend()
    # plt.ylim(-1.5, 3)
    # plt.xlim(-5, 5)
    plt.title(f"t = {t:05f}")
    plt.savefig(f"{folder}/{i}.png")
    # plt.show()
    plt.clf()


###############################################################################
def x_variance(x, dx, Ψ):
    # Calculate Spatial variance of wavefunction (Ψ) per unit time
    f = x * abs(Ψ ** 2)
    f_right = f[1:]  # right endpoints
    f_left = f[:-1]  # left endpoints
    expectation_value_x = (dx / 2) * np.sum(f_right + f_left)

    g = (x - expectation_value_x) ** 2 * abs(Ψ ** 2)
    g_right = g[1:]
    g_left = g[:-1]
    return dx / 2 * np.sum(g_right + g_left)


###############################################################################

#######################################################################################################


def FSS_2(Ψ, dt):
    V_real = V(x)
    K_fourier = K(
        kx
    )  # MEAN FIELD??? mod psi sqrd in FS? SHOULD THIS BE A NON LINEAR TERM?
    # First evolve using the potential term for half a timestep:
    Ψ = np.exp(-1j / hbar * V_real * 0.5 * dt) * Ψ
    # Then evolve using the kinetic term for a whole timestep, transforming to
    # and from Fourier space where the kinetic term is diagonal:
    Ψ = ifft(np.exp(-1j / hbar * K_fourier * dt) * fft(Ψ))
    # Then evolve with the potential term again for half a timestep:
    Ψ = np.exp(-1j / hbar * V_real * 0.5 * dt) * Ψ

    return Ψ


def TEV(x, wave):
    # spatial variance
    SIGMAS_x_SQUARED = []

    states = []

    # Create a new HDF5 file
    file = h5py.File('TEST.hdf5', 'w')

    # time evolution
    state = wave
    states.append(state)
    # store variance
    sigma_x_squared = x_variance(x, dx, state)
    SIGMAS_x_SQUARED.append(sigma_x_squared)
    dset = file.create_dataset("0.0", data=state)

    # generate timesteps
    times = np.arange(t_initial, t_final, dt)

    # ALL OTHER ts
    for time in times[1:]:
        print(f"t = {time}")
        state = FSS_2(state, time)
        states.append(state)
        # create a new dataset for each frame
        dset = file.create_dataset(f"{time}", data=state)

        # store variance
        sigma_x_squared = x_variance(x, dx, state)
        SIGMAS_x_SQUARED.append(sigma_x_squared)

    # Close the hdf5 file
    file.close()
    SIGMAS_x_SQUARED = np.array(SIGMAS_x_SQUARED)
    np.save(f"TEST_variance.npy", SIGMAS_x_SQUARED)

    i = 0
    PLOT_INTERVAL = 1
    for j, state in enumerate(states):
        if i % PLOT_INTERVAL == 0:
            print(f"t = {times[j]}")
            plot_evolution_frame(x, state, times[j], i)
        i += 1


def globals():
    # makes folder for simulation frames
    folder = Path(f'TEST')

    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # # Bender units
    # m = 1/2
    # ω = 2

    # natural units
    m = 1
    ω = 1
    hbar = 1

    # lengths for HO quench
    l1 = np.sqrt(hbar / (m * ω))

    # coefficient for quartic potential
    α = 1

    cut = 5

    dx = 0.1
    # hopping strength approximation
    t = 1 / (2 * dx ** 2)

    x_max = 45
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # for Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)

    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 2

    ## initial conditions: HO ground state
    wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))
    # # initial conditions: HO ground state
    # wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-((x - 1) ** 2) / (2 * l1 ** 2))

    return (
        folder,
        hbar,
        m,
        ω,
        l1,
        α,
        t,
        cut,
        dx,
        x_max,
        Nx,
        x,
        kx,
        dt,
        t_initial,
        t_final,
        wave,
    )


if __name__ == "__main__":

    (
        folder,
        hbar,
        m,
        ω,
        l1,
        α,
        t,
        cut,
        dx,
        x_max,
        Nx,
        x,
        kx,
        dt,
        t_initial,
        t_final,
        wave,
    ) = globals()

    TEV(x, wave)

    print(f"\n{np.sum(abs(wave)**2)*dx = }")  # is IC normalised???

    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"{kx.shape = }")
    print(f"x_cut_left = {x[cut]= }")
    print(f"x_cut_right = {x[Nx-cut]= }")

    print(f"\n{dx = }")
    print(f"{dt = }")

    print(f"\n{kx.max() * dx = }")
    print(f"{kx.min() * dx = }")

    """USING A STATE of the unitary hubbard simulation is easy given that its stored as 
    a separate dataset within the HDF5."""

    # file = h5py.File('Hubbard_FSS2.hdf5', 'r')
    # state_t10 = file['timestep_10'][:]  # Load the state at timestep 10 into memory
    # file.close()

    # file = h5py.File('10.?.hdf5', 'r')
    # state_t10 = file['timestep_10'][:]  # Load the state at timestep 10 into memory
    # file.close()

    # AM I DOING MANIPULATIONS IN BOTH FSPACE AND SPACE? Instead of one and then the other?
    # WHEN AM I DOING A MEAN FIELD APPROX IF I AM?
