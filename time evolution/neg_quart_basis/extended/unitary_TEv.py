import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import h5py

plt.rcParams['figure.dpi'] = 200


# def normalize_wavefunction(wavefunction, dx):
#     """
#     Normalize the given wavefunction.

#     Args:
#     - wavefunction: The wavefunction array.
#     - dx: Differential increment in x (spatial step).

#     Returns:
#     - The normalized wavefunction.
#     """
#     integral = np.sum(np.abs(wavefunction) ** 2) * dx
#     normalization_constant = np.sqrt(2 / integral)
#     return wavefunction * normalization_constant


# def save_to_hdf5(filename, eigenvalue, eigenfunction):
#     """
#     Save eigenvalue and eigenfunction to an HDF5 file.

#     Args:
#     - filename: Name of the file to save to.
#     - eigenvalue: The eigenvalue to save.
#     - eigenfunction: The eigenfunction to save.
#     """
#     with h5py.File(filename, 'w') as hf:
#         dataset = hf.create_dataset("eigenfunction", data=eigenfunction)
#         dataset.attrs["eigenvalue"] = eigenvalue


def plot_evolution_frame(y, state, time, i):
    # prob. density plot
    plt.plot(y, abs(state) ** 2, label=R"$|\psi(x, t)|^2$")
    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel(R"$x$")
    plt.legend()
    plt.ylim(-1.5, 3)
    plt.xlim(-5, 5)
    plt.title(f"t = {time:05f}")
    plt.savefig(f"{folder}/{i}.png")
    # plt.show()
    plt.clf()


def K_psi(wf):
    """
    Calculate the second derivative of wavefunction
    to evaluate the knetic energy of H
    """
    psiprime = np.gradient(wf) / dx
    return -np.gradient(psiprime) / dx  # hbar = 1, m = 1/2


def V(x):
    return -(x ** 4)


def E_matrix(eigenvalues):
    """
    Store the negative quartic Hamiltonian Eigenvalues in the diagonal matrix elements
    """
    N = len(eigenvalues)
    # Initialize a zero matrix to store eigenvalues
    E = np.zeros((N, N))
    for i in range(N):
        E[i, i] = eigenvalues[i]
    return E


def Unitary(M):
    return linalg.expm(-1j * M * dt)


def TEV(x, wave, eigenvalues):
    states = []

    # Create a new HDF5 file
    file = h5py.File('TEv.h5', 'w')

    # time evolution
    E_M = E_matrix(eigenvalues)
    U = Unitary(E_M)

    state = wave
    states.append(state)
    # initial time step
    dset = file.create_dataset("0.0", data=state)

    # generate timesteps
    times = np.arange(t_initial, t_final, dt)

    # ALL OTHER time steps
    for time in times[1:]:
        print(f"t = {time}")
        state = U @ state
        states.append(state)
        # create a new dataset for each frame
        dset = file.create_dataset(f"{time}", data=state)

    # Close the hdf5 file
    file.close()

    PLOT_INTERVAL = 20
    for j, state in enumerate(states):
        if j % PLOT_INTERVAL == 0:
            print(f"t = {times[j]}")
            plot_evolution_frame(x, state, times[j], j)
            # plot_vs_k(state, times[j], j)


def initialisation_parameters():
    # makes folder for simulation frames
    folder = Path(f'TEv_frames')
    os.makedirs(folder, exist_ok=True)
    os.system(f'rm {folder}/*.png')

    # Bender units
    m = 1 / 2
    hbar = 1
    ω = 2
    # length for HO quench
    l1 = np.sqrt(hbar / (m * ω))

    dx = 1e-5

    # space dimension
    x_max = 30
    Nx = int(x_max / dx)
    x = np.linspace(-x_max, x_max, 2 * Nx, endpoint=False)

    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 10

    # initial conditions: HO ground state
    wave = np.sqrt(1 / (np.sqrt(np.pi) * l1)) * np.exp(-(x ** 2) / (2 * l1 ** 2))
    NEED TO PROJECT THIS ONTO NEG QUARTIC BASIS

    return (
        folder,
        dx,
        x_max,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
        m,
        hbar,
        ω,
        l1,
        wave,
    )


if __name__ == "__main__":

    (
        folder,
        dx,
        x_max,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
        m,
        hbar,
        ω,
        l1,
        wave,
    ) = initialisation_parameters()

    eigenvalues = []
    wavefunctions = []

    for i in range(11):
        with h5py.File(f"EXTENDED_{i}.h5", "r") as file:
            # Get the eigenvalue data and convert it to a float
            evalue = file["eigenfunction"].attrs["eigenvalue"]
            eigenvalues.append(evalue)

            numpy_array = file["eigenfunction"][:]  # Get the wavefunction
            wavefunctions.append(numpy_array)

    print(f"\nWe got this many wavefunctions:{np.shape(wavefunctions)}\n")

    TEV(x, wave, eigenvalues)

    print(f"\n{np.sum(abs(wave)**2)*dx = }")  # is IC normalised???

    print(f"\n{x_max = }")
    print(f"{Nx = }")
    print(f"{x.shape = }")
    print(f"x_cut_left = {x[cut]= }")
    print(f"x_cut_right = {x[Nx-cut]= }")

    print(f"\n{dx = }")
    print(f"{dt = }")

