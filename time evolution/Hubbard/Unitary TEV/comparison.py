# Comparing HDF5 files
# Ana Fabela 11/07/2023

# Import necessary libraries and modules
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


def globals():
    """
    Function to define and return global variables used throughout the script.
    Includes physical constants, spatial and temporal discretizations.
    Returns:
    - A tuple containing all global parameters.
    """

    hbar = 1

    # Bender units
    m = 1 / 2
    omega = 2

    # space dimension
    x_max = 45
    dx = 0.08
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # time dimension
    dt = m * dx ** 2 / (np.pi * hbar) * (1 / 8)
    t_initial = 0
    t_final = 5.7

    return (
        hbar,
        m,
        omega,
        x_max,
        dx,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
    )


if __name__ == "__main__":

    # Retrieve global parameters

    (
        hbar,
        m,
        omega,
        x_max,
        dx,
        Nx,
        x,
        dt,
        t_initial,
        t_final,
    ) = globals()

    file1 = h5py.File('Unitary_hubbard.hdf5', 'r')

    # Print out the names of all items in the root of the HDF5 file
    times = []
    for key in file1.keys():
        print(key)
        times.append(float(key))
    times = np.array(times)
    print(f'{np.shape(times) = }')

    t0 = times[0]
    t1 = times[1000]
    t2 = times[2000]
    t3 = times[3000]
    t4 = times[4000]
    t5 = times[5000]

    # Print out the contents of a single timestep
    state1_0 = np.array(file1[f'{t0}'])

    state1_1 = np.array(file1[f'{t1}'])

    state1_2 = np.array(file1[f'{t2}'])

    state1_3 = np.array(file1[f'{t3}'])

    state1_4 = np.array(file1[f'{t4}'])

    state1_5 = np.array(file1[f'{t5}'])

    state1_list = [state1_0, state1_1, state1_2, state1_3, state1_4, state1_5]

    for i in range(6):
        plt.plot(
            x,
            abs(state1_list[i]) ** 2,
            # color=color1,
            label=f'U(t)' if i == 0 else None,
        )

    plt.ylabel(R"$|\psi(x, t)|^2$")
    plt.xlabel("x")
    plt.legend()
    # plt.title(f"")
    plt.show()

    file1.close()


