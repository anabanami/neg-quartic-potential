# Ana Fabela 14/10/2023

# Import necessary libraries and modules
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100

def pad_array(smaller_array, larger_size):
    diff = larger_size - len(smaller_array)
    pad_size = diff // 2
    padded_array = np.pad(smaller_array, (pad_size,), 'edge')
    return padded_array


if __name__ == "__main__":

    # space dimension
    dx = 0.08

    x_max1 = 25
    x_max = 45
    Nx = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, Nx, endpoint=False)

    # Pairs of filenames to compare
    # file1 = h5py.File('Unitary_hubbard_negHO_25.hdf5','r')
    # file2 = h5py.File('Unitary_hubbard_negHO_45.hdf5', 'r')

    # file1 = h5py.File('Unitary_hubbard_negquart_25.hdf5', 'r')
    # file2 = h5py.File('Unitary_hubbard_negquart_45.hdf5', 'r')

    file1 = h5py.File('Unitary_hubbard_negoct_25.hdf5', 'r')
    file2 = h5py.File('Unitary_hubbard_negoct_45.hdf5', 'r')

    
    times = []
    for key1, key2 in zip(file1.keys(), file2.keys()):
        # print(f"{key1 = }, {key2 = }")
        times.append(float(key1))
    
    print(f"\n{len(times) = }\n")  

    t0 = times[0]
    t1 = times[11781]
    t2 = times[23562]
    t3 = times[35343]
    t4 = times[47123]

    print(f"\n{t0 = }\n")
    print(f"\n{t1 = }\n")
    print(f"\n{t2 = }\n")
    print(f"\n{t3 = }\n")
    print(f"\n{t4 = }\n")


    # make contents of a single time step an array
    state1_0 = np.array(file1[f'{t0}'])
    state2_0 = np.array(file2[f'{t0}'])
    state1_0 = pad_array(state1_0, len(state2_0))
    
    state1_1 = np.array(file1[f'{t1}'])
    state2_1 = np.array(file2[f'{t1}'])
    state1_1 = pad_array(state1_1, len(state2_1))

    state1_2 = np.array(file1[f'{t2}'])
    state2_2 = np.array(file2[f'{t2}'])
    state1_2 = pad_array(state1_2, len(state2_2))
    
    state1_3 = np.array(file1[f'{t3}'])
    state2_3 = np.array(file2[f'{t3}'])
    state1_3 = pad_array(state1_3, len(state2_3))

    state1_4 = np.array(file1[f'{t4}'])
    state2_4 = np.array(file2[f'{t4}'])
    state1_4 = pad_array(state1_4, len(state2_4))

    # LISTS
    times_list = [t0, t1, t2, t3, t4]

    state1_list = [state1_0, state1_1, state1_2, state1_3, state1_4]
    state2_list = [state2_0, state2_1, state2_2, state2_3, state2_4]


    # Creating subplots
    fig, axes = plt.subplots(5, 1, figsize=(6, 15))
    axes = axes.flatten()

    fig.suptitle(f"Simulation box size comparison")
    for i in range(5):
        # Plots
        axes[i].plot(
            x,
            abs(state1_list[i]) ** 2,
            label=fR'$x_{{\mathrm{{max}}}}$ = {x_max1}',
        )
        axes[i].plot(
            x,
            abs(state2_list[i]) ** 2,
            linestyle='--',
            label=fR'$x_{{\mathrm{{max}}}}$ = {x_max}',
        )

        if i == 4:
            axes[i].set_xlabel("x")
        
        # Text box
        textstr = f"t = {times_list[i]:.5f}"
        axes[i].text(
            0.02,
            0.98,
            textstr,
            transform=axes[i].transAxes,
            verticalalignment='top',
        )

        # Labels and title
        axes[i].legend()
        axes[i].set_ylabel('Probability density')

    # To avoid overlapping of plots and labels
    plt.tight_layout()
    plt.savefig("probability_density_for_diff_times_and_x_maxs.pdf", dpi=300)

    plt.show()


