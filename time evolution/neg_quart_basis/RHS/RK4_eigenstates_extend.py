import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


def normalize_wavefunction(wavefunction, dx):
    """
    Normalize the given wavefunction.

    Args:
    - wavefunction: The wavefunction array.
    - dx: Differential increment in x (spatial step).

    Returns:
    - The normalized wavefunction.
    """
    integral = np.sum(np.abs(wavefunction) ** 2) * dx
    normalization_constant = np.sqrt(2 / integral)
    return wavefunction * normalization_constant


def save_to_hdf5(filename, eigenvalue, eigenfunction):
    """
    Save eigenvalue and eigenfunction to an HDF5 file.

    Args:
    - filename: Name of the file to save to.
    - eigenvalue: The eigenvalue to save.
    - eigenfunction: The eigenfunction to save.
    """
    with h5py.File(filename, 'w') as hf:
        dataset = hf.create_dataset("eigenfunction", data=eigenfunction)
        dataset.attrs["eigenvalue"] = eigenvalue


def even_extension(y):
    return np.concatenate([y[::-1][:-1], y])


def odd_extension(y):
    return np.concatenate([-y[::-1][:-1] ,y])


def initialisation_parameters():
    dx = 1e-5

    # space dimension
    x_max = 30
    Nx = int(x_max / dx)
    x = np.linspace(0, x_max, Nx, endpoint=False)

    return (
        dx,
        x_max,
        Nx,
        x,
    )


if __name__ == "__main__":

    dx, x_max, Nx, x = initialisation_parameters()

    eigenvalues = []
    wavefunctions = []
    extension_funcs = [even_extension, odd_extension, even_extension, odd_extension, even_extension, odd_extension, even_extension, odd_extension, even_extension, odd_extension, even_extension]

    for i in range(11):

        with h5py.File(f"{i}.h5", "r") as file:
            # Get the eigenvalue data and convert it to a float
            evalue = file["eigenfunction"].attrs["eigenvalue"]
            eigenvalues.append(evalue)

            numpy_array = file["eigenfunction"][:]  # Get the wavefunction
            wavefunctions.append(numpy_array)

    # print(f"\nWe got this many wavefunctions:{np.shape(wavefunctions)}")

    x = np.concatenate([-x[::-1][:-1], x])  # extend domain into negative numbers
    # print(x)

    # Using list comprehension to get extended wavefunctions
    extended_wavefunctions = [func(wf) for func, wf in zip(extension_funcs, wavefunctions)]

    for i, ext_wf in enumerate(extended_wavefunctions):
        print(f"*** ~~saving wavefunction for eigenvalue {i}~~ ***")
        normalized_wavefunction = normalize_wavefunction(np.array(ext_wf), dx)
        save_to_hdf5(f"EXTENDED_{i}.h5", eigenvalues[i], normalized_wavefunction)


    # # Plotting
    # for i, (wf, evalue) in enumerate(zip(extended_wavefunctions, eigenvalues)):
    #     ax = plt.gca()

    #     plt.plot(
    #         x,
    #         abs(wf **2) + evalue,
    #         "-",
    #         linewidth=1,
    #         label=fR'$E_{{{i}}}$',
    #         # color=color,
    #     )

    # textstr = '\n'.join((
    #     fR'$E_{0} = {eigenvalues[0]:.06f}$',
    #     fR'$E_{1} = {eigenvalues[1]:.06f}$',
    #     fR'$E_{2} = {eigenvalues[2]:.06f}$',
    #     fR'$E_{3} = {eigenvalues[3]:.06f}$',
    #     fR'$E_{4} = {eigenvalues[4]:.06f}$',
    #     )
    # )
    
    # # place a text box in upper left in axes coords
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')


    # # plt.legend()
    # plt.xlabel(R'$x$')
    # # plt.ylabel('Amplitude')    
    # plt.ylabel('Probability density')
    # plt.title("First 11 eigenstates")
    # plt.show()



