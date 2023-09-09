import os
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


def V(x):
    return 0.5 * x ** 2

def even_extension(y):
    return np.concatenate([y, y[::-1][:-1]])


def odd_extension(y):
    return np.concatenate([-y, y[::-1][:-1]])


def initialisation_parameters():
    dx = 9e-3

    # space dimension
    x_max = 8
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
    extension_funcs = [even_extension, odd_extension, even_extension, odd_extension, even_extension, odd_extension]  # Pattern of even and odd

    for i in range(6):

        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        with h5py.File(f"wavefunction_{i}.h5", "r") as file:
            # Get the eigenvalue data and convert it to a float
            evalue_data = list(file.values())[0][()]
            evalue = float(evalue_data)
            eigenvalues.append(evalue)

            numpy_array = np.array(list(file.values())[1])  # Get the wavefunction
            wavefunctions.append(numpy_array)

    # print(f"\nWe got this many wavefunctions:{np.shape(wavefunctions)}")

    x = np.concatenate([-x[::-1][:-1], x])  # extend domain into negative numbers
    # print(x)

    # Using list comprehension to get extended wavefunctions
    extended_wavefunctions = [func(wf) for func, wf in zip(extension_funcs, wavefunctions)]

    # Plotting
    for i, (wf, evalue) in enumerate(zip(extended_wavefunctions, eigenvalues)):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        plt.plot(x, np.real(wf) + evalue, linewidth=1, label=Rf"$\psi_{i}$", color=color)
        plt.plot(x, np.imag(wf) + evalue, "--", linewidth=1, color=color)


    plt.plot(x, V(x), linewidth=2, alpha=0.4, color='k')
    plt.legend()
    plt.xlabel(R'$x$')
    plt.ylabel('Amplitude')
    plt.title("First few eigenstates")
    plt.show()
