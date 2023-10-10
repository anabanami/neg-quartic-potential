import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


def V(x):
    return - (x ** 4)

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


    for i in range(1):

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

    # Plotting
    for i, (wf, evalue) in enumerate(zip(extended_wavefunctions, eigenvalues)):
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']

        # plt.plot(
        #     x,
        #     np.real(wf) + evalue,
        #     linewidth=1,
        #     label=Rf"$\psi_{i}$",
        #     color=color,
        # )
        # plt.plot(
        #     x,
        #     np.imag(wf) + evalue,
        #     "--",
        #     linewidth=1,
        #     color=color,
        # )

        plt.plot(
            x,
            (2 * abs(wf **2)),# + evalue,
            "-",
            linewidth=1,
            label=fR'$E_{{{i}}}$',
            color=color,
        )

    textstr = '\n'.join(
        (
            fr'$E_0 = {np.real(eigenvalues[0]):.06f}$',
            # fr'$E_1 = {np.real(eigenvalues[1]):.06f}$',
            # fr'$E_2 = {np.real(eigenvalues[2]):.06f}$',
            # fr'$E_3 = {np.real(eigenvalues[3]):.06f}$',
            # fr'$E_4 = {np.real(eigenvalues[4]):.06f}$',
            # fr'$E_5 = {np.real(eigenvalues[5]):.06f}$',
            # fr'$E_6 = {np.real(eigenvalues[6]):.06f}$',
            # fr'$E_7 = {np.real(eigenvalues[7]):.06f}$',
            # fr'$E_8 = {np.real(eigenvalues[8]):.06f}$',
            # fr'$E_9 = {np.real(eigenvalues[9]):.06f}$',
            # fr'$E_{{10}} = {np.real(eigenvalues[10]):.06f}$',
        )
    )
    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

    # plt.plot(x, V(x), linewidth=2, alpha=0.4, color='k')

    # # # INVESTIGATING THE DECAY
    # # # PLLOTTING AN ENVELOPE
    # envelope = 1 / x**2
    # plt.plot(x, envelope, alpha=0.4, color='blue', linestyle='--', label=R'$\frac{1}{x^2}$')
    
    # plt.legend()
    plt.xlabel(R'$x$')
    # plt.ylabel('Amplitude')    
    plt.ylabel('Probability density')
    plt.title("Ground state of negative quartic Hamiltonian")
    # plt.title("First 10 eigenstates in the real line")
    plt.grid(color="gray", linestyle=":")
    plt.show()

    # JMH_STATES_real_line_zoom.pdf