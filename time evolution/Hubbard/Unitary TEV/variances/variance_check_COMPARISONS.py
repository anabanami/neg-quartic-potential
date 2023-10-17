import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300

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

def plot_data(file_name, label, linestyle):
    with open(file_name, 'rb') as f:
        data = np.load(f)

    time = np.linspace(t_initial, t_final, len(data))
    plt.plot(time, data, label=label, linestyle=linestyle)

if __name__ == "__main__":

    # Retrieve global parameters
    (
        dt,
        t_initial,
        t_final,
    ) = globals()

    # File names list
    files = [
        # ("free_45.npy", R"$V(x) = 0$, $x_{\mathrm{max}} = 45$"),
        # ("free_65.npy", R"$V(x) = 0$, $x_{\mathrm{max}} = 65$"),
        # ("free_45_shiftedIC.npy", R"$V(x) = 0$, $x_{\mathrm{max}} = 45$"),

        # ("HO_45.npy", R"$V(x) = x^2$, $x_{\mathrm{max}} = 45$"),
        # ("HO_65.npy", R"$V(x) = x^2$, $x_{\mathrm{max}} = 65$"),
        # ("HO_45_shiftedIC.npy", R"$V(x) = x^2$, $x_{\mathrm{max}} = 45$"),

        # ("neg_HO_45.npy", R"$V(x) = -x^2$, $x_{\mathrm{max}} = 45$"),
        # ("neg_HO_65.npy", R"$V(x) = -x^2$, $x_{\mathrm{max}} = 65$"),
        # ("neg_HO_45_shiftedIC.npy", R"$V(x) = -x^2$, $x_{\mathrm{max}} = 45$"),

        # ("neg_quart_45.npy", R"$V(x) = -x^4$, $x_{\mathrm{max}} = 45$"),
        # ("neg_quart_65.npy", R"$V(x) = -x^4$, $x_{\mathrm{max}} = 65$"),
        # ("neg_quart_45_shiftedIC.npy", R"$V(x) = -x^4$, $x_{\mathrm{max}} = 45$"),

        # ("neg_oct_45.npy", R"$V(x) = -x^8$, $x_{\mathrm{max}} = 45$"),
        # ("neg_oct_65.npy", R"$V(x) = -x^8$, $x_{\mathrm{max}} = 65$"),
        # ("neg_oct_45_shiftedIC.npy", R"$V(x) = -x^8$, $x_{\mathrm{max}} = 45$"),
    ]

    for index, (file_name, label) in enumerate(files):
        linestyle = "--" if index % 2 != 0 else "-"  # Dashed for every second file, starting from the second file (1-indexed)
        plot_data(file_name, label, linestyle)

    plt.title("Spatial dispersion for various potentials")
    plt.ylabel(R"$\sigma_{(x)}^2$")
    plt.xlabel("t")
    plt.grid()
    plt.legend()
    plt.show()
