import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


# These are the (first 5) eigenvalues we found for epsilon = 2
E_Bender = [
    1.47714975357798685525,
    6.00338608330813595709,
    11.80243359513385665010,
    18.45881870407351787143,
    25.79179237850866277491,
]

# Dictionary to hold the file data
file_data = {}

for i in range(5):
    left_file = f"psi_left_{i}.csv"
    right_file = f"psi_right_{i}.csv"

    r_left = []
    wf_left = []
    with open(left_file, mode='r') as f:
        # open and read each file
        reader = csv.reader(f)
        # Skip the header
        next(reader)

        # Read the rows and negate the 'r' values
        for row in list(reader):
            r_left.append(-float(row[0]))  # Negating the r values for left files
            real = float(row[1])
            imag = float(row[2])
            wf_left.append(complex(real, imag))

    r_right = []
    wf_right = []
    with open(right_file, mode='r') as f:
        reader = csv.reader(f)
        # Skip the header
        next(reader)

        # Read the right file rows in reverse
        for row in reversed(list(reader)):
            r_right.append(float(row[0]))
            real = float(row[1])
            imag = float(row[2])
            wf_right.append(complex(real, imag))

    # Concatenating left and right data
    r = r_left + r_right
    wf = wf_left + wf_right

    file_data[f"n{i}"] = {"r": np.array(r), "wf": np.array(wf)}


# Plotting
for i, (name, data) in enumerate(file_data.items()):
    r = data["r"]
    wf = data["wf"]

    ax = plt.gca()
    color = next(ax._get_lines.prop_cycler)['color']

    # plt.plot(r, [np.real(val) + E_Bender[i] for val in wf], linewidth=1, label=Rf"$\psi_{i}$", color=color)
    # plt.plot(r, [np.imag(val) + E_Bender[i] for val in wf], "--", linewidth=1, color=color)

    # # probability density
    plt.plot(
        r,
        [abs(val ** 2) + E_Bender[i] for val in wf],
        linewidth=1,
        label=Rf"$\psi_{i}$",
        color=color,
    )

textstr = '\n'.join(
    (
        fr'$E_4 = {E_Bender[4]:.06f}$',
        fr'$E_3 = {E_Bender[3]:.06f}$',
        fr'$E_2 = {E_Bender[2]:.06f}$',
        fr'$E_1 = {E_Bender[1]:.06f}$',
        fr'$E_0 = {E_Bender[0]:.06f}$',
    )
)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top')

plt.legend()
plt.xlabel(R'$r$')
# plt.ylabel('Amplitude')
plt.ylabel('Absolute squared amplitude')
plt.title("First few eigenstates")
plt.show()
