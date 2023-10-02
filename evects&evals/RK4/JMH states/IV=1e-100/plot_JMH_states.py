import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200


# These are the (first 11) eigenvalues we found for epsilon = 2
E_Bender = [
    1.47714975357798686273,
    6.00338608330813547180,
    11.80243359513386549372,
    18.45881870407350603368,
    25.79179237850933880880,
    33.69427987657709945568,
    42.09380771103636153727,
    50.93740433237080572626,
    60.18436924385155633796,
    69.80209265698014350909,
    79.76551330248462000350
]

# Dictionary to hold the file data
file_data = {}

for i in range(11):
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

    plt.plot(
        r,
        abs(wf ** 2),
        linewidth=1,
        label=Rf"$\psi_{i}$",
        color=color,
    )


    # plt.semilogy(
    #     r,
    #     abs(wf) ** 2,
    #     linewidth=1,
    #     label=Rf"$\psi_{i}$",
    #     color=color,
    # )

plt.legend()
plt.xlabel(R'$r$')
# plt.ylabel('Absolute squared amplitude')
plt.ylabel('LOG Absolute squared amplitude')
plt.title("First few eigenstates")
plt.show()
