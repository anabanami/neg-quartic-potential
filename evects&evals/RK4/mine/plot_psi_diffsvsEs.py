import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

Es = []
psi_diffs = []

with open(f"psi_diffs.csv", mode='r') as f:
    # open and read file
    reader = csv.reader(f)
    # Skip the header
    next(reader)

    # Read the rows & save the values to lists
    for row in list(reader):
        Es.append(float(row[0]))
        real = float(row[1])
        imag = float(row[2])
        psi_diffs.append(complex(real, imag))


Es = np.array(Es)
psi_diffs = np.array(psi_diffs)

# plt.plot(Es, abs(psi_diffs)**2)
# plt.xlabel(R'$E$')
# plt.ylabel(R'|\psi_{\mathrm{diff}}^{2}|')
# # plt.title("")
# plt.show()

# ax = plt.gca()
# color = next(ax._get_lines.prop_cycler)['color']
# plt.plot(Es,np.real(psi_diffs), label=R"$\psi_{\mathrm{diff}}$", color=color)
# plt.plot(Es,np.imag(psi_diffs), linestyle= '--', color=color)
# plt.legend()
# plt.xlabel(R'$E$')
# plt.ylabel(R'$\psi_{\mathrm{diff}}$')
# plt.title(R"Real and imaginary parts of $\psi_{\mathrm{diff}}$")
# plt.show()

plt.semilogy(Es,abs(psi_diffs**2))
plt.xlabel(R'$E$')
plt.ylabel(R'$ln(|\psi_{\mathrm{diff}}|^2)$')
plt.title(R"LOG SCALE $|\psi_{\mathrm{diff}}|^2$")
plt.grid()
plt.show()