import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

# # Define constants
hbar = 1
m = 1/2

# spatial dimension
x_max = 30
dx = 1e-3
Nx = int(x_max / dx)
x = np.linspace(0, x_max, Nx, endpoint=False)

# Load the wavefunction
filename = '0.h5'
with h5py.File(filename, 'r') as f:
    psi = f['eigenfunction'][:]

V = - (x ** 4)

# Calculate the kinetic energy term (second derivative) using finite differences
K = -hbar**2/(2*m) * (np.roll(psi, 1) - 2*psi + np.roll(psi, -1)) / dx**2

# Calculate the potential energy term
potential_energy = V * psi

# Total energy
H_psi = K + potential_energy

# Comparing H_psi with E * psi to find the corresponding energy eigenvalue E.
# If psi is a correct eigenfunction, H_psi should be proportional to psi, with the proportionality constant being the energy eigenvalue.

E0 = 1.47714975357798685525

plt.plot(
    x,
    H_psi,
    label=R"$ H \psi_{0}$",
)

plt.plot(
    x,
    E0 * psi,
    label=R"$E_{0} \psi_{0}$",
)
plt.legend()
plt.xlabel(R'$x$')
plt.ylabel('Amplitude')    
plt.title(R"$H \psi = E \psi$??")

plt.show()

