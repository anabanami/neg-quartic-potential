import numpy as np
import matplotlib.pyplot as plt


def Hubbard_Hamiltonian_Harmonic(n_sites, t, U, ω):
    # Initialize the Hamiltonian as a zero matrix
    H = np.zeros((n_sites, n_sites))

    # Define the hopping and interaction terms
    # PERIODIC BCS
    for i in range(n_sites):
        # Hopping term
        H[i, (i+1)%n_sites] = -t
        H[(i+1)%n_sites, i] = -t

        # On-site interaction term with harmonic oscillator potential
        H[i, i] = U + 0.5 * ω **2 * (i - n_sites//2)**2  # assuming the potential is centred in the middle of the lattice

    return H


# natural units according to Wikipedia
hbar = 1
m = 1
ω = 1
# lengths for HO quench
l1 = np.sqrt(hbar / (m * ω))
l2 = 2 * l1


n_sites = 50
t = 1
U = 10


# Generate the Hamiltonian
H = Hubbard_Hamiltonian_Harmonic(n_sites, t, U, ω)

# Plot the Hamiltonian as a heat map
plt.imshow(H, cmap='hot', interpolation='nearest')
plt.colorbar(label='Matrix element value')
plt.title('Hubbard Hamiltonian')
plt.xlabel('Site index')
plt.ylabel('Site index')
plt.show()
5

# Calculate absolute values and add a small constant to avoid log(0)
H_abs = np.abs(H) + 1e-9

# Plot the absolute value of the Hamiltonian as a heat map on a logarithmic scale
plt.imshow(np.log(H_abs), cmap='hot', interpolation='nearest')
plt.colorbar(label='Log of absolute matrix element value')
plt.title('Absolute value of Hubbard Hamiltonian (log scale)')
plt.xlabel('Site index')
plt.ylabel('Site index')
plt.show()
