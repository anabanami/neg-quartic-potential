import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

def Bose_Hubbard_Hamiltonian(n_sites, t, U, ω, α):
    # Initialize the Hamiltonian as a zero matrix
    H = np.zeros((n_sites, n_sites))

    # Define the hopping and interaction terms
    for i in range(n_sites-1):
        # Hopping terms
        H[i, i+1] = -t
        H[i+1, i] = -t

        # On-site interaction term with harmonic oscillator potential and quartic potential
        H[i, i] = U * (i**2) - α * ((i - n_sites//2)**4) + 0.5 * ω **2 * (i - n_sites//2)**2

    # open BCS
    H[-1, -1] = U * ((n_sites-1)**2) - α * ((n_sites-1 - n_sites//2)**4) + 0.5 * ω **2 * ((n_sites-1) - n_sites//2)**2
    return H

# natural units according to Wikipedia
hbar = 1
m = 1
ω = 1
α = 0.01  # coefficient for quartic potential

n_sites = 50
t = 1
U = 1

# Generate the Hamiltonian
H = Bose_Hubbard_Hamiltonian(n_sites, t, U, ω, α)

# Plot the Hamiltonian as a heat map
plt.imshow(H, cmap='hot', interpolation='nearest')
plt.colorbar(label='Matrix element value')
plt.title('Bose-Hubbard Hamiltonian')
plt.xlabel('Site index')
plt.ylabel('Site index')
plt.show()

# Calculate absolute values and add a small constant to avoid log(0)
H_abs = np.abs(H) + 1e-9

# Plot the absolute value of the Hamiltonian as a heat map on a logarithmic scale
plt.imshow(np.log(H_abs), cmap='hot', interpolation='nearest')
plt.colorbar(label='Log of absolute matrix element value')
plt.title('Absolute value of Bose-Hubbard Hamiltonian (log scale)')
plt.xlabel('Site index')
plt.ylabel('Site index')
plt.show()
