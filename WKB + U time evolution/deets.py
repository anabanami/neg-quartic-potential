import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import special

def V(x, ϵ):
    if ϵ == 0:
        return x ** 2

Nx = 1024
x = np.linspace(-10, 10, Nx)
x[x==0] = 1e-200
delta_x = x[1] - x[0]

E = 15 # doesn't work for all Es
ϵ = 0



wkb = np.zeros(Nx, dtype=complex)
x0 = np.sqrt(E)
a = -x0
b = x0
F0 = -(2 * a)
F1 = 2 * b

δminus = 0.3
δplus = 0.8
       
Q = np.sqrt((V(x, ϵ) - E).astype(complex))
P = np.sqrt((E - V(x, ϵ)).astype(complex))

u1 = F0**(1/3) * (a - x[(a - δminus < x) & (x < a + δplus)])

# LHS of potential barrier
integral_left = np.cumsum(Q[x < a - δminus]) * delta_x
integral_left = -(integral_left - integral_left[-1])
wkb[x < a - δminus] = np.exp(-integral_left) / (3 * np.sqrt(Q[x < a - δminus]))

# approaching left turning point (from both sides of a)
Ai_a, Aip_a, Bi_a, Bip_a = special.airy(u1)
wkb[(a - δminus < x) & (x < a + δplus)] = Ai_a * np.sqrt(np.pi) / F0 ** (1/6)

# inside potential barrier 
np.cumsum(P[x > a]) * delta_x
integral_a_x = excessively_long_array[x[x > a] > a + δplus]

wkb[x > a + δplus] = np.cos(integral_a_x - np.pi/4) / np.sqrt(P[x > a + δplus]) 

# wkb[x > a + δplus] = (np.cos(integral_a_x) + np.sin(integral_a_x)) / 2  # indexing is correct... now my maths are dumb?


plt.plot(x, wkb)
plt.axvline(a, linestyle="--", color="red")
plt.fill_betweenx(y, a - δminus, a + δplus , alpha=0.2, color="pink")
plt.xlim(-7.5, 0)
plt.ylim(-0.2, 1)

plt.show()

