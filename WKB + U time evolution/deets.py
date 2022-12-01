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

y = np.linspace(-1, 1, Nx).T

E = 15
ϵ = 0

δminus = 0.3
δplus = 0.39

wkb = np.zeros(Nx, dtype=complex)
x0 = np.sqrt(E)
a = -x0
b = x0
F0 = -(2 * a)
F1 = 2 * b
       
Q = np.sqrt((V(x, ϵ) - E).astype(complex))
P = np.sqrt((E - V(x, ϵ)).astype(complex))

u1 = F0**(1/3) * (a - x[(a - δminus < x) & (x < a + δplus)])

# print(f"\n{np.shape(wkb)=}")
# LHS of potential barrier
integral_left = np.cumsum(Q[x < a - δminus]) * delta_x
integral_left = -(integral_left - integral_left[-1])
wkb[x < a - δminus] = np.exp(-integral_left) / (3 * np.sqrt(Q[x < a - δminus]))
# print(f"\n{np.shape(wkb)=}")

# approaching left turning point (from both sides of a)
Ai_a, Aip_a, Bi_a, Bip_a = special.airy(u1)
wkb[(a - δminus < x) & (x < a + δplus)] = Ai_a * np.sqrt(np.pi) / F0 ** (1/6)
# print(f"\n{np.shape(wkb)=}")

# inside potential barrier 
# since ψ1 = c ψ2
# get one array for ψ1 and one for ψ2 and divide them (make sure you avoid zero division!)
# print("===================== ")
# print(f"NEW INDEXING")
integral_aplusδ_x = np.cumsum(P[x > a + δplus]) * delta_x
print(f"{np.shape(integral_aplusδ_x)=}")

# ψ1 = np.cos(integral_a_x - np.pi / 4) / np.sqrt(P[x > a])
# ψ2 = np.cos(-integral_a_x + np.pi / 4) / np.sqrt(P[x > a])

# c = ψ1 / ψ2
# # print(f"{np.shape(ψ1)=}")
# # print(f"{np.shape(ψ2)=}")
# # print(f"{np.shape(c)=}")

wkb[x > a + δplus] = (np.cos(integral_aplusδ_x) + np.sin(integral_aplusδ_x)) / np.sqrt(2)   # indexing is correct... now my maths are dumb?
# print(f"\n{np.shape(wkb)=}")
# print("===================== ")

plt.plot(x, wkb)
plt.axvline(a, linestyle="--", color="red")
plt.fill_betweenx(y, a - δminus, a + δplus , alpha=0.2, color="pink")
plt.xlim(-7.5, 0)
plt.ylim(-0.2, 1)

plt.show()

