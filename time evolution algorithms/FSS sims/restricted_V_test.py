import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
from scipy.fft import fft, ifft 
from scipy.signal import convolve 
  

def gaussian_smoothing(data, pts): 
    """gaussian smooth an array by given number of points""" 
    x = np.arange(-4 * pts, 4 * pts + 1, 1) 
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2)) 
    smoothed = convolve(data, kernel, mode='same') 
    normalisation = convolve(np.ones_like(data), kernel, mode='same') 
    return smoothed / normalisation 

  
def smooth_restricted_V(x):
    V = α * np.ones_like(x) * x[cut] ** 4
    V[cut : Nx - cut] = α * x[cut : Nx - cut] ** 4
    ## smoooth by pts=3
    V = gaussian_smoothing(V, 3) # ??? make sure pts make sense
    return V


# natural units
hbar = 1
m = 1
ω = 1
# lengths for HO quench
l1 = np.sqrt(hbar / (m * ω))

# coefficient for quartic potential
α = 4


# N_sites = 1024
x_max = 5.12
dx = 0.01
Nx = int(2 * x_max / dx)
# print(f"{Nx=}")

cut = 255

x = np.linspace(-x_max, x_max, Nx, endpoint=False)

y = smooth_restricted_V(x)

plt.plot(x, y)
plt.show()
