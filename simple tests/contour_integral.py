# Ana Fabela Hinojosa, 23/05/2022
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.special as sc
from scipy.integrate import quad

y_max = 2 * np.pi
y = np.linspace(0, y_max, 1024, endpoint=False)
y_step = y[1] - y[0]


def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integrate.quad() for a complex function
    def real_func(*args):
        return np.real(func(*args))

    def imag_func(*args):
        return np.imag(func(*args))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]

def integrand(y):
    z = np.exp(1j * y)
    dz_dy = 1j * np.exp(1j * y)
    f = 1 / z
    # print(f"{f * dz_dy = }")
    return f * dz_dy

residue = complex_quad(integrand, 0, y_max, epsabs=1.49e-08, limit=3000)

print(f"calc residue = {1j * np.imag(residue)}")

print(f"expe residue = {2j * np.pi}")