# WKB approximation for NHH
# Ana Fabela Hinojosa, 31/03/2022
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import gamma
# from tabulate import tabulate


def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integratequad() for a complex function
    def real_func(*args):
        return np.real(func(*args))

    def imag_func(*args):
        return np.imag(func(*args))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]


def complex_fsolve(func, E0, **kwargs):
    # root finding algorithm. FINDS: Energy values from error() function
    # call: complex_fsolve(error, E0, args=(ϵ, n))
    def real_func(*args):
        return np.real(func(*args))

    real_root = fsolve(real_func, E0, **kwargs)
    # Check that the imaginary part of func is also zero
    value = func(real_root[0], *kwargs['args'])
    assert abs(np.imag(value)) < 1e-10, "Imaginary part wasn't zero"
    # print(f"E = {Energies[0]:.04f}")
    return real_root[0]


def integrand(x_prime, tp_minus, E, ϵ):
    # Change of variables integrand
    x = x_prime + 1j * np.imag(tp_minus)
    return np.sqrt(E - x ** 2 * (1j * x) ** ϵ)


def LHS(n):
    # Quantization condition
    return (n + 1 / 2) * np.pi


def RHS(E, ϵ):
    # Integral defining E
    # integration LIMITS
    tp_minus = E ** (1 / (ϵ + 2)) * np.exp(1j * np.pi * (3 / 2 - (1 / (ϵ + 2))))
    tp_plus = E ** (1 / (ϵ + 2)) * np.exp(-1j * np.pi * (1 / 2 - (1 / (ϵ + 2))))
    tp_minus_prime = np.real(
        E ** (1 / (ϵ + 2)) * np.exp(1j * np.pi * (3 / 2 - (1 / (ϵ + 2))))
        - 1j * np.imag(tp_minus)
    )
    tp_plus_prime = np.real(
        E ** (1 / (ϵ + 2)) * np.exp(-1j * np.pi * (1 / 2 - (1 / (ϵ + 2))))
        - 1j * np.imag(tp_minus)
    )
    # print(tp_minus_prime)
    # print(tp_plus_prime)
    return complex_quad(integrand, tp_minus_prime, tp_plus_prime, args=(tp_minus, E, ϵ))


def error(E, ϵ, n):
    return RHS(E, ϵ) - LHS(n)


def analytic_E(ϵ, n):
    # Bender equation (34) pg. 960
    top = gamma(3 / 2 + 1 / (ϵ + 2)) * np.sqrt(np.pi) * (n + 1 / 2)
    bottom = np.sin(np.pi / (ϵ + 2)) * gamma(1 + 1 / (ϵ + 2))
    return (top / bottom) ** ((2 * ϵ + 4) / (ϵ + 4))


# Schrödinger equation
def Schrodinger_eqn(x, Ψ):
    psi, psi_prime = Ψ
    psi_primeprime = (x ** 2 * (1j * x) ** ϵ - E) * psi
    Ψ_prime = np.array([psi_prime, psi_primeprime])
    return Ψ_prime


#######################################function calls############################################

# WKB approximation
# IC based on RK results given (ϵ, n) = (1, 0)
E0 = 1.1563
E = E0

N = 100
# ϵ = 2
ϵ = 0

# change of variables
tp_minus = E ** (1 / (ϵ + 2)) * np.exp(1j * np.pi * (3 / 2 - (1 / (ϵ + 2))))
tp_plus = E ** (1 / (ϵ + 2)) * np.exp(-1j * np.pi * (1 / 2 - (1 / (ϵ + 2))))

tp_minus_prime = E ** (1 / (ϵ + 2)) * np.exp(
    1j * np.pi * (3 / 2 - (1 / (ϵ + 2)))
) - 1j * np.imag(tp_minus)
tp_plus_prime = E ** (1 / (ϵ + 2)) * np.exp(
    -1j * np.pi * (1 / 2 - (1 / (ϵ + 2)))
) - 1j * np.imag(tp_minus)

######################## eigenvalues ###################################

Energies = []
for n in range(N):
    E_s = []
    E = complex_fsolve(error, E0, args=(ϵ, n))
    E_s.append(E)
    Energies.append(E_s)
np.save(f"Energies_HO_WKB_{N=}.npy", Energies)
# Energies = np.load("Energies_WKB.npy")

# print(tabulate(Energies, headers=['WKB Eigenvalues']))
