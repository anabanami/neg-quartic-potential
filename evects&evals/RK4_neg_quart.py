# RK4 to solve negative quartic Eigenvalue problem with shooting method
# Ana Fabela 15/08/2023

"""
This code provides a way to numerically solve the Time-Independent Schrödinger Equation (TISE) for a specific potential, using the shooting method and the Runge-Kutta 4 (RK4) algorithm. The goal is to find the energy eigenvalues of a quantum system for a given potential. Let me break down the main aspects of the code:

    Imports and Settings:
        Essential Python libraries and modules like numpy, matplotlib, scipy.fft, and scipy.signal are imported.
        Default settings for plotting (with matplotlib) and printing (with numpy) are set.

    Potential Function (V):
        V(x)=0.5x4V(x)=0.5x4: Defines the quartic potential function in terms of a position variable xx.

    Schrödinger's Equation:
        This function returns the spatial derivatives of the wavefunction (ΨΨ) and its first spatial derivative (ΦΦ) using the Schrödinger equation. The potential VV and the energy EE are parameters of this equation.

    Runge-Kutta 4 (RK4) Method:
        Schrödinger_RK4 is a numerical integration method to solve ordinary differential equations (ODEs). Here, it's used to solve Schrödinger's equation for the given potential.

    Shooting Method:
        The Solve function uses the shooting method. Given two initial energy guesses (E1 and E2), the function integrates Schrödinger's equation from some boundary towards another boundary and checks if the solution matches the desired boundary condition at the other end. The energies E1 and E2 are then updated using the secant method until the solution converges or a maximum number of iterations is reached.

    Finding Eigenvalues:
        find_eigenvalues divides the energy range into intervals. For each interval, the Solve function is called to find an energy eigenvalue. Duplicate eigenvalues (from neighboring intervals) are filtered out.

    Global Parameters:
        globals function returns the global parameters like conv_crit (convergence criteria), m (mass), hbar (Planck's reduced constant), and space discretization parameters.

    Main Execution (__main__):
        This section initializes all parameters and calls the find_eigenvalues function.
        The found eigenvalues are printed out and compared with a set of known values (E_bender_RK4 and E_bender_wkb).
        Various parameters and results are printed to the console for inspection.

Key Points:
    The code is built to solve the TISE for a quartic potential. This potential is non-analytic, meaning it doesn't have a known exact solution, so numerical methods like the shooting method are appropriate.
    The Runge-Kutta 4 (RK4) method is chosen as the numerical integrator because of its accuracy.
    The shooting method, combined with the secant method, is applied to adjust the energy guesses iteratively until a solution meeting the desired boundary condition is found.
    The eigenvalues found represent the allowed energy levels of the quantum system under the defined quartic potential.

In summary, this code serves as a tool for finding the allowed energy levels of a quantum system governed by a quartic potential, providing an essential part of understanding the behavior of quantum systems in such potentials.
"""


    ONLY Checking solution in the form: 2 Real(Ψ) =  2 B cos(y) 
    y = x_max ** 3 / (3 * np.sqrt(2)) # NEGATIVE QUARTIC POTENTIAL 
    Ψ1_init, Φ1_init = (np.cos(y), - x_max**2 * np.sin(y) * np.sqrt(2))

    # Bender energies to compare to
    E_bender_RK4 = np.array([1.477150, 6.003386, 11.802434, 18.458819, 25.791792])
    # more Bender energies to compare to
    E_bender_wkb = np.array([1.3765, 5.9558, 11.7690, 18.4321, 25.7692])
