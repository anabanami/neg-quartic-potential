// These are the (first 5) eigenvalues for epsilon = 2
// E_Bender = [
//         1.477150,
//         6.003386,
//         11.802434,
//         18.458819,
//         25.791792,
//     ]

// This code includes standard libraries. stdio.h is for input/output functions like printf.
 // math.h provides mathematical functions. complex.h provides support for complex numbers.
#include <stdio.h>
#include <math.h>
#include <complex.h>

// Macros are defined for convenience. ldc is a shorthand for long double complex, pi 
// represents the mathematical constant π, and epsilon is defined as 0.0 
// (though this might change if needed).
#define ldc long double complex
#define pi 3.14159265358979323846
#define epsilon 2.0


// Function prototypes are declared so that they can be used before their definitions.
void rkintegrate(ldc x0, ldc *y, ldc *z, long double h,
                 long double E, long double theta, char *output_filename);
ldc F(long double r, long double theta, ldc psi, ldc psiprime, long double E);
ldc G(long double r, long double theta, ldc psi, ldc psiprime, long double E);
long double shoot(long double E1, long double E2, long double A1, long double A2);


// The main function initializes a series of variables and complex values and runs a loop 
// to integrate certain differential equations using the rkintegrate function. 
// The results and energies are printed out. There's an energy adjustment mechanism 
// that appears to be searching for particular solutions based on the difference in the 
// solutions from two starting energies (E1 and E2).
int main(void) 
{
    // long double E1 = 1, E2 = 1.5, tempE;
    // long double E1 = 5.9, E2 = 6.2, tempE;
    // long double E1 = 11.5, E2 = 11.9, tempE;
    // long double E1 = 18, E2 = 18.5, tempE;
    long double E1 = 25, E2 = 26, tempE;
    long double three = 3.0;
    long double stepsize = 0.0005;
    long double theta_left, theta_right;

    ldc psi1_left, psiprime1_left;
    ldc psi1_right, psiprime1_right;
    ldc psi2_left, psiprime2_left;
    ldc psi2_right, psiprime2_right;
    ldc psi1_diff, psi2_diff;
    ldc x_left, x_right;
    
    int i, loop;

    // Save the initial energy bounds for use in output filenames
    long double E1_init = E1, E2_init = E2;

    x_right = 10.0 * cexpl(I * pi * ((4.0 * 1.0 - epsilon) / (4.0 + 2.0 * epsilon)));
    x_left = 10.0 * cexpl(I * pi * ((4.0 * -2.0 - epsilon) / (4.0 + 2.0 * epsilon)));

    theta_right = cargl(x_right);
    theta_left = cargl(x_left);

    loop = (int) (x_right / stepsize);
    loop++;

    psi1_right = 1.0;
    psiprime1_right = -cpowl(I, epsilon) * cpowl(x_right, (2.0 + epsilon)) * psi1_right;
    psi2_right = 1.0;
    psiprime2_right = -cpowl(I, epsilon) * cpowl(x_right, (2.0 + epsilon)) * psi2_right;
    
    psi1_left = 1.0;
    psiprime1_left = -cpowl(I, epsilon) * cpowl(x_left, (2.0 + epsilon)) * psi1_left;
    psi2_left = 1.0;
    psiprime2_left = -cpowl(I, epsilon) * cpowl(x_left, (2.0 + epsilon)) * psi2_left;

    for (i = 0; i < 40; i++)
    {
        rkintegrate(x_right, &psi1_right, &psiprime1_right, -stepsize, E1, theta_right, NULL);
        rkintegrate(x_right, &psi2_right, &psiprime2_right, -stepsize, E2, theta_right, NULL);
        rkintegrate(x_left, &psi1_left, &psiprime1_left, -stepsize, E1, theta_left, NULL);
        rkintegrate(x_left, &psi2_left, &psiprime2_left, -stepsize, E2, theta_left, NULL);

        psi1_diff = cexpl(-I * theta_left) * (psiprime1_left / psi1_left);
        psi1_diff -= cexpl(-I * theta_right) * (psiprime1_right / psi1_right);
        psi2_diff = cexpl(-I * theta_left) * (psiprime2_left / psi2_left);
        psi2_diff -= cexpl(-I * theta_right) * (psiprime2_right / psi2_right);

        printf("%d\t\t%.10Lf\t\t%.10Lf\n", i + 1, E1, E2);

        if (!isnan(E1))
        {
            tempE = E1;
            E1 = shoot(E1, E2, cabsl(psi1_diff), cabsl(psi2_diff));
            E2 = tempE;
        } 

        else 
        {
            printf("%.20Lf\n", E2);
            return 0;
        }

        psi1_right = 1.0;
        psiprime1_right = -cpowl(I, epsilon) * cpowl(x_right, (2.0 + epsilon)) * psi1_right;
        psi2_right = 1.0;
        psiprime2_right = -cpowl(I, epsilon) * cpowl(x_right, (2.0 + epsilon)) * psi2_right;
        
        psi1_left = 1.0;
        psiprime1_left = -cpowl(I, epsilon) * cpowl(x_left, (2.0 + epsilon)) * psi1_left;
        psi2_left = 1.0;
        psiprime2_left = -cpowl(I, epsilon) * cpowl(x_left, (2.0 + epsilon)) * psi2_left;
    }

    // Integrate once more, to save the results to output csv files
    char output_filename[256];

    sprintf(output_filename, "psi_left_%.1Lf-%.1Lf.csv", E1_init, E2_init);
    rkintegrate(x_left, &psi1_left, &psiprime1_left, -stepsize, E1, theta_left, output_filename);

    sprintf(output_filename, "psi_right_%.1Lf-%.1Lf.csv", E1_init, E2_init);
    rkintegrate(x_right, &psi1_right, &psiprime1_right, -stepsize, E1, theta_right, output_filename);

    printf("%.20Lf\n", E1);

    return 0;

}

// This function appears to be an implementation of the Runge-Kutta 4th order method (RK4)
// for solving ordinary differential equations. The method is used to numerically integrate
// the differential equations defined by functions F and G.
void rkintegrate(ldc x0, ldc *y, ldc *z, long double h,
                 long double E, long double theta, char *output_filename)
{
    ldc k1, k2, k3, k4;
    ldc l1, l2, l3, l4;
    ldc psi, psiprime;
    ldc x;
    long double r;

    int loop;

    r = cabsl(x0);
    psi = *y; psiprime = *z;

    FILE *output;
    if (output_filename != NULL)
    {
        output = fopen(output_filename, "w");
        fprintf(output, "r,psi_real,psi_imag\n");
    }

    for(loop = 0; loop < 20000; loop++)
    {
        k1 = h * F(r, theta, psi , psiprime, E);
        l1 = h * G(r, theta, psi , psiprime, E);

        k2 = h * F(r + h * 0.5, theta, psi + k1 * 0.5, psiprime + l1 * 0.5 , E);
        l2 = h * G(r + h * 0.5, theta , psi + k1 * 0.5, psiprime + l1 * 0.5 , E) ;

        k3 = h * F(r + h * 0.5, theta, psi + k2 * 0.5, psiprime + l2 * 0.5, E) ;
        l3 = h * G(r + h * 0.5, theta, psi + k2 * 0.5, psiprime + l2 * 0.5, E) ;

        k4 = h * F(r + h , theta, psi + k3, psiprime + l3, E);
        l4 = h * G(r + h , theta, psi + k3, psiprime + l3, E) ;

        psi += (k1 + k2 + k2 + k3 + k3 + k4 ) / 6.0;
        psiprime += (l1 + l2 + l2 + l3 + l3 + l4 ) / 6.0 ;

        r += h ;

        if (output_filename != NULL)
        {
            fprintf(output, "%.10Lf,%.10Lf,%.10Lf,\n", r, creall(psi), cimagl(psi));
        }

        // //  NORMALISEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE <><><><><><><>
        //     if (output_filename != NULL)
        // {
        //     long double magnitude = cabsl(psi); // calculates the magnitude of psi
        //     if (magnitude == 0) // to avoid division by zero
        //     {
        //         fprintf(output, "%.10Lf,%.10Lf,%.10Lf,\n", r, 0.0L, 0.0L);
        //     }
        //     else
        //     {
        //         fprintf(output, "%.10Lf,%.10Lf,%.10Lf,\n", r, creall(psi) / magnitude, cimagl(psi) / magnitude);
        //     }
        // }

    }

    if (output_filename != NULL)
    {
        fclose(output);
    }

    *y = psi ;
    *z = psiprime;
}

// F and G represent the differential equations to be integrated.
 // The system appears to be a set of two first-order equations, 
// which might have originally been a single second-order differential equation.
ldc F(long double r, long double theta, ldc psi, ldc psiprime, long double E)
{
    return psiprime;
}

ldc G(long double r, long double theta, ldc psi, ldc psiprime, long double E)
{
    ldc retval;
    ldc x;

    x = r * cexpl(I * theta);

    retval = (cpowl(x, 2.0) * cpowl(I * x, epsilon) - E) * psi;
    retval = retval * cexpl(2.0 * I * theta);
    return retval;
}
// The shoot function appears to be part of the shooting method to find solutions 
// to boundary value problems. Given two trial energy values and the results of 
// their integration, it adjusts the energy to get closer to the desired boundary condition.
long double shoot(long double E1, long double E2, long double A1, long double A2)
{
    long double E3;
    E3 = (E2 * A1 - E1 * A2) / (A1 - A2);

    return E3;
}
