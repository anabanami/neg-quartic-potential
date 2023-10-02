// This code includes standard libraries. stdio.h is for input/output functions like printf.
 // math.h provides mathematical functions. complex.h provides support for complex numbers.
#include <stdio.h>
#include <math.h>
#include <complex.h>

// Macros are defined for convenience. ldc is a shorthand for long double complex, pi 
// represents the mathematical constant π, and epsilon is defined as 2.0 
// for the negative quartic Hamiltonian.
#define ldc long double complex
#define pi 3.14159265358979323846
#define epsilon 2.0


// Function prototypes are declared so that they can be used before their definitions.
void rkintegrate(ldc x0, ldc *y, ldc *z, long double h,
                 long double E, long double theta, char *output_filename);
ldc F(long double r, long double theta, ldc psi, ldc psiprime, long double E);
ldc G(long double r, long double theta, ldc psi, ldc psiprime, long double E);


int main(void) 
{
    // COARSE -- SEARCH parameters
    long double E;
    long double step = 0.1;// coarse grid interval
    long double E_min = 0; // lower bound 
    int N = 1000; // Number of points in coarse grid
    ldc x_left, x_right;
    ldc psi_left, psiprime_left;
    ldc psi_right, psiprime_right;
    ldc psi_diff;

    long double stepsize = 0.0005;
    long double theta_left, theta_right;

    // Saving psi-diff as function of E to a csv file 
    char *output_filename = "psi_diffs.csv";

    FILE *output;
    output = fopen(output_filename, "w");
    fprintf(output, "E,psi_diff_real,psi_diff_imag\n");

    x_right = 10.0 * cexpl(I * pi * ((4.0 * 1.0 - epsilon) / (4.0 + 2.0 * epsilon)));
    x_left = 10.0 * cexpl(I * pi * ((4.0 * -2.0 - epsilon) / (4.0 + 2.0 * epsilon)));

    theta_right = cargl(x_right);
    theta_left = cargl(x_left);

    for(int i = 0; i < N; i++)
    {
        E = E_min + i * step;
        sprintf("%.20Lf",E,"\n");

        psi_right = 1.0;
        psiprime_right = -cpowl(I, epsilon) * cpowl(x_right, (2.0 + epsilon)) * psi_right;

        psi_left = 1.0;
        psiprime_left = -cpowl(I, epsilon) * cpowl(x_left, (2.0 + epsilon)) * psi_left;

        rkintegrate(x_right, &psi_right, &psiprime_right, -stepsize, E, theta_right, NULL);
        rkintegrate(x_left, &psi_left, &psiprime_left, -stepsize, E, theta_left, NULL);

        psi_diff = cexpl(-I * theta_left) * (psiprime_left / psi_left);
        psi_diff -= cexpl(-I * theta_right) * (psiprime_right / psi_right);

        fprintf(output, "%.10Lf,%.20Lf,%.20Lf\n", E, creall(psi_diff), cimagl(psi_diff)); // Writing E, real and imag parts of psi_diff to file

    }
    return 0;
    
}

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

    }

    if (output_filename != NULL)
    {
        fclose(output);
    }

    *y = psi;
    *z = psiprime;
}


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
