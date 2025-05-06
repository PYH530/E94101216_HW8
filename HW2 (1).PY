import numpy as np
import sympy as sp

# Define the function f(x)
def f(x):
    return 0.5 * sp.cos(x) + 0.25 * sp.sin(2*x)

# Define the polynomial basis functions
def phi0(x):
    return 1

def phi1(x):
    return x

def phi2(x):
    return x**2

# Set up the normal equations
# We need to calculate the following integrals:
# integral(phi_i * phi_j dx) and integral(f(x) * phi_i dx) for i, j = 0, 1, 2

# Calculate the integrals
def calculate_integral(func1, func2):
    integral = sp.integrate(func1 * func2, (x, -1, 1))
    return float(integral)

x = sp.Symbol('x')  # Define x as a symbolic variable

# Calculate the coefficients of the normal equations
a00 = calculate_integral(phi0(x), phi0(x))
a01 = calculate_integral(phi0(x), phi1(x))
a02 = calculate_integral(phi0(x), phi2(x))
a10 = a01
a11 = calculate_integral(phi1(x), phi1(x))
a12 = calculate_integral(phi1(x), phi2(x))
a20 = a02
a21 = a12
a22 = calculate_integral(phi2(x), phi2(x))

b0 = calculate_integral(f(x), phi0(x))
b1 = calculate_integral(f(x), phi1(x))
b2 = calculate_integral(f(x), phi2(x))

# Construct the matrix and vector
A = np.array([[a00, a01, a02],
              [a10, a11, a12],
              [a20, a21, a22]])
B = np.array([b0, b1, b2])

# Solve the system of equations
coefficients = np.linalg.solve(A, B)
a, b, c = coefficients

# Define the least squares polynomial approximation
def p2(x_val):  # Use a different variable name to avoid shadowing
    return a + b * x_val + c * x_val**2

# Print the results
print("Least Squares Polynomial Approximation of Degree 2:")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")
print(f"P2(x) = {a:.6f} + {b:.6f}x + {c:.6f}x^2")