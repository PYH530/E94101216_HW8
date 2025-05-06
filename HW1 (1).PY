import numpy as np
from scipy.optimize import curve_fit

# Data from the table
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# a. Least squares approximation of degree two
def polynomial_fit(x, a, b, c):
    return a * x**2 + b * x + c

# Use curve_fit to find the best fit parameters
popt_poly, pcov_poly = curve_fit(polynomial_fit, x, y)
a_opt, b_opt, c_opt = popt_poly

# Calculate the error
y_fit_poly = polynomial_fit(x, a_opt, b_opt, c_opt)
error_poly = np.sum((y - y_fit_poly)**2)

print("a. Quadratic Approximation:")
print(f"  y = {a_opt:.4f}x^2 + {b_opt:.4f}x + {c_opt:.4f}")
print(f"  Error: {error_poly:.4f}")

# b. Least squares approximation of the form be^(ax)
def exponential_fit(x, a, b):
    return b * np.exp(a * x)

# Use curve_fit to find the best fit parameters
popt_exp, pcov_exp = curve_fit(exponential_fit, x, y, p0=(0.2, 80))  # Provide an initial guess
a_opt_exp, b_opt_exp = popt_exp

# Calculate the error
y_fit_exp = exponential_fit(x, a_opt_exp, b_opt_exp)
error_exp = np.sum((y - y_fit_exp)**2)

print("\nb. Exponential Approximation:")
print(f"  y = {b_opt_exp:.4f} * exp({a_opt_exp:.4f} * x)")
print(f"  Error: {error_exp:.4f}")

# c. Least squares approximation of the form bx^n
def power_fit(x, b, n):
    return b * x**n

# Use curve_fit to find the best fit parameters
popt_power, pcov_power = curve_fit(power_fit, x, y, p0=(10, 2))  # Provide an initial guess
b_opt_power, n_opt_power = popt_power

# Calculate the error
y_fit_power = power_fit(x, b_opt_power, n_opt_power)
error_power = np.sum((y - y_fit_power)**2)

print("\nc. Power Function Approximation:")
print(f"  y = {b_opt_power:.4f} * x^{n_opt_power:.4f}")
print(f"  Error: {error_power:.4f}")