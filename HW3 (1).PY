import numpy as np
from scipy.integrate import quad

# === (a) Construct the discrete least squares trigonometric polynomial S_4 ===

# Define the function f(x)
f = lambda x: x**2 * np.sin(x)

# Interval and sample size
c, d = 0, 1
m = 16
n = 4
delta_x = (d - c) / (2 * m - 1)

# Discrete points x_i and corresponding transformed points z_i
x_i = np.array([c + i * delta_x for i in range(2 * m)])
z_i = np.pi * (2 * x_i - 1)  # Transform to [-π, π]
y_i = f(x_i)

# Compute coefficients
a0 = (1 / m) * np.sum(y_i)

a_coeffs = [(1 / m) * np.sum(y_i * np.cos(k * z_i)) for k in range(1, n + 1)]
b_coeffs = [(1 / m) * np.sum(y_i * np.sin(k * z_i)) for k in range(1, n)]

# Define S_4(x) using transformed variable z = π(2x - 1)
def S4(x):
    z = np.pi * (2 * x - 1)
    result = a0 / 2 + a_coeffs[-1] * np.cos(n * z)
    for k in range(1, n):
        result += a_coeffs[k - 1] * np.cos(k * z) + b_coeffs[k - 1] * np.sin(k * z)
    return result

# === (b) Compute ∫₀¹ S₄(x) * sin(4πx) dx ===
integral_S4_sin4pi, _ = quad(lambda x: S4(x) * np.sin(4 * np.pi * x), 0, 1)

# === (c) Compute ∫₀¹ f(x) * sin(4πx) dx ===
integral_f_sin4pi, _ = quad(lambda x: f(x) * np.sin(4 * np.pi * x), 0, 1)

# === (d) Compute discrete error E(S_4) ===
error_ES = np.sum((y_i - S4(x_i))**2)

# === Output ===
print("=== (a) Trigonometric Polynomial Coefficients ===")
print(f"a_0 = {a0:.6f}")
for i, a_k in enumerate(a_coeffs, 1):
    print(f"a_{i} = {a_k:.6f}")
for i, b_k in enumerate(b_coeffs, 1):
    print(f"b_{i} = {b_k:.6f}")

print("\n=== (b) Integral ∫₀¹ S₄(x) * sin(4πx) dx ===")
print(f"{integral_S4_sin4pi:.6f}")

print("\n=== (c) Integral ∫₀¹ f(x) * sin(4πx) dx ===")
print(f"{integral_f_sin4pi:.6f}")

print("\n=== (d) Discrete Least Squares Error E(S_4) ===")
print(f"{error_ES:.6f}")
