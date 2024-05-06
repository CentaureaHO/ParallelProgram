import numpy as np
from scipy.optimize import curve_fit

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

x = np.linspace(-5, 5, 1000)
y = np.arctan(x)

initial_guess = [0] * 20 
coeffs, _ = curve_fit(polynomial, x, y, p0=initial_guess)

print("多项式系数:", coeffs)
