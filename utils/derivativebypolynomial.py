import numpy as np

def derivative_by_polynomial(x, y):
    poly_coeffs = np.polyfit(x, y, 3)
    poly = np.poly1d(poly_coeffs)
    deriv_poly = poly.deriv()
    return deriv_poly(x)