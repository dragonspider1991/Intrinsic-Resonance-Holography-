"""
cosmology.py - Dynamical Dark Energy Prediction

Formalism v9.4 Prediction:
    w(a) = -1 + 0.25 * (1 + a)^{-1.5}

This formula predicts the equation of state parameter for dark energy
as a function of the scale factor a, where a=1 corresponds to the present day.

Key predictions:
    - w_0 = w(a=1) ≈ -0.911
    - w_a = thawing parameter (CPL parameterization)
"""

import numpy as np
from scipy.integrate import quad


def dark_energy_eos(a):
    """
    Calculate the dark energy equation of state w(a).
    
    Formalism v9.4: w(a) = -1 + 0.25 * (1 + a)^{-1.5}
    
    Args:
        a (float or np.array): Scale factor. a=1 corresponds to present day.
        
    Returns:
        float or np.array: Equation of state parameter w(a).
    """
    return -1.0 + 0.25 * (1.0 + a) ** (-1.5)


def calculate_w0():
    """
    Calculate w_0 = w(a=1), the present-day equation of state.
    
    In the CPL parameterization: w(a) = w_0 + w_a * (1 - a)
    
    Returns:
        float: w_0 value at present epoch (a=1).
    """
    return dark_energy_eos(1.0)


def calculate_wa():
    """
    Calculate the thawing parameter w_a.
    
    In the CPL parameterization: w(a) = w_0 + w_a * (1 - a)
    
    w_a is the derivative of w with respect to (1-a), evaluated at a=1.
    For our formula: w(a) = -1 + 0.25 * (1 + a)^{-1.5}
    
    dw/d(1-a) = -dw/da = -0.25 * (-1.5) * (1 + a)^{-2.5} = 0.375 * (1 + a)^{-2.5}
    At a=1: w_a = 0.375 * 2^{-2.5} ≈ 0.0663
    
    Returns:
        float: Thawing parameter w_a.
    """
    # Analytical derivative: dw/da = -0.375 * (1 + a)^{-2.5}
    # w_a = -dw/da at a=1
    a = 1.0
    dw_da = -0.375 * (1.0 + a) ** (-2.5)
    w_a = -dw_da
    return w_a


def dark_energy_density_ratio(a, omega_de_0=0.685):
    """
    Calculate the dark energy density ratio Omega_DE(a).
    
    For a dynamical w(a), the dark energy density evolves as:
        rho_DE(a) / rho_DE_0 = exp(3 * integral_{a}^{1} (1 + w(a')) da' / a')
    
    Args:
        a (float): Scale factor.
        omega_de_0 (float): Present-day dark energy density parameter.
        
    Returns:
        float: Dark energy density ratio at scale factor a.
    """
    def integrand(a_prime):
        return (1.0 + dark_energy_eos(a_prime)) / a_prime
    
    if np.isscalar(a):
        integral, _ = quad(integrand, a, 1.0)
        return omega_de_0 * np.exp(3.0 * integral)
    else:
        result = []
        for a_val in a:
            integral, _ = quad(integrand, a_val, 1.0)
            result.append(omega_de_0 * np.exp(3.0 * integral))
        return np.array(result)


if __name__ == "__main__":
    # Quick demonstration
    print(f"w_0 = w(a=1) = {calculate_w0():.6f}")
    print(f"w_a (thawing) = {calculate_wa():.6f}")
    
    # Test at various redshifts
    for z in [0, 0.5, 1.0, 2.0]:
        a = 1.0 / (1.0 + z)
        print(f"z = {z}, a = {a:.3f}, w(a) = {dark_energy_eos(a):.6f}")
