"""
Thawing Dark Energy - w(a) evolution

IRH v10.0 predicts a specific dark energy equation of state:
    w(a) = -1 + 0.25(1+a)^(-1.5)

This is "thawing quintessence": w evolves from -0.75 → -1.

Reference: IRH v10.0 manuscript, Section VI.C "Dark Energy"
"""

import numpy as np
from typing import Tuple


def w_dark_energy(a: float | np.ndarray) -> float | np.ndarray:
    """
    Dark energy equation of state from IRH v10.0.
    
    w(a) = -1 + 0.25(1+a)^(-1.5)
    
    Args:
        a: Scale factor (a=1 today, a=0 at Big Bang)
    
    Returns:
        w: Equation of state parameter
    
    Example:
        >>> w_today = w_dark_energy(1.0)
        >>> print(f"w_0 = {w_today:.4f}")
    """
    return -1.0 + 0.25 * (1 + a)**(-1.5)


def w_cpl_parameters() -> Tuple[float, float]:
    """
    Compute CPL parameterization: w(a) = w_0 + w_a(1-a).
    
    Fit IRH formula to CPL at a=1.
    
    Returns:
        w_0: Present-day equation of state
        w_a: Evolution parameter
    """
    a_today = 1.0
    w_0 = w_dark_energy(a_today)
    
    # Derivative at a=1
    # dw/da = 0.25 * (-1.5) * (1+a)^(-2.5)
    dw_da = 0.25 * (-1.5) * (1 + a_today)**(-2.5)
    
    # CPL: w_a = -dw/da at a=1
    w_a = -dw_da
    
    return w_0, w_a


def rho_dark_energy(
    a: float | np.ndarray,
    rho_0: float = 1.0,
) -> float | np.ndarray:
    """
    Dark energy density evolution.
    
    ρ_DE(a) = ρ_0 × exp(-3 ∫ [1+w(a')] da'/a')
    
    Args:
        a: Scale factor
        rho_0: Present-day dark energy density
    
    Returns:
        rho: Dark energy density at scale factor a
    """
    # For IRH formula w(a) = -1 + 0.25(1+a)^(-1.5)
    # Integral can be done semi-analytically
    
    # Simple numerical integration
    if np.isscalar(a):
        a_vals = np.linspace(a, 1.0, 100)
    else:
        a_vals = a
    
    # Integrate [1 + w(a)] / a
    integrand = (1 + w_dark_energy(a_vals)) / a_vals
    
    # Trapezoidal integration from a to 1
    if np.isscalar(a):
        integral = np.trapz(integrand, a_vals)
        rho = rho_0 * np.exp(-3 * integral)
    else:
        rho = np.zeros_like(a)
        for i, a_val in enumerate(a):
            a_range = np.linspace(a_val, 1.0, 50)
            integrand_i = (1 + w_dark_energy(a_range)) / a_range
            integral_i = np.trapz(integrand_i, a_range)
            rho[i] = rho_0 * np.exp(-3 * integral_i)
    
    return rho


def hubble_parameter(
    a: float | np.ndarray,
    Omega_m: float = 0.3,
    Omega_de: float = 0.7,
    H_0: float = 1.0,
) -> float | np.ndarray:
    """
    Hubble parameter evolution H(a).
    
    H²(a) = H_0² [Ω_m a^(-3) + Ω_DE ρ_DE(a)/ρ_DE(1)]
    
    Args:
        a: Scale factor
        Omega_m: Matter density parameter
        Omega_de: Dark energy density parameter
        H_0: Present-day Hubble constant
    
    Returns:
        H: Hubble parameter at a
    """
    # Matter contribution
    rho_m_ratio = a**(-3)
    
    # Dark energy contribution
    rho_de_ratio = rho_dark_energy(a, rho_0=1.0) / rho_dark_energy(1.0, rho_0=1.0)
    
    # Total
    H_squared = H_0**2 * (Omega_m * rho_m_ratio + Omega_de * rho_de_ratio)
    H = np.sqrt(H_squared)
    
    return H


def thawing_test(a_range: np.ndarray = None) -> bool:
    """
    Test that w(a) is "thawing" (increasing toward -1).
    
    Thawing: dw/da > 0 for a > 0
    
    Args:
        a_range: Range of scale factors to test (default: [0.1, 3.0])
    
    Returns:
        is_thawing: True if w increases with a
    """
    if a_range is None:
        a_range = np.linspace(0.1, 3.0, 100)
    
    w_vals = w_dark_energy(a_range)
    
    # Check that w is increasing
    dw = np.diff(w_vals)
    is_thawing = np.all(dw >= -1e-6)  # Allow tiny numerical errors
    
    return is_thawing
