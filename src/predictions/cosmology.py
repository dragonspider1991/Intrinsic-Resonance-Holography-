"""
cosmology.py - Dynamical Dark Energy Prediction

Formalism v9.5 Prediction:
    w(a) = -1 + 0.25 * (1 + a)^{-1.5}

This formula predicts the equation of state parameter for dark energy
as a function of the scale factor a, where a=1 corresponds to the present day.

Key predictions:
    - w_0 = w(a=1) ≈ -0.911
    - w_a = thawing parameter (CPL parameterization)

The CPL (Chevallier-Polarski-Linder) parameterization:
    w(a) = w_0 + w_a * (1 - a)

can be fitted to the RIRH dark energy formula using scipy.optimize.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit


def dark_energy_eos(a):
    """
    Calculate the dark energy equation of state w(a).
    
    Formalism v9.5: w(a) = -1 + 0.25 * (1 + a)^{-1.5}
    
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


def _cpl_model(a, w0, wa):
    """
    CPL (Chevallier-Polarski-Linder) parameterization of w(a).
    
    The CPL parameterization is:
        w(a) = w_0 + w_a * (1 - a)
    
    Args:
        a (float or np.array): Scale factor.
        w0 (float): Present-day equation of state w_0 = w(a=1).
        wa (float): Thawing parameter w_a = dw/d(1-a).
        
    Returns:
        float or np.array: CPL w(a) value.
    """
    return w0 + wa * (1.0 - a)


def cpl_fit(a_range=None, n_points=100):
    """
    Fit the CPL parameterization to the RIRH dark energy EoS.
    
    Uses scipy.optimize.curve_fit to determine the effective w_0 and w_a
    parameters that best approximate the RIRH formula:
        w(a) = -1 + 0.25 * (1 + a)^{-1.5}
    
    over the specified scale factor range.
    
    Args:
        a_range (tuple, optional): Range of scale factors (a_min, a_max).
            Defaults to (0.3, 1.0) covering redshift z ~ 0 to z ~ 2.3.
        n_points (int, optional): Number of points for fitting. Default 100.
        
    Returns:
        dict: Contains:
            - 'w0': Fitted present-day equation of state
            - 'wa': Fitted thawing parameter
            - 'w0_err': Uncertainty in w0
            - 'wa_err': Uncertainty in wa
            - 'residual_rms': RMS residual between fit and RIRH formula
            - 'a_values': Scale factor values used for fitting
            - 'w_rirh': RIRH w(a) values
            - 'w_cpl': CPL fit w(a) values
    """
    if a_range is None:
        a_range = (0.3, 1.0)  # z ~ 0 to z ~ 2.3
    
    # Generate scale factor array
    a_values = np.linspace(a_range[0], a_range[1], n_points)
    
    # Compute RIRH w(a) values
    w_rirh = dark_energy_eos(a_values)
    
    # Initial guess from analytical expressions
    w0_init = calculate_w0()
    wa_init = calculate_wa()
    
    # Perform the fit
    try:
        popt, pcov = curve_fit(
            _cpl_model, 
            a_values, 
            w_rirh,
            p0=[w0_init, wa_init],
            maxfev=5000
        )
        w0_fit, wa_fit = popt
        w0_err, wa_err = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        # Fallback to analytical values if fit fails
        w0_fit = w0_init
        wa_fit = wa_init
        w0_err = 0.0
        wa_err = 0.0
    
    # Compute fitted CPL values
    w_cpl = _cpl_model(a_values, w0_fit, wa_fit)
    
    # Compute RMS residual
    residual_rms = np.sqrt(np.mean((w_rirh - w_cpl) ** 2))
    
    return {
        'w0': float(w0_fit),
        'wa': float(wa_fit),
        'w0_err': float(w0_err),
        'wa_err': float(wa_err),
        'residual_rms': float(residual_rms),
        'a_values': a_values,
        'w_rirh': w_rirh,
        'w_cpl': w_cpl
    }


if __name__ == "__main__":
    # Quick demonstration
    print("=" * 60)
    print("RIRH v9.5 Dark Energy Predictions")
    print("=" * 60)
    
    print(f"\nDirect calculation:")
    print(f"  w_0 = w(a=1) = {calculate_w0():.6f}")
    print(f"  w_a (thawing) = {calculate_wa():.6f}")
    
    # CPL fit
    print(f"\nCPL Fit (over a ∈ [0.3, 1.0]):")
    fit_result = cpl_fit()
    print(f"  w_0 = {fit_result['w0']:.6f} ± {fit_result['w0_err']:.6f}")
    print(f"  w_a = {fit_result['wa']:.6f} ± {fit_result['wa_err']:.6f}")
    print(f"  RMS residual = {fit_result['residual_rms']:.6e}")
    
    # Test at various redshifts
    print(f"\nw(a) at various redshifts:")
    for z in [0, 0.5, 1.0, 2.0]:
        a = 1.0 / (1.0 + z)
        print(f"  z = {z}, a = {a:.3f}, w(a) = {dark_energy_eos(a):.6f}")
    
    # Verification against DESI expectations
    print(f"\nComparison to DESI constraints:")
    print(f"  RIRH w_0 = {calculate_w0():.4f} (DESI: -0.45 ± 0.21)")
    print(f"  RIRH w_a = {calculate_wa():.4f} (DESI: -1.79 ± 0.65)")
    print(f"\nNote: RIRH predicts thawing quintessence behavior.")
