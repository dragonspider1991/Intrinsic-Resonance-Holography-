"""
fine_structure.py - Fine Structure Constant Error Budget Calculator

Formalism v9.4 Error Budget:
    Δ_total = sqrt(σ_sys² + σ_stat²)
    
Based on the v9.4 Error Budget section:
    - σ_sys: Systematic errors from graph discretization
    - σ_stat: Statistical errors scaling as 1/sqrt(N)

Target: α⁻¹ = 137.035999084(15)
"""

import numpy as np


# CODATA 2022 value
ALPHA_INVERSE_TARGET = 137.035999084
ALPHA_INVERSE_UNCERTAINTY = 0.000000015  # (15) in last two digits


def calculate_alpha_error(N_min, N_max):
    """
    Calculate total error budget for α⁻¹ prediction.
    
    Implements the error budget from Formalism v9.4:
        Δ_total = sqrt(σ_sys² + σ_stat²)
    
    Args:
        N_min (int): Minimum graph size used in computation.
        N_max (int): Maximum graph size used in computation.
        
    Returns:
        dict: Error budget components including:
            - sigma_sys: Systematic error estimate
            - sigma_stat: Statistical error (1/sqrt(N) scaling)
            - delta_total: Total combined error
            - relative_error: Relative error compared to target
    """
    # Systematic error estimate
    # Based on discretization effects, scales as 1/N_min
    sigma_sys = ALPHA_INVERSE_TARGET * (1.0 / N_min)
    
    # Statistical error estimate
    # Standard 1/sqrt(N) scaling for ensemble averaging
    sigma_stat = ALPHA_INVERSE_TARGET * (1.0 / np.sqrt(N_max))
    
    # Total error (quadrature sum)
    delta_total = np.sqrt(sigma_sys ** 2 + sigma_stat ** 2)
    
    # Relative error
    relative_error = delta_total / ALPHA_INVERSE_TARGET
    
    return {
        "sigma_sys": sigma_sys,
        "sigma_stat": sigma_stat,
        "delta_total": delta_total,
        "relative_error": relative_error,
        "N_min": N_min,
        "N_max": N_max,
        "target": ALPHA_INVERSE_TARGET,
        "target_uncertainty": ALPHA_INVERSE_UNCERTAINTY,
    }


def required_graph_size_for_precision(target_precision):
    """
    Calculate required graph size to achieve a given precision.
    
    Given target relative precision ε, find N such that:
        σ_stat / α⁻¹ < ε
        
    Since σ_stat ~ 1/sqrt(N), we need N > 1/ε²
    
    Args:
        target_precision (float): Target relative precision (e.g., 1e-9 for 10^-9).
        
    Returns:
        dict: Required graph parameters including:
            - N_min: Minimum N for systematic error control
            - N_max: Minimum N for statistical precision
    """
    # For statistical precision: N > 1/ε²
    N_stat = int(np.ceil(1.0 / (target_precision ** 2)))
    
    # For systematic precision: N > 1/ε
    N_sys = int(np.ceil(1.0 / target_precision))
    
    return {
        "N_min_systematic": N_sys,
        "N_max_statistical": N_stat,
        "target_precision": target_precision,
        "achievable_at": max(N_sys, N_stat),
    }


def extrapolate_to_continuum(values, N_values):
    """
    Extrapolate α⁻¹ predictions to the continuum limit (N → ∞).
    
    Uses Richardson extrapolation assuming 1/N corrections.
    
    Args:
        values (array-like): α⁻¹ values at different N.
        N_values (array-like): Corresponding N values.
        
    Returns:
        dict: Continuum extrapolation result including:
            - continuum_value: Extrapolated value
            - correction_coefficient: Leading 1/N coefficient
    """
    values = np.array(values)
    N_values = np.array(N_values)
    
    # Fit to: α⁻¹(N) = α⁻¹_∞ + c/N
    # Rewrite as: α⁻¹(N) = α⁻¹_∞ + c * (1/N)
    x = 1.0 / N_values
    
    # Linear regression
    A = np.vstack([np.ones_like(x), x]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, values, rcond=None)
    
    alpha_inf = coeffs[0]
    c = coeffs[1]
    
    return {
        "continuum_value": alpha_inf,
        "correction_coefficient": c,
        "deviation_from_target": abs(alpha_inf - ALPHA_INVERSE_TARGET),
    }


if __name__ == "__main__":
    # Quick demonstration
    print("Fine Structure Constant Error Budget Calculator")
    print("=" * 50)
    
    # Example calculation
    result = calculate_alpha_error(N_min=100, N_max=4096)
    print(f"\nFor N_min={result['N_min']}, N_max={result['N_max']}:")
    print(f"  σ_sys  = {result['sigma_sys']:.6f}")
    print(f"  σ_stat = {result['sigma_stat']:.6f}")
    print(f"  Δ_total = {result['delta_total']:.6f}")
    print(f"  Relative error = {result['relative_error']:.2e}")
    
    # What graph size needed for experimental precision?
    exp_precision = ALPHA_INVERSE_UNCERTAINTY / ALPHA_INVERSE_TARGET
    print(f"\nExperimental precision: {exp_precision:.2e}")
    req = required_graph_size_for_precision(exp_precision)
    print(f"Required N_max for same precision: {req['N_max_statistical']:.2e}")
