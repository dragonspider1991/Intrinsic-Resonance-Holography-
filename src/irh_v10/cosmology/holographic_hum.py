"""
Holographic Hum - Dark energy from spectral entropy

The "Holographic Hum" is the contribution of spectral entropy
to the vacuum energy density, causing accelerated expansion.

ρ_Hum ~ S_dissonance / V

Reference: IRH v10.0 manuscript, Section VI "Cosmology"
"""

import numpy as np


def holographic_hum_density(
    eigenvalues: np.ndarray,
    volume: float = 1.0,
) -> float:
    """
    Compute vacuum energy density from Holographic Hum.
    
    ρ_Hum = S_dissonance / V
    
    where S_dissonance is spectral entropy.
    
    Args:
        eigenvalues: Eigenvalues of Interference Matrix ℒ
        volume: Spatial volume (in natural units)
    
    Returns:
        rho_hum: Vacuum energy density from Holographic Hum
    """
    # Filter non-zero eigenvalues
    lambdas = eigenvalues[eigenvalues > 1e-12]
    
    if len(lambdas) == 0:
        return 0.0
    
    # Normalize to probability distribution
    total = lambdas.sum()
    probs = lambdas / total
    
    # Shannon entropy
    S_dissonance = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Energy density
    rho_hum = S_dissonance / volume
    
    return rho_hum


def holographic_hum_pressure(
    eigenvalues: np.ndarray,
    volume: float = 1.0,
) -> float:
    """
    Compute pressure from Holographic Hum.
    
    For dark energy: p = w × ρ
    where w(a) = -1 + 0.25(1+a)^(-1.5)
    
    Args:
        eigenvalues: Eigenvalues of ℒ
        volume: Spatial volume
    
    Returns:
        p_hum: Pressure from Holographic Hum
    """
    rho = holographic_hum_density(eigenvalues, volume)
    
    # At present (a=1): w = -1 + 0.25(2)^(-1.5) ≈ -0.912
    a = 1.0
    w = -1.0 + 0.25 * (1 + a)**(-1.5)
    
    p_hum = w * rho
    
    return p_hum


def spectral_entropy_evolution(
    eigenvalues_history: list,
    scale_factors: np.ndarray,
) -> np.ndarray:
    """
    Track evolution of spectral entropy with cosmic expansion.
    
    Args:
        eigenvalues_history: List of eigenvalue arrays at different epochs
        scale_factors: Corresponding scale factors
    
    Returns:
        S_history: Spectral entropy vs scale factor
    """
    S_history = []
    
    for eigenvalues in eigenvalues_history:
        lambdas = eigenvalues[eigenvalues > 1e-12]
        if len(lambdas) > 0:
            total = lambdas.sum()
            probs = lambdas / total
            S = -np.sum(probs * np.log2(probs + 1e-12))
        else:
            S = 0.0
        S_history.append(S)
    
    return np.array(S_history)


def holographic_bound_check(
    entropy: float,
    area: float,
) -> bool:
    """
    Check holographic bound: S ≤ A/(4ℏG).
    
    In natural units (ℏ=G=1): S ≤ A/4
    
    Args:
        entropy: Total entropy
        area: Boundary area
    
    Returns:
        satisfied: True if bound is satisfied
    """
    bound = area / 4.0  # Natural units
    satisfied = entropy <= bound * (1 + 1e-6)  # Small tolerance
    
    return satisfied
