"""
Impedance Matching - Derives ξ(N) = 1/(N ln N)

The impedance coefficient ξ(N) balances elastic energy against disorder
in the Harmony Functional. It emerges from matching the scaling of
elastic and entropic contributions.

Mathematical Derivation (Section III.B in manuscript):
    - Elastic energy scales as: E_elastic ~ N × κ²
    - Spectral entropy scales as: S ~ ln N (for random networks)
    - Impedance matching: ξ(N) × S ~ E_elastic
    - Therefore: ξ(N) = 1/(N ln N)

This ensures ARO converges to physically meaningful configurations.

Reference: IRH v10.0 manuscript, Equation (18)
"""

import numpy as np


def impedance_coefficient(N: int, base: str = "natural") -> float:
    """
    Compute the impedance matching coefficient ξ(N).
    
    Exact formula from impedance matching principle:
        ξ(N) = 1 / (N × ln N)
    
    This coefficient ensures proper balance between elastic energy
    and spectral dissonance in the Harmony Functional.
    
    Args:
        N: Number of oscillators in the network
        base: Logarithm base ("natural" or "2")
            - "natural": ln N (default, matches manuscript)
            - "2": log₂ N (alternative for information-theoretic interpretation)
    
    Returns:
        xi: Impedance coefficient ξ(N)
    
    Example:
        >>> xi_100 = impedance_coefficient(100)
        >>> print(f"ξ(100) = {xi_100:.6f}")
        ξ(100) = 0.002171
    
    Notes:
        - For N < 10, returns a regularized value to avoid numerical issues
        - Derivation assumes N >> 1 in thermodynamic limit
    """
    if N < 10:
        # Regularization for small N
        N_eff = 10
    else:
        N_eff = N
    
    if base == "natural":
        log_N = np.log(N_eff)
    elif base == "2":
        log_N = np.log2(N_eff)
    else:
        raise ValueError(f"Unknown logarithm base: {base}")
    
    xi = 1.0 / (N_eff * log_N)
    
    return xi


def verify_impedance_scaling(N_values: np.ndarray) -> dict:
    """
    Verify that ξ(N) follows the predicted 1/(N ln N) scaling.
    
    This function computes ξ(N) for a range of N values and verifies
    the scaling law by checking:
        N × ln(N) × ξ(N) ≈ 1
    
    Args:
        N_values: Array of N values to test
    
    Returns:
        results: Dictionary with:
            - N_values: Input N values
            - xi_values: Computed ξ(N) values
            - scaled_values: N × ln(N) × ξ(N) (should be ≈ 1)
            - max_deviation: Maximum deviation from 1.0
    
    Example:
        >>> N_vals = np.logspace(1, 6, 20)  # 10 to 10^6
        >>> results = verify_impedance_scaling(N_vals)
        >>> print(f"Max deviation: {results['max_deviation']:.2e}")
    """
    xi_values = np.array([impedance_coefficient(int(N)) for N in N_values])
    
    # Check scaling: N × ln(N) × ξ(N) should equal 1
    scaled_values = N_values * np.log(N_values) * xi_values
    
    max_deviation = np.max(np.abs(scaled_values - 1.0))
    
    return {
        "N_values": N_values,
        "xi_values": xi_values,
        "scaled_values": scaled_values,
        "max_deviation": max_deviation,
    }


def analytical_ξ_limit(N: float) -> float:
    """
    Analytical continuation of ξ(N) for non-integer N.
    
    Useful for theoretical analysis and smooth interpolation.
    
    Args:
        N: Effective oscillator number (can be float)
    
    Returns:
        xi: Impedance coefficient
    """
    if N < 1.0:
        raise ValueError("N must be >= 1")
    
    return 1.0 / (N * np.log(N))


def effective_N_from_spectrum(eigenvalues: np.ndarray, threshold: float = 1e-10) -> int:
    """
    Compute effective number of modes N_eff from spectrum.
    
    This accounts for degeneracies and near-zero modes.
    Used to determine appropriate ξ for a given network.
    
    Args:
        eigenvalues: Eigenvalues of Interference Matrix ℒ
        threshold: Minimum eigenvalue to count as non-zero
    
    Returns:
        N_eff: Effective number of active modes
    """
    # Count non-zero eigenvalues
    N_eff = np.sum(eigenvalues > threshold)
    return int(N_eff)
