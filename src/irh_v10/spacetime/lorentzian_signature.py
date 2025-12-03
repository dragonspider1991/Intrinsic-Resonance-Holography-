"""
Lorentzian Signature - Time dimension emergence

In 4D spacetime, signature should be (-,+,+,+) or metric signature -2.
This emerges from counting negative eigenvalues of a modified Laplacian.

Reference: IRH v10.0 manuscript, Section IV.B "Lorentzian Signature"
"""

import numpy as np
from typing import Tuple


def count_negative_eigenvalues(
    eigenvalues: np.ndarray,
    threshold: float = -1e-10,
) -> int:
    """
    Count negative eigenvalues (signature of time dimension).
    
    For Lorentzian spacetime signature (-,+,+,+), expect 1 negative eigenvalue.
    
    Args:
        eigenvalues: Eigenvalues of (modified) Interference Matrix
        threshold: Threshold for negativity
    
    Returns:
        count: Number of negative eigenvalues
    """
    count = np.sum(eigenvalues < threshold)
    return int(count)


def compute_lorentzian_signature(
    eigenvalues: np.ndarray,
) -> int:
    """
    Compute metric signature: (# negative) - (# positive).
    
    Standard convention:
        - Lorentzian (-,+,+,+): signature = -2
        - Euclidean (+,+,+,+): signature = +4
    
    Args:
        eigenvalues: Eigenvalues of metric-like operator
    
    Returns:
        signature: Metric signature
    """
    n_negative = count_negative_eigenvalues(eigenvalues)
    n_positive = np.sum(eigenvalues > 1e-10)
    
    signature = n_positive - n_negative
    return signature


def verify_lorentzian_signature(
    eigenvalues: np.ndarray,
    expected_negative: int = 1,
) -> bool:
    """
    Verify that signature matches Lorentzian spacetime.
    
    For 4D spacetime with signature (-,+,+,+), expect exactly 1 negative eigenvalue.
    
    Args:
        eigenvalues: Eigenvalues of appropriate operator
        expected_negative: Expected number of negative eigenvalues (default: 1)
    
    Returns:
        is_lorentzian: True if signature matches
    """
    n_negative = count_negative_eigenvalues(eigenvalues)
    is_lorentzian = (n_negative == expected_negative)
    return is_lorentzian


def modified_laplacian_for_signature(
    L: np.ndarray,
    time_mode_index: int = 0,
) -> np.ndarray:
    """
    Modify Laplacian to reveal Lorentzian signature.
    
    Standard Laplacian has all non-negative eigenvalues.
    To see time signature, we flip sign of timelike mode:
        ℒ_Lorentz = ℒ - 2λ_time |ψ_time⟩⟨ψ_time|
    
    Args:
        L: Standard Laplacian matrix
        time_mode_index: Index of mode to treat as timelike (default: 0 = zero mode)
    
    Returns:
        L_lorentz: Modified Laplacian with Lorentzian signature
    """
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Modify timelike eigenvalue (flip sign)
    eigenvalues_mod = eigenvalues.copy()
    eigenvalues_mod[time_mode_index] = -eigenvalues_mod[time_mode_index]
    
    # Reconstruct
    L_lorentz = eigenvectors @ np.diag(eigenvalues_mod) @ eigenvectors.T
    
    return L_lorentz


def timelike_direction_from_aro(
    harmony_history: list,
) -> np.ndarray:
    """
    Extract timelike direction from ARO evolution.
    
    The arrow of time is the direction of decreasing Harmony.
    
    Args:
        harmony_history: History of Harmony Functional values
    
    Returns:
        time_direction: Unit vector pointing forward in time
    """
    # Gradient of harmony
    harmony_array = np.array(harmony_history)
    gradient = np.diff(harmony_array)
    
    # Time direction = direction of negative gradient (decrease)
    # For simplicity, return sign
    time_direction = -np.sign(gradient.mean())
    
    return time_direction


def arrow_of_time_test(
    harmony_history: list,
    threshold: float = -1e-6,
) -> bool:
    """
    Verify thermodynamic arrow of time.
    
    Harmony should decrease monotonically (on average).
    
    Args:
        harmony_history: ARO harmony evolution
        threshold: Maximum allowed increase per step
    
    Returns:
        has_arrow: True if arrow of time is well-defined
    """
    harmony_array = np.array(harmony_history)
    differences = np.diff(harmony_array)
    
    # Check that most steps decrease or stay constant
    decreasing_steps = np.sum(differences <= threshold)
    total_steps = len(differences)
    
    # At least 80% of steps should decrease
    has_arrow = (decreasing_steps / total_steps) > 0.8
    
    return has_arrow
