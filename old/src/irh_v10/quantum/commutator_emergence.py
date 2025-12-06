"""
Commutator Emergence - [q, p] = iℏ from symplectic structure

The canonical commutation relations emerge geometrically from
the symplectic structure of real phase space.

Reference: IRH v10.0 manuscript, Section II.D "Symplectic → U(N)"
"""

import numpy as np
from typing import Tuple


def verify_canonical_commutator(
    q: np.ndarray,
    p: np.ndarray,
    hbar: float = 1.0,
    tolerance: float = 1e-10,
) -> Tuple[bool, float]:
    """
    Verify canonical commutation relation [q, p] = iℏ.
    
    For discrete phase space, check:
        q_i p_j - p_j q_i = iℏ δ_ij
    
    Args:
        q: Position operators (as matrices)
        p: Momentum operators (as matrices)
        hbar: Planck's constant (default: 1.0 in natural units)
        tolerance: Numerical tolerance
    
    Returns:
        verified: True if CCR holds
        max_error: Maximum deviation from iℏ δ_ij
    """
    # Compute commutator [q, p] = qp - pq
    commutator = q @ p - p @ q
    
    # Expected: iℏ I
    N = len(q)
    expected = 1j * hbar * np.eye(N)
    
    # Error
    error = commutator - expected
    max_error = np.abs(error).max()
    
    verified = max_error < tolerance
    
    return verified, max_error


def position_operator_from_spectrum(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    Construct position operator from spectrum.
    
    q = Σ_k √λ_k |ψ_k⟩⟨ψ_k|
    
    Args:
        eigenvalues: Eigenvalues of ℒ
        eigenvectors: Eigenvectors of ℒ
    
    Returns:
        q: Position operator
    """
    # Filter positive eigenvalues
    mask = eigenvalues > 1e-10
    lambdas = eigenvalues[mask]
    vecs = eigenvectors[:, mask]
    
    # q = Σ √λ_k |ψ_k⟩⟨ψ_k|
    q = vecs @ np.diag(np.sqrt(lambdas)) @ vecs.T
    
    return q


def momentum_operator_from_laplacian(
    L: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """
    Construct momentum operator from Laplacian.
    
    p = -iℏ ∇ ≈ -iℏ L (discrete derivative)
    
    Args:
        L: Interference Matrix (Laplacian)
        hbar: Planck's constant
    
    Returns:
        p: Momentum operator
    """
    # Momentum = -iℏ times Laplacian (discrete gradient)
    p = -1j * hbar * L
    
    return p


def commutator_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute commutator [A, B] = AB - BA.
    
    Args:
        A: First operator
        B: Second operator
    
    Returns:
        comm: Commutator [A, B]
    """
    return A @ B - B @ A


def verify_heisenberg_uncertainty(
    q_operator: np.ndarray,
    p_operator: np.ndarray,
    state: np.ndarray,
    hbar: float = 1.0,
) -> Tuple[bool, float, float]:
    """
    Verify Heisenberg uncertainty principle for a state.
    
    Δq Δp ≥ ℏ/2
    
    Args:
        q_operator: Position operator
        p_operator: Momentum operator
        state: Quantum state vector (normalized)
        hbar: Planck's constant
    
    Returns:
        satisfied: True if uncertainty principle holds
        delta_q: Position uncertainty
        delta_p: Momentum uncertainty
    """
    # Expectation values
    q_mean = np.real(state.conj() @ q_operator @ state)
    p_mean = np.real(state.conj() @ p_operator @ state)
    
    # Variances
    q2_mean = np.real(state.conj() @ (q_operator @ q_operator) @ state)
    p2_mean = np.real(state.conj() @ (p_operator @ p_operator) @ state)
    
    delta_q = np.sqrt(q2_mean - q_mean**2)
    delta_p = np.sqrt(p2_mean - p_mean**2)
    
    # Check uncertainty principle
    product = delta_q * delta_p
    bound = hbar / 2
    
    satisfied = product >= bound * (1 - 1e-6)  # Small tolerance for numerics
    
    return satisfied, delta_q, delta_p
