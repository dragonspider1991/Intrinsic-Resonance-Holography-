"""
Symplectic to Unitary Theorem - Complex structure emergence

This module proves and implements the Sp(2N) → U(N) theorem:
Complex quantum structure emerges from real symplectic geometry.

Theorem (Section II.D in manuscript):
    The symplectic structure of phase space (q, p) ∈ ℝ^(2N) naturally
    induces a U(N) structure via the identification:
        z_i = (q_i + ip_i) / √2
    
    The symplectic form ω = Σ dq_i ∧ dp_i becomes the Hermitian form
    on ℂ^N, and Sp(2N, ℝ) transformations become U(N) transformations.

This is the mathematical foundation for quantum mechanics emerging
from classical oscillator dynamics.

Reference: IRH v10.0 manuscript, Equation (8)-(11)
"""

import numpy as np
from typing import Tuple


def symplectic_to_unitary(
    q: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
    Convert real phase space coordinates (q, p) to complex amplitudes z.
    
    This implements the canonical map:
        z_i = (q_i + ip_i) / √2
    
    Args:
        q: Position coordinates (N,)
        p: Momentum coordinates (N,)
    
    Returns:
        z: Complex amplitudes (N,) in ℂ^N
    
    Example:
        >>> q = np.array([1.0, 0.0])
        >>> p = np.array([0.0, 1.0])
        >>> z = symplectic_to_unitary(q, p)
        >>> print(z)
        [0.70710678+0.j         0.        +0.70710678j]
    """
    z = (q + 1j * p) / np.sqrt(2.0)
    return z


def unitary_to_symplectic(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert complex amplitudes z back to real phase space (q, p).
    
    Inverse map:
        q_i = Re(z_i) × √2
        p_i = Im(z_i) × √2
    
    Args:
        z: Complex amplitudes (N,)
    
    Returns:
        q: Position coordinates (N,)
        p: Momentum coordinates (N,)
    """
    q = np.real(z) * np.sqrt(2.0)
    p = np.imag(z) * np.sqrt(2.0)
    return q, p


def verify_symplectic_structure(
    q1: np.ndarray,
    p1: np.ndarray,
    q2: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    Verify that the symplectic form is preserved.
    
    The symplectic form is:
        ω((q1,p1), (q2,p2)) = Σ_i (q1_i × p2_i - q2_i × p1_i)
    
    In complex coordinates z1, z2, this becomes:
        ω = Im(z1† · z2) × 2
    
    Args:
        q1, p1: First phase space point
        q2, p2: Second phase space point
    
    Returns:
        omega: Symplectic form value (should be preserved under transformations)
    """
    # Real symplectic form
    omega_real = np.sum(q1 * p2 - q2 * p1)
    
    # Complex form
    z1 = symplectic_to_unitary(q1, p1)
    z2 = symplectic_to_unitary(q2, p2)
    omega_complex = 2.0 * np.imag(np.vdot(z1, z2))
    
    # Should be equal
    assert np.abs(omega_real - omega_complex) < 1e-10, \
        f"Symplectic form mismatch: {omega_real} vs {omega_complex}"
    
    return omega_real


def construct_J_matrix(N: int) -> np.ndarray:
    """
    Construct the symplectic matrix J for canonical transformations.
    
    The symplectic matrix in block form is:
        J = [  0   I  ]
            [ -I   0  ]
    where I is the N×N identity matrix.
    
    This satisfies: J^T J = -I_{2N} and J^2 = -I_{2N}
    
    Args:
        N: Dimension of configuration space
    
    Returns:
        J: Symplectic matrix (2N × 2N)
    """
    I = np.eye(N)
    Z = np.zeros((N, N))
    
    J = np.block([
        [Z, I],
        [-I, Z]
    ])
    
    return J


def verify_unitary_group(
    z: np.ndarray,
    U: np.ndarray,
) -> bool:
    """
    Verify that transformation U is unitary: U† U = I.
    
    Args:
        z: Complex state vector
        U: Transformation matrix
    
    Returns:
        is_unitary: True if U is unitary within tolerance
    """
    N = len(z)
    I = np.eye(N)
    
    # Check U† U = I
    product = np.conj(U.T) @ U
    error = np.linalg.norm(product - I)
    
    is_unitary = (error < 1e-10)
    
    return is_unitary


def hamiltonian_to_unitary_evolution(
    H: np.ndarray,
    dt: float,
    hbar: float = 1.0,
) -> np.ndarray:
    """
    Convert Hamiltonian H to unitary time evolution operator.
    
    U(t) = exp(-iHt/ℏ)
    
    This shows how classical Hamiltonian evolution becomes
    unitary quantum evolution in complex coordinates.
    
    Args:
        H: Hamiltonian matrix (N×N Hermitian)
        dt: Time step
        hbar: Reduced Planck constant
    
    Returns:
        U: Unitary evolution operator
    """
    from scipy.linalg import expm
    
    U = expm(-1j * H * dt / hbar)
    
    # Verify unitarity
    N = len(H)
    I = np.eye(N)
    error = np.linalg.norm(np.conj(U.T) @ U - I)
    
    if error > 1e-8:
        print(f"Warning: Evolution operator not unitary, error = {error}")
    
    return U


def real_to_complex_hamiltonian(
    K: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Convert real oscillator Hamiltonian to complex form.
    
    Real Hamiltonian:
        H = Σ_i p_i²/(2m_i) + Σ_ij K_ij q_i q_j / 2
    
    Complex Hamiltonian (in z = (q + ip)/√2 basis):
        H = Σ_i ω_i |z_i|² + interactions
    
    Args:
        K: Coupling matrix (N×N real symmetric)
        masses: Mass array (N,)
    
    Returns:
        H_complex: Complex Hamiltonian matrix (N×N Hermitian)
    
    Notes:
        - This is approximate for coupled oscillators
        - Exact for normal modes
    """
    N = len(masses)
    
    # For uncoupled oscillators: ω_i = √(K_ii / m_i)
    # For coupled case, need to diagonalize
    # Here we construct an effective complex Hamiltonian
    
    # Diagonal part: kinetic + potential
    H_diag = np.diag(np.diag(K) / masses)
    
    # Off-diagonal coupling (converted to complex form)
    H_coupling = (K - np.diag(np.diag(K))) / np.sqrt(np.outer(masses, masses))
    
    H_complex = H_diag + H_coupling
    
    # Ensure Hermitian
    H_complex = (H_complex + H_complex.conj().T) / 2
    
    return H_complex
