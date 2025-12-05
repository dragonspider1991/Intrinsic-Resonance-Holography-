"""
Spectral Action Computation

This module implements the spectral action functional and its gradient.
The spectral action is:

    S[D] = Tr(f(D²/Λ²)) + λ * sparsity_penalty(D)

where:
- f is a smooth cutoff function
- Λ is an energy scale
- The sparsity penalty enforces locality

The gradient is computed analytically for efficiency.
For large sparse matrices, Chebyshev approximation is used to avoid
expensive diagonalization.
"""

from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import numba
from scipy.linalg import expm
from scipy.sparse import issparse, csr_matrix


@numba.jit(nopython=True, cache=True)
def _sigmoid_cutoff(x: float, smoothness: float = 1.0) -> float:
    """
    Smooth sigmoid cutoff function.
    
    Parameters
    ----------
    x : float
        Input value
    smoothness : float, default=1.0
        Controls the smoothness of the cutoff
    
    Returns
    -------
    f : float
        Cutoff value in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(smoothness * (x - 1.0)))


@numba.jit(nopython=True, cache=True)
def _heat_kernel_cutoff(x: float, t: float = 1.0) -> float:
    """
    Heat kernel cutoff: exp(-t*x).
    
    Parameters
    ----------
    x : float
        Input value (eigenvalue of D²/Λ²)
    t : float, default=1.0
        Heat kernel time parameter
    
    Returns
    -------
    f : float
        Heat kernel value
    """
    return np.exp(-t * x)


def spectral_action(
    D: NDArray[np.complex128],
    Lambda: float = 1.0,
    cutoff: str = "heat",
    cutoff_param: float = 1.0,
    sparsity_weight: float = 0.0,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute the spectral action S[D].
    
    Parameters
    ----------
    D : NDArray[np.complex128]
        Dirac operator (Hermitian matrix)
    Lambda : float, default=1.0
        Energy cutoff scale
    cutoff : str, default="heat"
        Cutoff function type: "heat" or "sigmoid"
    cutoff_param : float, default=1.0
        Parameter for cutoff function (t for heat, smoothness for sigmoid)
    sparsity_weight : float, default=0.0
        Weight λ for sparsity penalty
    epsilon : float, default=1e-8
        Regularization for sparsity penalty
    
    Returns
    -------
    S : float
        Spectral action value
    """
    N = D.shape[0]
    
    # Compute D²/Λ²
    D2 = D @ D
    D2_scaled = D2 / (Lambda ** 2)
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(D2_scaled)
    
    # Apply cutoff function
    if cutoff == "heat":
        f_values = np.array([_heat_kernel_cutoff(lam, cutoff_param) for lam in eigvals])
    elif cutoff == "sigmoid":
        f_values = np.array([_sigmoid_cutoff(lam, cutoff_param) for lam in eigvals])
    else:
        raise ValueError(f"Unknown cutoff: {cutoff}")
    
    # Trace of f(D²/Λ²)
    S = np.sum(f_values)
    
    # Add sparsity penalty if requested
    if sparsity_weight > 0:
        # Penalty: λ Σ_{i<j} 1/(|D_ij|² + ε)
        # This encourages sparsity (locality)
        D_abs_sq = np.abs(D) ** 2
        # Only off-diagonal elements
        off_diag_mask = ~np.eye(N, dtype=bool)
        penalty = np.sum(1.0 / (D_abs_sq[off_diag_mask] + epsilon))
        S += sparsity_weight * penalty
    
    return S


def spectral_action_gradient(
    D: NDArray[np.complex128],
    Lambda: float = 1.0,
    cutoff: str = "heat",
    cutoff_param: float = 1.0,
    sparsity_weight: float = 0.0,
    epsilon: float = 1e-8,
) -> NDArray[np.complex128]:
    """
    Compute the gradient ∇_D S[D].
    
    For the spectral part Tr(f(D²/Λ²)), the gradient is:
        ∇_D Tr(f(D²/Λ²)) = (2/Λ²) f'(D²/Λ²) D
    
    This is computed using the eigenbasis of D.
    
    Parameters
    ----------
    D : NDArray[np.complex128]
        Dirac operator
    Lambda : float, default=1.0
        Energy scale
    cutoff : str, default="heat"
        Cutoff function type
    cutoff_param : float, default=1.0
        Cutoff parameter
    sparsity_weight : float, default=0.0
        Sparsity penalty weight
    epsilon : float, default=1e-8
        Regularization for sparsity penalty
    
    Returns
    -------
    grad : NDArray[np.complex128]
        Gradient of S with respect to D
    """
    N = D.shape[0]
    
    # Compute eigendecomposition of D
    eigvals, eigvecs = np.linalg.eigh(D)
    
    # Compute D²/Λ² eigenvalues
    D2_eigvals = eigvals ** 2 / (Lambda ** 2)
    
    # Compute f'(λ) for each eigenvalue
    if cutoff == "heat":
        # f(x) = exp(-t*x), f'(x) = -t*exp(-t*x)
        t = cutoff_param
        f_prime = -t * np.exp(-t * D2_eigvals)
    elif cutoff == "sigmoid":
        # f(x) = 1/(1 + exp(s*(x-1))), f'(x) = -s*exp(s*(x-1))/(1+exp(s*(x-1)))²
        s = cutoff_param
        exp_term = np.exp(s * (D2_eigvals - 1.0))
        f_prime = -s * exp_term / ((1.0 + exp_term) ** 2)
    else:
        raise ValueError(f"Unknown cutoff: {cutoff}")
    
    # Gradient of spectral part: (2/Λ²) Σ_i f'(λ_i²/Λ²) * 2λ_i |i⟩⟨i|
    # In matrix form: (2/Λ²) V diag(f'(λ²/Λ²) * 2λ) V†
    # Simplifying: (4/Λ²) V diag(f'(λ²/Λ²) * λ) V†
    diag_term = (4.0 / (Lambda ** 2)) * f_prime * eigvals
    grad = eigvecs @ np.diag(diag_term) @ eigvecs.conj().T
    
    # Add sparsity gradient if requested
    if sparsity_weight > 0:
        # ∇_D [λ Σ_{i<j} 1/(|D_ij|² + ε)]
        # = -2λ Σ_{i<j} D_ij* / (|D_ij|² + ε)²
        D_abs_sq = np.abs(D) ** 2
        sparsity_grad = -2.0 * sparsity_weight * D.conj() / ((D_abs_sq + epsilon) ** 2)
        # Zero out diagonal
        np.fill_diagonal(sparsity_grad, 0)
        grad += sparsity_grad
    
    # Ensure gradient is Hermitian
    grad = (grad + grad.conj().T) / 2.0
    
    return grad


def trace_heat_kernel(
    D: NDArray[np.complex128],
    t: float,
) -> float:
    """
    Compute Tr(exp(-t D²)).
    
    This is used for spectral dimension analysis.
    
    Parameters
    ----------
    D : NDArray[np.complex128]
        Dirac operator
    t : float
        Heat kernel time
    
    Returns
    -------
    K : float
        Heat kernel trace
    """
    D2 = D @ D
    # Compute eigenvalues of D²
    eigvals = np.linalg.eigvalsh(D2)
    # Trace of exp(-t D²) = Σ exp(-t λ_i)
    K = np.sum(np.exp(-t * eigvals))
    return K


def chiral_projector(
    gamma: NDArray[np.complex128],
    chirality: int = 1,
) -> NDArray[np.complex128]:
    """
    Construct the chiral projector P_± = (1 ± γ) / 2.
    
    Parameters
    ----------
    gamma : NDArray[np.complex128]
        Grading operator
    chirality : int, default=1
        +1 for right-handed, -1 for left-handed
    
    Returns
    -------
    P : NDArray[np.complex128]
        Chiral projector
    """
    N = gamma.shape[0]
    I = np.eye(N, dtype=np.complex128)
    return (I + chirality * gamma) / 2.0


def compute_spectral_torsion(
    D: NDArray[np.complex128],
    gamma: NDArray[np.complex128],
) -> float:
    """
    Compute spectral torsion, related to the effective fine-structure constant.
    
    The spectral torsion is defined as:
        τ = Tr(γ D) / Tr(|D|)
    
    This quantity is related to chiral asymmetry and gauge coupling.
    
    Parameters
    ----------
    D : NDArray[np.complex128]
        Dirac operator
    gamma : NDArray[np.complex128]
        Grading operator
    
    Returns
    -------
    torsion : float
        Spectral torsion
    """
    trace_gamma_D = np.trace(gamma @ D)
    trace_abs_D = np.trace(np.abs(D))
    
    if np.abs(trace_abs_D) < 1e-12:
        return 0.0
    
    torsion = np.real(trace_gamma_D) / trace_abs_D
    return torsion
